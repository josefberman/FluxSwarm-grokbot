"""Batched fluid–swarm solver (GPU Brinkman + DCT Poisson, two-way only)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from fluxswarm.config import Config, SimConfig
from fluxswarm.physics.domain import Domain
from fluxswarm.physics.swarm import (
    SwarmState,
    apply_force_integrate,
    pressure_force,
    project_actions_to_unit_disk,
    resolve_collisions,
    stokes_drag_2d,
    thrust_force,
)
from fluxswarm.physics.torch_fluid import TorchFluidSolver


@dataclass
class FieldSnapshot:
    vx: np.ndarray
    vy: np.ndarray
    p: np.ndarray
    t: float


class BatchedFluidSolver:
    """Batched two-way fluid–swarm solver on GPU (or CPU)."""

    def __init__(self, cfg: Config, batch: int, device: torch.device | str = "cuda"):
        self.cfg = cfg
        self.sim: SimConfig = cfg.sim
        self.batch = batch
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.domain = Domain.from_config(self.sim)
        self.swarm = SwarmState(batch, cfg.swarm, cfg.sim, self.device)
        self.torch_fluid = TorchFluidSolver(self.sim, batch, self.device)
        self.episode_time = torch.zeros(batch, dtype=torch.float64, device=self.device)
        self.ring_k = cfg.obs.ring_points
        self.angles = torch.linspace(0, 2 * math.pi, self.ring_k + 1, device=self.device, dtype=torch.float64)[:-1]
        self.last_pressure_ring: Optional[torch.Tensor] = None
        self.last_vel_ring_u: Optional[torch.Tensor] = None
        self.last_vel_ring_v: Optional[torch.Tensor] = None
        self.last_fluid_center_u: Optional[torch.Tensor] = None
        self.last_fluid_center_v: Optional[torch.Tensor] = None
        # Force breakdown from the most recent step (for diagnostics / TensorBoard).
        self.last_drag: Optional[torch.Tensor] = None
        self.last_pressure_force: Optional[torch.Tensor] = None
        self.last_thrust: Optional[torch.Tensor] = None

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.batch, device=self.device)
        self.swarm.reset_envs(env_ids, self.cfg.swarm, self.sim)
        self.episode_time[env_ids] = 0.0
        self.torch_fluid.reset(env_ids)

    def _sample_rings(self) -> None:
        """Sample pressure and velocity on rings in the fluid around each disc."""
        delta = min(self.sim.dx, self.sim.dy)
        ly, lx = self.sim.length_y, self.sim.length_x
        r_agent = self.swarm.radii
        r = torch.maximum(r_agent * self.cfg.obs.ring_radius_factor, r_agent + delta)
        cx = self.swarm.pos[..., 0:1]
        cy = self.swarm.pos[..., 1:2]
        sx = cx + r.unsqueeze(-1) * torch.cos(self.angles)
        sy = cy + r.unsqueeze(-1) * torch.sin(self.angles)
        r_min = (r_agent + delta).unsqueeze(-1)
        for _ in range(2):
            sy = sy.clamp(delta, ly - delta)
            dx = sx - cx
            dy = sy - cy
            dist = torch.sqrt(dx * dx + dy * dy).clamp(min=1e-12)
            scale = torch.clamp(r_min / dist, min=1.0)
            sx = cx + dx * scale
            sy = cy + dy * scale
        sy = sy.clamp(delta, ly - delta)
        sx = sx.clamp(0.0, lx)
        p, u, v = self.torch_fluid.sample_at(sx, sy)
        self.last_pressure_ring = p
        self.last_vel_ring_u = u
        self.last_vel_ring_v = v
        self.last_fluid_center_u = u.mean(dim=-1)
        self.last_fluid_center_v = v.mean(dim=-1)

    def step(self, actions: torch.Tensor) -> None:
        """One RL step: swarm forces then fluid substeps. actions: (B, N, 2)."""
        actions = project_actions_to_unit_disk(actions.to(self.device, dtype=torch.float64))
        # Hidden −x override on the first RL step. Gated on the trainer flag so
        # `--no-action-x-prior` is actually off-policy-free (log-probs match).
        if self.cfg.train.use_action_x_prior:
            early = self.episode_time <= 0.005
            if bool(torch.as_tensor(early).any()):
                forced = actions.clone()
                forced[early] = torch.tensor([-1.0, 0.0], device=self.device, dtype=torch.float64)
                actions = forced

        self.swarm.prev_action = self.swarm.action.clone()
        self.swarm.action = actions

        t0 = self.episode_time
        if float(t0.min()) > 0:
            self._sample_rings()
        else:
            B, N, k = self.batch, self.swarm.n, self.ring_k
            z = torch.zeros(B, N, k, device=self.device, dtype=torch.float64)
            self.last_pressure_ring = z
            self.last_vel_ring_u = z
            self.last_vel_ring_v = z
            self.last_fluid_center_u = z.mean(-1)
            self.last_fluid_center_v = z.mean(-1)

        drag = stokes_drag_2d(
            self.swarm.vel,
            self.last_fluid_center_u,
            self.last_fluid_center_v,
            mu=self.sim.viscosity,
        )
        pforce = pressure_force(self.last_pressure_ring, self.swarm.radii, self.angles)
        thrust = thrust_force(actions, self.swarm.max_forces)
        total = drag + pforce + thrust

        self.last_drag = drag
        self.last_pressure_force = pforce
        self.last_thrust = thrust

        dt = self.sim.dt
        max_thrust = float(self.swarm.max_force)
        max_force_mag = 5.0 * max_thrust
        v_max = 3.0 * abs(self.sim.inflow_velocity)
        pos, vel = apply_force_integrate(
            self.swarm.pos,
            self.swarm.vel,
            total,
            self.swarm.masses,
            self.swarm.radii,
            self.domain,
            dt,
            max_force_mag=max_force_mag,
            v_max=v_max,
        )
        pos, vel = resolve_collisions(pos, vel, self.swarm.radii, self.swarm.masses, self.domain)
        self.swarm.pos, self.swarm.vel = pos, vel

        self._update_imu(total)

        Re_inv = self.sim.viscosity / max(self.sim.inflow_velocity * self.sim.length_y, 1e-8)
        self.torch_fluid.substep_loop(
            t0=float(self.episode_time.mean().item()),
            dt=dt,
            n_sub=self.sim.substeps,
            pos=self.swarm.pos,
            vel=self.swarm.vel,
            radii=self.swarm.radii,
            nu=Re_inv,
        )
        self.episode_time = self.episode_time + dt
        self._sample_rings()

    def _update_imu(self, force: torch.Tensor) -> None:
        dt = self.sim.dt
        true_accel = force / self.swarm.masses.unsqueeze(-1)
        if self.cfg.obs.localization == "imu":
            noise = torch.randn_like(true_accel) * self.cfg.obs.imu_noise
            measured = true_accel + self.swarm.imu_bias + noise
        else:
            measured = true_accel
        self.swarm.imu_vel = self.swarm.imu_vel + measured * dt
        self.swarm.imu_pos = self.swarm.imu_pos + self.swarm.imu_vel * dt

    def reset_imu_bias(self, env_ids: torch.Tensor) -> None:
        b = self.cfg.obs.imu_bias
        for i in env_ids.tolist():
            self.swarm.imu_bias[i] = torch.randn(self.swarm.n, 2, device=self.device, dtype=torch.float64) * b
            self.swarm.imu_vel[i] = 0
            self.swarm.imu_pos[i] = 0

    def export_fields(self, env_index: int = 0) -> FieldSnapshot:
        """Export numpy field arrays for one env."""
        vx, vy, p = self.torch_fluid.export(env_index)
        return FieldSnapshot(vx=vx, vy=vy, p=p, t=float(self.episode_time[env_index].item()))
