"""Cheap point-mass dummy env with the BatchedSwarmEnv step API.

Used to isolate trainer bugs from the fluid solver. Agents are drifted
downstream and must thrust in −x to succeed.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from fluxswarm.config import Config


class _DummySwarm:
    def __init__(self, batch: int, n: int, device: torch.device, start_x: float, length_y: float):
        self.pos = torch.zeros(batch, n, 2, device=device, dtype=torch.float32)
        self.vel = torch.zeros(batch, n, 2, device=device, dtype=torch.float32)
        self.action = torch.zeros(batch, n, 2, device=device, dtype=torch.float32)
        self.prev_action = torch.zeros(batch, n, 2, device=device, dtype=torch.float32)
        self.start_x = start_x
        self.length_y = length_y
        self.n = n
        self._layout_y = torch.linspace(0.25, length_y - 0.25, n, device=device)

    def reset_all(self) -> None:
        self.pos[..., 0] = self.start_x
        self.pos[..., 1] = self._layout_y
        self.vel.zero_()
        self.action.zero_()
        self.prev_action.zero_()

    def reset_ids(self, ids: torch.Tensor) -> None:
        self.pos[ids, :, 0] = self.start_x
        self.pos[ids, :, 1] = self._layout_y
        self.vel[ids] = 0
        self.action[ids] = 0
        self.prev_action[ids] = 0


class _DummySolver:
    def __init__(self, swarm: _DummySwarm, batch: int, device: torch.device):
        self.swarm = swarm
        self.episode_time = torch.zeros(batch, device=device, dtype=torch.float32)
        self.last_drag = None
        self.last_pressure_force = None
        self.last_thrust = None
        self.torch_fluid = None


class DummyUpstreamEnv:
    """Vector env: (B, N) point masses, 6-D local obs (pos, vel, prev action)."""

    def __init__(
        self,
        cfg: Config,
        batch: Optional[int] = None,
        device: Optional[str] = None,
        n_agents: Optional[int] = None,
        drift: float = 0.4,
        speed: float = 1.5,
        dt: float = 1.0,
        max_steps: int = 80,
    ):
        self.cfg = cfg
        self.batch = batch or cfg.train.batch_envs
        self.n = int(n_agents or min(cfg.swarm.num_members, 4))
        self.device = torch.device(device or "cpu")
        self.obs_dim = 6
        self.drift = float(drift)
        self.speed = float(speed)
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.success_x = float(cfg.task.success_x)
        self.failure_x = float(cfg.task.failure_x)
        self.start_x = 0.5 * (self.success_x + self.failure_x)
        self.length_y = float(cfg.sim.length_y)
        self.w = torch.tensor(
            [cfg.task.w_progress, cfg.task.w_energy, cfg.task.w_smooth],
            device=self.device,
            dtype=torch.float32,
        )
        swarm = _DummySwarm(self.batch, self.n, self.device, self.start_x, self.length_y)
        self.solver = _DummySolver(swarm, self.batch, self.device)
        self.episode_steps = torch.zeros(self.batch, dtype=torch.int64, device=self.device)
        self.episode_id = torch.zeros(self.batch, dtype=torch.int64, device=self.device)
        self.cum_objectives = torch.zeros(self.batch, 3, dtype=torch.float64, device=self.device)
        self.single_action_space_shape = (self.n, 2)
        self.single_observation_space_shape = (self.n, self.obs_dim)

    def _obs(self) -> torch.Tensor:
        s = self.solver.swarm
        return torch.cat([s.pos, s.vel, s.prev_action], dim=-1)

    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.solver.swarm.reset_all()
        self.solver.episode_time.zero_()
        self.episode_steps.zero_()
        self.cum_objectives.zero_()
        return self._obs().cpu().numpy(), {}

    def reset_done(self, done_mask: torch.Tensor) -> None:
        ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() == 0:
            return
        self.solver.swarm.reset_ids(ids)
        self.solver.episode_time[ids] = 0
        self.episode_steps[ids] = 0
        self.cum_objectives[ids] = 0
        self.episode_id[ids] += 1

    def get_positions(self) -> np.ndarray:
        return self.solver.swarm.pos.detach().cpu().numpy()

    def get_velocities(self) -> np.ndarray:
        return self.solver.swarm.vel.detach().cpu().numpy()

    def get_actions(self) -> np.ndarray:
        return self.solver.swarm.action.detach().cpu().numpy()

    def step(self, actions: np.ndarray | torch.Tensor) -> tuple:
        if isinstance(actions, np.ndarray):
            actions = torch.tensor(actions, device=self.device, dtype=torch.float32)
        else:
            actions = actions.to(self.device, dtype=torch.float32)
        norms = torch.linalg.norm(actions, dim=-1, keepdim=True).clamp_min(1e-8)
        actions = torch.where(norms > 1.0, actions / norms, actions)

        s = self.solver.swarm
        prev_pos = s.pos.clone()
        s.prev_action = s.action.clone()
        s.action = actions
        vel = torch.zeros_like(s.pos)
        vel[..., 0] = actions[..., 0] * self.speed + self.drift
        vel[..., 1] = actions[..., 1] * self.speed
        s.vel = vel
        s.pos = s.pos + vel * self.dt
        s.pos[..., 1] = s.pos[..., 1].clamp(0.05, self.length_y - 0.05)

        xs = s.pos[..., 0]
        dx = prev_pos[..., 0] - xs
        moving_left = dx > 0
        progress = torch.where(moving_left, torch.ones_like(dx), torch.full_like(dx, -0.1))
        mag = torch.linalg.norm(actions, dim=-1).clamp(0.0, 1.0)
        energy = 1.0 - mag
        delta = actions - s.prev_action
        d2 = (delta * delta).sum(dim=-1)
        smooth = (2.0 * torch.exp(-d2 / (2.0 * 0.25**2)) - 1.0).clamp(-1.0, 1.0)
        objectives = torch.stack([progress, energy, smooth], dim=-1)
        reward_matrix = objectives * self.w
        scalar = reward_matrix.sum(dim=-1).mean(dim=-1)

        self.cum_objectives = self.cum_objectives + objectives.mean(dim=1).double()
        self.episode_steps += 1
        self.solver.episode_time = self.solver.episode_time + self.dt

        mean_x = xs.mean(dim=-1)
        success = mean_x <= self.success_x
        any_fail = (xs > self.failure_x).any(dim=-1)
        terminations = success | any_fail
        truncations = (self.episode_steps >= self.max_steps) & ~terminations
        done = terminations | truncations

        infos: list[dict[str, Any]] = []
        for b in range(self.batch):
            reason = ""
            if success[b]:
                reason = "success"
            elif any_fail[b]:
                reason = "failure"
            elif truncations[b]:
                reason = "timeout"
            info = {
                "reward_matrix": reward_matrix[b].detach().cpu().numpy().astype(np.float32),
                "objectives_matrix": objectives[b].detach().cpu().numpy(),
                "objectives": {
                    "progress": float(objectives[b, :, 0].mean()),
                    "energy_efficiency": float(objectives[b, :, 1].mean()),
                    "smoothness": float(objectives[b, :, 2].mean()),
                },
                "episode_steps": int(self.episode_steps[b]),
                "episode_id": int(self.episode_id[b]),
                "termination_reason": reason if done[b] else None,
            }
            if done[b]:
                info["episode"] = {
                    "steps": int(self.episode_steps[b]),
                    "cum_progress": float(self.cum_objectives[b, 0]),
                    "cum_energy_efficiency": float(self.cum_objectives[b, 1]),
                    "cum_smoothness": float(self.cum_objectives[b, 2]),
                    "status": "terminated" if terminations[b] else "truncated",
                    "reason": reason,
                }
            infos.append(info)

        self.reset_done(done)
        obs = self._obs()
        return (
            obs.detach().cpu().numpy(),
            scalar.detach().cpu().numpy().astype(np.float32),
            terminations.detach().cpu().numpy(),
            truncations.detach().cpu().numpy(),
            {
                "reward_matrix": reward_matrix.detach().cpu().numpy().astype(np.float32),
                "objectives_matrix": objectives.detach().cpu().numpy(),
                "infos": infos,
            },
        )
