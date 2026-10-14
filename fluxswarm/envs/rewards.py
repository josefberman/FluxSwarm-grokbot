"""Multi-objective rewards: progress, energy, smoothness."""
from __future__ import annotations

import math

import torch

from fluxswarm.config import TaskConfig
from fluxswarm.physics.solver import BatchedFluidSolver


def compute_rewards(
    solver: BatchedFluidSolver,
    task: TaskConfig,
    prev_pos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (reward_matrix, objectives_matrix, scalar_reward).

    reward_matrix / objectives_matrix: (B, N, 3) — progress, energy, smooth.
    reward_matrix is weighted; objectives_matrix is unweighted.
    scalar_reward: (B,) mean over agents of sum of weighted objectives.
    """
    B, N, _ = solver.swarm.pos.shape
    device = solver.swarm.pos.device
    progress = _progress(solver, task, prev_pos)
    energy = _energy(solver)
    smooth = _smoothness(solver)

    objectives = torch.stack([progress, energy, smooth], dim=-1)  # (B,N,3)
    weights = torch.tensor(
        [task.w_progress, task.w_energy, task.w_smooth],
        device=device,
        dtype=objectives.dtype,
    )
    reward_matrix = objectives * weights
    scalar = reward_matrix.sum(dim=-1).mean(dim=-1)
    return reward_matrix, objectives, scalar


def _progress(solver: BatchedFluidSolver, task: TaskConfig, prev_pos: torch.Tensor) -> torch.Tensor:
    pos = solver.swarm.pos
    mode = task.progress_reward
    L = max(task.failure_x - task.success_x, 1e-6)

    if mode == "potential":
        dx = prev_pos[..., 0] - pos[..., 0]  # positive when moving upstream
        prog = 100.0 * dx / L
        # No clamping
        # Terminal bonuses applied by env on done; keep step shaping clean
        return prog

    if mode == "fluid-relative":
        fu = solver.last_fluid_center_u
        if fu is None:
            fu = torch.zeros(pos.shape[0], pos.shape[1], device=pos.device, dtype=pos.dtype)
        v_ref = max(abs(solver.cfg.sim.inflow_velocity), 1e-6)
        return (fu - solver.swarm.vel[..., 0]) / v_ref  # No clamping

    # legacy
    prog = torch.zeros(pos.shape[0], pos.shape[1], device=pos.device, dtype=pos.dtype)
    xs = pos[..., 0]
    mean_x = xs.mean(dim=-1, keepdim=True)
    success = mean_x <= task.success_x
    prog = torch.where(success.expand_as(prog), torch.full_like(prog, 1.0), prog)

    x_t = xs
    x_prev = prev_pos[..., 0]
    moving_left = x_t < x_prev
    delta = (x_prev - x_t) / L
    # tanh_term = torch.tanh(3.0 * delta) / math.tanh(3.0) * 100.0 - 0.1
    tanh_term = delta * 300.0
    linear_term = delta * 100.0
    step_r = torch.where(moving_left, tanh_term, linear_term)
    failed = x_t >= task.failure_x
    step_r = torch.where(failed, torch.full_like(step_r, -1.0), step_r)
    # Don't overwrite success bonus unless failure
    prog = torch.where(success.expand_as(prog) & ~failed, prog, step_r)
    # No clamping
    return prog


def _energy(solver: BatchedFluidSolver) -> torch.Tensor:
    """1 - ||a|| (actions already in unit disk)."""
    a = solver.swarm.action
    mag = torch.linalg.norm(a, dim=-1).clamp(0.0, 1.0)
    return 1.0 - mag


_SMOOTH_SIGMA = 0.25


def _smoothness(solver: BatchedFluidSolver) -> torch.Tensor:
    """r = 2 exp(-||a_t − a_{t-1}||² / (2σ²)) − 1, σ = 0.25."""
    delta = solver.swarm.action - solver.swarm.prev_action
    d2 = (delta * delta).sum(dim=-1)
    return (2.0 * torch.exp(-d2 / (2.0 * _SMOOTH_SIGMA**2)) - 1.0).clamp(-1.0, 1.0)
