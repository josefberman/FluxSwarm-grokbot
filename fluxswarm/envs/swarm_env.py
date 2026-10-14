"""Batched swarm-in-fluid Gymnasium-style environment."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from fluxswarm.config import Config
from fluxswarm.envs.observations import ObservationBuilder, obs_dim_stacked
from fluxswarm.envs.rewards import compute_rewards
from fluxswarm.physics.solver import BatchedFluidSolver


class BatchedSwarmEnv:
    """Vector-native env: steps all batch_envs in one GPU fluid call.

    API mirrors gymnasium VectorEnv enough for our trainers:
      reset() -> obs (B,N,D), info
      step(actions) -> obs, rewards, terminations, truncations, infos
    """

    def __init__(self, cfg: Config, batch: Optional[int] = None, device: Optional[str] = None):
        self.cfg = cfg
        self.batch = batch or cfg.train.batch_envs
        self.n = cfg.swarm.num_members
        dev = device or cfg.train.device
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"
        self.device = torch.device(dev)
        self.solver = BatchedFluidSolver(cfg, self.batch, self.device)
        self.obs_builder = ObservationBuilder(cfg.obs, self.n, self.batch, self.device)
        self.obs_dim = obs_dim_stacked(cfg.obs, self.n)
        self.episode_steps = torch.zeros(self.batch, dtype=torch.int64, device=self.device)
        self.episode_id = torch.zeros(self.batch, dtype=torch.int64, device=self.device)
        self.cum_objectives = torch.zeros(self.batch, 3, dtype=torch.float64, device=self.device)
        self._prev_pos = self.solver.swarm.pos.clone()
        self.single_action_space_shape = (self.n, 2)
        self.single_observation_space_shape = (self.n, self.obs_dim)

    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.solver.reset()
        if self.cfg.obs.localization == "imu":
            self.solver.reset_imu_bias(torch.arange(self.batch, device=self.device))
        self.obs_builder.reset()
        self.episode_steps.zero_()
        self.cum_objectives.zero_()
        self._prev_pos = self.solver.swarm.pos.clone()
        obs = self.obs_builder.build(self.solver)
        return obs.detach().cpu().numpy(), {}

    def reset_done(self, done_mask: torch.Tensor) -> None:
        ids = done_mask.nonzero(as_tuple=False).squeeze(-1)
        if ids.numel() == 0:
            return
        self.solver.reset(ids)
        if self.cfg.obs.localization == "imu":
            self.solver.reset_imu_bias(ids)
        self.episode_steps[ids] = 0
        self.cum_objectives[ids] = 0
        self.episode_id[ids] += 1
        self._prev_pos[ids] = self.solver.swarm.pos[ids]
        self.obs_builder.reset_envs(ids)

    def step(self, actions: np.ndarray | torch.Tensor) -> tuple:
        if isinstance(actions, np.ndarray):
            actions = torch.tensor(actions, device=self.device, dtype=torch.float64)
        else:
            actions = actions.to(self.device, dtype=torch.float64)

        self._prev_pos = self.solver.swarm.pos.clone()
        self.solver.step(actions)

        reward_matrix, objectives_matrix, scalar = compute_rewards(
            self.solver, self.cfg.task, self._prev_pos
        )
        self.cum_objectives = self.cum_objectives + objectives_matrix.mean(dim=1)
        self.episode_steps += 1

        terminations, truncations, reasons = self._done_flags()
        # Terminal progress bonuses for potential mode
        if self.cfg.task.progress_reward == "potential":
            success = torch.tensor([r == "success" for r in reasons], device=reward_matrix.device)
            failure = torch.tensor([r == "failure" for r in reasons], device=reward_matrix.device)
            if success.any():
                reward_matrix[success, :, 0] += self.cfg.task.w_progress * 10.0
            if failure.any():
                reward_matrix[failure, :, 0] -= self.cfg.task.w_progress * 10.0
            scalar = reward_matrix.sum(dim=-1).mean(dim=-1)

        done = terminations | truncations

        infos: list[dict[str, Any]] = []
        for b in range(self.batch):
            info = {
                "reward_matrix": reward_matrix[b].detach().cpu().numpy().astype(np.float32),
                "objectives_matrix": objectives_matrix[b].detach().cpu().numpy(),
                "objectives": {
                    "progress": float(objectives_matrix[b, :, 0].mean()),
                    "energy_efficiency": float(objectives_matrix[b, :, 1].mean()),
                    "smoothness": float(objectives_matrix[b, :, 2].mean()),
                },
                "episode_steps": int(self.episode_steps[b]),
                "episode_id": int(self.episode_id[b]),
                "termination_reason": reasons[b] if done[b] else None,
            }
            if done[b]:
                info["episode"] = {
                    "steps": int(self.episode_steps[b]),
                    "cum_progress": float(self.cum_objectives[b, 0]),
                    "cum_energy_efficiency": float(self.cum_objectives[b, 1]),
                    "cum_smoothness": float(self.cum_objectives[b, 2]),
                    "status": "terminated" if terminations[b] else "truncated",
                    "reason": reasons[b],
                }
            infos.append(info)

        # Auto-reset done envs, then observe the *current* state. Returning the
        # terminal snapshot after reset would make the next policy action act on
        # the previous episode while the simulator is already in the next one.
        self.reset_done(done)
        obs = self.obs_builder.build(self.solver)

        return (
            obs.detach().cpu().numpy(),
            scalar.detach().cpu().numpy().astype(np.float32),
            terminations.detach().cpu().numpy(),
            truncations.detach().cpu().numpy(),
            {
                "reward_matrix": reward_matrix.detach().cpu().numpy().astype(np.float32),
                "objectives_matrix": objectives_matrix.detach().cpu().numpy(),
                "infos": infos,
            },
        )

    def _done_flags(self) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        xs = self.solver.swarm.pos[..., 0]
        mean_x = xs.mean(dim=-1)
        any_fail = (xs > self.cfg.task.failure_x).any(dim=-1)
        success = mean_x <= self.cfg.task.success_x
        terminations = success | any_fail
        truncations = self.solver.episode_time > self.cfg.task.episode_duration
        # Don't double-count
        truncations = truncations & ~terminations
        reasons = []
        for b in range(self.batch):
            if success[b]:
                reasons.append("success")
            elif any_fail[b]:
                reasons.append("failure")
            elif truncations[b]:
                reasons.append("timeout")
            else:
                reasons.append("")
        return terminations, truncations, reasons

    def get_positions(self) -> np.ndarray:
        return self.solver.swarm.pos.detach().cpu().numpy()

    def get_velocities(self) -> np.ndarray:
        return self.solver.swarm.vel.detach().cpu().numpy()

    def get_actions(self) -> np.ndarray:
        return self.solver.swarm.action.detach().cpu().numpy()
