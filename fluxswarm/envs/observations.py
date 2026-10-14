"""Position-blind local observations and localization ladder."""
from __future__ import annotations

from collections import deque

import torch

from fluxswarm.config import ObsConfig
from fluxswarm.physics.solver import BatchedFluidSolver


def obs_dim(cfg: ObsConfig, num_members: int) -> int:
    """Per-agent observation dimension (single frame, before history stack)."""
    if cfg.preset == "legacy":
        # legacy = 8 means the observation vector is 8 dimensional, which is the position, velocity, and pressure
        return 8
    k = cfg.ring_points
    nk = max(0, cfg.neighbor_k)
    # base = k (ring pressure) + 2*k (ring velocities) +
    #        neighbor_k*4 (neighbor Δpos, Δvel) + 2 (prev action)
    base = k + 2 * k + nk * 4 + 2
    if cfg.localization == "none":
        loc = 0
    elif cfg.localization == "absolute-y":
        loc = 1
    elif cfg.localization in ("displacement", "imu", "full"):
        # displacement/imu: dead-reckoned (Δpos, Δvel); full: lab (pos, vel)
        loc = 4
    else:
        loc = 0
    return base + loc


def obs_dim_stacked(cfg: ObsConfig, num_members: int) -> int:
    if cfg.preset == "legacy":
        return 8
    return obs_dim(cfg, num_members) * max(1, cfg.history)


class ObservationBuilder:
    def __init__(self, cfg: ObsConfig, num_members: int, batch: int, device: torch.device):
        self.cfg = cfg
        self.n = num_members
        self.batch = batch
        self.device = device
        self.frame_dim = obs_dim(cfg, num_members)
        self.dim = obs_dim_stacked(cfg, num_members)
        hist = 1 if cfg.preset == "legacy" else max(1, cfg.history)
        self._history: deque[torch.Tensor] = deque(maxlen=hist)
        self._hist_len = hist

    def reset(self) -> None:
        self._history.clear()

    def reset_envs(self, ids: torch.Tensor) -> None:
        """Zero stacked frames for auto-reset env indices so history does not leak."""
        if ids is None or ids.numel() == 0 or not self._history:
            return
        for frame in self._history:
            frame[ids] = 0

    def build(self, solver: BatchedFluidSolver) -> torch.Tensor:
        if self.cfg.preset == "legacy":
            return self._legacy(solver).float()

        frame = self._rich_frame(solver)
        self._history.append(frame)
        while len(self._history) < self._hist_len:
            self._history.appendleft(torch.zeros_like(frame))
        return torch.cat(list(self._history), dim=-1).float()

    def _legacy(self, solver: BatchedFluidSolver) -> torch.Tensor:
        pos = solver.swarm.pos
        vel = solver.swarm.vel
        pr = solver.last_pressure_ring
        if pr is None:
            pr = torch.zeros(self.batch, self.n, 4, device=self.device, dtype=torch.float64)
        else:
            pr = pr[..., :4]
        return torch.cat([pos, vel, pr], dim=-1)

    def _rich_frame(self, solver: BatchedFluidSolver) -> torch.Tensor:
        cfg = self.cfg
        B, N = self.batch, self.n
        pos = solver.swarm.pos
        vel = solver.swarm.vel

        pr = solver.last_pressure_ring
        vu = solver.last_vel_ring_u
        vv = solver.last_vel_ring_v
        k = cfg.ring_points
        if pr is None:
            pr = torch.zeros(B, N, k, device=self.device, dtype=torch.float64)
            vu = torch.zeros_like(pr)
            vv = torch.zeros_like(pr)
        ring_u = vu - vel[..., 0:1]
        ring_v = vv - vel[..., 1:2]
        ring_rel = torch.stack([ring_u, ring_v], dim=-1).reshape(B, N, 2 * k)

        neigh = self._neighbors(pos, vel, cfg.neighbor_k, cfg.neighbor_radius)
        prev_a = solver.swarm.prev_action
        parts = [pr, ring_rel, neigh, prev_a]

        loc = cfg.localization
        if loc == "absolute-y":
            parts.append(pos[..., 1:2])
        elif loc in ("displacement", "imu"):
            # displacement = IMU integrator without bias/noise; imu adds both
            parts.append(solver.swarm.imu_pos)
            parts.append(solver.swarm.imu_vel)
        elif loc == "full":
            parts.append(pos)
            parts.append(vel)

        return torch.cat(parts, dim=-1)

    def _neighbors(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        k: int,
        radius: float,
    ) -> torch.Tensor:
        B, N, _ = pos.shape
        # rel[b,i,j] = pos[j] - pos[i]
        rel = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = torch.linalg.norm(rel, dim=-1)
        eye = torch.eye(N, device=pos.device, dtype=torch.bool).unsqueeze(0)
        dist = dist.masked_fill(eye, float("inf"))
        k_take = min(k, max(N - 1, 0))
        if k_take == 0:
            return torch.zeros(B, N, k * 4, device=pos.device, dtype=pos.dtype)
        knn_dist, knn_idx = dist.topk(k_take, dim=-1, largest=False)
        # Pad if N-1 < k (self is excluded, so k == N leaves one masked slot)
        if knn_idx.shape[-1] < k:
            pad = k - knn_idx.shape[-1]
            knn_idx = torch.cat([knn_idx, knn_idx[..., :1].expand(*knn_idx.shape[:-1], pad)], dim=-1)
            knn_dist = torch.cat(
                [knn_dist, torch.full((*knn_dist.shape[:-1], pad), float("inf"), device=pos.device)],
                dim=-1,
            )
        batch_idx = torch.arange(B, device=pos.device).view(B, 1, 1).expand(B, N, k)
        member_idx = torch.arange(N, device=pos.device).view(1, N, 1).expand(B, N, k)
        rel_pos = rel[batch_idx, member_idx, knn_idx]
        vel_j = vel[batch_idx, knn_idx]
        rel_vel = vel_j - vel.unsqueeze(2)
        mask = (knn_dist <= radius).unsqueeze(-1)
        feat = torch.cat([rel_pos, rel_vel], dim=-1) * mask.to(rel_pos.dtype)
        return feat.reshape(B, N, k * 4)
