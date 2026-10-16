"""Training-stability helpers: success metrics, return normalization, LR schedule."""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class SuccessTracker:
    """Track episode outcomes and expose several success-rate estimators.

    ``window_rate`` reproduces the historical ``objectives/success_rate``: the
    mean over the last ``window`` completed episodes, counting every source.
    The other estimators exist because that number is noisy when episodes end
    in synchronized bursts or when short (failure) episodes crowd the window.

    * ``ema`` – episode-count-weighted EMA across PPO updates:
      ``num = beta*num + successes``; ``den = beta*den + episodes``. Updates
      without finished episodes leave it unchanged instead of injecting zeros.
    * ``update_rate`` – success over episodes finished in the current update.
    * ``train_window_rate`` – window mean over stochastic training envs only
      (excludes the deterministic CPU-mirror env).
    """

    def __init__(self, window: int, ema_beta: float = 0.9, min_episodes: Optional[int] = None):
        self.window = int(window)
        self.min_episodes = int(min_episodes if min_episodes is not None else max(1, window // 4))
        self.ema_beta = float(ema_beta)
        self._all: deque[float] = deque(maxlen=self.window)
        self._train: deque[float] = deque(maxlen=self.window)
        self._mirror: deque[float] = deque(maxlen=self.window)
        self.total = 0
        self._upd_succ = 0.0
        self._upd_n = 0
        self._ema_num = 0.0
        self._ema_den = 0.0
        self.last_update_rate: Optional[float] = None
        self.last_update_n = 0

    def add(self, success: bool, source: str = "train") -> None:
        v = 1.0 if success else 0.0
        self._all.append(v)
        self.total += 1
        if source == "mirror":
            self._mirror.append(v)
        else:
            self._train.append(v)
            self._upd_succ += v
            self._upd_n += 1

    @property
    def ready(self) -> bool:
        return self.total >= self.min_episodes

    @property
    def episodes_in_window(self) -> int:
        return len(self._all)

    def window_rate(self) -> float:
        return float(np.mean(self._all)) if self._all else 0.0

    def train_window_rate(self) -> float:
        return float(np.mean(self._train)) if self._train else 0.0

    def mirror_window_rate(self) -> Optional[float]:
        return float(np.mean(self._mirror)) if self._mirror else None

    def end_update(self) -> None:
        """Close a PPO update: fold this update's train episodes into the EMA."""
        b = self.ema_beta
        self._ema_num = b * self._ema_num + self._upd_succ
        self._ema_den = b * self._ema_den + self._upd_n
        self.last_update_n = self._upd_n
        self.last_update_rate = (self._upd_succ / self._upd_n) if self._upd_n else None
        self._upd_succ = 0.0
        self._upd_n = 0

    def ema(self) -> Optional[float]:
        return (self._ema_num / self._ema_den) if self._ema_den > 0 else None


class ReturnNormalizer:
    """Per-objective running mean/std of returns (PopArt-lite).

    The critic is trained on normalized targets; values are de-normalized for
    GAE. With ``preserve_outputs`` the critic heads are rescaled whenever the
    statistics move so de-normalized predictions are unchanged (PopArt's
    "preserving outputs precisely"). Heads output one value per member, so
    every row of head ``k`` shares objective ``k``'s statistics.
    """

    def __init__(self, n_obj: int, device: torch.device, beta: float = 0.0, eps: float = 1e-4):
        self.mean = torch.zeros(n_obj, device=device)
        self.var = torch.ones(n_obj, device=device)
        self.count = eps
        self.beta = float(beta)  # 0 => exact running (Welford); else EMA weight on old stats

    @property
    def std(self) -> torch.Tensor:
        return self.var.clamp_min(1e-8).sqrt()

    def update(self, returns: torch.Tensor, heads: Optional[nn.ModuleList] = None) -> None:
        """returns: (..., n_obj)."""
        flat = returns.reshape(-1, returns.shape[-1]).float()
        old_mean, old_std = self.mean.clone(), self.std.clone()
        bm = flat.mean(0)
        bv = flat.var(0, unbiased=False)
        if self.beta > 0:
            self.mean = self.beta * self.mean + (1 - self.beta) * bm
            sq = self.beta * (self.var + old_mean.pow(2)) + (1 - self.beta) * (bv + bm.pow(2))
            self.var = (sq - self.mean.pow(2)).clamp_min(1e-8)
        else:
            n = flat.shape[0]
            delta = bm - self.mean
            tot = self.count + n
            self.mean = self.mean + delta * n / tot
            m_a = self.var * self.count
            m_b = bv * n
            self.var = (m_a + m_b + delta.pow(2) * self.count * n / tot) / tot
            self.count = tot
        if heads is not None:
            new_std = self.std
            with torch.no_grad():
                for k, head in enumerate(heads):
                    head.weight.mul_(old_std[k] / new_std[k])
                    head.bias.mul_(old_std[k]).add_(old_mean[k] - self.mean[k]).div_(new_std[k])

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., n_obj) with objectives on the last axis."""
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean

    def state_dict(self) -> dict:
        return {"mean": self.mean.cpu(), "var": self.var.cpu(), "count": float(self.count)}

    def load_state_dict(self, state: dict, device: torch.device) -> None:
        self.mean = state["mean"].to(device)
        self.var = state["var"].to(device)
        self.count = float(state["count"])


def lr_at(update: int, num_updates: int, base_lr: float, schedule: str, final_fraction: float) -> float:
    """Learning rate for PPO update ``update`` (0-based)."""
    if schedule == "constant" or num_updates <= 1:
        return base_lr
    frac = min(1.0, update / max(1, num_updates - 1))
    if schedule == "linear":
        return base_lr * (1.0 - frac * (1.0 - final_fraction))
    if schedule == "cosine":
        return base_lr * (final_fraction + (1 - final_fraction) * 0.5 * (1 + np.cos(np.pi * frac)))
    raise ValueError(f"unknown lr schedule {schedule!r}")
