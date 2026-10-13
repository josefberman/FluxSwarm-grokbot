"""Actor and multi-head critic networks for CTDE MOMAPPO."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def orthogonal_init_linear(module: nn.Module, gain: float = 1.0, bias: float = 0.0) -> None:
    """Orthogonal weight init (CleanRL PPO). No-op for non-linear modules."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, bias)


class Actor(nn.Module):
    """Shared-weight decentralized actor: local obs -> tanh-squashed Gaussian."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 2,
        hidden_sizes: tuple[int, ...] = (256, 256),
        log_std_init: float = -1.0,
        recurrent: bool = False,
    ):
        super().__init__()
        self.recurrent = recurrent
        self.action_dim = action_dim
        if recurrent:
            h0 = hidden_sizes[0]
            h1 = hidden_sizes[-1]
            self.gru = nn.GRU(obs_dim, h0, batch_first=True)
            self.torso = nn.Sequential(nn.Linear(h0, h1), nn.Tanh())
            self.mu_head = nn.Linear(h1, action_dim)
        else:
            layers: list[nn.Module] = []
            prev = obs_dim
            for h in hidden_sizes:
                layers += [nn.Linear(prev, h), nn.Tanh()]
                prev = h
            self.torso = nn.Sequential(*layers)
            self.mu_head = nn.Linear(prev, action_dim)
            self.gru = None
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)

    def apply_orthogonal_init(self) -> None:
        if self.gru is not None:
            for name, param in self.gru.named_parameters():
                if "weight" in name:
                    nn.init.orthogonal_(param, gain=1.0)
                elif "bias" in name:
                    nn.init.constant_(param, 0.0)
        for mod in self.torso.modules():
            orthogonal_init_linear(mod, gain=1.0)
        orthogonal_init_linear(self.mu_head, gain=0.01)

    def forward(self, obs: torch.Tensor, hidden: Optional[torch.Tensor] = None):
        shape = obs.shape
        flat = obs.reshape(-1, shape[-1])
        if self.recurrent and self.gru is not None:
            x = flat.unsqueeze(1)
            out, hidden = self.gru(x, hidden)
            feat = self.torso(out.squeeze(1))
        else:
            feat = self.torso(flat)
            hidden = None
        mu = self.mu_head(feat)
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp().expand_as(mu)
        mu = mu.reshape(*shape[:-1], -1)
        std = std.reshape(*shape[:-1], -1)
        return mu, std, hidden

    def dist(self, obs: torch.Tensor, hidden=None):
        mu, std, hidden = self.forward(obs, hidden)
        return Normal(mu, std), hidden


def tanh_sample(dist: Normal) -> tuple[torch.Tensor, torch.Tensor]:
    z = dist.rsample()
    action = torch.tanh(z)
    log_prob = dist.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)
    return action, log_prob


def tanh_log_prob(dist: Normal, action: torch.Tensor) -> torch.Tensor:
    a = action.clamp(-0.999, 0.999)
    z = 0.5 * torch.log((1 + a) / (1 - a))
    return dist.log_prob(z) - torch.log(1.0 - a.pow(2) + 1e-6)


class MultiHeadCritic(nn.Module):
    def __init__(
        self,
        joint_dim: int,
        num_members: int,
        hidden_sizes: tuple[int, ...] = (256, 256),
        n_objectives: int = 3,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = joint_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        self.torso = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(prev, num_members) for _ in range(n_objectives)])
        self.num_members = num_members

    def apply_orthogonal_init(self) -> None:
        for mod in self.torso.modules():
            orthogonal_init_linear(mod, gain=1.0)
        for head in self.heads:
            orthogonal_init_linear(head, gain=1.0)

    def forward(self, obs_joint: torch.Tensor) -> torch.Tensor:
        feat = self.torso(obs_joint)
        return torch.stack([h(feat) for h in self.heads], dim=1)


class SingleCritic(nn.Module):
    """Single-objective centralized critic for baseline PPO."""

    def __init__(self, joint_dim: int, num_members: int, hidden_sizes: tuple[int, ...] = (256, 256)):
        super().__init__()
        layers: list[nn.Module] = []
        prev = joint_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        self.net = nn.Sequential(*layers, nn.Linear(prev, num_members))

    def forward(self, obs_joint: torch.Tensor) -> torch.Tensor:
        return self.net(obs_joint)


class ActorCriticMO(nn.Module):
    def __init__(
        self,
        num_members: int,
        obs_local_dim: int,
        hidden_sizes: tuple[int, ...] = (256, 256),
        log_std_init: float = -1.0,
        recurrent: bool = False,
    ):
        super().__init__()
        self.num_members = num_members
        self.obs_local_dim = obs_local_dim
        self.actor = Actor(obs_local_dim, 2, hidden_sizes, log_std_init, recurrent=recurrent)
        self.critic = MultiHeadCritic(num_members * obs_local_dim, num_members, hidden_sizes)

    def apply_orthogonal_init(self) -> None:
        self.actor.apply_orthogonal_init()
        self.critic.apply_orthogonal_init()

    def actor_parameters(self):
        return self.actor.parameters()

    def critic_parameters(self):
        return self.critic.parameters()
