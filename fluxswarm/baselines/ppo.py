"""Standard single-objective PPO baseline (comparison only)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from fluxswarm.agents.momappo import RunningNorm, RolloutBuffer, blend_preset_x, compute_gae
from fluxswarm.agents.networks import Actor, SingleCritic, tanh_log_prob, tanh_sample
from fluxswarm.config import Config
from fluxswarm.envs.swarm_env import BatchedSwarmEnv
from fluxswarm.runs.recorder import RunRecorder


class ActorCriticPPO(nn.Module):
    def __init__(self, num_members, obs_dim, hidden_sizes=(256, 128), log_std_init=-1.0):
        super().__init__()
        self.actor = Actor(obs_dim, 2, hidden_sizes, log_std_init)
        self.critic = SingleCritic(num_members * obs_dim, num_members, hidden_sizes)
        self.num_members = num_members
        self.obs_dim = obs_dim


def train_ppo_baseline(
    cfg: Config,
    env: Optional[BatchedSwarmEnv] = None,
    recorder: Optional[RunRecorder] = None,
) -> Path:
    """PPO on scalarized reward; same actor, single value head, no PCGrad."""
    device = torch.device(
        cfg.train.device if torch.cuda.is_available() and cfg.train.device == "cuda" else "cpu"
    )
    cfg = Config.from_dict(cfg.to_dict())
    cfg.run.tag = cfg.run.tag or "ppo"
    if env is None:
        env = BatchedSwarmEnv(cfg, device=str(device))
    if recorder is None:
        recorder = RunRecorder(cfg, algorithm="ppo")
    recorder.save_config(cfg)

    B, N, obs_dim = env.batch, env.n, env.obs_dim
    model = ActorCriticPPO(N, obs_dim, cfg.train.hidden_sizes, cfg.train.log_std_init).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)
    obs_norm = RunningNorm(obs_dim, device)
    writer = SummaryWriter(log_dir=str(recorder.run_dir / "tb"))

    models_dir = recorder.run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    obs_np, _ = env.reset(seed=cfg.train.seed)
    obs = torch.tensor(obs_np, device=device)
    obs_norm.update(obs)
    obs = obs_norm.normalize(obs)

    n_steps = cfg.train.n_steps
    total = cfg.train.total_timesteps_per_env * B
    num_updates = max(1, total // (n_steps * B))
    global_steps = 0
    w = torch.tensor(
        [cfg.task.w_progress, cfg.task.w_energy, cfg.task.w_smooth],
        device=device,
        dtype=torch.float32,
    )

    pbar = trange(num_updates, desc="PPO-baseline")
    for update in pbar:
        # Reuse MO buffer but store scalarized reward in channel 0
        buf = RolloutBuffer(n_steps, B, N, obs_dim, device)
        for t in range(n_steps):
            with torch.no_grad():
                dist, _ = model.actor.dist(obs)
                action, logp_pre = tanh_sample(dist)
                logp = logp_pre.sum(-1)
                joint = obs.reshape(B, N * obs_dim)
                values = model.critic(joint)  # (B, N)

            next_obs_np, scalar_r, terms, truncs, info = env.step(action.cpu().numpy())
            rm = torch.tensor(info["reward_matrix"], device=device)
            # Scalarized per-agent reward
            r = (rm * w).sum(-1)  # (B, N)
            dones = torch.tensor(terms | truncs, device=device)

            # Pack into 3-channel buffer (only channel 0 used)
            rewards = torch.zeros(B, N, 3, device=device)
            rewards[..., 0] = r
            vals = torch.zeros(B, N, 3, device=device)
            vals[..., 0] = values
            buf.add(obs, action.detach(), logp.detach(), rewards, vals, dones)

            recorder.log_step(global_steps, env, scalar_r, info)
            next_obs = torch.tensor(next_obs_np, device=device)
            obs_norm.update(next_obs)
            obs = obs_norm.normalize(next_obs)
            global_steps += B

        with torch.no_grad():
            last_val = model.critic(obs.reshape(B, N * obs_dim))
            last_done = torch.zeros(B, device=device)

        adv, ret = compute_gae(
            buf.rewards[..., 0],
            buf.values[..., 0],
            buf.dones,
            last_val,
            last_done,
            cfg.train.gamma,
            cfg.train.gae_lambda,
        )
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Fake 3-channel for buffer.get
        adv3 = torch.zeros(*adv.shape, 3, device=device)
        ret3 = torch.zeros(*ret.shape, 3, device=device)
        adv3[..., 0] = adv
        ret3[..., 0] = ret
        data = buf.get(adv3, ret3)

        n_samples = data["obs"].shape[0]
        batch_size = min(cfg.train.batch_size, n_samples)
        idxs = np.arange(n_samples)
        for _ in range(cfg.train.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, n_samples, batch_size):
                mb = idxs[start : start + batch_size]
                b_obs = data["obs"][mb]
                b_act = data["actions"][mb]
                b_old = data["logprobs"][mb]
                b_adv = data["adv"][mb][..., 0]
                b_ret = data["ret"][mb][..., 0]

                dist, _ = model.actor.dist(b_obs)
                new_logp = tanh_log_prob(dist, b_act).sum(-1)
                ratio = torch.exp(new_logp - b_old)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - cfg.train.clip_coef, 1 + cfg.train.clip_coef) * b_adv
                loss_pi = -torch.min(surr1, surr2).mean()
                values = model.critic(b_obs.reshape(b_obs.shape[0], N * obs_dim))
                loss_v = 0.5 * ((values - b_ret) ** 2).mean()
                entropy = dist.entropy().sum(-1).mean()
                loss = loss_pi + cfg.train.vf_coef * loss_v - cfg.train.ent_coef * entropy
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
                optimizer.step()

        torch.save(
            {"model": model.state_dict(), "num_members": N, "obs_local_dim": obs_dim},
            models_dir / "model_latest.pt",
        )
        pbar.set_postfix(steps=global_steps)

    writer.close()
    recorder.close()
    return recorder.run_dir
