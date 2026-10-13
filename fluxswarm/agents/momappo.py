"""Multi-objective multi-agent PPO (CTDE) with corrected log-probs and PCGrad."""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from fluxswarm.agents.networks import ActorCriticMO, tanh_log_prob, tanh_sample
from fluxswarm.agents.pcgrad import pcgrad_merge_actor
from fluxswarm.config import Config
from fluxswarm.envs.swarm_env import BatchedSwarmEnv
from fluxswarm.runs.recorder import RunRecorder


class RunningNorm:
    def __init__(self, dim: int, device: torch.device):
        self.mean = torch.zeros(dim, device=device)
        self.var = torch.ones(dim, device=device)
        self.count = 1e-4

    def update(self, x: torch.Tensor) -> None:
        flat = x.reshape(-1, x.shape[-1])
        batch_mean = flat.mean(0)
        batch_var = flat.var(0, unbiased=False)
        batch_count = flat.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta.pow(2) * self.count * batch_count / total) / total
        self.count = total

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.var.sqrt() + 1e-8)

    def state_dict(self) -> dict:
        return {
            "mean": self.mean.detach().cpu(),
            "var": self.var.detach().cpu(),
            "count": float(self.count),
        }

    def load_state_dict(self, state: dict, device: torch.device) -> None:
        self.mean = state["mean"].to(device=device, dtype=self.mean.dtype)
        self.var = state["var"].to(device=device, dtype=self.var.dtype)
        self.count = float(state["count"])


def resolve_resume_dir(spec: str, output_root: str | Path) -> Path:
    """Resolve a former run folder from a name, relative path, or absolute path."""
    raw = Path(spec).expanduser()
    candidates = [raw]
    if not raw.is_absolute():
        candidates.append(Path(output_root) / spec)
        candidates.append(Path.cwd() / spec)
    for cand in candidates:
        if cand.is_dir() and ((cand / "models").is_dir() or (cand / "config.yaml").exists()):
            return cand.resolve()
    raise FileNotFoundError(
        f"resume-from run folder not found: {spec!r} (looked under {output_root})"
    )


def find_resume_checkpoint(run_dir: Path) -> Path:
    latest = run_dir / "models" / "model_latest.pt"
    if latest.is_file():
        return latest
    numbered = sorted((run_dir / "models").glob("model_*.pt")) if (run_dir / "models").is_dir() else []
    if numbered:
        return numbered[-1]
    raise FileNotFoundError(f"no checkpoint in {run_dir / 'models'}")


def pack_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    obs_norm: RunningNorm,
    *,
    num_members: int,
    obs_local_dim: int,
    global_steps: int,
    use_pcgrad: bool,
) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "obs_norm": obs_norm.state_dict(),
        "num_members": num_members,
        "obs_local_dim": obs_local_dim,
        "total_timesteps": global_steps,
        "use_pcgrad": use_pcgrad,
    }


def apply_checkpoint(
    ckpt: dict,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    obs_norm: RunningNorm,
    device: torch.device,
    *,
    num_members: int,
    obs_local_dim: int,
) -> int:
    if int(ckpt.get("num_members", num_members)) != num_members:
        raise ValueError(
            f"checkpoint num_members={ckpt.get('num_members')} != current {num_members}"
        )
    if int(ckpt.get("obs_local_dim", obs_local_dim)) != obs_local_dim:
        raise ValueError(
            f"checkpoint obs_local_dim={ckpt.get('obs_local_dim')} != current {obs_local_dim}"
        )
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if "obs_norm" in ckpt:
        obs_norm.load_state_dict(ckpt["obs_norm"], device)
    return int(ckpt.get("total_timesteps", 0))


def blend_preset_x(raw: torch.Tensor, relax: float, preset_x: float = -0.95) -> torch.Tensor:
    """Blend toward upstream thrust, staying strictly inside the open unit disk.

    ``preset_x`` must be in (-1, 0] so atanh stays finite for PPO log-probs.
    """
    r = float(np.clip(relax, 0.0, 1.0))
    out = raw.clone()
    out[..., 0] = (1.0 - r) * preset_x + r * raw[..., 0]
    out = out.clamp(-0.999, 0.999)
    norms = torch.linalg.norm(out, dim=-1, keepdim=True).clamp_min(1e-8)
    return torch.where(norms > 1.0, out / norms, out)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    last_done: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE with per-step done masking.

    rewards/values: (T, B, N), dones: (T, B), last_*: (B, N) / (B,)
    """
    T, B, N = rewards.shape
    adv = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, N, device=rewards.device)
    next_values = torch.cat([values[1:], last_value.unsqueeze(0)], dim=0)
    for t in reversed(range(T)):
        if t == T - 1:
            next_v = last_value
        else:
            next_v = next_values[t]
        # dones[t] True => episode ended; do not bootstrap across the boundary
        mask = (1.0 - dones[t].float()).unsqueeze(-1)
        delta = rewards[t] + gamma * next_v * mask - values[t]
        last_gae = delta + gamma * lam * mask * last_gae
        adv[t] = last_gae
    returns = adv + values
    return adv, returns


def normalize_objective_advantages(
    adv: torch.Tensor,
    *,
    mode: str = "independent",
    weight: float = 1.0,
    min_std: float = 1e-3,
) -> torch.Tensor:
    """Whiten advantages; optionally re-apply the task weight.

    Independent per-objective unit-variance scaling cancels ``w_progress`` etc.
    and amplifies noise when an objective is nearly constant (e.g. all-zero
    energy rewards). Degenerate batches are zeroed instead of divided by 1e-8.
    """
    adv = torch.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "off":
        return adv
    std = adv.std()
    if not torch.isfinite(std) or float(std) < min_std:
        return torch.zeros_like(adv)
    z = (adv - adv.mean()) / (std + 1e-8)
    z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "weighted":
        z = z * float(weight)
    return z


def explained_variance(y_pred: torch.Tensor, y: torch.Tensor) -> float:
    y = y.detach().flatten()
    y_pred = y_pred.detach().flatten()
    var_y = y.var()
    if float(var_y) < 1e-8:
        return 0.0
    return float(1.0 - (y - y_pred).var() / (var_y + 1e-8))


class RolloutBuffer:
    def __init__(self, T: int, B: int, N: int, obs_dim: int, device: torch.device):
        self.T, self.B, self.N, self.obs_dim = T, B, N, obs_dim
        self.device = device
        self.obs = torch.zeros(T, B, N, obs_dim, device=device)
        self.actions = torch.zeros(T, B, N, 2, device=device)
        self.logprobs = torch.zeros(T, B, N, device=device)  # per-agent
        self.rewards = torch.zeros(T, B, N, 3, device=device)
        self.values = torch.zeros(T, B, N, 3, device=device)
        self.dones = torch.zeros(T, B, device=device)
        self.ptr = 0

    def add(self, obs, actions, logprobs, rewards, values, dones):
        t = self.ptr
        self.obs[t] = obs
        self.actions[t] = actions
        self.logprobs[t] = logprobs
        self.rewards[t] = rewards
        self.values[t] = values
        self.dones[t] = dones.float()
        self.ptr += 1

    def get(self, adv, ret):
        T, B, N = self.T, self.B, self.N
        return {
            "obs": self.obs.reshape(T * B, N, self.obs_dim),
            "actions": self.actions.reshape(T * B, N, 2),
            "logprobs": self.logprobs.reshape(T * B, N),
            "adv": adv.reshape(T * B, N, 3),
            "ret": ret.reshape(T * B, N, 3),
            "values": self.values.reshape(T * B, N, 3),
        }


def train_momappo(
    cfg: Config,
    env: Optional[BatchedSwarmEnv] = None,
    recorder: Optional[RunRecorder] = None,
) -> Path:
    from concurrent.futures import ThreadPoolExecutor

    from fluxswarm.perf import resolve_device, split_batch_sizes

    device = torch.device(resolve_device(cfg.train.device))
    gpu_n, cpu_n = split_batch_sizes(cfg)
    if env is None:
        # Primary env on GPU (or CPU if no CUDA). CPU shard is optional and
        # stepped after the GPU batch when device_split=cpu-shard.
        env = BatchedSwarmEnv(cfg, batch=gpu_n, device=str(device))
    cpu_env = None
    cpu_pool = None
    if cpu_n > 0 and device.type == "cuda":
        cpu_env = BatchedSwarmEnv(cfg, batch=cpu_n, device="cpu")
        cpu_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cpu-shard")
    if recorder is None:
        recorder = RunRecorder(cfg, algorithm="momappo")
    recorder.save_config(cfg)

    B = env.batch
    N = env.n
    obs_dim = env.obs_dim
    model = ActorCriticMO(
        num_members=N,
        obs_local_dim=obs_dim,
        hidden_sizes=cfg.train.hidden_sizes,
        log_std_init=cfg.train.log_std_init,
        recurrent=(cfg.train.actor == "recurrent"),
    ).to(device)
    if cfg.train.orthogonal_init:
        model.apply_orthogonal_init()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, eps=cfg.train.adam_eps)
    obs_norm = RunningNorm(obs_dim, device)

    models_dir = recorder.run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    latest = models_dir / "model_latest.pt"

    global_steps = 0
    skip_prior_warmup = False
    resume_ckpt_path: Optional[Path] = None
    if cfg.train.resume_from:
        src = resolve_resume_dir(cfg.train.resume_from, cfg.run.output_root)
        resume_ckpt_path = find_resume_checkpoint(src)
        skip_prior_warmup = True
    elif cfg.train.resume and latest.exists():
        resume_ckpt_path = latest

    if resume_ckpt_path is not None:
        ckpt = torch.load(resume_ckpt_path, map_location=device, weights_only=False)
        global_steps = apply_checkpoint(
            ckpt,
            model,
            optimizer,
            obs_norm,
            device,
            num_members=N,
            obs_local_dim=obs_dim,
        )
        print(f"Resumed from {resume_ckpt_path} at global_steps={global_steps}")

    writer = SummaryWriter(log_dir=str(recorder.run_dir / "tb"))
    obs_np, _ = env.reset(seed=cfg.train.seed)
    cpu_obs_np = None
    if cpu_env is not None:
        cpu_obs_np, _ = cpu_env.reset(seed=cfg.train.seed + 1)
    obs = torch.tensor(obs_np, device=device)
    obs_norm.update(obs)
    obs = obs_norm.normalize(obs)

    n_steps = cfg.train.n_steps
    # Count true env-steps across all shards
    envs_per_step = B + (cpu_n if cpu_env is not None else 0)
    total = cfg.train.total_timesteps_per_env * envs_per_step
    num_updates = max(1, total // (n_steps * envs_per_step))
    prior_warmup_rows = max(1, int(num_updates * n_steps * cfg.train.action_x_prior_warmup_fraction))
    success_window = max(32, 8 * envs_per_step)
    success_outcomes: list[float] = []
    success_min = max(1, success_window // 4)

    metrics_path = recorder.run_dir / "metrics.jsonl"
    ep_reasons: deque[str] = deque(maxlen=256)
    ep_lens: deque[int] = deque(maxlen=256)
    ep_progress: deque[float] = deque(maxlen=256)
    task_weights = (
        cfg.task.w_progress,
        cfg.task.w_energy,
        cfg.task.w_smooth,
    )

    pbar = trange(num_updates, desc="MOMAPPO")
    for update in pbar:
        buf = RolloutBuffer(n_steps, B, N, obs_dim, device)
        for t in range(n_steps):
            with torch.no_grad():
                dist, _ = model.actor.dist(obs)
                raw_action, logp_pre = tanh_sample(dist)
                logp = logp_pre.sum(dim=-1)
                relax = 1.0
                if cfg.train.use_action_x_prior and not skip_prior_warmup:
                    row = update * n_steps + t
                    relax = min(1.0, (row + 1) / prior_warmup_rows)
                    action = blend_preset_x(raw_action, relax)
                    logp = tanh_log_prob(dist, action).sum(dim=-1)
                else:
                    action = raw_action

                joint = obs.reshape(B, N * obs_dim)
                values = model.critic(joint).permute(0, 2, 1)

            action_np = action.detach().cpu().numpy()
            cpu_fut = None
            if cpu_env is not None and cpu_obs_np is not None:
                with torch.no_grad():
                    c_obs = obs_norm.normalize(torch.tensor(cpu_obs_np, device=device))
                    c_dist, _ = model.actor.dist(c_obs)
                    c_act = torch.tanh(c_dist.mean).detach().cpu().numpy()
                cpu_fut = cpu_pool.submit(cpu_env.step, c_act)
            next_obs_np, scalar_r, terms, truncs, info = env.step(action_np)
            pbar.set_postfix(rollout=f"{t + 1}/{n_steps}", steps=global_steps + B, refresh=True)
            c_info = None
            if cpu_fut is not None:
                cpu_obs_np, c_scalar, _, _, c_info = cpu_fut.result()
                recorder.log_step(global_steps + B, cpu_env, c_scalar, c_info)

            reward_matrix = torch.tensor(info["reward_matrix"], device=device, dtype=torch.float32)
            dones = torch.tensor(terms | truncs, device=device)

            buf.add(obs, action.detach(), logp.detach(), reward_matrix, values, dones)
            for inf in info["infos"]:
                ep = inf.get("episode")
                if ep is not None:
                    ep_reasons.append(str(ep.get("reason", "")))
                    ep_lens.append(int(ep.get("steps", 0)))
                    ep_progress.append(float(ep.get("cum_progress", 0.0)))

            recorder.log_step(
                global_steps,
                env,
                scalar_r,
                info,
                save_fields=cfg.train.save_fields and (global_steps % 50 == 0),
            )
            for inf in info["infos"]:
                ep = inf.get("episode")
                if ep is not None:
                    success_outcomes.append(1.0 if ep.get("reason") == "success" else 0.0)
            if c_info is not None:
                for inf in c_info["infos"]:
                    ep = inf.get("episode")
                    if ep is not None:
                        success_outcomes.append(1.0 if ep.get("reason") == "success" else 0.0)
            n_done = len(success_outcomes)
            if n_done >= success_min:
                writer.add_scalar(
                    "objectives/success_rate",
                    float(np.mean(success_outcomes[-success_window:])),
                    global_steps,
                )
            if B > 0:
                writer.add_scalar("objectives/progress", float(reward_matrix[0, :, 0].mean()), global_steps)
                writer.add_scalar("objectives/energy", float(reward_matrix[0, :, 1].mean()), global_steps)
                writer.add_scalar("objectives/smoothness", float(reward_matrix[0, :, 2].mean()), global_steps)
                a0 = env.solver.swarm
                writer.add_scalar("agent0/pos_x", float(a0.pos[0, 0, 0]), global_steps)
                writer.add_scalar("agent0/pos_y", float(a0.pos[0, 0, 1]), global_steps)
                writer.add_scalar("agent0/act_x", float(a0.action[0, 0, 0]), global_steps)
                writer.add_scalar("agent0/act_y", float(a0.action[0, 0, 1]), global_steps)
                writer.add_scalar("swarm/mean_x", float(a0.pos[..., 0].mean()), global_steps)
                writer.add_scalar("swarm/mean_act_x", float(a0.action[..., 0].mean()), global_steps)
                sol = env.solver
                if sol.last_drag is not None and sol.last_pressure_force is not None:
                    writer.add_scalar(
                        "agent0/force_drag_mag",
                        float(torch.linalg.norm(sol.last_drag[0, 0])),
                        global_steps,
                    )
                    writer.add_scalar(
                        "agent0/force_thrust_mag",
                        float(torch.linalg.norm(sol.last_thrust[0, 0])),
                        global_steps,
                    )

            next_obs = torch.tensor(next_obs_np, device=device)
            next_obs = torch.nan_to_num(next_obs, nan=0.0, posinf=0.0, neginf=0.0)
            obs_norm.update(next_obs)
            obs = obs_norm.normalize(next_obs)
            global_steps += envs_per_step

        # Bootstrap
        with torch.no_grad():
            joint = obs.reshape(B, N * obs_dim)
            last_val = model.critic(joint).permute(0, 2, 1)
            last_done = torch.zeros(B, device=device)

        advs, rets = [], []
        for k in range(3):
            adv_k, ret_k = compute_gae(
                buf.rewards[..., k],
                buf.values[..., k],
                buf.dones,
                last_val[..., k],
                last_done,
                cfg.train.gamma,
                cfg.train.gae_lambda,
            )
            adv_k = normalize_objective_advantages(
                adv_k,
                mode=cfg.train.adv_norm,
                weight=task_weights[k],
            )
            ret_k = torch.nan_to_num(ret_k, nan=0.0, posinf=0.0, neginf=0.0)
            advs.append(adv_k)
            rets.append(ret_k)
        adv = torch.stack(advs, dim=-1)
        ret = torch.stack(rets, dim=-1)

        data = buf.get(adv, ret)
        n_samples = data["obs"].shape[0]
        batch_size = min(max(int(cfg.train.batch_size), 1), n_samples)
        idxs = np.arange(n_samples)
        last_entropy = torch.tensor(0.0, device=device)
        last_approx_kl = 0.0
        last_ev = explained_variance(buf.values, ret)

        for _epoch in range(cfg.train.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, n_samples, batch_size):
                mb = idxs[start : start + batch_size]
                b_obs = data["obs"][mb]
                if not torch.isfinite(b_obs).all():
                    continue
                b_act = data["actions"][mb].clamp(-0.999, 0.999)
                b_old_logp = data["logprobs"][mb]
                b_adv = data["adv"][mb]
                b_ret = data["ret"][mb]

                dist, _ = model.actor.dist(b_obs)
                new_logp = tanh_log_prob(dist, b_act).sum(dim=-1)
                log_ratio = (new_logp - b_old_logp).clamp(-10.0, 10.0)
                ratio = log_ratio.exp()

                loss_pis = []
                clip = cfg.train.clip_coef
                for k in range(3):
                    adv_k = b_adv[..., k]
                    surr1 = ratio * adv_k
                    surr2 = ratio.clamp(1.0 - clip, 1.0 + clip) * adv_k
                    surr = torch.min(surr1, surr2)
                    # Dual-clip: PPO's min() does not bound the loss when A<0 and ratio→∞.
                    surr = torch.where(adv_k < 0, torch.max(surr, 3.0 * adv_k), surr)
                    loss_pis.append(-surr.mean())

                if not all(torch.isfinite(lp) for lp in loss_pis):
                    continue

                joint = b_obs.reshape(b_obs.shape[0], N * obs_dim)
                values = model.critic(joint).permute(0, 2, 1)
                loss_v = 0.5 * ((values - b_ret) ** 2).mean()
                entropy = dist.entropy().sum(dim=-1).mean()
                if not torch.isfinite(loss_v) or not torch.isfinite(entropy):
                    continue

                optimizer.zero_grad(set_to_none=True)
                if cfg.train.use_pcgrad:
                    pcgrad_merge_actor(
                        model.actor,
                        loss_pis,
                        protect_primary=cfg.train.pcgrad_protect_progress,
                    )
                else:
                    sum(loss_pis).backward(retain_graph=True)

                aux = cfg.train.vf_coef * loss_v - cfg.train.ent_coef * entropy
                aux.backward()
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                ):
                    optimizer.zero_grad(set_to_none=True)
                    continue
                nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
                optimizer.step()
                last_entropy = entropy.detach()
                last_approx_kl = float((b_old_logp - new_logp).mean().detach())

        ckpt = pack_checkpoint(
            model,
            optimizer,
            obs_norm,
            num_members=N,
            obs_local_dim=obs_dim,
            global_steps=global_steps,
            use_pcgrad=cfg.train.use_pcgrad,
        )
        torch.save(ckpt, latest)
        if update % 10 == 0 or update == num_updates - 1:
            torch.save(ckpt, models_dir / f"model_{update:05d}.pt")

        n_ep = max(len(ep_reasons), 1)
        success_rate = sum(r == "success" for r in ep_reasons) / n_ep if ep_reasons else 0.0
        fail_rate = sum(r == "failure" for r in ep_reasons) / n_ep if ep_reasons else 0.0
        mean_x = float(env.solver.swarm.pos[..., 0].mean().detach())
        mean_act_x = float(env.solver.swarm.action[..., 0].mean().detach())
        mean_prog = float(np.mean(list(ep_progress)[-64:])) if ep_progress else 0.0
        mean_len = float(np.mean(list(ep_lens)[-64:])) if ep_lens else 0.0
        writer.add_scalar("episode/success_rate", success_rate, global_steps)
        writer.add_scalar("episode/failure_rate", fail_rate, global_steps)
        writer.add_scalar("episode/mean_length", mean_len, global_steps)
        writer.add_scalar("episode/cum_progress", mean_prog, global_steps)
        writer.add_scalar("training/entropy", float(last_entropy), global_steps)
        writer.add_scalar("training/approx_kl", last_approx_kl, global_steps)
        writer.add_scalar("training/explained_variance", last_ev, global_steps)
        writer.add_scalar("swarm/mean_x", mean_x, global_steps)
        row = {
            "update": int(update),
            "global_steps": int(global_steps),
            "success_rate": success_rate,
            "failure_rate": fail_rate,
            "mean_ep_len": mean_len,
            "mean_cum_progress": mean_prog,
            "mean_x": mean_x,
            "mean_act_x": mean_act_x,
            "entropy": float(last_entropy),
            "approx_kl": last_approx_kl,
            "explained_variance": last_ev,
        }
        with metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        pbar.set_postfix(
            steps=global_steps,
            succ=f"{success_rate:.2f}",
            mx=f"{mean_x:.1f}",
            ent=float(last_entropy),
        )

    writer.close()
    recorder.close()
    if cpu_pool is not None:
        cpu_pool.shutdown(wait=True)
    if getattr(getattr(env, "solver", None), "torch_fluid", None) is not None:
        try:
            from fluxswarm.analysis.swarm_viz import animate_swarm_from_run

            anim = animate_swarm_from_run(recorder.run_dir)
            if anim is not None:
                print(f"Wrote swarm animation {anim}")
        except Exception as exc:
            print(f"Swarm animation skipped: {exc}")
    return recorder.run_dir
