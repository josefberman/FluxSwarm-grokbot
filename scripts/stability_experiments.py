#!/usr/bin/env python3
"""Success-rate stability experiments (dummy env + optional cheap physics).

Base settings mirror Josef's GPU run: legacy progress reward, full obs
localization, no x action prior, 16 envs, batch_size 256, protect-progress
PCGrad, weighted adv-norm, target_kl 0.02, lr 1e-3, (256,128) layers.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from fluxswarm.agents.momappo import train_momappo
from fluxswarm.config import Config
from fluxswarm.envs.dummy import DummyUpstreamEnv
from fluxswarm.runs.recorder import RunRecorder


class LiteRecorder(RunRecorder):
    """Skip the per-step parquet log (dominant CPU cost); metrics/episodes/TB still written."""

    def log_step(self, *a, **k) -> None:  # noqa: D401
        return None

OUT = "runs_new/_stability"


def base_cfg(seed: int, steps: int, n_envs: int = 16) -> Config:
    cfg = Config()
    cfg.task.progress_reward = "legacy"
    cfg.obs.localization = "full"
    cfg.train.use_action_x_prior = False
    cfg.train.live_swarm_tb_every = 0
    cfg.train.save_fields = False
    cfg.train.resume = False
    cfg.train.device = "cpu"
    cfg.train.seed = seed
    cfg.train.batch_envs = n_envs
    cfg.train.n_steps = 200
    cfg.train.total_timesteps_per_env = steps
    cfg.run.output_root = OUT
    return cfg


def apply_variant(cfg: Config, variant: str) -> dict:
    """Mutate cfg; return extra env kwargs (dummy only). '+'-joined combos allowed."""
    env_kw: dict = {}
    for v in variant.split("+"):
        if v == "base":
            pass
        elif v == "clock":
            cfg.sim.per_env_clock = True
        elif v == "popart":
            cfg.train.value_norm = "popart"
        elif v == "retnorm":
            cfg.train.value_norm = "running"
        elif v == "vclip":
            cfg.train.clip_vloss = True
        elif v.startswith("bonus"):
            b = float(v[5:])
            cfg.task.terminal_bonus = b
            cfg.task.terminal_bonus_legacy = True
            env_kw["terminal_bonus"] = b
        elif v.startswith("kl"):
            cfg.train.target_kl = float(v[2:])
        elif v.startswith("lr") and v[2:].replace("e-", "").replace(".", "").isdigit():
            cfg.train.lr = float(v[2:])
        elif v == "linlr":
            cfg.train.lr_schedule = "linear"
        elif v == "sympcg":
            cfg.train.pcgrad_protect_progress = False
        elif v == "nopcg":
            cfg.train.use_pcgrad = False
        elif v.startswith("ent"):
            cfg.train.ent_coef = float(v[3:])
        else:
            raise ValueError(v)
    return env_kw


def run(env_kind: str, variant: str, seed: int, steps: int, n_agents: int) -> Path:
    cfg = base_cfg(seed, steps)
    env_kw = apply_variant(cfg, variant)
    cfg.run.tag = f"{env_kind}_{variant}_s{seed}"
    t0 = time.time()
    if env_kind in ("dummy", "hard"):
        if env_kind == "hard":
            env_kw.update(drift=2.5, drift_profile="poiseuille", drift_jitter=0.3)
        env = DummyUpstreamEnv(cfg, batch=16, n_agents=n_agents, device="cpu", reward_mode="legacy", **env_kw)
        rd = train_momappo(cfg, env=env, recorder=LiteRecorder(cfg))
    else:
        cfg.sim.resolution_scale = 2.0
        cfg.sim.dt_substeps = 4
        rd = train_momappo(cfg, recorder=LiteRecorder(cfg))
    rd = Path(rd)
    (rd / "wall.json").write_text(json.dumps({"variant": variant, "seed": seed, "env": env_kind,
                                              "steps": steps, "wall_s": time.time() - t0}))
    print(f"DONE {env_kind} {variant} s{seed} {time.time()-t0:.0f}s -> {rd}", flush=True)
    return rd


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="dummy", choices=["dummy", "hard", "physics"])
    ap.add_argument("--variants", nargs="+", default=["base"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--n-agents", type=int, default=16)
    a = ap.parse_args()
    for v in a.variants:
        for s in a.seeds:
            run(a.env, v, s, a.steps, a.n_agents)
