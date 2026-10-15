#!/usr/bin/env python3
"""Frozen-policy success of saved checkpoints (separates metric noise from policy change).

For every models/model_XXXXX.pt, roll the *stochastic* training policy with
frozen obs-norm in fresh envs and count successes. If consecutive checkpoints
have similar frozen success while objectives/success_rate swings, the swing is
measurement noise; if frozen success itself swings, the policy is unstable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fluxswarm.agents.momappo import RunningNorm
from fluxswarm.agents.networks import ActorCriticMO, tanh_sample
from fluxswarm.config import Config
from fluxswarm.envs.swarm_env import BatchedSwarmEnv


def eval_ckpt(cfg: Config, path: Path, steps: int, seed: int, deterministic: bool) -> dict:
    env = BatchedSwarmEnv(cfg, batch=cfg.train.batch_envs, device="cpu")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = ActorCriticMO(env.n, env.obs_dim, cfg.train.hidden_sizes, cfg.train.log_std_init)
    model.load_state_dict(ck["model"])
    norm = RunningNorm(env.obs_dim, torch.device("cpu"))
    norm.load_state_dict(ck["obs_norm"], torch.device("cpu"))
    torch.manual_seed(seed)
    obs_np, _ = env.reset(seed=seed)
    succ = n = 0
    for _ in range(steps):
        with torch.no_grad():
            o = norm.normalize(torch.nan_to_num(torch.tensor(obs_np)))
            dist, _ = model.actor.dist(o)
            a = torch.tanh(dist.mean) if deterministic else tanh_sample(dist)[0]
        obs_np, _, _, _, info = env.step(a.numpy())
        for inf in info["infos"]:
            ep = inf.get("episode")
            if ep is not None:
                n += 1
                succ += ep.get("reason") == "success"
    return {"ckpt": path.name, "update": int(path.stem.split("_")[1]), "episodes": n,
            "success": succ / n if n else None}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--max-update", type=int, default=10**9)
    ap.add_argument("--every", type=int, default=1, help="use every k-th saved checkpoint")
    a = ap.parse_args()
    run = Path(a.run)
    cfg = Config.load(run / "config.yaml")
    cks = sorted((run / "models").glob("model_[0-9]*.pt"))[:: a.every]
    cks = [p for p in cks if int(p.stem.split("_")[1]) <= a.max_update]
    out = []
    for p in cks:
        r = eval_ckpt(cfg, p, a.steps, a.seed, a.deterministic)
        print(json.dumps(r), flush=True)
        out.append(r)
    tag = "det" if a.deterministic else "stoch"
    (run / f"frozen_eval_{tag}.json").write_text(json.dumps(out, indent=1))
