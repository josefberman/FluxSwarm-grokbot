#!/usr/bin/env python3
"""Cheap-physics MOMAPPO pilots (CPU). Not the full 100k grid; scale is logged."""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fluxswarm.agents.momappo import train_momappo
from fluxswarm.config import Config
from fluxswarm.envs.swarm_env import BatchedSwarmEnv

FIG = Path(__file__).resolve().parents[1] / "docs" / "research_figures"
OUT = "runs_new/_research"


def cheap_cfg(tag: str, *, winner: bool, seed: int, steps_per_env: int, batch: int) -> Config:
    cfg = Config()
    cfg.task.progress_reward = "legacy"
    cfg.obs.localization = "full"
    cfg.train.use_action_x_prior = False
    cfg.train.live_swarm_tb_every = 0
    cfg.train.save_fields = False
    cfg.train.resume = False
    cfg.train.device = "cpu"
    cfg.train.seed = seed
    cfg.train.batch_envs = batch
    cfg.train.n_steps = 64
    cfg.train.total_timesteps_per_env = steps_per_env
    cfg.sim.resolution_scale = 2.0
    cfg.sim.dt_substeps = 4
    cfg.swarm.num_x = 4
    cfg.swarm.num_y = 2
    cfg.task.episode_duration = 3.0
    cfg.obs.history = 2
    cfg.run.output_root = OUT
    cfg.run.tag = tag
    if winner:
        cfg.train.batch_size = 256
        cfg.train.pcgrad_protect_progress = True
        cfg.train.adv_norm = "weighted"
    else:
        cfg.train.batch_size = 4
        cfg.train.pcgrad_protect_progress = False
        cfg.train.adv_norm = "independent"
    return cfg


def brute_rollout(steps: int = 200) -> dict:
    cfg = cheap_cfg("brute", winner=True, seed=0, steps_per_env=steps, batch=2)
    env = BatchedSwarmEnv(cfg, batch=2, device="cpu")
    env.reset(seed=0)
    start = float(env.solver.swarm.pos[..., 0].mean())
    act = np.zeros((2, env.n, 2), dtype=np.float64)
    act[..., 0] = -1.0
    t0 = time.time()
    reasons = []
    for i in range(steps):
        _, _, terms, truncs, info = env.step(act)
        for inf in info["infos"]:
            if inf.get("episode"):
                reasons.append(inf["episode"]["reason"])
    dt = time.time() - t0
    end = float(env.solver.swarm.pos[..., 0].mean())
    return {
        "policy": "brute_upstream",
        "steps": steps,
        "seconds": dt,
        "sec_per_step": dt / steps,
        "start_mean_x": start,
        "end_mean_x": end,
        "delta_x": end - start,
        "n_episodes": len(reasons),
        "successes": sum(r == "success" for r in reasons),
        "failures": sum(r == "failure" for r in reasons),
        "resolution": list(cfg.sim.resolution),
        "dt_substeps": cfg.sim.dt_substeps,
        "batch": 2,
    }


def train_one(winner: bool, seed: int, steps_per_env: int, batch: int) -> dict:
    tag = f"phys_{'win' if winner else 'base'}_s{seed}"
    cfg = cheap_cfg(tag, winner=winner, seed=seed, steps_per_env=steps_per_env, batch=batch)
    t0 = time.time()
    run_dir = Path(train_momappo(cfg))
    elapsed = time.time() - t0
    rows = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    tail = df.tail(max(1, len(df) // 4))
    return {
        "winner": winner,
        "seed": seed,
        "run": run_dir.name,
        "seconds": elapsed,
        "steps_per_env": steps_per_env,
        "batch": batch,
        "final_success_rate": float(df["success_rate"].iloc[-1]),
        "tail_success_rate": float(tail["success_rate"].mean()),
        "final_mean_x": float(df["mean_x"].iloc[-1]),
        "final_mean_act_x": float(df["mean_act_x"].iloc[-1]),
        "final_cum_progress": float(df["mean_cum_progress"].iloc[-1]),
        "curve": df[["global_steps", "success_rate", "mean_x", "mean_act_x", "mean_cum_progress"]].to_dict(
            orient="list"
        ),
    }


def plot_physics(results: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    metrics = [
        ("success_rate", "success_rate"),
        ("mean_x", "mean x (lower=upstream)"),
        ("mean_act_x", "mean action x"),
    ]
    for ax, (key, ylab) in zip(axes, metrics):
        for r in results:
            label = ("winner" if r["winner"] else "old hparams") + f" s{r['seed']}"
            ax.plot(r["curve"]["global_steps"], r["curve"][key], label=label)
        ax.set_xlabel("env steps")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "physics_pilot_curves.png", dpi=150)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    brute = brute_rollout(120)
    print("BRUTE", json.dumps(brute, indent=2))
    (FIG / "physics_brute.json").write_text(json.dumps(brute, indent=2))

    results = []
    for winner in (False, True):
        results.append(train_one(winner, seed=0, steps_per_env=512, batch=4))
        print("TRAIN", {k: results[-1][k] for k in results[-1] if k != "curve"})
    plot_physics(results)
    slim = [{k: v for k, v in r.items() if k != "curve"} for r in results]
    (FIG / "physics_pilot_summaries.json").write_text(json.dumps({"brute": brute, "trains": slim}, indent=2))


if __name__ == "__main__":
    main()
