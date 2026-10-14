#!/usr/bin/env python3
"""Dummy-env run at the reported 16 envs x 100k steps/env scale."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from fluxswarm.agents.momappo import train_momappo
from fluxswarm.config import Config
from fluxswarm.envs.dummy import DummyUpstreamEnv

FIG = Path(__file__).resolve().parents[1] / "docs" / "research_figures"
OUT = "runs_new/_research"


def make_cfg(winner: bool, seed: int) -> Config:
    cfg = Config()
    cfg.task.progress_reward = "legacy"
    cfg.obs.localization = "full"
    cfg.train.use_action_x_prior = False
    cfg.train.live_swarm_tb_every = 0
    cfg.train.resume = False
    cfg.train.device = "cpu"
    cfg.train.seed = seed
    cfg.train.batch_envs = 16
    cfg.train.n_steps = 200
    cfg.train.total_timesteps_per_env = 100_000
    cfg.train.hidden_sizes = (64, 64)
    cfg.run.output_root = OUT
    cfg.run.tag = f"dummy16x100k_{'win' if winner else 'base'}_s{seed}"
    if winner:
        cfg.train.batch_size = 256
        cfg.train.pcgrad_protect_progress = True
        cfg.train.adv_norm = "weighted"
    else:
        cfg.train.batch_size = 4
        cfg.train.pcgrad_protect_progress = False
        cfg.train.adv_norm = "independent"
    return cfg


def run(winner: bool, seed: int = 0) -> Path:
    cfg = make_cfg(winner, seed)
    env = DummyUpstreamEnv(cfg, batch=16, n_agents=16, device="cpu")
    return Path(train_momappo(cfg, env=env))


def plot(paths: dict[str, Path]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    keys = [
        ("success_rate", "success_rate"),
        ("mean_x", "mean x"),
        ("mean_act_x", "mean action x"),
    ]
    for ax, (col, ylab) in zip(axes, keys):
        for name, p in paths.items():
            df = pd.DataFrame(
                [json.loads(l) for l in (p / "metrics.jsonl").read_text().splitlines() if l.strip()]
            )
            ax.plot(df["global_steps"], df[col], label=name)
        ax.set_xlabel("env steps")
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "dummy_16x100k_curves.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["win", "base", "both"], default="both")
    args = parser.parse_args()
    paths = {}
    if args.only in ("win", "both"):
        paths["winner"] = run(True, 0)
        print("winner", paths["winner"])
    if args.only in ("base", "both"):
        paths["baseline"] = run(False, 0)
        print("baseline", paths["baseline"])
    if len(paths) == 2:
        plot(paths)
        summ = {}
        for name, p in paths.items():
            df = pd.DataFrame(
                [json.loads(l) for l in (p / "metrics.jsonl").read_text().splitlines() if l.strip()]
            )
            tail = df.tail(max(1, len(df) // 4))
            summ[name] = {
                "run": p.name,
                "final_success_rate": float(df["success_rate"].iloc[-1]),
                "tail_success_rate": float(tail["success_rate"].mean()),
                "final_mean_x": float(df["mean_x"].iloc[-1]),
                "final_mean_act_x": float(df["mean_act_x"].iloc[-1]),
            }
        (FIG / "dummy_16x100k_summaries.json").write_text(json.dumps(summ, indent=2))
        print(json.dumps(summ, indent=2))
