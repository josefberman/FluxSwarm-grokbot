#!/usr/bin/env python3
"""Cheap-physics MOMAPPO at 16 envs (optional 100k steps/env)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from fluxswarm.agents.momappo import train_momappo
from fluxswarm.config import Config

FIG = Path(__file__).resolve().parents[1] / "docs" / "research_figures"


def make_cfg(*, winner: bool, seed: int, steps: int, tag: str) -> Config:
    cfg = Config()
    cfg.task.progress_reward = "legacy"
    cfg.obs.localization = "full"
    cfg.train.use_action_x_prior = False
    cfg.train.live_swarm_tb_every = 0
    cfg.train.save_fields = False
    cfg.train.resume = False
    cfg.train.device = "cpu"
    cfg.train.seed = seed
    cfg.train.batch_envs = 16
    cfg.train.n_steps = 200
    cfg.train.total_timesteps_per_env = steps
    cfg.sim.resolution_scale = 2.0
    cfg.sim.dt_substeps = 4
    cfg.run.output_root = "runs_new/_research"
    cfg.run.tag = tag
    if winner:
        cfg.train.batch_size = 256
        cfg.train.pcgrad_protect_progress = True
        cfg.train.adv_norm = "weighted"
    else:
        cfg.train.batch_size = 4
        cfg.train.pcgrad_protect_progress = False
        cfg.train.adv_norm = "independent"
        cfg.train.target_kl = 0.0
    return cfg


def summarize(run_dir: Path) -> dict:
    rows = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    tail = df.tail(max(1, len(df) // 4))
    return {
        "run": run_dir.name,
        "updates": int(len(df)),
        "global_steps": int(df["global_steps"].iloc[-1]),
        "final_success_rate": float(df["success_rate"].iloc[-1]),
        "peak_success_rate": float(df["success_rate"].max()),
        "tail_success_rate": float(tail["success_rate"].mean()),
        "final_mean_x": float(df["mean_x"].iloc[-1]),
        "min_mean_x": float(df["mean_x"].min()),
        "final_mean_act_x": float(df["mean_act_x"].iloc[-1]),
        "final_cum_progress": float(df["mean_cum_progress"].iloc[-1]),
        "final_entropy": float(df["entropy"].iloc[-1]),
    }


def plot(paths: dict[str, Path], out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex="col")
    specs = [
        (0, 0, "success_rate", "success_rate"),
        (0, 1, "mean_x", "mean x (lower=upstream)"),
        (1, 0, "mean_act_x", "mean action x"),
        (1, 1, "mean_cum_progress", "cum. progress"),
    ]
    for r, c, col, ylab in specs:
        ax = axes[r][c]
        for name, p in paths.items():
            df = pd.DataFrame(
                [json.loads(l) for l in (p / "metrics.jsonl").read_text().splitlines() if l.strip()]
            )
            ax.plot(df["global_steps"], df[col], label=name, lw=2)
        ax.set_ylabel(ylab)
        ax.grid(True, alpha=0.3)
        if r == 1:
            ax.set_xlabel("env steps")
    axes[0][0].legend()
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--winner", action="store_true")
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    paths = {}
    if args.baseline:
        cfg = make_cfg(winner=False, seed=args.seed, steps=args.steps, tag=f"phys16_base_{args.steps}_s{args.seed}")
        paths["old hparams"] = Path(train_momappo(cfg))
        print("BASE", summarize(paths["old hparams"]))
    if args.winner:
        cfg = make_cfg(winner=True, seed=args.seed, steps=args.steps, tag=f"phys16_win_{args.steps}_s{args.seed}")
        paths["winner"] = Path(train_momappo(cfg))
        print("WIN", summarize(paths["winner"]))
    for name, p in paths.items():
        (FIG / f"{p.name}_summary.json").write_text(json.dumps(summarize(p), indent=2))
    if len(paths) >= 1:
        plot(paths, FIG / f"physics_16env_{args.steps}_curves.png")


if __name__ == "__main__":
    main()
