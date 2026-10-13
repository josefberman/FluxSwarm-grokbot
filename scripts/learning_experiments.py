#!/usr/bin/env python3
"""Controlled MOMAPPO learning experiments (dummy env, then optional physics).

Fixed task flags for the reported failure mode:
  --progress-reward legacy --obs-localization full --no-action-x-prior
  --batch-envs 16 --total-timesteps-per-env 100000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from fluxswarm.agents.momappo import train_momappo
from fluxswarm.config import Config
from fluxswarm.envs.dummy import DummyUpstreamEnv

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "research_figures"
OUT_ROOT = "runs_new/_research"


def _read_metrics(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def _plot_curves(series: dict[str, list[pd.DataFrame]], metric: str, ylabel: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = plt.cm.tab10.colors
    for i, (name, dfs) in enumerate(series.items()):
        if not dfs:
            continue
        xs = [df["global_steps"].to_numpy() for df in dfs]
        ys = [df[metric].to_numpy() for df in dfs]
        n = min(len(x) for x in xs)
        grid = xs[0][:n]
        stacked = np.stack([y[:n] for y in ys], axis=0)
        mu = stacked.mean(0)
        sd = stacked.std(0)
        c = colors[i % len(colors)]
        ax.plot(grid, mu, color=c, label=name, lw=2)
        if stacked.shape[0] > 1:
            ax.fill_between(grid, mu - sd, mu + sd, color=c, alpha=0.18)
    ax.set_xlabel("env steps (all envs)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _apply_variant(cfg: Config, variant: str) -> Config:
    # Shared "reported failure" flags
    cfg.task.progress_reward = "legacy"
    cfg.obs.localization = "full"
    cfg.train.use_action_x_prior = False
    cfg.train.live_swarm_tb_every = 0
    cfg.train.save_fields = False
    cfg.train.resume = False
    cfg.train.hidden_sizes = (64, 64)
    cfg.train.device = "cpu"
    cfg.run.output_root = OUT_ROOT

    if variant == "baseline":
        pass
    elif variant == "minibatch256":
        cfg.train.batch_size = 256
    elif variant == "adv_weighted":
        cfg.train.adv_norm = "weighted"
    elif variant == "protect_progress":
        cfg.train.pcgrad_protect_progress = True
    elif variant == "ortho_init":
        cfg.train.orthogonal_init = True
    elif variant == "lr_3e4":
        cfg.train.lr = 3e-4
    elif variant == "combined":
        cfg.train.batch_size = 256
        cfg.train.adv_norm = "weighted"
        cfg.train.pcgrad_protect_progress = True
        cfg.train.orthogonal_init = True
        cfg.train.lr = 3e-4
        cfg.train.adam_eps = 1e-5
    elif variant == "mb256_protect":
        cfg.train.batch_size = 256
        cfg.train.pcgrad_protect_progress = True
        cfg.train.adv_norm = "weighted"
    elif variant == "mb256_ortho":
        cfg.train.batch_size = 256
        cfg.train.orthogonal_init = True
    elif variant == "mb256_lr3e4":
        cfg.train.batch_size = 256
        cfg.train.lr = 3e-4
    else:
        raise ValueError(variant)
    return cfg


def run_dummy(variant: str, seed: int, updates: int = 40) -> Path:
    cfg = Config()
    cfg.train.seed = seed
    cfg.train.batch_envs = 8
    cfg.train.n_steps = 32
    cfg.train.total_timesteps_per_env = updates * 32
    cfg.train.batch_size = 4
    cfg.run.tag = f"dummy_{variant}_s{seed}"
    cfg = _apply_variant(cfg, variant)
    env = DummyUpstreamEnv(cfg, batch=cfg.train.batch_envs, n_agents=4, device="cpu")
    t0 = time.time()
    run_dir = train_momappo(cfg, env=env)
    dt = time.time() - t0
    print(f"{variant} seed={seed} -> {run_dir} ({dt:.1f}s)")
    return Path(run_dir)


def summarize(run_dir: Path) -> dict:
    df = _read_metrics(run_dir)
    tail = df.tail(max(1, len(df) // 4))
    return {
        "run": run_dir.name,
        "updates": int(len(df)),
        "final_success_rate": float(df["success_rate"].iloc[-1]),
        "tail_success_rate": float(tail["success_rate"].mean()),
        "final_mean_x": float(df["mean_x"].iloc[-1]),
        "tail_mean_x": float(tail["mean_x"].mean()),
        "final_mean_act_x": float(df["mean_act_x"].iloc[-1]),
        "final_cum_progress": float(df["mean_cum_progress"].iloc[-1]),
        "final_explained_variance": float(df["explained_variance"].iloc[-1]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "baseline",
            "minibatch256",
            "adv_weighted",
            "protect_progress",
            "ortho_init",
            "lr_3e4",
            "combined",
        ],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--updates", type=int, default=40)
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[pd.DataFrame]] = {v: [] for v in args.variants}
    summaries = []
    for variant in args.variants:
        for seed in args.seeds:
            run_dir = run_dummy(variant, seed, updates=args.updates)
            df = _read_metrics(run_dir)
            grouped[variant].append(df)
            row = summarize(run_dir)
            row["variant"] = variant
            row["seed"] = seed
            summaries.append(row)
            print(json.dumps(row, indent=2))

    _plot_curves(grouped, "success_rate", "success_rate (rolling)", FIG_DIR / "dummy_success_rate.png")
    _plot_curves(grouped, "mean_x", "mean member x (lower is upstream)", FIG_DIR / "dummy_mean_x.png")
    _plot_curves(grouped, "mean_act_x", "mean action x (want negative)", FIG_DIR / "dummy_mean_act_x.png")
    _plot_curves(grouped, "mean_cum_progress", "mean episode cum. progress", FIG_DIR / "dummy_cum_progress.png")

    out = FIG_DIR / "dummy_summaries.json"
    out.write_text(json.dumps(summaries, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
