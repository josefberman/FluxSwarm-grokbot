#!/usr/bin/env python3
"""Rebuild dummy learning-curve plots from runs_new/_research."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs_new" / "_research"
FIG = ROOT / "docs" / "research_figures"

# Display order and grouping
VARIANTS = [
    "baseline",
    "minibatch256",
    "protect_progress",
    "mb256_protect",
    "mb256_ortho",
    "mb256_lr3e4",
    "adv_weighted",
    "ortho_init",
    "lr_3e4",
    "combined",
]


def load_groups() -> dict[str, list[pd.DataFrame]]:
    groups: dict[str, list[pd.DataFrame]] = {v: [] for v in VARIANTS}
    for d in sorted(RUNS.glob("*_dummy_*")):
        metrics = d / "metrics.jsonl"
        if not metrics.exists():
            continue
        name = d.name
        variant = None
        for v in VARIANTS:
            if f"_dummy_{v}_s" in name or name.endswith(f"_dummy_{v}"):
                variant = v
                break
        if variant is None:
            continue
        rows = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
        if rows:
            groups[variant].append(pd.DataFrame(rows))
    return {k: v for k, v in groups.items() if v}


def plot_metric(groups, metric, ylabel, out):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = plt.cm.tab10.colors
    for i, (name, dfs) in enumerate(groups.items()):
        xs = [df["global_steps"].to_numpy() for df in dfs]
        ys = [df[metric].to_numpy() for df in dfs]
        n = min(len(x) for x in xs)
        grid = xs[0][:n]
        stacked = np.stack([y[:n] for y in ys], axis=0)
        mu, sd = stacked.mean(0), stacked.std(0)
        c = colors[i % len(colors)]
        ax.plot(grid, mu, color=c, label=name, lw=2)
        if stacked.shape[0] > 1:
            ax.fill_between(grid, mu - sd, mu + sd, color=c, alpha=0.15)
    ax.set_xlabel("env steps (all envs)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    groups = load_groups()
    summaries = []
    for name, dfs in groups.items():
        for df in dfs:
            tail = df.tail(max(1, len(df) // 4))
            summaries.append(
                {
                    "variant": name,
                    "final_success_rate": float(df["success_rate"].iloc[-1]),
                    "tail_success_rate": float(tail["success_rate"].mean()),
                    "final_mean_x": float(df["mean_x"].iloc[-1]),
                    "final_mean_act_x": float(df["mean_act_x"].iloc[-1]),
                    "final_cum_progress": float(df["mean_cum_progress"].iloc[-1]),
                }
            )
    (FIG / "dummy_summaries.json").write_text(json.dumps(summaries, indent=2))
    plot_metric(groups, "success_rate", "success_rate (rolling)", FIG / "dummy_success_rate.png")
    plot_metric(groups, "mean_x", "mean member x (lower is upstream)", FIG / "dummy_mean_x.png")
    plot_metric(groups, "mean_act_x", "mean action x (want negative)", FIG / "dummy_mean_act_x.png")
    plot_metric(groups, "mean_cum_progress", "mean episode cum. progress", FIG / "dummy_cum_progress.png")
    # Focused winner vs baseline
    focus = {k: groups[k] for k in ("baseline", "minibatch256", "protect_progress", "mb256_protect") if k in groups}
    plot_metric(focus, "success_rate", "success_rate (rolling)", FIG / "dummy_success_rate_focus.png")
    print("variants", {k: len(v) for k, v in groups.items()})


if __name__ == "__main__":
    main()
