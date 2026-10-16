#!/usr/bin/env python3
"""Figures for docs/research_log_stability.md."""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_stability as A  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "docs" / "research_figures"
R = Path("runs_new/_stability")


def runs(pattern: str) -> list[Path]:
    return sorted(Path(p) for p in glob.glob(str(R / pattern)))


def estimators_fig(run: Path, out: Path) -> None:
    m, e = A.load(run)
    pu = A.per_update(m, e, 128, 16)
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    x = pu["update"]
    ax[0].plot(x, pu["update_rate"], ".", ms=3, alpha=0.5, label="per-update success")
    ax[0].plot(x, pu["window"], lw=2, label="objectives/success_rate (last 128 eps)")
    ax[0].plot(x, pu["ema"], lw=2, label="success_rate_ema (beta=0.9/update)")
    ax[0].plot(x, pu["window_x4"], lw=2, label="last 512 eps")
    ax[0].set_ylabel("success"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].plot(x, pu["episodes_this_update"], label="episodes finished per update")
    ax[1].plot(x, pu["window_span_updates"], label="updates spanned by 128-ep window")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    ax[2].plot(x, m["log_std"], label="actor log_std (mean)")
    ax[2].plot(x, m["epochs_run"], label="PPO epochs actually run")
    ax[2].plot(x, np.array(m["explained_variance"]), label="explained variance")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3); ax[2].set_xlabel("PPO update")
    fig.suptitle(run.name); fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def compare_fig(groups: dict[str, list[Path]], out: Path, col: str = "window", max_u: int = 150) -> None:
    A.MAX_UPDATES = max_u
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, (name, rs) in enumerate(groups.items()):
        for j, r in enumerate(rs):
            m, e = A.load(r)
            pu = A.per_update(m, e, 128, 16)
            ax.plot(pu["update"], pu[col], color=f"C{i}", lw=1.6, ls=["-", "--", ":"][j % 3],
                    label=name if j == 0 else None)
    ax.set_xlabel("PPO update (16 envs x 200 steps)"); ax.set_ylabel(f"success ({col})")
    ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    base100 = runs("2026-10-15_22-35-01_physics_base_s0")[0]
    estimators_fig(base100, FIG / "stab_base100k_estimators.png")
    g = {}
    for v in ["base", "clock", "bonus50", "popart", "kl0.05", "lr3e-4", "sympcg", "nopcg", "clock+bonus50"]:
        rs = [r for r in runs(f"*physics_{v}_s[0-9]") if "22-35-01" not in r.name]
        if rs:
            g[v] = rs
    compare_fig({k: g[k] for k in g if k in ("base", "clock", "clock+bonus50")}, FIG / "stab_clock_vs_base.png")
    compare_fig({k: g[k] for k in g if k in ("base", "bonus50", "popart")}, FIG / "stab_bonus_popart.png")
    compare_fig({k: g[k] for k in g if k in ("base", "kl0.05", "lr3e-4", "sympcg", "nopcg")}, FIG / "stab_kl_lr_pcgrad.png")
    for k, rs in g.items():
        if k.startswith("clock"):
            estimators_fig(rs[0], FIG / f"stab_{k.replace('+', '_')}_s0_estimators.png")
    print("ok")
