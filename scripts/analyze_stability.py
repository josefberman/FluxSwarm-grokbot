#!/usr/bin/env python3
"""Offline analysis of success-rate noise for runs with episodes.jsonl.

For each run: reconstruct the logged rolling window, compare alternative
estimators on the *same* episodes, quantify burstiness / short-episode bias,
and compute the stability score (std + mean over the last 50% of updates).
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load(run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = pd.DataFrame([json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines() if l.strip()])
    ep_path = run / "episodes.jsonl"
    e = pd.DataFrame([json.loads(l) for l in ep_path.read_text().splitlines() if l.strip()]) if ep_path.exists() else pd.DataFrame()
    return m, e


def window_series(e: pd.DataFrame, window: int) -> np.ndarray:
    s = e["success"].astype(float).to_numpy()
    c = np.cumsum(np.r_[0.0, s])
    idx = np.arange(1, len(s) + 1)
    lo = np.maximum(0, idx - window)
    return (c[idx] - c[lo]) / (idx - lo)


def per_update(m: pd.DataFrame, e: pd.DataFrame, window: int, n_envs: int) -> pd.DataFrame:
    """Value of each estimator at the end of every update."""
    out = []
    w_log = window_series(e, window)
    w_big = window_series(e, 4 * window)
    upd_end_idx = e.groupby("update").size().reindex(range(len(m)), fill_value=0).cumsum().to_numpy()
    for u in range(len(m)):
        k = upd_end_idx[u]
        row = {"update": u, "global_steps": m["global_steps"].iloc[u]}
        row["window"] = w_log[k - 1] if k > 0 else np.nan
        row["window_x4"] = w_big[k - 1] if k > 0 else np.nan
        if k > 0:
            lo = max(0, k - window)
            span = e.iloc[lo:k]
            row["window_span_updates"] = int(span["update"].iloc[-1] - span["update"].iloc[0] + 1)
            row["window_mean_len"] = float(span["steps"].mean())
            # time-weighted (per env-step) success: episodes weighted by length
            row["window_len_weighted"] = float((span["success"] * span["steps"]).sum() / span["steps"].sum())
        out.append(row)
    df = pd.DataFrame(out)
    df["update_rate"] = m["success_rate_update"].astype(float)
    df["ema"] = m["success_rate_ema"].astype(float)
    df["episodes_this_update"] = m["episodes_this_update"]
    return df


def stability(series: pd.Series) -> dict:
    s = series.dropna().to_numpy()
    half = s[len(s) // 2:]
    last10 = s[int(len(s) * 0.9):]
    return {"std_last50": float(np.std(half)) if len(half) else np.nan,
            "mean_last50": float(np.mean(half)) if len(half) else np.nan,
            "final_mean_last10": float(np.mean(last10)) if len(last10) else np.nan}


def burstiness(e: pd.DataFrame, n_envs: int) -> dict:
    g = e[e.source == "train"].groupby("global_steps").size()
    frac_big = float(g[g >= n_envs // 2].sum() / g.sum())
    return {"mean_ends_per_ending_step": float(g.mean()), "frac_eps_in_bursts_ge_half_envs": frac_big,
            "max_ends_per_step": int(g.max())}


def short_bias(e: pd.DataFrame) -> dict:
    half = e.iloc[len(e) // 2:]
    by = half.groupby("reason")["steps"].agg(["mean", "count"])
    return {r: {"mean_len": float(by.loc[r, "mean"]), "count": int(by.loc[r, "count"])} for r in by.index}


def analyze(run: Path) -> dict:
    m, e = load(run)
    cfg_txt = (run / "config.yaml").read_text()
    import yaml
    cfg = yaml.safe_load(cfg_txt)
    n_envs = int(cfg["train"]["batch_envs"])
    window = max(32, 8 * n_envs)
    pu = per_update(m, e, window, n_envs)
    res = {"run": run.name, "updates": len(m), "episodes": len(e), "window": window,
           "burst": burstiness(e, n_envs), "length_by_reason_last50": short_bias(e),
           "median_window_span_updates": float(pu["window_span_updates"].median())}
    for col in ["window", "window_x4", "ema", "update_rate", "window_len_weighted"]:
        res[col] = stability(pu[col])
    tail = m.iloc[len(m) // 2:]
    for col in ["epochs_run", "minibatches", "early_stop", "approx_kl_max", "log_std", "grad_norm",
                "grad_norm_std", "actor_grad_norm", "actor_grad_norm_std", "explained_variance", "clipfrac"]:
        if col in m:
            res[f"tail_{col}"] = float(tail[col].astype(float).mean())
    if "value_loss" in m:
        vl = np.array(tail["value_loss"].tolist())
        res["tail_value_loss_heads"] = vl.mean(0).round(3).tolist()
        ev = np.array(tail["ev_heads"].tolist())
        res["tail_ev_heads"] = ev.mean(0).round(3).tolist()
    pu.to_csv(run / "per_update_estimators.csv", index=False)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    runs = sorted({Path(p) for g in a.globs for p in glob.glob(g)})
    results = [analyze(r) for r in runs]
    txt = json.dumps(results, indent=1)
    if a.out:
        Path(a.out).write_text(txt)
    print(txt)
