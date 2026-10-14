"""Figure generators writing to runs_new/<run>/figures/ and _comparisons/."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

_FIG_FONT = 14
plt.rcParams.update(
    {
        "font.size": _FIG_FONT,
        "axes.titlesize": _FIG_FONT + 2,
        "axes.labelsize": _FIG_FONT,
        "xtick.labelsize": _FIG_FONT - 1,
        "ytick.labelsize": _FIG_FONT - 1,
        "legend.fontsize": _FIG_FONT - 1,
        "figure.titlesize": _FIG_FONT + 4,
    }
)


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _write_sidecar(path: Path, meta: dict[str, Any]) -> None:
    with path.with_suffix(".json").open("w") as f:
        json.dump({**meta, "git_commit": _git_commit()}, f, indent=2)


def _read_resume_from(run_dir: Path) -> Optional[str]:
    meta = run_dir / "meta.json"
    if meta.exists():
        try:
            spec = json.loads(meta.read_text()).get("resume_from")
            if spec:
                return str(spec)
        except (OSError, json.JSONDecodeError):
            pass
    spec = (_load_config(run_dir).get("train") or {}).get("resume_from")
    return str(spec) if spec else None


def resume_chain(run_dir: Path) -> list[Path]:
    """Oldest-first ``resume_from`` chain ending at ``run_dir``."""
    from fluxswarm.agents.momappo import resolve_resume_dir

    seen: set[str] = set()
    chain: list[Path] = []
    cur = Path(run_dir).resolve()
    while True:
        key = str(cur)
        if key in seen:
            break
        seen.add(key)
        chain.append(cur)
        spec = _read_resume_from(cur)
        if not spec:
            break
        try:
            prev = resolve_resume_dir(spec, cur.parent)
        except FileNotFoundError:
            break
        cur = prev
    chain.reverse()
    return chain


def _chain_ids(run_dir: Path) -> list[str]:
    return [p.name for p in resume_chain(run_dir)]


def _read_steps_file(run_dir: Path) -> Optional[pd.DataFrame]:
    pq = run_dir / "steps.parquet"
    return pd.read_parquet(pq) if pq.exists() else None


def _read_episodes_file(run_dir: Path) -> Optional[pd.DataFrame]:
    pq = run_dir / "episodes.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    csv = run_dir / "episodes.csv"
    return pd.read_csv(csv) if csv.exists() else None


def _stitch_logs(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate resume-chain tables; keep episode / global_step monotonic."""
    parts: list[pd.DataFrame] = []
    ep_off = 0
    g_off = 0
    for df in frames:
        d = df.copy()
        if "episode" in d.columns and len(d):
            d["episode"] = d["episode"].astype(np.int64) + ep_off
            ep_off = int(d["episode"].max()) + 1
        if "global_step" in d.columns and len(d):
            gmin = int(d["global_step"].min())
            if gmin < g_off:
                d["global_step"] = d["global_step"].astype(np.int64) + g_off
            g_off = int(d["global_step"].max()) + 1
        parts.append(d)
    cat = pd.concat(parts, ignore_index=True)
    if "global_step" in cat.columns:
        subset = ["global_step", "env"] if "env" in cat.columns else ["global_step"]
        cat = cat.sort_values(subset).drop_duplicates(subset, keep="last")
    return cat.reset_index(drop=True)


def _load_steps(run_dir: Path) -> Optional[pd.DataFrame]:
    frames = [df for p in resume_chain(run_dir) if (df := _read_steps_file(p)) is not None]
    if not frames:
        return None
    return frames[0] if len(frames) == 1 else _stitch_logs(frames)


def _load_episodes(run_dir: Path) -> Optional[pd.DataFrame]:
    frames = [df for p in resume_chain(run_dir) if (df := _read_episodes_file(p)) is not None]
    if not frames:
        return None
    return frames[0] if len(frames) == 1 else _stitch_logs(frames)


def _load_config(run_dir: Path) -> dict:
    p = run_dir / "config.yaml"
    if p.exists():
        with p.open() as f:
            return yaml.safe_load(f) or {}
    return {}


def _binned_mean_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 40,
    *,
    n_out: int = 200,
    smooth: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Bin (x, y) and return a smoothed mean and 95% CI half-width."""
    from scipy.ndimage import gaussian_filter1d

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size == 0:
        return None
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi <= lo:
        return np.array([lo]), np.array([float(np.mean(y))]), np.array([0.0])
    bins = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(x, bins) - 1, 0, n_bins - 1)
    centers, means, half = [], [], []
    for i in range(n_bins):
        sl = y[idx == i]
        if sl.size == 0:
            continue
        centers.append(0.5 * (bins[i] + bins[i + 1]))
        means.append(float(sl.mean()))
        if sl.size > 1:
            half.append(1.96 * float(sl.std(ddof=1)) / np.sqrt(sl.size))
        else:
            half.append(0.0)
    if not centers:
        return None
    centers = np.asarray(centers)
    means = np.asarray(means)
    half = np.asarray(half)
    t = np.linspace(float(centers[0]), float(centers[-1]), n_out)
    mu = np.interp(t, centers, means)
    hw = np.interp(t, centers, half)
    if t.size >= 3 and smooth > 0:
        mu = gaussian_filter1d(mu, sigma=smooth, mode="nearest")
        hw = np.maximum(gaussian_filter1d(hw, sigma=smooth, mode="nearest"), 0.0)
    return t, mu, hw


def comparison_dir(output_root: Path, slug_parts: Sequence[str]) -> Path:
    raw = "|".join(slug_parts)
    h = hashlib.sha1(raw.encode()).hexdigest()[:10]
    slug = "_".join(slug_parts)[:80] + "_" + h
    d = output_root / "_comparisons" / slug
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_reward_per_episode(run_dir: Path, out: Optional[Path] = None) -> Path:
    run_dir = Path(run_dir)
    out = out or (run_dir / "figures" / "reward_per_episode.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    steps = _load_steps(run_dir)
    eps = _load_episodes(run_dir)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    metrics = ["progress", "energy_efficiency", "smoothness"]
    colors = ["#2F67B1", "#2E8B57", "#E07A2F"]
    if steps is not None and "episode" in steps.columns:
        for ax, m, c in zip(axes, metrics, colors):
            if m not in steps.columns:
                continue
            how = "sum" if m == "progress" else "mean"
            keys = ["episode", "env"] if "env" in steps.columns else ["episode"]
            ep = steps.groupby(keys, sort=True)[m].agg(how)
            if "env" in steps.columns and steps["env"].nunique() > 1:
                per = ep.groupby("episode")
                roll, std = per.mean(), per.std().fillna(0)
            else:
                if isinstance(ep.index, pd.MultiIndex):
                    ep = ep.droplevel("env")
                roll = ep.rolling(5, center=True, min_periods=1).mean()
                std = ep.rolling(5, min_periods=1).std().fillna(0)
            ax.plot(roll.index, roll.values, color=c, label=m)
            ax.fill_between(roll.index, roll - std, roll + std, color=c, alpha=0.2)
            ax.set_ylabel(m)
            ax.grid(True, alpha=0.3)
    elif eps is not None:
        for ax, m, c in zip(axes, metrics, colors):
            cum = f"cum_{m}"
            if m == "progress":
                col = cum if cum in eps.columns else m
                series = eps[col] if col in eps.columns else None
            elif cum in eps.columns and "steps" in eps.columns:
                series = eps[cum] / eps["steps"].clip(lower=1)
                col = m
            else:
                col = cum if cum in eps.columns else m
                series = eps[col] if col in eps.columns else None
            if series is None:
                continue
            if "env" in eps.columns and "episode" in eps.columns and eps["env"].nunique() > 1:
                tmp = eps.assign(_v=series)
                per = tmp.groupby("episode")["_v"]
                roll, std = per.mean(), per.std().fillna(0)
                x = roll.index
            else:
                x = eps["episode"] if "episode" in eps.columns else eps.index
                roll = series.rolling(5, center=True, min_periods=1).mean()
                std = series.rolling(5, min_periods=1).std().fillna(0)
            ax.plot(x, roll, color=c)
            ax.fill_between(x, roll - std, roll + std, color=c, alpha=0.2)
            ax.set_ylabel(col)
    axes[-1].set_xlabel("episode")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    _write_sidecar(out, {"figure": "reward_per_episode", "run_ids": _chain_ids(run_dir)})
    return out


def plot_episode_length(run_dir: Path, out: Optional[Path] = None) -> Path:
    run_dir = Path(run_dir)
    out = out or (run_dir / "figures" / "episode_length_per_episode.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    eps = _load_episodes(run_dir)
    fig, ax = plt.subplots(figsize=(10, 4))
    if eps is not None and "steps" in eps.columns:
        if "env" in eps.columns and "episode" in eps.columns and eps["env"].nunique() > 1:
            per = eps.groupby("episode", sort=True)["steps"]
            roll, std = per.mean(), per.std().fillna(0)
            x = roll.index.to_numpy()
            y = roll.to_numpy()
            band = std.to_numpy()
        else:
            work = eps.sort_values("episode") if "episode" in eps.columns else eps
            y = work["steps"].rolling(5, center=True, min_periods=1).mean().to_numpy()
            band = work["steps"].rolling(5, center=True, min_periods=1).std().fillna(0).to_numpy()
            x = work["episode"].to_numpy() if "episode" in work.columns else np.arange(len(work))
        ax.plot(x, y, color="gray")
        ax.fill_between(x, np.maximum(y - band, 0.0), y + band, color="gray", alpha=0.2)
    ax.set_xlabel("episode")
    ax.set_ylabel("Episode length (steps)")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    _write_sidecar(out, {"figure": "episode_length_per_episode", "run_ids": _chain_ids(run_dir)})
    return out


def success_rate_figure(success: np.ndarray, window: int) -> plt.Figure:
    """Moving-window success rate. ``success`` is 1 for a successful episode, else 0."""
    fig, ax = plt.subplots(figsize=(10, 4))
    success = np.asarray(success, dtype=float)
    if success.size:
        w = min(max(1, int(window)), int(success.size))
        rate = pd.Series(success).rolling(w, min_periods=max(1, w // 4)).mean()
        ax.plot(np.arange(success.size), rate.to_numpy(), color="#2F67B1", lw=2.0)
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(0.0, color="0.7", lw=0.6)
        ax.axhline(1.0, color="0.7", lw=0.6)
    ax.set_xlabel("Completed episodes")
    ax.set_ylabel("Success rate")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_success_rate(run_dir: Path, out: Optional[Path] = None, window: Optional[int] = None) -> Path:
    """Moving-window success rate over completed env-episodes."""
    run_dir = Path(run_dir)
    out = out or (run_dir / "figures" / "success_rate.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    eps = _load_episodes(run_dir)
    if eps is not None and "reason" in eps.columns and len(eps):
        work = eps.copy()
        sort_cols = [c for c in ("episode", "env") if c in work.columns]
        if sort_cols:
            work = work.sort_values(sort_cols)
        n_env = int(work["env"].nunique()) if "env" in work.columns else 1
        w = window if window is not None else max(32, 8 * n_env)
        success = (work["reason"].astype(str) == "success").to_numpy(dtype=float)
    else:
        w = 1
        success = np.zeros(0)
    fig = success_rate_figure(success, w)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    _write_sidecar(out, {"figure": "success_rate", "run_ids": _chain_ids(run_dir)})
    return out


def plot_progress_vs_episode_length(run_dir: Path, out: Optional[Path] = None) -> Path:
    from fluxswarm.physics.inflow import beat_waveform

    run_dir = Path(run_dir)
    out = out or (run_dir / "figures" / "progress_vs_episode_length.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    steps = _load_steps(run_dir)
    cfg = _load_config(run_dir).get("sim", {})
    amp = float(cfg.get("inflow_velocity", 400.0))
    period = float(cfg.get("inflow_period", 1.0))
    fig, ax = plt.subplots(figsize=(8, 6))
    if steps is not None and "episode" in steps.columns:
        xs = [c for c in steps.columns if c.startswith("location_") and c.endswith("_x")]
        if xs:
            work = steps.copy()
            work["_com_x"] = work[xs].mean(axis=1)
            keys = ["episode", "env"] if "env" in work.columns else ["episode"]
            g = work.groupby(keys, sort=True)
            first = g["_com_x"].first()
            last = g["_com_x"].last()
            dx = last - first
            length = g.size().rename("length")
            if "episode_time" in work.columns:
                t_last = g["episode_time"].last()
            else:
                t_last = pd.Series(0.0, index=dx.index)
            inflow = np.asarray(
                beat_waveform(torch.tensor(t_last.to_numpy(), dtype=torch.float64), amp, period)
            )
            sc = ax.scatter(
                length.to_numpy(),
                dx.to_numpy(),
                c=inflow,
                cmap="coolwarm",
                s=28,
                alpha=0.85,
                edgecolors="none",
                zorder=3,
            )
            cb = fig.colorbar(sc, ax=ax)
            cb.set_label("Last inflow (mm/s)")
    ax.set_xlabel("Episode length (steps)")
    ax.set_ylabel(r"$\Delta x_\mathrm{COM}$ (mm)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    _write_sidecar(out, {"figure": "progress_vs_episode_length", "run_ids": _chain_ids(run_dir)})
    return out


def plot_center_of_mass(run_dir: Path, out: Optional[Path] = None) -> Path:
    from fluxswarm.physics.inflow import beat_waveform

    run_dir = Path(run_dir)
    out = out or (run_dir / "figures" / "center_of_mass.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    steps = _load_steps(run_dir)
    cfg = _load_config(run_dir).get("sim", {})
    amp = float(cfg.get("inflow_velocity", 400.0))
    period = float(cfg.get("inflow_period", 1.0))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    inflow_color = "#C0392B"
    color = "#2F67B1"
    if steps is not None:
        xs = [c for c in steps.columns if c.startswith("location_") and c.endswith("_x")]
        if xs and "episode_time" in steps.columns:
            com_x = steps[xs].mean(axis=1)
            n = len(steps)
            early, late = slice(0, max(1, n // 10)), slice(max(0, n - n // 10), n)
            for i, (ax, sl) in enumerate(zip(axes, [early, late])):
                t = steps["episode_time"].iloc[sl].to_numpy()
                y = com_x.iloc[sl].to_numpy()
                binned = _binned_mean_ci(t, y)
                if binned is not None:
                    xc, mu, half = binned
                    ax.fill_between(xc, mu - half, mu + half, color=color, alpha=0.25, zorder=3)
                    ax.plot(xc, mu, color=color, lw=2.0, zorder=4)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("COM x (mm)")
                tmin, tmax = (float(np.min(t)), float(np.max(t))) if t.size else (0.0, 1.0)
                t_grid = np.linspace(tmin, tmax, 400)
                inflow = np.asarray(
                    beat_waveform(torch.tensor(t_grid, dtype=torch.float64), amp, period)
                )
                ax2 = ax.twinx()
                ax2.plot(t_grid, inflow, color=inflow_color, ls="--", lw=1.6, zorder=1)
                ax2.tick_params(axis="y", colors=inflow_color)
                ax2.set_ylim(min(0.0, float(np.min(inflow))), amp * 1.05)
                if i == 1:
                    ax2.set_ylabel("Inflow (mm/s)", color=inflow_color)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    _write_sidecar(out, {"figure": "center_of_mass", "run_ids": _chain_ids(run_dir)})
    return out


def plot_dispersion(run_dir: Path, out: Optional[Path] = None) -> Path:
    from fluxswarm.physics.inflow import beat_waveform

    run_dir = Path(run_dir)
    out = out or (run_dir / "figures" / "dispersion.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    steps = _load_steps(run_dir)
    cfg = _load_config(run_dir).get("sim", {})
    amp = float(cfg.get("inflow_velocity", 400.0))
    period = float(cfg.get("inflow_period", 1.0))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    inflow_color = "#C0392B"
    if steps is not None and "episode_time" in steps.columns:
        xs = [c for c in steps.columns if c.startswith("location_") and c.endswith("_x")]
        ys = [c for c in steps.columns if c.startswith("location_") and c.endswith("_y")]
        sx = steps[xs].std(axis=1) if xs else None
        sy = steps[ys].std(axis=1) if ys else None
        n = len(steps)
        early, late = slice(0, max(1, n // 10)), slice(max(0, n - n // 10), n)
        specs = [
            (0, 0, early, r"$\sigma_x$ (mm)", "#2F67B1", sx),
            (0, 1, late, r"$\sigma_x$ (mm)", "#2F67B1", sx),
            (1, 0, early, r"$\sigma_y$ (mm)", "#2E8B57", sy),
            (1, 1, late, r"$\sigma_y$ (mm)", "#2E8B57", sy),
        ]
        for row, col, sl, ylabel, color, series in specs:
            ax = axes[row, col]
            t = steps["episode_time"].iloc[sl].to_numpy()
            if series is not None:
                y = series.iloc[sl].to_numpy()
                binned = _binned_mean_ci(t, y)
                if binned is not None:
                    xc, mu, half = binned
                    ax.fill_between(xc, mu - half, mu + half, color=color, alpha=0.25, zorder=3)
                    ax.plot(xc, mu, color=color, lw=2.0, zorder=4)
            ax.set_ylabel(ylabel)
            if row == 1:
                ax.set_xlabel("Time (s)")
            tmin, tmax = (float(np.min(t)), float(np.max(t))) if t.size else (0.0, 1.0)
            t_grid = np.linspace(tmin, tmax, 400)
            inflow = np.asarray(
                beat_waveform(torch.tensor(t_grid, dtype=torch.float64), amp, period)
            )
            ax2 = ax.twinx()
            ax2.plot(t_grid, inflow, color=inflow_color, ls="--", lw=1.6, zorder=1)
            ax2.tick_params(axis="y", colors=inflow_color)
            ax2.set_ylim(min(0.0, float(np.min(inflow))), amp * 1.05)
            if col == 1:
                ax2.set_ylabel("Inflow (mm/s)", color=inflow_color)
            else:
                ax2.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    _write_sidecar(out, {"figure": "dispersion", "run_ids": _chain_ids(run_dir)})
    return out


def plot_velocity_temporal_profile(run_dir: Path, out: Optional[Path] = None) -> Path:
    from fluxswarm.physics.inflow import beat_waveform

    run_dir = Path(run_dir)
    out = out or (run_dir / "figures" / "velocity_temporal_profile.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = _load_config(run_dir).get("sim", {})
    amp = float(cfg.get("inflow_velocity", 400.0))
    period = float(cfg.get("inflow_period", 1.0))
    inflow_color = "#C0392B"
    fields = (
        sorted((run_dir / "fields").glob("step_*.npz"))
        if len(resume_chain(run_dir)) == 1 and (run_dir / "fields").exists()
        else []
    )
    t = vx = vy = None
    if fields:
        ts, mean_vx, mean_vy = [], [], []
        for f in fields:
            d = np.load(f)
            ts.append(float(d["timestep"]))
            mean_vx.append(float(np.mean(d["vx"])))
            mean_vy.append(float(np.mean(d["vy"])))
        t, vx, vy = np.asarray(ts), np.asarray(mean_vx), np.asarray(mean_vy)
    else:
        steps = _load_steps(run_dir)
        if steps is not None:
            vxs = [c for c in steps.columns if c.startswith("velocity_") and c.endswith("_x")]
            vys = [c for c in steps.columns if c.startswith("velocity_") and c.endswith("_y")]
            t = (
                steps["episode_time"].to_numpy()
                if "episode_time" in steps.columns
                else np.arange(len(steps), dtype=float)
            )
            vx = steps[vxs].mean(axis=1).to_numpy() if vxs else None
            vy = steps[vys].mean(axis=1).to_numpy() if vys else None

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    panels = [
        (axes[0], vx, "#2F67B1", r"$v_x$ (mm/s)"),
        (axes[1], vy, "#2E8B57", r"$v_y$ (mm/s)"),
    ]
    for ax, series, color, ylabel in panels:
        if t is not None and series is not None:
            binned = _binned_mean_ci(t, series)
            if binned is not None:
                xc, mu, half = binned
                ax.fill_between(xc, mu - half, mu + half, color=color, alpha=0.25, zorder=3)
                ax.plot(xc, mu, color=color, lw=2.0, zorder=4)
            tmin, tmax = (float(np.min(t)), float(np.max(t))) if t.size else (0.0, 1.0)
            t_grid = np.linspace(tmin, tmax, 400)
            inflow = np.asarray(
                beat_waveform(torch.tensor(t_grid, dtype=torch.float64), amp, period)
            )
            ax2 = ax.twinx()
            ax2.plot(t_grid, inflow, color=inflow_color, ls="--", lw=1.6, zorder=1)
            ax2.tick_params(axis="y", colors=inflow_color)
            ax2.set_ylim(min(0.0, float(np.min(inflow))), amp * 1.05)
            ax2.set_ylabel("Inflow (mm/s)", color=inflow_color)
        ax.set_ylabel(ylabel)
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    _write_sidecar(out, {"figure": "velocity_temporal_profile", "run_ids": _chain_ids(run_dir)})
    return out


def plot_domain(run_dir: Path, out: Optional[Path] = None) -> Path:
    run_dir = Path(run_dir)
    out = out or (run_dir / "figures" / "domain.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = _load_config(run_dir)
    sim, swarm = cfg.get("sim", {}), cfg.get("swarm", {})
    Lx, Ly = sim.get("length_x", 100), sim.get("length_y", 2)
    fig, ax = plt.subplots(figsize=(12, 3.0))
    ax.add_patch(plt.Rectangle((0, 0), Lx, Ly, fill=False, linewidth=2))
    y_in = Ly + 0.22
    ax.annotate(
        "",
        xy=(16, y_in),
        xytext=(5, y_in),
        arrowprops=dict(arrowstyle="->", color="C0", lw=1.6),
        clip_on=False,
    )
    ax.text(10.5, y_in + 0.08, "inflow", color="C0", ha="center", va="bottom", clip_on=False)
    nx, ny = swarm.get("num_x", 8), swarm.get("num_y", 2)
    r = swarm.get("member_radius", 0.25)
    gap = (Ly - 2 * ny * r) / (ny + 1)
    left = Lx / 2 - (2 * (nx - 1) * r + (nx - 1) * gap) / 2
    bottom = gap + r
    for iy in range(ny):
        for ix in range(nx):
            ax.add_patch(
                plt.Circle(
                    (left + ix * (2 * r + gap), bottom + iy * (2 * r + gap)),
                    r,
                    color="0.3",
                    alpha=0.8,
                )
            )
    ax.set_xlim(-2, Lx + 2)
    ax.set_ylim(-0.3, Ly + 0.7)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    _write_sidecar(out, {"figure": "domain", "run_ids": _chain_ids(run_dir)})
    return out


def plot_pcgrad_comparison(with_dir: Path, without_dir: Path, out_dir: Optional[Path] = None) -> Path:
    with_dir, without_dir = Path(with_dir), Path(without_dir)
    out_dir = out_dir or comparison_dir(with_dir.parent, ["pcgrad", with_dir.name, without_dir.name])
    out = Path(out_dir) / "reward_with_without_PCGrad.png"
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for ax, m in zip(axes, ["progress", "energy_efficiency", "smoothness"]):
        for run, color, label in [
            (with_dir, "#2F67B1", "with PCGrad"),
            (without_dir, "#BF2C23", "without PCGrad"),
        ]:
            steps = _load_steps(run)
            if steps is None or m not in steps.columns:
                continue
            g = steps.groupby("episode")[m].mean().rolling(5, center=True, min_periods=1).mean()
            ax.plot(g.index, g.values, color=color, label=label)
        ax.set_ylabel(m)
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("episode")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    _write_sidecar(
        out,
        {
            "figure": "reward_with_without_PCGrad",
            "run_ids": _chain_ids(with_dir) + _chain_ids(without_dir),
        },
    )
    return out


def generate_all_for_run(run_dir: Path, *, video: bool = False) -> list[Path]:
    run_dir = Path(run_dir)
    ids = _chain_ids(run_dir)
    if len(ids) > 1:
        print(f"Stitching resume chain: {' -> '.join(ids)}")
    paths = [
        plot_reward_per_episode(run_dir),
        plot_episode_length(run_dir),
        plot_success_rate(run_dir),
        plot_progress_vs_episode_length(run_dir),
        plot_center_of_mass(run_dir),
        plot_dispersion(run_dir),
        plot_velocity_temporal_profile(run_dir),
        plot_domain(run_dir),
    ]
    if video:
        try:
            from fluxswarm.analysis.swarm_viz import animate_swarm_from_run

            anim = animate_swarm_from_run(run_dir, steps=_load_steps(run_dir))
            if anim is not None:
                paths.append(anim)
        except Exception:
            pass
    return paths
