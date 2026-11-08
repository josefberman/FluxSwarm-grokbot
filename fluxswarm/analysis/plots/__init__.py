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


_EPISODE_COLUMNS = (
    "episode",
    "env",
    "steps",
    "reason",
    "status",
    "cum_progress",
    "cum_energy_efficiency",
    "cum_smoothness",
    "global_step",
)

# Figure extraction never materializes the step log. A long run is tens of
# millions of wide rows; loading it (once per plot) is what gets the process killed.
_MAX_STEP_SAMPLE = 80_000
_MAX_LINE_POINTS = 2_000
_MAX_SCATTER = 20_000
_summary_cache: dict[tuple[str, ...], Optional["_StepSummary"]] = {}


def _read_parquet_frame(path: Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    import pyarrow.dataset as ds

    dataset = ds.dataset(str(path), format="parquet")
    cols = None
    if columns is not None:
        have = set(dataset.schema.names)
        cols = [c for c in columns if c in have] or None
    return dataset.to_table(columns=cols).to_pandas()


def _read_episodes_file(run_dir: Path) -> Optional[pd.DataFrame]:
    pq = run_dir / "episodes.parquet"
    if pq.exists():
        return _read_parquet_frame(pq, _EPISODE_COLUMNS)
    csv = run_dir / "episodes.csv"
    if not csv.exists():
        return None
    return pd.read_csv(csv, usecols=lambda c: c in _EPISODE_COLUMNS)


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


class _StepSummary:
    """Bounded stand-in for the step log: one row per episode, plus a stride sample."""

    def __init__(self, n_rows: int, sample: pd.DataFrame, episodes: pd.DataFrame):
        self.n_rows = int(n_rows)
        self.sample = sample
        self.episodes = episodes


def _thin_line(
    x,
    y,
    band=None,
    max_points: int = _MAX_LINE_POINTS,
    how: str = "mean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Reduce a line to ``max_points`` so matplotlib is not handed millions of vertices."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    b = None if band is None else np.asarray(band, dtype=float)
    n = int(x.size)
    if n <= max_points or max_points < 2:
        return x, y, b
    if how == "stride":
        k = int(np.ceil(n / max_points))
        sl = slice(None, None, k)
        return x[sl], y[sl], None if b is None else b[sl]
    edges = np.linspace(0, n, max_points + 1).astype(int)
    xs, ys, bs = [], [], []
    for a, c in zip(edges[:-1], edges[1:]):
        if c <= a:
            continue
        xs.append(float(np.nanmean(x[a:c])))
        ys.append(float(np.nanmean(y[a:c])))
        if b is not None:
            bs.append(float(np.nanmean(b[a:c])))
    return np.asarray(xs), np.asarray(ys), (np.asarray(bs) if b is not None else None)


def _plot_band(ax, x, y, band, color, **plot_kw) -> None:
    x, y, band = _thin_line(x, y, band)
    ax.plot(x, y, color=color, **plot_kw)
    if band is not None:
        ax.fill_between(x, y - band, y + band, color=color, alpha=0.2)


def _step_sources(run_dir: Path) -> list[Path]:
    return [p / "steps.parquet" for p in resume_chain(run_dir) if (p / "steps.parquet").exists()]


def _step_summary(run_dir: Path) -> Optional[_StepSummary]:
    sources = _step_sources(run_dir)
    key = tuple(str(p.resolve()) for p in sources)
    if key in _summary_cache:
        return _summary_cache[key]
    summary = _scan_steps(sources) if sources else None
    _summary_cache[key] = summary
    return summary


def _scan_steps(sources: list[Path]) -> Optional[_StepSummary]:
    """One single-threaded pass. Keeps per-episode aggregates and a stride sample."""
    import pyarrow as pa
    import pyarrow.dataset as ds

    plans: list[tuple[int, object, list[str]]] = []
    total = 0
    for source_i, path in enumerate(sources):
        dataset = ds.dataset(str(path), format="parquet")
        names = list(dataset.schema.names)
        fragments = sorted(dataset.get_fragments(), key=lambda f: f.path)
        for frag in fragments:
            total += int(frag.count_rows())
            plans.append((source_i, frag, names))
    if total <= 0:
        return None

    stride = max(1, total // _MAX_STEP_SAMPLE)
    if total > 500_000:
        print(
            f"Scanning {total} step rows for figures (one pass, stride {stride})",
            flush=True,
        )

    held: dict[str, list[np.ndarray]] = {k: [] for k in (
        "row", "episode_time", "com_x", "sig_x", "sig_y", "vx", "vy",
    )}
    epi: dict[str, list] = {k: [] for k in (
        "episode", "env", "length", "dx", "t_last", "progress_sum", "energy_sum", "smooth_sum",
    )}
    states: dict[int, dict] = {}
    row_base = 0
    ep_offset = 0
    wanted = ("episode", "env", "episode_time", "progress", "energy_efficiency", "smoothness")
    prev_threads = pa.cpu_count()
    pa.set_cpu_count(1)
    try:
        current_source = -1
        max_ep = -1
        for source_i, frag, names in plans:
            if source_i != current_source:
                if current_source >= 0:
                    _flush_open_episodes(states, epi, ep_offset)
                    if max_ep >= 0:
                        ep_offset += max_ep + 1
                current_source = source_i
                max_ep = -1
            name_set = set(names)
            xcols = sorted(c for c in names if c.startswith("location_") and c.endswith("_x"))
            ycols = sorted(c for c in names if c.startswith("location_") and c.endswith("_y"))
            vxcols = sorted(c for c in names if c.startswith("velocity_") and c.endswith("_x"))
            vycols = sorted(c for c in names if c.startswith("velocity_") and c.endswith("_y"))
            cols = [c for c in wanted if c in name_set] + xcols + ycols + vxcols + vycols
            for batch in frag.to_batches(columns=cols or None, batch_size=65_536):
                pdf = batch.to_pandas()
                n = len(pdf)
                if n == 0:
                    continue
                ep = pdf["episode"].to_numpy() if "episode" in pdf.columns else np.zeros(n)
                env = pdf["env"].to_numpy() if "env" in pdf.columns else np.zeros(n, dtype=np.int64)
                t = (
                    pdf["episode_time"].to_numpy(dtype=float)
                    if "episode_time" in pdf.columns
                    else np.zeros(n)
                )
                prog = pdf["progress"].to_numpy(dtype=float) if "progress" in pdf.columns else np.zeros(n)
                energy = (
                    pdf["energy_efficiency"].to_numpy(dtype=float)
                    if "energy_efficiency" in pdf.columns
                    else np.zeros(n)
                )
                smooth = (
                    pdf["smoothness"].to_numpy(dtype=float)
                    if "smoothness" in pdf.columns
                    else np.zeros(n)
                )
                com = pdf[xcols].to_numpy(dtype=float).mean(axis=1) if xcols else np.full(n, np.nan)
                sig_x = pdf[xcols].to_numpy(dtype=float).std(axis=1, ddof=1) if len(xcols) > 1 else np.full(n, np.nan)
                sig_y = pdf[ycols].to_numpy(dtype=float).std(axis=1, ddof=1) if len(ycols) > 1 else np.full(n, np.nan)
                vx = pdf[vxcols].to_numpy(dtype=float).mean(axis=1) if vxcols else np.full(n, np.nan)
                vy = pdf[vycols].to_numpy(dtype=float).mean(axis=1) if vycols else np.full(n, np.nan)
                idx = np.arange(row_base, row_base + n)
                keep = (idx % stride) == 0
                if keep.any():
                    held["row"].append(idx[keep])
                    held["episode_time"].append(t[keep])
                    held["com_x"].append(com[keep])
                    held["sig_x"].append(sig_x[keep])
                    held["sig_y"].append(sig_y[keep])
                    held["vx"].append(vx[keep])
                    held["vy"].append(vy[keep])
                finite_ep = ep[np.isfinite(ep.astype(float))] if ep.dtype.kind == "f" else ep
                if len(finite_ep):
                    max_ep = max(max_ep, int(np.max(finite_ep)))
                _consume_episodes(states, epi, ep_offset, env, ep, com, t, prog, energy, smooth)
                row_base += n
        _flush_open_episodes(states, epi, ep_offset)
    finally:
        pa.set_cpu_count(prev_threads)

    sample = (
        pd.DataFrame({k: np.concatenate(v) for k, v in held.items()})
        if held["row"]
        else pd.DataFrame(columns=list(held))
    )
    episodes = pd.DataFrame(epi) if epi["episode"] else pd.DataFrame(columns=list(epi))
    return _StepSummary(total, sample, episodes)


def _flush_open_episodes(states: dict[int, dict], epi: dict[str, list], offset: int) -> None:
    for env, st in states.items():
        epi["episode"].append(int(st["episode"]) + offset)
        epi["env"].append(int(env))
        epi["length"].append(int(st["n"]))
        epi["dx"].append(float(st["last"] - st["first"]))
        epi["t_last"].append(float(st["t"]))
        epi["progress_sum"].append(float(st["p"]))
        epi["energy_sum"].append(float(st["e"]))
        epi["smooth_sum"].append(float(st["s"]))
    states.clear()


def _consume_episodes(
    states: dict[int, dict],
    epi: dict[str, list],
    offset: int,
    env: np.ndarray,
    ep: np.ndarray,
    com: np.ndarray,
    t: np.ndarray,
    prog: np.ndarray,
    energy: np.ndarray,
    smooth: np.ndarray,
) -> None:
    for e in np.unique(env):
        mask = env == e
        ep_e = ep[mask]
        if ep_e.size == 0:
            continue
        com_e, t_e = com[mask], t[mask]
        p_e, en_e, sm_e = prog[mask], energy[mask], smooth[mask]
        cuts = np.flatnonzero(ep_e[1:] != ep_e[:-1]) + 1
        starts = np.concatenate(([0], cuts))
        ends = np.concatenate((cuts, [ep_e.size]))
        st = states.get(int(e))
        key = int(e)
        for a, b in zip(starts, ends):
            eid = ep_e[a]
            eid_i = int(eid) if np.isfinite(float(eid)) else -1
            if st is None or st["episode"] != eid_i:
                if st is not None:
                    epi["episode"].append(int(st["episode"]) + offset)
                    epi["env"].append(key)
                    epi["length"].append(int(st["n"]))
                    epi["dx"].append(float(st["last"] - st["first"]))
                    epi["t_last"].append(float(st["t"]))
                    epi["progress_sum"].append(float(st["p"]))
                    epi["energy_sum"].append(float(st["e"]))
                    epi["smooth_sum"].append(float(st["s"]))
                st = {
                    "episode": eid_i,
                    "first": float(com_e[a]),
                    "last": float(com_e[b - 1]),
                    "n": int(b - a),
                    "t": float(t_e[b - 1]),
                    "p": float(p_e[a:b].sum()),
                    "e": float(en_e[a:b].sum()),
                    "s": float(sm_e[a:b].sum()),
                }
            else:
                st["last"] = float(com_e[b - 1])
                st["n"] += int(b - a)
                st["t"] = float(t_e[b - 1])
                st["p"] += float(p_e[a:b].sum())
                st["e"] += float(en_e[a:b].sum())
                st["s"] += float(sm_e[a:b].sum())
        if st is not None:
            states[key] = st


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


def _env_episode_order(eps: pd.DataFrame) -> pd.DataFrame:
    """One row per finished env-episode, ordered so every env is on the x-axis."""
    cols = [c for c in ("episode", "env") if c in eps.columns]
    if not cols:
        return eps
    return eps.sort_values(cols, kind="mergesort")


def plot_reward_per_episode(run_dir: Path, out: Optional[Path] = None) -> Path:
    run_dir = Path(run_dir)
    out = out or (run_dir / "figures" / "reward_per_episode.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    eps = _load_episodes(run_dir)
    if eps is None or "cum_progress" not in eps.columns:
        summary = _step_summary(run_dir)
        if summary is not None and len(summary.episodes):
            ep = summary.episodes
            eps = pd.DataFrame({
                "episode": ep["episode"].to_numpy(),
                "env": ep["env"].to_numpy(),
                "steps": ep["length"].to_numpy(),
                "cum_progress": ep["progress_sum"].to_numpy(),
                "cum_energy_efficiency": ep["energy_sum"].to_numpy(),
                "cum_smoothness": ep["smooth_sum"].to_numpy(),
            })
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    metrics = ["progress", "energy_efficiency", "smoothness"]
    colors = ["#2F67B1", "#2E8B57", "#E07A2F"]
    if eps is not None:
        work = _env_episode_order(eps)
        x_all = np.arange(len(work), dtype=float)
        for ax, m, c in zip(axes, metrics, colors):
            cum = f"cum_{m}"
            if m == "progress":
                col = cum if cum in work.columns else m
                series = work[col] if col in work.columns else None
            elif cum in work.columns and "steps" in work.columns:
                series = work[cum] / work["steps"].clip(lower=1)
                col = m
            else:
                col = cum if cum in work.columns else m
                series = work[col] if col in work.columns else None
            if series is None:
                continue
            roll = series.rolling(5, center=True, min_periods=1).mean()
            x, y, _ = _thin_line(x_all, roll.to_numpy(dtype=float), how="stride")
            ax.plot(x, y, color=c)
            ax.set_ylabel(col)
    axes[-1].set_xlabel("Completed episodes")
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
        x, y, band = _thin_line(x, y, band)
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
        x, y, _ = _thin_line(np.arange(success.size), rate.to_numpy(), how="stride")
        ax.plot(x, y, color="#2F67B1", lw=2.0)
        ax.set_ylim(-0.02, 1.02)
        ax.axhline(0.0, color="0.7", lw=0.6)
        ax.axhline(1.0, color="0.7", lw=0.6)
    ax.set_xlabel("Completed episodes")
    ax.set_ylabel("Success rate")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_success_rate(run_dir: Path, out: Optional[Path] = None, window: Optional[int] = None) -> Path:
    """Rolling success rate over every finished env-episode, in the same order as reward."""
    run_dir = Path(run_dir)
    out = out or (run_dir / "figures" / "success_rate.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    eps = _load_episodes(run_dir)
    if eps is not None and "reason" in eps.columns and len(eps):
        work = _env_episode_order(eps)
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
    summary = _step_summary(run_dir)
    cfg = _load_config(run_dir).get("sim", {})
    amp = float(cfg.get("inflow_velocity", 400.0))
    period = float(cfg.get("inflow_period", 1.0))
    fig, ax = plt.subplots(figsize=(8, 6))
    if summary is not None and len(summary.episodes):
        ep = summary.episodes
        if len(ep) > _MAX_SCATTER:
            ep = ep.sample(n=_MAX_SCATTER, random_state=0)
        inflow = np.asarray(
            beat_waveform(torch.tensor(ep["t_last"].to_numpy(), dtype=torch.float64), amp, period)
        )
        sc = ax.scatter(
            ep["length"].to_numpy(),
            ep["dx"].to_numpy(),
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
    summary = _step_summary(run_dir)
    cfg = _load_config(run_dir).get("sim", {})
    amp = float(cfg.get("inflow_velocity", 400.0))
    period = float(cfg.get("inflow_period", 1.0))
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    inflow_color = "#C0392B"
    color = "#2F67B1"
    if summary is not None and len(summary.sample):
        sample = summary.sample
        n = max(summary.n_rows, 1)
        windows = (sample[sample["row"] < n * 0.1], sample[sample["row"] >= n * 0.9])
        for i, (ax, sl) in enumerate(zip(axes, windows)):
            if sl.empty:
                continue
            t = sl["episode_time"].to_numpy()
            y = sl["com_x"].to_numpy()
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
    summary = _step_summary(run_dir)
    cfg = _load_config(run_dir).get("sim", {})
    amp = float(cfg.get("inflow_velocity", 400.0))
    period = float(cfg.get("inflow_period", 1.0))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
    inflow_color = "#C0392B"
    if summary is not None and len(summary.sample):
        sample = summary.sample
        n = max(summary.n_rows, 1)
        early = sample["row"] < n * 0.1
        late = sample["row"] >= n * 0.9
        specs = [
            (0, 0, early, r"$\sigma_x$ (mm)", "#2F67B1", "sig_x"),
            (0, 1, late, r"$\sigma_x$ (mm)", "#2F67B1", "sig_x"),
            (1, 0, early, r"$\sigma_y$ (mm)", "#2E8B57", "sig_y"),
            (1, 1, late, r"$\sigma_y$ (mm)", "#2E8B57", "sig_y"),
        ]
        for row, col, sl, ylabel, color, series in specs:
            ax = axes[row, col]
            part = sample.loc[sl]
            t = part["episode_time"].to_numpy()
            if series in part.columns and len(part):
                y = part[series].to_numpy()
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
    if len(fields) > _MAX_LINE_POINTS:
        fields = fields[:: max(1, int(np.ceil(len(fields) / _MAX_LINE_POINTS)))]
    if fields:
        ts, mean_vx, mean_vy = [], [], []
        for f in fields:
            d = np.load(f)
            ts.append(float(d["timestep"]))
            mean_vx.append(float(np.mean(d["vx"])))
            mean_vy.append(float(np.mean(d["vy"])))
        t, vx, vy = np.asarray(ts), np.asarray(mean_vx), np.asarray(mean_vy)
    else:
        summary = _step_summary(run_dir)
        if summary is not None and len(summary.sample):
            s = summary.sample
            t = s["episode_time"].to_numpy()
            vx = s["vx"].to_numpy()
            vy = s["vy"].to_numpy()

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
            summary = _step_summary(run)
            if summary is None or not len(summary.episodes):
                continue
            ep = summary.episodes
            col = {"progress": "progress_sum", "energy_efficiency": "energy_sum", "smoothness": "smooth_sum"}[m]
            agg = ep.groupby("episode", sort=True)[[col, "length"]].sum()
            g = (agg[col] / agg["length"].clip(lower=1)).rolling(5, center=True, min_periods=1).mean()
            x, y, _ = _thin_line(g.index.to_numpy(), g.to_numpy(), how="stride")
            ax.plot(x, y, color=color, label=label)
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

            anim = animate_swarm_from_run(run_dir)
            if anim is not None:
                paths.append(anim)
        except Exception:
            pass
    return paths
