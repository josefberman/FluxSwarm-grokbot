"""Run folder recorder: config snapshot, parquet logs, npz fields."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from fluxswarm.config import Config


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def make_run_id(tag: str = "") -> str:
    now = datetime.now()
    base = now.strftime("%Y-%m-%d_%H-%M-%S")
    return f"{base}_{tag}" if tag else base


def _part_count(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for p in directory.glob("part-*.parquet"))


def _append_parquet(rows: list[dict[str, Any]], directory: Path, part: int) -> int:
    """Write rows to a new part file. ``pd.read_parquet(directory)`` still concatenates them."""
    if not rows:
        return part
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(directory / f"part-{part:06d}.parquet", index=False)
    return part + 1


class RunRecorder:
    def __init__(self, cfg: Config, algorithm: str = "momappo", run_id: Optional[str] = None):
        self.cfg = cfg
        self.algorithm = algorithm
        root = Path(cfg.run.output_root)
        root.mkdir(parents=True, exist_ok=True)
        tag = cfg.run.tag or algorithm
        self.run_id = run_id or make_run_id(tag)
        self.run_dir = root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "fields").mkdir(exist_ok=True)
        (self.run_dir / "figures").mkdir(exist_ok=True)
        self._steps: list[dict[str, Any]] = []
        self._episodes: list[dict[str, Any]] = []
        # One new part file per flush. Never reread the log: a full rewrite is
        # O(rows²) and dominates the update once the run is long.
        self._step_flush_every = 8192
        self._episode_flush_every = 1024
        self._step_part = _part_count(self.run_dir / "steps.parquet")
        self._episode_part = _part_count(self.run_dir / "episodes.parquet")

    def save_config(self, cfg: Config) -> None:
        cfg.save(self.run_dir / "config.yaml")
        meta = {
            "algorithm": self.algorithm,
            "git_commit": _git_commit(),
            "run_id": self.run_id,
            "use_pcgrad": cfg.train.use_pcgrad,
            "resume_from": cfg.train.resume_from,
        }
        with (self.run_dir / "meta.json").open("w") as f:
            json.dump(meta, f, indent=2)

    def log_step(
        self,
        global_step: int,
        env,
        scalar_rewards: np.ndarray,
        info: dict,
        save_fields: bool = False,
    ) -> None:
        B = env.batch
        pos = env.get_positions()
        vel = env.get_velocities()
        act = env.get_actions()
        for b in range(B):
            row: dict[str, Any] = {
                "global_step": global_step,
                "env": b,
                "episode": int(env.episode_id[b]),
                "episode_time": float(env.solver.episode_time[b]),
                "step_reward": float(scalar_rewards[b]),
            }
            objs = info["infos"][b]["objectives"]
            row.update(objs)
            for i in range(env.n):
                row[f"location_{i}_x"] = float(pos[b, i, 0])
                row[f"location_{i}_y"] = float(pos[b, i, 1])
                row[f"velocity_{i}_x"] = float(vel[b, i, 0])
                row[f"velocity_{i}_y"] = float(vel[b, i, 1])
                row[f"action_{i}_x"] = float(act[b, i, 0])
                row[f"action_{i}_y"] = float(act[b, i, 1])
            self._steps.append(row)

            ep = info["infos"][b].get("episode")
            if ep is not None:
                self._episodes.append(
                    {
                        "env": b,
                        "episode": int(env.episode_id[b]) - 1,
                        **ep,
                    }
                )
                if len(self._episodes) >= self._episode_flush_every:
                    self._flush_episodes()

        if save_fields:
            # Fluid snapshots stay env-0 only; step/episode rows log every env.
            snap = env.solver.export_fields(0)
            path = self.run_dir / "fields" / f"step_{global_step:06d}.npz"
            np.savez_compressed(
                path,
                vx=snap.vx,
                vy=snap.vy,
                p=snap.p,
                timestep=snap.t,
                length_x=env.cfg.sim.length_x,
                length_y=env.cfg.sim.length_y,
                resolution=np.array(env.cfg.sim.resolution),
            )

        if len(self._steps) >= self._step_flush_every:
            self._flush_steps()

    def _flush_steps(self) -> None:
        self._step_part = _append_parquet(
            self._steps, self.run_dir / "steps.parquet", self._step_part
        )
        self._steps.clear()

    def _flush_episodes(self) -> None:
        self._episode_part = _append_parquet(
            self._episodes, self.run_dir / "episodes.parquet", self._episode_part
        )
        self._episodes.clear()

    def close(self) -> None:
        self._flush_steps()
        self._flush_episodes()
        ep_path = self.run_dir / "episodes.parquet"
        if _part_count(ep_path):
            pd.read_parquet(ep_path).to_csv(self.run_dir / "episodes.csv", index=False)
