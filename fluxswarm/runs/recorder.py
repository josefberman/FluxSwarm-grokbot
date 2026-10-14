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
        self._step_flush_every = 256

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
        if not self._steps:
            return
        df = pd.DataFrame(self._steps)
        path = self.run_dir / "steps.parquet"
        if path.exists():
            prev = pd.read_parquet(path)
            df = pd.concat([prev, df], ignore_index=True)
        df.to_parquet(path, index=False)
        self._steps.clear()

    def close(self) -> None:
        self._flush_steps()
        if self._episodes:
            ep_path = self.run_dir / "episodes.parquet"
            df = pd.DataFrame(self._episodes)
            if ep_path.exists():
                prev = pd.read_parquet(ep_path)
                df = pd.concat([prev, df], ignore_index=True)
            df.to_parquet(ep_path, index=False)
            self._episodes.clear()
            df.to_csv(self.run_dir / "episodes.csv", index=False)
