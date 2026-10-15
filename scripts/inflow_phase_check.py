#!/usr/bin/env python3
"""Check whether each env's inflow follows its own episode clock.

TorchFluidSolver.substep_loop receives t0 = mean(episode_time) over the batch,
so after desynchronized resets every env is driven by the *batch-mean* clock.
This script rolls random actions in cheap physics and compares the inlet
centerline velocity of each env with beat_waveform(own episode time).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from fluxswarm.config import Config
from fluxswarm.envs.swarm_env import BatchedSwarmEnv
from fluxswarm.physics.inflow import beat_waveform

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
cfg = Config()
cfg.task.progress_reward = "legacy"
cfg.obs.localization = "full"
cfg.train.use_action_x_prior = False
cfg.sim.resolution_scale = 2.0
cfg.sim.dt_substeps = 4
cfg.sim.per_env_clock = len(sys.argv) > 3 and sys.argv[3] == "fixed"
env = BatchedSwarmEnv(cfg, batch=16, device="cpu")
env.reset(seed=0)
rng = np.random.default_rng(0)
fl = env.solver.torch_fluid
ny = fl.u.shape[-1]
par_c = float(fl.parabolic.max())
rows = []
for t in range(steps):
    a = rng.uniform(-1, 1, size=(16, env.n, 2)).astype(np.float32) * 0.7
    a[..., 0] += rng.normal(0.0, 0.3, size=(16, 1))  # per-env bias => different episode lengths
    env.step(np.clip(a, -1, 1))
    own_t = env.solver.episode_time.numpy()
    u_in = fl.u[:, 0, :].max(dim=-1).values.numpy() / par_c
    rows.append({"step": t, "u_in": u_in.tolist(), "own": [float(beat_waveform(x, 1.0, 1.0)) for x in own_t],
                 "mean_t": float(own_t.mean())})
u = np.array([r["u_in"] for r in rows])
own = np.array([r["own"] for r in rows]) * cfg.sim.inflow_velocity
err = u - own
res = {"steps": steps, "inflow_velocity": cfg.sim.inflow_velocity,
       "mean_abs_err_vs_own_clock": float(np.abs(err).mean()),
       "p95_abs_err": float(np.percentile(np.abs(err), 95)),
       "max_abs_err": float(np.abs(err).max()),
       "cross_env_corr_u": float(np.nanmean(np.corrcoef(u[200:].T)[np.triu_indices(16, 1)])),
       "cross_env_corr_own_clock": float(np.nanmean(np.corrcoef(own[200:].T)[np.triu_indices(16, 1)]))}
print(json.dumps(res, indent=1))
out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
if out:
    out.write_text(json.dumps(res, indent=1))
    np.savez_compressed(out.with_suffix(".npz"), u=u, own=own)
