# FluxSwarm

Physics-informed multi-objective multi-agent RL for controlling a microrobot swarm in a 2D fluid channel.

## Highlights

- **GPU Brinkman/DCT fluid solver** — two-way coupling; all environments share one process and a GPU batch dimension.
- **MOMAPPO** — CTDE multi-objective PPO (progress, energy, smoothness) with corrected tanh-Gaussian log-probs, per-agent ratios, and actor-only PCGrad.
- **Position-blind local observations** with an observability ladder (`--obs-localization`).
- **Baselines**: brute-upstream, brute-wall, and standard single-objective PPO.
- **Outputs** under `runs_new/` (legacy `run/` is never touched).

## Install

```bash
conda env create -f environment.yml
conda activate fluxswarm
```

`environment.yml` installs the project in editable mode (`pip install -e .`). Recreate or update with `conda env update -f environment.yml --prune`.

## Train

```bash
python -m fluxswarm.cli train \
  --batch-envs 8 \
  --total-timesteps-per-env 25000 \
  --obs-localization none \
  --progress-reward potential \
  --tag momappo_exp1
```

Prefer `--progress-reward potential` (dense upstream displacement shaping) over `legacy` for stable learning under pulsatile inflow.

Continue a finished run in a **new** folder (weights, optimizer, obs-norm, global steps; no action-x-prior warmup). `--total-timesteps-per-env` is additional budget:

```bash
python -m fluxswarm.cli train \
  --resume-from 2026-09-21_11-30-51_momappo \
  --total-timesteps-per-env 25000 \
  --tag continued
```

Live swarm positions appear in TensorBoard under `swarm/positions` (every `--live-swarm-tb-every` steps, default 100). After training a subsampled MP4 is written to `figures/swarm_motion.mp4`. The figures CLI writes it only with `--video`.

## Baselines

```bash
python -m fluxswarm.cli baseline --policy upstream --max-steps 5000
python -m fluxswarm.cli baseline --policy wall --wall-policy-mode static
python -m fluxswarm.cli baseline --policy ppo --total-timesteps-per-env 25000
```

## Figures

```bash
python -m fluxswarm.cli figures --run runs_new/<run_id>
python -m fluxswarm.cli figures --run runs_new/<run_id> --video
python -m fluxswarm.cli figures --run runs_new/<with_pcgrad> --compare-pcgrad runs_new/<without>
```

Single-run figures land in `runs_new/<run_id>/figures/`. Comparisons go to `runs_new/_comparisons/<slug>/`.

## Key flags

| Flag | Meaning |
|------|---------|
| `--obs-localization {none,imu,displacement,absolute-y,full}` | Localization ablation |
| `--progress-reward {potential,fluid-relative,legacy}` | Progress objective (prefer `potential`) |
| `--dt-substeps` | Fluid substeps per RL step (default 40; CFL headroom) |
| `--live-swarm-tb-every` | TB swarm image every N steps (default 100; `0`=off) |
| `--resume-from` | Former run folder name or path; new run continues from that checkpoint |
| `--no-pcgrad` | Disable gradient surgery |
| `--output-root` | Default `runs_new` |
| `--batch-envs` | Env batch size (default 64) |

See `python -m fluxswarm.cli train --help` for the full list.

## Layout

```
fluxswarm/
  config.py          CLI + dataclasses
  physics/           GPU Brinkman/DCT two-way fluid + swarm forces
  envs/              BatchedSwarmEnv, observations, rewards
  agents/            MOMAPPO, networks, PCGrad
  baselines/         brute + PPO
  runs/              recorder → runs_new/
  analysis/plots/    figure generators
  cli/               train / evaluate / baseline / figures
```

## Citation

See `CITATION.cff`. License: Apache-2.0.
