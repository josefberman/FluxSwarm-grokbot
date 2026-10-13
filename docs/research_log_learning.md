# MOMAPPO learning diagnosis

Fixed owner flags (must keep):

```bash
python -m fluxswarm.cli train \
  --progress-reward legacy \
  --obs-localization full \
  --no-action-x-prior \
  --batch-envs 16 \
  --total-timesteps-per-env 100000
```

Hardware: CPU-only, 4 Xeon cores, no GPU. Dummy-env pilots isolate the trainer. Cheap physics uses `resolution_scale=2` (200×4 grid) and `dt_substeps=4`. A full 1000×20 × 40-substep fluid 16×100k run is not CPU-feasible.

## Root causes (checked in this order)

### Trainer / env bugs

1. **PPO minibatch size default was 4.** With `n_steps=200` and 16 envs that is ~800 SGD steps per epoch × 4 epochs on a single rollout. Dummy baseline success_rate = **0.00** (2 seeds). `--batch-size 256` → **0.55**.
2. **Symmetric PCGrad projects the progress gradient away** when it conflicts with energy (`1−‖a‖`). Protecting progress (never project it off energy/smoothness) → dummy success **0.19** alone, and **0.67–0.82** stacked on minibatch 256.
3. **Per-objective advantage whitening cancels `w_*`** and amplifies near-constant objectives. Independent unit-variance scaling makes `w_progress=9` identical to `w_progress=1`. Degenerate std now zeros that head; `adv_norm=weighted` re-applies task weights after whitening.
4. **Entropy collapse on long runs.** Allowing `log_std` below −2 let the tanh-Gaussian lock onto a hover policy. Dummy 16 envs × 100k: success **1.0 at 89.6k global steps**, then **0** by update 50 (entropy −6.8). Restored `LOG_STD_MIN=-2` and added `target_kl=0.02`. Rerun: **final success 0.977**, tail **0.975**.
5. **Auto-reset returned the terminal observation** while the simulator was already in the next episode. The next action was computed on the previous episode’s last state.
6. **`--no-action-x-prior` did not disable** the solver’s first-step forced `[-1, 0]` (log-prob ≠ executed action).
7. **Observation history was not cleared** on auto-reset, so stacked frames leaked across episode boundaries.

### Hypotheses that did **not** help (dummy, 2 seeds, 50 updates)

| Change | tail success vs baseline 0.00 |
|--------|-------------------------------:|
| orthogonal init alone | 0.00 |
| `lr=3e-4` alone | 0.00 |
| `adv_norm=weighted` alone (weights 1,1,1) | 0.00 |
| all knobs at once (ortho + low LR + …) | ~0.00 (hurt) |
| minibatch 256 + ortho | 0.36 (worse than mb256 alone 0.55) |
| minibatch 256 + `lr=3e-4` | 0.31 (worse) |

## Dummy-env results

**Scale A** — 8 envs, 4 agents, `n_steps=32`, 50 updates (~1.6k steps/env), 2 seeds, no x-prior, legacy-like progress.

| Variant | tail success | final mean act_x |
|---------|-------------:|-----------------:|
| baseline (`batch_size=4`) | **0.00** | ~0 / −0.2 |
| minibatch 256 | **0.55** | −0.51 / −0.60 |
| protect-progress only | **0.19** | −0.49 / −0.52 |
| **mb256 + protect + weighted** | **0.67 / 0.82** | **−0.71 / −0.73** |

Plots: `dummy_success_rate.png`, `dummy_success_rate_focus.png`, `dummy_mean_x.png`, `dummy_mean_act_x.png`.

**Scale B** — dummy **16 envs × 100k steps/env**, 16 agents, `n_steps=200` (requested parallelism × budget; point-mass dynamics).

| Run | peak success | final success | notes |
|-----|-------------:|--------------:|-------|
| winner, `LOG_STD_MIN=-5` | **1.00** (update 27) | **0.00** | catastrophic forgetting; `dummy_16x100k_collapse.png` |
| winner, floor −2 + `target_kl=0.02` | **1.00** | **0.977** (tail 0.975) | `dummy_16x100k_before_after.png` |
| 80-update check after floor | 1.00 | 0.88 | entropy floored at −1.16 |

## Cheap physics

Brute-upstream, 120 steps, 200×4 grid: mean x **50 → 30.9** (`physics_brute.json`). Upstream swimming is possible.

**16 envs × 4k steps/env** (20 PPO updates, 8 members? no — default 16 members, 200×4, 4 substeps):

| | peak success | final success | final act_x |
|--|-------------:|--------------:|------------:|
| old hparams (`batch_size=4`, vanilla PCGrad) | 0.50 | 0.34 | +0.018 |
| winner | **1.00** | **0.42** | **−0.091** |

Plot: `physics_16env_4000_compare.png`.

A 16 env × 100k cheap-physics winner run is in `runs_new/_research/` (CPU ~0.018 s/env-step plus PPO). Full-resolution fluid at this budget is not CPU-feasible (~0.05 s/step already on a 400×8 × 8-substep grid).

## Defaults after ablations

Kept only what measurably helped:

- `batch_size=256` (was 4)
- `pcgrad_protect_progress=True`
- `adv_norm="weighted"` + degenerate-advantage zeroing
- `target_kl=0.02`
- `LOG_STD_MIN=-2` (restored)
- `lr=1e-3` unchanged
- orthogonal init **off**
- TB scalar dump every 10 steps (was every step; needed for long runs)

Correctness fixes (always on): post-reset observations, gated first-step x-prior, per-env obs-history reset, `model_best.pt` by rolling success_rate.

## Suggested next steps

1. Re-run 16×100k on a **GPU** at default `resolution_scale=10`, `dt_substeps=40`, 16 members — cheap 200×4 physics is easier than the production grid.
2. Sweep `w_progress` now that weighted advantages make weights real (legacy used 9 before it was equalized to 1).
3. Evaluate `model_best.pt` not only `model_latest.pt`.
4. Wall-seeking vs straight upstream: cheap grid may under-resolve Poiseuille shear; confirm on the full mesh.
