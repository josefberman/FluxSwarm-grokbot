# Research log: `objectives/success_rate` oscillation

Branch `success-rate-stability` (from `origin/refactor` @ 1e14070, which includes PRs #3–#5). CPU only.

**Base settings (Josef's run):** legacy progress reward, `--obs-localization full`, `--no-action-x-prior`, 16 envs,
`n_steps=200`, `batch_size=256`, protect-progress PCGrad, weighted advantage normalization, `LOG_STD_MIN=-2`,
`target_kl=0.02`, **`lr=1e-3`** (checked in `TrainConfig`), no return/value normalization, `(256,128)` layers.
Note: in legacy mode the `±10` terminal bonus is **not** applied. `swarm_env.py` only added it in potential
mode; legacy uses `+1`/step while succeeded and `-1` on failure.

**Scales used**
- *Dummy*: `DummyUpstreamEnv`, 16 envs × 16 agents, 100k steps/env (500 updates), 3 seeds, plus a harder
  "Poiseuille-drift" dummy, 50k steps/env, 3 seeds. Both were solved to 100% with zero oscillation, so the
  dummy cannot reproduce the problem. Everything below uses cheap physics.
- *Cheap physics*: 200×4 grid, 4 substeps, 16 members, 16 envs. One 100k-steps/env reference run was stopped
  at 366 updates (~1.17M env steps). The controlled comparisons all cover **150 PPO updates (30k steps/env =
  480k env steps)**, 2–3 seeds (0,1[,2]), same seeds for every variant. Runs that went past 150 updates were
  truncated to 150 in the analysis, so budgets are equal. Each comparison run took roughly 25–60 min of
  wall time on a shared 8-core box.
- **Score:** `std` of the logged `objectives/success_rate` (sampled at each update end) over the last 50% of
  updates, plus its mean over the last 50% and the last 10% ("final").

Tools: `scripts/stability_experiments.py` (variants), `scripts/analyze_stability.py` (estimators, burstiness,
diagnostics), `scripts/outcome_correlation.py`, `scripts/inflow_phase_check.py`, `scripts/eval_checkpoints.py`,
`scripts/plot_stability.py`. Raw summaries: `research_figures/stab_summary_150u.json`,
`stab_base100k_summary.json`.

## H0: Measurement artifact?

Reference run (physics base, seed 0, 366 updates; `stab_base100k_estimators.png`):

| estimator (same episodes) | std last 50% | mean last 50% |
|---|---:|---:|
| `objectives/success_rate` (last 128 eps) | 0.153 | 0.31 |
| last 512 eps | 0.077 | 0.29 |
| EMA over updates (β=0.9, episode-weighted) | 0.118 | 0.31 |
| per-update success (~13 eps/update) | 0.428 | 0.34 |
| length-weighted window | 0.200 | 0.39 |

- **Window vs completion rate.** About 13 episodes finish per update, so the 128-episode window covers about
  10 updates. If outcomes were independent, the binomial noise floor for p≈0.3 and n=128 would be
  std ≈ 0.04. The observed std is 0.15, which is 3.7× that floor.
- **Synchronized resets.** Only 0.2–0.4% of episodes end in bursts of at least 8 envs on the same step, and
  the most envs ending on one step was 8. Once training starts, ends are not synchronized at the step level.
- **CPU mirror env.** The mirror exists only with `--device-split cpu-shard`, which needs CUDA. When present,
  its *deterministic* `tanh(mean)` episodes were mixed into the same window as the stochastic ones. It is
  not active by default. The new `success_rate_train` / `success_rate_mirror` tags separate the two.
- **Short episodes.** Failures are shorter (mean 217 steps) than successes (329), so the count-based window
  over-represents failures. The length-weighted window reads about 0.08 higher but is just as noisy. This
  biases the level, not the swing.
- **Decisive test: outcome correlation under the same policy.** Within one update the policy is frozen.
  Pairs of episodes in *different* envs that end within 20 steps of each other have outcome correlation
  **0.84–0.97**. Pairs that end ≥100 steps apart have −0.3 to 0.5. The variance of per-update success is
  **11–12× binomial** in every legacy-clock run (all variants). Outcomes are therefore driven by a shared,
  time-dependent condition across envs, not by independent draws and not by the policy changing.

**Root cause found: the physics env couples all envs through one clock.**
`BatchedFluidSolver.step` passed `t0 = episode_time.mean()` (the batch mean) to
`TorchFluidSolver.substep_loop`, and the inflow is applied *incrementally*
(`beat(t+dt) − beat(t)`). After desynchronized resets:
- every env is driven by the same batch-mean pulse phase;
- each env's inflow amplitude drifts away from `beat(own time)` every time any other env resets, because
  the batch mean jumps backwards.

Separately, the fluid rings were zeroed for **all** envs whenever **any** env had `episode_time == 0`. That
gives every env "zero fluid velocity" drag and observations on every step after any reset (about 7% of steps).

Random-policy check (`stab_inflow_clock_*.json`): with the legacy clock, the inlet amplitude has mean |error|
30 mm/s (max 400 = full peak) vs its own clock, and cross-env correlation is 0.93. With the fix, the error is
18 mm/s and correlation is 0.77, matching the 0.80 expected from the envs' own clocks.

Fix: `sim.per_env_clock` (new; **default True**, `--no-per-env-clock` restores the old behaviour). It uses a
per-env clock for the inflow and blanks rings only for envs that just reset. With the fix, the variance of
per-update success drops to **1.1–1.5× binomial** and near-pair correlation drops to −0.01…0.23.

**Verdict.** The window itself is not the main problem. Its size, burst counting and the mirror add only
second-order noise; the larger window and the EMA cut the std by a further ~25–50%. Most of the swing is
real, correlated outcome variance caused by the env coupling bug.

A frozen-checkpoint evaluation of the legacy-clock run (`eval_checkpoints.py`, 800 steps × 16 envs per
checkpoint) also swung from 0.0 to 1.0 between checkpoints 10 updates apart. However, that evaluation
suffers from the same coupling, so it does not separate policy instability from env noise.

## Controlled comparison (cheap physics, 150 updates, same seeds)

| variant | seeds | std(success_rate) last 50% | mean last 50% | final (last 10%) | std(ema) | epochs run | early-stop frac | actor grad-norm std | value loss (prog,en,sm) |
|---|---|---|---|---|---|---|---|---|---|
| base | 3 | 0.126 | 0.265 | 0.208 | 0.094 | 3.83 | 0.08 | 0.55 | [20.4, 6.6, 2.5] |
|  per-seed std/final |  | 0.104 / 0.133 / 0.142 | | 0.23 / 0.33 / 0.05 | | | | | |
| popart | 2 | 0.161 | 0.454 | 0.402 | 0.127 | 3.98 | 0.04 | 0.41 | [14.1, 4.1, 2.0] |
|  per-seed std/final |  | 0.203 / 0.120 | | 0.16 / 0.64 | | | | | |
| bonus50 | 2 | 0.132 | 0.442 | 0.462 | 0.098 | 3.72 | 0.10 | 0.44 | [38.3, 8.9, 2.7] |
|  per-seed std/final |  | 0.119 / 0.146 | | 0.39 / 0.53 | | | | | |
| clock | 3 | 0.064 | 0.874 | 0.925 | 0.042 | 4.00 | 0.00 | 0.38 | [3.8, 2.6, 1.8] |
|  per-seed std/final |  | 0.002 / 0.107 / 0.082 | | 1.00 / 0.78 / 1.00 | | | | | |
| kl0.05 | 2 | 0.142 | 0.513 | 0.479 | 0.113 | 4.00 | 0.00 | 0.37 | [14.3, 5.6, 2.3] |
|  per-seed std/final |  | 0.169 / 0.115 | | 0.43 / 0.52 | | | | | |
| lr3e-4 | 2 | 0.148 | 0.306 | 0.268 | 0.100 | 3.72 | 0.11 | 0.43 | [26.1, 7.9, 2.7] |
|  per-seed std/final |  | 0.172 / 0.123 | | 0.33 / 0.21 | | | | | |
| clock+bonus50 | 2 | 0.014 | 0.987 | 1.000 | 0.009 | 3.92 | 0.03 | 0.44 | [3.6, 2.7, 1.9] |
|  per-seed std/final |  | 0.028 / 0.000 | | 1.00 / 1.00 | | | | | |
| sympcg | 2 | 0.210 | 0.381 | 0.243 | 0.154 | 3.59 | 0.14 | 0.28 | [22.0, 8.6, 2.8] |
|  per-seed std/final |  | 0.247 / 0.172 | | 0.13 / 0.35 | | | | | |
| nopcg | 2 | 0.147 | 0.345 | 0.207 | 0.111 | 3.94 | 0.05 | 0.58 | [15.0, 7.0, 2.4] |
|  per-seed std/final |  | 0.130 / 0.165 | | 0.22 / 0.20 | | | | | |

(`stab_clock_vs_base.png`, `stab_bonus_popart.png`, `stab_kl_lr_pcgrad.png`)

### H1: Return/value normalization
Hypothesis: with an unclamped legacy reward, the progress value head dominates the shared critic loss.
- Evidence for it: the progress value loss is about 20, vs 6.6 for energy and 2.5 for smoothness.
  Progress-head explained variance is about 0.66 (energy 0.76, smoothness 0.70). The progress return std
  per batch swings between 5 and 40.
- PopArt (`--value-norm popart`) lowered the value loss (14.1) and total grad norm (6.3 → 1.7). It did
  **not** stabilize success: std 0.161 vs 0.126. Seed 1 was much better (final 0.64) and seed 0 worse (0.16).
- **Not kept as default.** It stays available as an option. `--clip-vloss` was implemented but not
  screened, for time.

### H2: Bigger terminal bonus (opt-in `--terminal-bonus-legacy --terminal-bonus 50`)
- Legacy-clock: mean success rises from 0.27 to 0.44 (final 0.21 → 0.46), but std is unchanged
  (0.132 vs 0.126).
- Combined with the clock fix: std **0.014**, final **1.00** (2/2 seeds). This is the best result.
- Kept as an **opt-in** recommendation, so Josef's reward is unchanged by default.

### H3: target_kl / epochs
- In base, early stop triggers in 8–20% of updates and on average 3.8 of 4 epochs run. KL is rarely binding.
- `target_kl=0.05`: no early stops and 4 epochs every update. Mean success 0.51 (vs 0.27), but std 0.142.
  It does not reduce oscillation.
- `lr=3e-4`: std 0.148, mean 0.31. No benefit. Not kept.

### H4: PCGrad mode
- Symmetric PCGrad: std **0.210** (worse), mean 0.38. It has the lowest actor-grad-norm std (0.28), so
  gradient-norm variance is not what drives the success swings.
- No PCGrad: std 0.147, mean 0.35, actor-grad-norm std 0.58.
- protect-progress stays.

### H5: Entropy / log_std
`training/log_std` reaches the `LOG_STD_MIN=-2` floor by about update 40 in every physics run and stays
there. It neither collapses below the floor nor swings. Entropy is constant after that, so the oscillation is
not an exploration swing.

### H6: Learning rate
`lr=3e-4` constant: no improvement (above). Linear annealing (`--lr-schedule linear`) is implemented and
unit-tested but was not screened, for time.

## Combined winner and verification
`per_env_clock=True` (3 seeds): std **0.064** (vs 0.126), final **0.93** (vs 0.21). Dispersion is about
binomial. Adding `--terminal-bonus-legacy --terminal-bonus 50` (2 seeds): std **0.014**, final **1.00**.

**Caveat:** fixing the clock also makes cheap physics *easier* for a random policy. Over 1200 steps, legacy
gives 35 failures / 16 successes and the fix gives 1 / 5. A zero-thrust swarm reaches `success_x` in about
1196 steps under either clock, so passive drift alone succeeds on the cheap grid. Part of the higher final
success is therefore "easier, correct env" rather than a better trainer. The stability claim rests on the
dispersion and correlation test, which does not depend on the success level.

## How much is artifact vs real policy instability
- About 85–90% of the per-update variance in legacy-clock runs is excess correlated variance (11–12×
  binomial), and it disappears with the per-env clock. The cause is env coupling, not PPO.
- The remaining binomial/window noise is about std 0.04 for a 128-episode window, which
  `success_rate_ema` and a larger window cut further.
- No trainer knob (PopArt, KL, lr, PCGrad) reduced the std under the legacy clock. There is no evidence of
  real PPO policy oscillation beyond that.

## Defaults and options after this work
- **New default:** `sim.per_env_clock=True` (env bug fix; `--no-per-env-clock` gives legacy).
- Opt-in: `--terminal-bonus-legacy --terminal-bonus 50` (recommended), `--value-norm {running,popart}`,
  `--clip-vloss`, `--lr-schedule {linear,cosine}` with `--lr-final-fraction`.
- Unchanged: reward, (256,128), `lr=1e-3`, `target_kl=0.02`, `batch_size=256`, protect-progress, weighted adv-norm.
- New TB tags (`objectives/success_rate` is unchanged): `objectives/success_rate_ema`, `success_rate_update`,
  `success_rate_train`, `success_rate_mirror`, `episodes_in_window`, `episodes_this_update`; `training/epochs_run`,
  `minibatches`, `early_stop`, `approx_kl_max`, `clipfrac`, `grad_norm(_std)`, `actor_grad_norm`, `log_std`, `lr`;
  `losses/value_{progress,energy,smooth}`, `losses/return_std_progress`. There is also a per-episode
  `episodes.jsonl` in the run dir.

## What to check on the GPU run
The full-resolution GPU env has the same batch-mean-clock code, so expect the same coupling.
1. Plot `objectives/success_rate` against `objectives/success_rate_ema` and `success_rate_update`.
2. Run `python scripts/outcome_correlation.py <run_dir>` on the old code. A dispersion ratio well above 1
   (e.g. ~10) confirms the coupling.
3. Rerun with the new default (`per_env_clock`) and check that the ratio is about 1.
4. Check `training/epochs_run`, `training/early_stop` and `training/log_std` (should sit at −2).
5. Check `losses/value_progress` against the other heads.
6. If it uses `--device-split cpu-shard`, compare `success_rate_train` with `success_rate_mirror`.
