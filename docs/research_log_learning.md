# MOMAPPO learning diagnosis

Fixed run flags for the owner-reported setting:

```bash
python -m fluxswarm.cli train \
  --progress-reward legacy \
  --obs-localization full \
  --no-action-x-prior \
  --batch-envs 16 \
  --total-timesteps-per-env 100000
```

Hardware for this log: CPU-only (4 cores, no GPU). Dummy-env pilots isolate the trainer; physics runs use reduced grid/substeps unless noted.

## Hypotheses

| ID | Hypothesis | Predicted signature |
|----|------------|---------------------|
| H0 | Auto-reset returns the terminal obs while the sim is already reset, so the next action is computed on the previous episode. | First step of every episode is off-policy. |
| H1 | PPO minibatch size 4 with 4 epochs overfits each rollout (thousands of SGD steps). | Policy noise; success_rate stays ~0. |
| H2 | Per-objective advantage whitening cancels `w_*` and amplifies constant objectives (energy/smooth). | Actor ignores progress; `mean_act_x → 0`. |
| H3 | Symmetric PCGrad projects the progress gradient away when it conflicts with energy. | Same as H2, worse with PCGrad on. |
| H4 | Default Linear init + `lr=1e-3` is unstable for tanh-Gaussian PPO. | Explained variance noisy; KL spikes. |
| H5 | Solver still forces `action=[-1,0]` on the first step even with `--no-action-x-prior` (log-prob mismatch). | Off-policy first step every episode. |
| H6 | Observation history is not cleared on auto-reset. | Stale frames after episode boundaries. |

Correctness fixes H0, H5, H6 are applied first (they do not change the loss). Ablations below toggle H1–H4.

## Experiments

Results are appended as runs complete.
