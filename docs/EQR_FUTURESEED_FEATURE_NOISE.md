# EqR FutureSeed and Feature Noise Patch

Timestamp: 2026-06-03

Source repo: `repos/eqr`
Base upstream SHA: `aba94e9cde0f273ce644db5261cd6915ba6561f0`
Local EqR implementation commit: `a4d22ac`
Archived patch: `docs/eqr_patches/0001-Add-FutureSeed-and-feature-diff-noise-to-EqR.patch`

## What Changed

- Added opt-in learned cross-level FutureSeed mixers to EqR.
- Added opt-in `feature_diff` Langevin noise using a non-persistent hidden-feature ring buffer.
- Kept default EqR behavior unchanged: `future_seed_scale=0.0`, `noise_mode=gaussian`.
- Added eval-time runtime overrides for `noise_mode`, `feature_noise_fallback`, and `future_seed_scale`.
- Added validation and warnings so typoed noise modes or old checkpoints do not create silent false positives.
- Added a CUDA-safe attention fallback: use FlashAttention when present, otherwise use PyTorch SDPA instead of requiring `flash_attn_3`.

## Mechanism Hypothesis

EqR already learns attractors, but its stochastic breadth is Gaussian and not tied to the learned hidden manifold. Feature-diff noise asks whether attractor exploration improves when perturbations follow differences between recent model features. FutureSeed asks whether the H/L recursion can benefit from a learned cross-level prior before each level's update, instead of relying only on additive input injection inside the existing reasoning module.

This is not a table-filling ablation. The first experiment should answer one decision: do manifold-shaped perturbations plus cross-level seeding improve EqR's depth/breadth efficiency enough to justify making them the main branch?

## Suggested First GPU1 Experiment

Use the EqR patch and run one Sudoku training job with:

```bash
scripts/train.sh eqr_sudoku \
  arch.future_seed_scale=1.0 \
  arch.noise_mode=feature_diff \
  arch.feature_noise_buffer_size=512 \
  arch.feature_noise_buffer_add=64 \
  arch.noise_scale=0.01
```

Decision rules:

- Keep both if convergence top-k or different-init metrics improve without worse train loss or wall time.
- Keep FutureSeed but revert noise to Gaussian if train loss slows and breadth does not improve.
- Lower `arch.noise_scale` before abandoning feature-diff if train is stable but breadth is noisy.

## Local Validation

- `python -m py_compile models/eqr.py evaluate.py`
- `git diff --check`
- Direct CPU forward smoke for default EqR and `futureseed_featurediff`
- CPU `torch.compile` forward smoke for `futureseed_featurediff`
- GPU1 no-Hydra smoke uses PyTorch SDPA when FlashAttention is not installed.

Hydra job-config parsing was not run locally because the Mac environment lacks `hydra`.
