# FutureSeed Sudoku Engine

This directory contains the maintained model runner and CUDA kernels behind the
project baseline.

## Maintained

- `study_rwkv_futureseed_loop.py`: shared FutureSeed + loop model, data, train,
  evaluation, checkpoint, and visualization runner.
- `rwkv7_cuda/`: RWKV7 state-passing CUDA path used by the fair benchmark.
- `gdn_triton.py`: local GDN recurrent Triton path used by the scale baseline.
- `check_fla_delta_backbones.py`: official FLA provenance and CUDA
  forward/state/backward checks for GDN, GDN2, and KDA.
- `check_gdn_triton_kernel.py`: local GDN Triton reference tests.
- `uv.lock`: fixed Python dependency resolution.

The runner still accepts old experimental flags so archived checkpoints remain
readable. Those flags are disabled in both canonical configs and are not active
research directions. See `../../docs/DEPRECATIONS.md`.

## Two GDN Names

- Public benchmark `gdn` maps to internal `BACKBONE=fla_gdn`, the exact official
  FLA `GatedDeltaNet` layer.
- Internal `BACKBONE=gdn` is the local GDN-Triton implementation retained for
  the long scale baseline.

This distinction is deliberate. Do not compare one implementation's score with
another implementation's cost without naming it.

## Invariants

- GPU1 only: `CUDA_VISIBLE_DEVICES=0`.
- No CPU model smoke.
- No automatic kernel fallback in a reported benchmark.
- FutureSeed is cross-layer terminal-state seeding, never a right-to-left scan.
- No search, repair, selector, oracle rollout, or Sudoku-specific model rule.
- Every loop receives CE supervision in canonical training.
