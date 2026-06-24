# official-eqr-fs-mixer-triton-sudoku-trainonly-20260624T0930Z-d9c2600

Purpose: verify that the Triton FutureSeed scan backend trains inside the
official EqR FutureSeed mixer replacement path on GPU1.

This run is a backend viability check, not a model-quality evaluation.

## Result

- Source SHA: `d9c2600e47d0b5e0950bf884ba8a155de3e6e198`
- Backend: `FUTURESEED_SCAN_BACKEND=triton`
- GPU: AIStation GPU1, A800 80GB
- Dataset: official `sudoku-extreme-1k-aug-1000`
- Steps: `448/448`
- Final train loss: `1.626443`
- Final logged speed: `7.97 it/s`
- Checkpoints:
  - `/huyang2/double-loop/official_eqr_compare/outputs/futureseed-mixer/outputs/2zycadqg/2026-6-24/9-25-39/checkpoints/step_250_2zycadqg.pth`
  - `/huyang2/double-loop/official_eqr_compare/outputs/futureseed-mixer/outputs/2zycadqg/2026-6-24/9-25-39/checkpoints/step_448_2zycadqg.pth`

The preceding full train-time-eval launch was killed because the official
training script expanded the step250 eval into `3304` batches, projected near an
hour under `DISABLE_COMPILE=1`. This train-only relaunch avoids burning the GPU
on low-ROI eval while still proving that the CUDA scan backend is trainable.

