# official-eqr-futureseed-mixer-triton-trainonly-e256-20260624T0950Z-ce2e76b

Matched e256 FutureSeed mixer replacement run using the Triton CUDA scan
backend.

- Source SHA: `ce2e76bed92d2f271518f0bf23f251c8eec492e5`
- GPU: AIStation GPU1 only
- Dataset: official `sudoku-extreme-1k-aug-1000`
- Config: official EqR Sudoku with `FutureSeedScanMixer`, `h192`, `heads6`,
  `halt16`, `noise0.01`, `batch128`, `epochs256`
- Backend: `FUTURESEED_SCAN_BACKEND=triton`
- Steps: `1792/1792`
- Final train loss: `1.448282`
- Final checkpoint:
  `/huyang2/double-loop/official_eqr_compare/outputs/futureseed-mixer/outputs/njrnbwbc/2026-6-24/9-46-52/checkpoints/step_1792_njrnbwbc.pth`

