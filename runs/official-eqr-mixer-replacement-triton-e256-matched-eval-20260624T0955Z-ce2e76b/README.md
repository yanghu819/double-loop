# official-eqr-mixer-replacement-triton-e256-matched-eval-20260624T0955Z-ce2e76b

Purpose: test whether the CUDA FutureSeed mixer replacement still beats the
official EqR mixer after a longer matched train-only budget, or whether the
step448 signal was just early optimization noise.

## Setup

- Source SHA: `ce2e76bed92d2f271518f0bf23f251c8eec492e5`
- GPU: AIStation GPU1 only
- Dataset: official `sudoku-extreme-1k-aug-1000`
- Train budget: `1792` steps for both arms (`epochs=256`)
- Eval: official `evaluate.py`, `max_eval_steps=16`, `batch=128`, `D16`,
  `noise=0.01`, `different_init=1`
- FutureSeed backend: `FUTURESEED_SCAN_BACKEND=triton`

## Result

| metric | official mixer-base | FutureSeed mixer | FS - base |
|---|---:|---:|---:|
| eval accuracy | 0.366766 | 0.420693 | +0.053928 |
| exact accuracy | 0.000000 | 0.000000 | 0.000000 |
| lm loss | 1.803480 | 1.523322 | -0.280157 |
| total loss | 1.803888 | 1.524001 | -0.279888 |
| residual@1 | 165.761230 | 188.403320 | +22.642090 |
| residual@4 | 63.062988 | 35.649902 | -27.413086 |
| residual@8 | 60.984375 | 13.391113 | -47.593262 |
| residual@16 | 61.583008 | 7.972656 | -53.610352 |
| eval wall time | 7.541340s | 15.192253s | +7.650913s |

Interpretation: the FutureSeed mixer keeps a real sample-efficiency and
late-loop state-contraction advantage at 4x the step448 budget. It is not a
solved-task result because exact is still zero. The next high-ROI check is one
longer scale gate; do not run seeds or loss tables.

