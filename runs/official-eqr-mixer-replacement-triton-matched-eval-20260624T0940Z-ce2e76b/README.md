# official-eqr-mixer-replacement-triton-matched-eval-20260624T0940Z-ce2e76b

Purpose: a tiny matched official-eval gate after adding the Triton CUDA
FutureSeed scan backend.

This answers one question: under the same official EqR Sudoku train budget and
a bounded official `evaluate.py` run, does the FutureSeed mixer replacement
retain its sample-efficiency signal against the official mixer baseline?

## Setup

- Source SHA: `ce2e76bed92d2f271518f0bf23f251c8eec492e5`
- GPU: AIStation GPU1 only
- Dataset: official `sudoku-extreme-1k-aug-1000`
- Train budget: `448` steps for both arms
- Eval: official `evaluate.py`, `max_eval_steps=16`, `batch=128`, `D16`,
  `noise=0.01`, `different_init=1`
- FutureSeed backend: `FUTURESEED_SCAN_BACKEND=triton`

## Result

| metric | official mixer-base | FutureSeed mixer | FS - base |
|---|---:|---:|---:|
| eval accuracy | 0.095498 | 0.188079 | +0.092581 |
| exact accuracy | 0.000000 | 0.000000 | 0.000000 |
| lm loss | 2.534932 | 2.313245 | -0.221687 |
| total loss | 2.538283 | 2.316542 | -0.221741 |
| residual@1 | 238.799805 | 226.280762 | -12.519043 |
| residual@16 | 229.125488 | 150.869141 | -78.256348 |
| eval wall time | 8.557675s | 13.011803s | +4.454128s |

Interpretation: positive mechanism gate, not solved Sudoku. FutureSeed roughly
doubles token accuracy and gives much lower recurrent residual at the same
small training budget, but exact accuracy is still zero. The next high-ROI
test is a single longer matched scale gate, not seeds or loss-weight tables.

