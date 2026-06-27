# GDN Real-Backward Official Sudoku D192 Clean 3000-Step Scaling

## 1. Metainfo

- run family: `gdn-realbwd-official-sudoku-d192-clean-s3000-20260627`
- plan ID: `P-GDN-010`
- machine: AIStation GPU1 only
- remote root: `/huyang2/double-loop`
- source commit: pending launch
- local branch: `codex/gpu1-experiment-tracking`
- scheduled time: 2026-06-27 UTC

## 2. Hypothesis

P-GDN-007 showed clean D192/L10 GDN+FutureSeed improves from 600 to 1500 steps. P-GDN-008 showed blind D256 width does not help, and P-GDN-009 showed a small loop-residual FutureSeed memory also does not improve exact. The next high-ROI better-lesson question is whether the clean backbone simply needs to consume more training examples/time.

Prediction: if the current limit is data/optimization budget, a fixed clean D192/L10 FutureSeed backbone should keep improving at 2200/3000 steps over the 1500 baseline (`loop5 exact 0.0293`, `blank_acc 0.5337`). If it stays flat, then same-config longer training is not enough and the next move should be more data/curriculum or a larger clean backbone, not more epochs on this exact config.

## 3. Configuration

- task: official EqR Sudoku arrays, 9x9
- data: `sudoku-extreme-1k-aug-1000`
- model: `BACKBONE=gdn`, `GDN_MODE=triton_recurrent`
- short convolution: disabled with `GDN_USE_SHORT_CONV=0`
- size: `D_MODEL=192`, `LAYERS=10`, `HEADS=12`, `HEAD_DIM=16`
- loops: `MAX_LOOPS=5`
- batch: `128`
- steps: `3000`
- eval_n: `2048`
- checkpoint eval: `1500,2200`, no train checkpoint save
- dtype: `bfloat16`
- optimizer: `LR=0.0015`, `WEIGHT_DECAY=0.001`
- FutureSeed: `FUTURE_SEED_SCALE=1`, `FUTURE_SEED_UPDATE=fixed`

Forbidden mechanisms: no CPU smoke, no GPU2, no selector, no repair, no Sudoku rule, no seed sweep, no decay/gate sweep.

## 4. Environment

Pending GPU1 probe.

## 5. Commands

Pending launch at the final source SHA:

```bash
SOURCE_SHA=<sha> RUN_STAMP=<timestamp> bash /huyang2/double-loop/artifacts/launch/start_gdn_d192_fs_clean_s3000_20260627.sh
```

Kill criteria:

- OOM or CUDA kernel failure: stop exactly this run and record as failed infrastructure.
- Step100 wall time above 15 minutes: stop; same-config long training is too slow for ROI.
- Checkpoint eval at step1500 is materially worse than the known 1500 baseline and CE is not improving: stop before 3000.
- If step2200 exact/blank are flat relative to step1500 and CE is noisy upward, stop as same-config plateau.

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

None.
