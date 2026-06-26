# GDN Real-Backward Official Sudoku D192 Long Scaling

## 1. Metainfo

- run family: `gdn-realbwd-official-sudoku-d192-long-s1500-20260626`
- plan ID: `P-GDN-007`
- machine: AIStation GPU1 only
- remote root: `/huyang2/double-loop`
- source commit: pending launch
- local branch: `codex/gpu1-experiment-tracking`
- run time: 2026-06-26 UTC

## 2. Hypothesis

P-GDN-006 showed a clean mechanism split: D192/L10 no-FS did not open, while D192/L10 FutureSeed opened at 600 steps. The next high-ROI question is not another no-FS control. It is whether the opened GDN+FutureSeed model has a useful scaling slope when trained longer.

Prediction: if GDN+FutureSeed is a viable backbone rather than a tiny opening trick, 1500 steps should materially improve exact/blank accuracy over the 600-step result (`loop5 exact 0.0176`, `blank_acc 0.5021`) and should preserve a positive loop1-to-loop5 gain.

## 3. Configuration

- task: official EqR Sudoku arrays, 9x9
- data: `sudoku-extreme-1k-aug-1000`
- model: `BACKBONE=gdn`, `GDN_MODE=triton_recurrent`
- short convolution: disabled with `GDN_USE_SHORT_CONV=0`
- size: `D_MODEL=192`, `LAYERS=10`, `HEADS=12`, `HEAD_DIM=16`
- loops: `MAX_LOOPS=5`
- batch: `128`
- steps: `1500`
- eval_n: `2048`
- dtype: `bfloat16`
- only arm: `FUTURE_SEED_SCALE=1`

Forbidden mechanisms: no CPU smoke, no GPU2, no selector, no repair, no Sudoku rule, no seed sweep, no no-FS repeat.

## 4. Environment

Pending GPU1 probe.

## 5. Commands

Pending launch at the final source SHA:

```bash
SOURCE_SHA=<sha> RUN_STAMP=<timestamp> bash /huyang2/double-loop/artifacts/launch/start_gdn_d192_fs_long_s1500_20260626.sh
```

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

None.
