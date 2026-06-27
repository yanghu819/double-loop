# GDN Real-Backward Official Sudoku D192 Loop-Residual FutureSeed

## 1. Metainfo

- run family: `gdn-realbwd-official-sudoku-d192-loopresid-s1500-20260627`
- plan ID: `P-GDN-009`
- machine: AIStation GPU1 only
- remote root: `/huyang2/double-loop`
- source commit: pending launch
- local branch: `codex/gpu1-experiment-tracking`
- run time: 2026-06-27 UTC

## 2. Hypothesis

P-GDN-008 showed that simple D192->D256 widening does not improve official Sudoku quality. The next high-ROI question is whether the bottleneck is loop state dynamics: fixed FutureSeed recomputes seed state each loop, while `loop_residual` lets the model carry a damped FutureSeed memory across loops and learn how much to update it.

Prediction: if later loops are limited because FutureSeed state cannot persistently revise itself, loop-residual seeding should improve loop5 exact or blank accuracy over fixed D192/L10, or at least make loop3-5 continue improving instead of saturating at loop2. If it matches or regresses, the next axis should be data/curriculum rather than more state-memory plumbing.

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
- FutureSeed: `FUTURE_SEED_SCALE=1`, `FUTURE_SEED_UPDATE=loop_residual`, `FUTURE_SEED_DECAY=0.5`

Forbidden mechanisms: no CPU smoke, no GPU2, no selector, no repair, no Sudoku rule, no seed sweep, no no-FS repeat.

## 4. Environment

Pending GPU1 probe.

## 5. Commands

Pending launch at the final source SHA:

```bash
SOURCE_SHA=<sha> RUN_STAMP=<timestamp> bash /huyang2/double-loop/artifacts/launch/start_gdn_d192_fs_loopresid_s1500_20260627.sh
```

Kill criteria:

- OOM or CUDA kernel failure: stop exactly this run and record as failed mechanism gate.
- Step100 wall time above 15 minutes: stop; this mechanism adds too much overhead.
- Step300 CE much worse than fixed D192/L10 without compensating loop gain: stop as optimization failure, not a score result.

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

None.
