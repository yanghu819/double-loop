# GDN Real-Backward Official Sudoku D192 Scale

## 1. Metainfo

- run family: `gdn-realbwd-official-sudoku-d192-scale-20260626`
- plan ID: `P-GDN-006`
- machine: AIStation GPU1 only
- remote root: `/huyang2/double-loop`
- source commit: pending launch
- local branch: `codex/gpu1-experiment-tracking`
- run time: 2026-06-26 UTC

## 2. Hypothesis

P-GDN-005 showed that native FutureSeed transfers to Gated DeltaNet: D128/L6 no-FS plateaued, while D128/L6 FS opened with loop4 exact `0.0107`.

The next question is scale: if GDN is a viable FutureSeed backbone, D192/L10/loop5 should approach the earlier RWKV full-backbone 600-step official Sudoku opening (`loop5 exact 0.0137`, blank_acc `0.4972`) instead of staying at the D128 gate level.

This is not a table. It is staged: run FS first; only run matched no-FS if FS opens enough to make the control worth the GPU time.

## 3. Configuration

- task: official EqR Sudoku arrays, 9x9
- data: `sudoku-extreme-1k-aug-1000`
- model: `BACKBONE=gdn`, `GDN_MODE=triton_recurrent`
- short convolution: disabled with `GDN_USE_SHORT_CONV=0`
- size: `D_MODEL=192`, `LAYERS=10`, `HEADS=12`, `HEAD_DIM=16`
- loops: `MAX_LOOPS=5`
- batch target: `128`, fallback `80` only for OOM or step100 speed kill
- steps: `600`
- eval_n: `2048`
- dtype: `bfloat16`
- first arm: `FUTURE_SEED_SCALE=1`
- conditional arm: `FUTURE_SEED_SCALE=0` only if FS opens

Forbidden mechanisms: no CPU smoke, no GPU2, no selector, no repair, no Sudoku rule, no seed sweep.

## 4. Environment

Pending GPU1 probe.

## 5. Commands

Pending launch script under `/huyang2/double-loop/artifacts/launch/`.

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

None.
