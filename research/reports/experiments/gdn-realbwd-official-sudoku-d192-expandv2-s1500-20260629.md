# GDN Real-Backward Official Sudoku D192 ExpandV2 State-Capacity Probe

## 1. Metainfo

- run family: `gdn-realbwd-official-sudoku-d192-expandv2-s1500-20260629`
- plan ID: `P-GDN-014`
- machine: AIStation GPU1 only
- remote root: `/huyang2/double-loop`
- source commit: pending launch SHA
- local branch: `codex/gpu1-experiment-tracking`
- planned launch time: 2026-06-29 UTC

## 2. Hypothesis

The last several official Sudoku GDN+FutureSeed probes all point to the same bottleneck: FutureSeed opens training, but hard full-board exact stalls around `0.03`. P-GDN-013 showed that simply evaluating/training more loops does not keep fixing hard boards after loop3/5.

Mechanism question: is the fixed D192/L10 GDN state too small to retain enough alternative constraint information for later loops, even though the token width is already adequate to improve blank accuracy?

Prediction: if recurrent value/state capacity is the bottleneck, doubling GDN value/state dimension per head should improve exact and/or visibly reduce hard-case wrong cells without adding task rules. If exact remains around `0.03` while CE/blank accuracy only move softly, then value-state expansion alone is not the missing piece.

## 3. Configuration

- task: official EqR Sudoku arrays, 9x9
- data: `sudoku-extreme-1k-aug-1000`
- training blank-count range: full official train split, effectively `46-64`
- model: `BACKBONE=gdn`, `GDN_MODE=triton_recurrent`
- short convolution: disabled with `GDN_USE_SHORT_CONV=0`
- size: `D_MODEL=192`, `LAYERS=10`, `HEADS=12`, `HEAD_DIM=16`
- state/value expansion: `GDN_EXPAND_V=2.0`, per-head value/state dim `32`
- loops: `MAX_LOOPS=5`
- microbatch: `64`, gradient accumulation `2`, effective batch `128`
- steps: `1500`
- eval_n: `2048`
- checkpoint eval: `800,1200`, no train checkpoint save
- case bank: official eval fixed batch, `CASE_BANK_N=3`, `CASE_BANK_EVAL_N=256`, loops `1,2,3,5`
- dtype: `bfloat16`
- optimizer: `LR=0.0015`, `WEIGHT_DECAY=0.001`
- blank CE weight: `8`
- FutureSeed: `FUTURE_SEED_SCALE=1`, `FUTURE_SEED_UPDATE=fixed`
- no exact-margin or extra loss

Forbidden mechanisms: no CPU smoke, no GPU2, no selector, no repair, no Sudoku rule in training, no seed sweep, no loss table.

## 4. Environment

- host: AIStation `GPU1`
- GPU target: A100 80GB, `CUDA_VISIBLE_DEVICES=0`
- remote run worktree: `/huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-d192-fs-expandv2-s1500-<sha>-20260629`
- git dirty at run start: must be `false` after detached checkout

## 5. Commands

Launch from GitHub-truth SHA:

```bash
SOURCE_SHA=<launch-sha> \
RUN_STAMP=<utc-stamp> \
bash /huyang2/double-loop/artifacts/launch/start_gdn_d192_fs_expandv2_s1500_20260629.sh
```

Kill criteria:

- OOM or CUDA kernel failure: stop exact PID and record infrastructure failure.
- Step100 wall time above 12 minutes: expandV2 is too slow for this diagnostic at this batch; relaunch only with lower microbatch if no source change is needed.
- If checkpoint800 has CE above `1.5` and loop5 exact is still zero with blank_acc below `0.45`, stop as optimization failure.
- If checkpoint1200 exact is near baseline and CE/blank show no slope, finish only if final eval is near; otherwise archive as capacity boundary.

Success criteria:

- Strong: loop5 exact `>=0.05`.
- Useful: loop5 exact improves by at least `+0.01` over the clean D192/L10 1500 baseline (`0.0293`), or official hard-case bank shows a clear wrong-cell reduction without just moving errors.
- Negative: exact remains around `0.03` or only CE/blank accuracy improve.

## 6. Artifacts

Pending launch.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

None.
