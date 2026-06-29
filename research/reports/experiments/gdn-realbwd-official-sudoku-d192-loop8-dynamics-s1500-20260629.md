# GDN Real-Backward Official Sudoku D192 Loop8 Dynamics Probe

## 1. Metainfo

- run family: `gdn-realbwd-official-sudoku-d192-loop8-dynamics-s1500-20260629`
- plan ID: `P-GDN-013`
- machine: AIStation GPU1 only
- remote root: `/huyang2/double-loop`
- source commit: pending launch SHA
- local branch: `codex/gpu1-experiment-tracking`
- planned launch time: 2026-06-29 UTC

## 2. Hypothesis

The recent D192/L10 GDN+FutureSeed results all have the same shape: soft metrics improve, exact stalls around `0.03`, and indirect objectives do not open the plateau. The next question is whether the loop itself still has useful late-loop computation, or whether the state has already settled by loop3/5.

Mechanism question: if we train/evaluate with loop8, do loops 5->8 keep correcting official Sudoku boards, or do they copy the same operating point?

Prediction: if loop depth is a real remaining scaling axis, loop8 should improve exact beyond loop5 and hardest-case visualizations should show wrong hidden cells decreasing from loop5 to loop8. If loop3/5/8 are nearly identical, the bottleneck is not simply more recurrent compute; it is state capacity or state update dynamics.

## 3. Configuration

- task: official EqR Sudoku arrays, 9x9
- data: `sudoku-extreme-1k-aug-1000`
- training blank-count range: `45-64`, full official train/test range
- model: `BACKBONE=gdn`, `GDN_MODE=triton_recurrent`
- short convolution: disabled with `GDN_USE_SHORT_CONV=0`
- size: `D_MODEL=192`, `LAYERS=10`, `HEADS=12`, `HEAD_DIM=16`
- loops: `MAX_LOOPS=8`
- batch: `96`
- steps: `1500`
- eval_n: `2048`
- checkpoint eval: `800,1200`, no train checkpoint save
- case bank: official eval fixed batch, `CASE_BANK_N=3`, `CASE_BANK_EVAL_N=256`, loops `1,2,3,5,8`
- dtype: `bfloat16`
- optimizer: `LR=0.0015`, `WEIGHT_DECAY=0.001`
- blank CE weight: `8`
- FutureSeed: `FUTURE_SEED_SCALE=1`, `FUTURE_SEED_UPDATE=fixed`
- no exact-margin or extra loss

Forbidden mechanisms: no CPU smoke, no GPU2, no selector, no repair, no Sudoku rule in training, no seed sweep, no loss table.

## 4. Environment

- host: AIStation `GPU1`
- GPU target: A100 80GB, `CUDA_VISIBLE_DEVICES=0`
- remote run worktree: `/huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-d192-fs-loop8-dynamics-s1500-<sha>-20260629`
- git dirty at run start: must be `false` after detached checkout

## 5. Commands

Launch from GitHub-truth SHA:

```bash
SOURCE_SHA=<launch-sha> \
RUN_STAMP=<utc-stamp> \
bash /huyang2/double-loop/artifacts/launch/start_gdn_d192_fs_loop8_dynamics_s1500_20260629.sh
```

Kill criteria:

- OOM or CUDA kernel failure: stop exact PID and record infrastructure failure.
- Step100 wall time above 10 minutes: loop8 is too slow for this diagnostic.
- If checkpoint800 has CE above `1.5` and loop8 exact is still zero with blank_acc below `0.45`, stop as optimization failure.
- If checkpoint1200 shows loop8 equals loop5 and runtime is already high, finish only if final eval is near; otherwise archive as loop-depth boundary.

Success criteria:

- Strong: loop8 exact `>=0.04`.
- Useful: loop5->loop8 exact gain `>=0.005`, or official hardest cases show loop5->loop8 wrong_count reduction without new wrong-cell inflation.
- Negative: loop3/5/8 exact and wrong_count are nearly identical.

## 6. Artifacts

Pending launch.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

None.
