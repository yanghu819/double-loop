# GDN FutureSeed D192/L12 All-Loop Scaling

## 1. Metainfo

- Plan ID: `P-DIAG-009`
- Status: planned
- Local branch: `codex/gpu1-experiment-tracking`
- Scheduled time: `2026-07-01 16:36:00 +0800`
- Machine: AIStation `GPU1` only
- Remote work dir: `/huyang2/double-loop`
- Source SHA: pending launch commit
- Parent evidence:
  P-DIAG-008 finite-through-step700 no-checkpoint stability result; user
  correction that EqR uses per-loop supervision.

## 2. Hypothesis

EqR supervises every recurrent step. Our standalone FutureSeed/GDN scaling runs
have mostly used final-loop supervision, which gives sparser gradients and may
let early recurrent states become poor answer states.

Mechanism question:

Can EqR-aligned every-loop supervision improve the `51-55` blank transition by
making each loop produce a readable, board-consistent state, instead of asking
only the final loop to fix the board?

Prediction:

- If sparse final-loop supervision was a major bottleneck, `LOOP_LOSS=all`
  should improve holes53 and official `51-55` exact earlier than P-DIAG-004/005,
  while preserving loop1-to-loop5 refinement.
- If all-loop supervision only makes every loop copy the same operating point,
  exact will remain near the prior plateau and loop gain will shrink. Then the
  bottleneck is state/update capacity, not loss placement.

This is not a loss-weight table. It is one recipe-alignment probe against EqR.

## 3. Configuration

- Data: official EqR Sudoku arrays, `sudoku-extreme-1k-aug-1000`
- Backbone: GDN Triton recurrent real backward + native terminal-state
  FutureSeed
- Architecture: `D_MODEL=192`, `LAYERS=12`, `HEADS=12`, `HEAD_DIM=16`,
  `CHANNEL_MULT=4`
- Memory/state: `GDN_EXPAND_V=4.0`
- Loop: `L_CYCLES=2`, `MAX_LOOPS=5`
- Loop supervision: `LOOP_LOSS=all`
- Curriculum: `HOLE_STAGES=46-50:500,51-55:2500`
- Train: `FULL_STEPS=3000`, microbatch `FULL_BATCH=48`,
  `GRAD_ACCUM_STEPS=3`, effective batch `144`
- Eval: official test split, `FULL_EVAL_N=1024`
- Checkpoint eval: steps `1000,2000,3000`, holes53
- Official blank-range eval: `46-50,51-55,56-64`
- Case-bank visualization: loops `1,2,3,5`
- CUDA path: `GDN_MODE=triton_recurrent`, `GDN_USE_SHORT_CONV=0`,
  `FORWARD_DTYPE=bfloat16`, no activation checkpoint
- Optimization: `LR=0.0015`, `WEIGHT_DECAY=0.001`, `BLANK_LOSS_WEIGHT=8`
- Forbidden: no CPU smoke, no GPU2, no selector/search/repair/oracle rollout,
  no Sudoku-specific rule, no seed/LR/loss-weight sweep.

## 4. Environment

Pending launch.

## 5. Commands

Remote launch shape:

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 \
SOURCE_SNAPSHOT_MODE=lean UPDATE_LEADERBOARD=0 \
XDG_CACHE_HOME=/huyang2/double-loop/.cache \
PIP_CACHE_DIR=/huyang2/double-loop/.cache/pip \
HF_HOME=/huyang2/double-loop/.cache/huggingface \
TORCH_HOME=/huyang2/double-loop/.cache/torch \
TORCH_EXTENSIONS_DIR=/huyang2/double-loop/.cache/torch_extensions \
PYTHON_BIN=/opt/conda/bin/python \
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
SUDOKU_SIZE=9 BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 \
GDN_EXPAND_V=4.0 D_MODEL=192 LAYERS=12 HEADS=12 HEAD_DIM=16 CHANNEL_MULT=4 \
L_CYCLES=2 MAX_LOOPS=5 LOOP_LOSS=all \
HOLE_STAGES=46-50:500,51-55:2500 \
FULL_STEPS=3000 FULL_BATCH=48 GRAD_ACCUM_STEPS=3 FULL_EVAL_N=1024 \
HOLES_MIN=46 HOLES_MAX=55 EVAL_HOLES=53 EVAL_HOLES_LIST=50,53,56 \
OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64 \
EVAL_CHECKPOINT_STEPS=1000,2000,3000 EVAL_CHECKPOINT_HOLES_LIST=53 \
SAVE_TRAIN_CHECKPOINT_EVERY=1000 \
FULL_ROLLOUT_KS= ROLLOUT_LOOP_VALUES= FULL_LOG_EVERY=100 \
BLANK_LOSS_WEIGHT=8 LR=0.0015 WEIGHT_DECAY=0.001 \
FORWARD_DTYPE=bfloat16 FUTURE_SEED_SCALE=1 \
CASE_BANK_N=2 CASE_BANK_EVAL_N=256 CASE_BANK_LOOP_VALUES=1,2,3,5 \
RUN_NAME=<run_name> ./run.sh full
```

Kill criteria:

- If CUDA OOMs, stop exact PID and archive.
- If step100 is NaN, stop exact PID and archive.
- If step1000 holes53 loop5 exact is below D192/L10 expand_v4 step1000
  (`0.0205`) and blank_acc/CE are not better, stop unless near checkpoint.
- If loop1 and loop5 become identical while exact remains low, classify as
  all-loop-copy failure rather than continue same recipe longer.

Success criteria:

- Strong: official `51-55` loop5 exact `>= 0.05`, or holes53 loop5 exact
  `>= 0.06` by step3000.
- Mechanism-positive: all-loop improves holes53 step1000/2000 vs D192/L10
  expand_v4 and preserves loop gain without relying on selector/repair.

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

Not a submission run.
