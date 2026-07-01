# GDN FutureSeed D224/L12 No-Checkpoint Viability

## 1. Metainfo

- Plan ID: `P-DIAG-010`
- Status: planned
- Local branch: `codex/gpu1-experiment-tracking`
- Scheduled time: `2026-07-01 18:48:31 +0800`
- Machine: AIStation `GPU1` only
- Remote work dir: `/huyang2/double-loop`
- Source SHA: pending plan commit
- Parent evidence:
  - P-DIAG-006 D224/L12 with activation checkpoint reached NaN at step100.
  - P-DIAG-008 D192/L12 no-checkpoint stayed finite through step700.
  - P-DIAG-009 showed all-loop supervision makes intermediate states readable
    but does not solve the 51-55 cliff at D192.

## 2. Hypothesis

The next real scale axis is a larger model, not another loop-loss or objective
variant. The earlier D224/L12 failure is confounded by activation checkpointing.
If the bigger network is viable without checkpointing at a smaller microbatch,
then D224/L12 should at least fit, stay finite, and start learning the 51-55
stage.

Prediction:

- If D224/L12 no-checkpoint fits in 80GB and stays finite, the activation
  checkpoint/custom-backward interaction was the main D224 failure mode.
- If it OOMs or NaNs even without checkpointing, D224/L12 expand_v4 is currently
  outside the stable single-A800 scale envelope.

This is a short viability gate because GPU1 has about 20 minutes left. It is not
a score claim and not a sweep.

## 3. Configuration

- Data: official EqR Sudoku arrays, `sudoku-extreme-1k-aug-1000`
- Backbone: GDN Triton recurrent real backward + native terminal-state
  FutureSeed
- Architecture: `D_MODEL=224`, `LAYERS=12`, `HEADS=14`, `HEAD_DIM=16`,
  `CHANNEL_MULT=4`
- Memory/state: `GDN_EXPAND_V=4.0`
- Loop: `L_CYCLES=2`, `MAX_LOOPS=5`
- Loop supervision: `LOOP_LOSS=all`
- Curriculum: `HOLE_STAGES=46-50:100,51-55:200`
- Train: `FULL_STEPS=300`, microbatch `FULL_BATCH=32`,
  `GRAD_ACCUM_STEPS=4`, effective batch `128`
- Eval: official test split, `FULL_EVAL_N=512`
- Checkpoint eval: steps `100,300`, holes53
- Official blank-range eval: `46-50,51-55,56-64`
- CUDA path: `GDN_MODE=triton_recurrent`, `GDN_USE_SHORT_CONV=0`,
  `FORWARD_DTYPE=bfloat16`, no activation checkpoint
- Optimization: `LR=0.0015`, `WEIGHT_DECAY=0.001`, `BLANK_LOSS_WEIGHT=8`
- Forbidden: no CPU smoke, no GPU2, no activation checkpoint, no
  selector/search/repair/oracle rollout, no Sudoku-specific rule, no
  LR/seed/loss-weight sweep.

## 4. Environment

Pending launch.

## 5. Commands

Remote launch shape:

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 \
SOURCE_SNAPSHOT_MODE=lean UPDATE_LEADERBOARD=0 \
PYTHON_BIN=/opt/conda/bin/python \
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
SUDOKU_SIZE=9 BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 \
GDN_EXPAND_V=4.0 D_MODEL=224 LAYERS=12 HEADS=14 HEAD_DIM=16 CHANNEL_MULT=4 \
L_CYCLES=2 MAX_LOOPS=5 LOOP_LOSS=all \
HOLE_STAGES=46-50:100,51-55:200 \
FULL_STEPS=300 FULL_BATCH=32 GRAD_ACCUM_STEPS=4 FULL_EVAL_N=512 \
HOLES_MIN=46 HOLES_MAX=55 EVAL_HOLES=53 EVAL_HOLES_LIST=50,53,56 \
OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64 \
EVAL_CHECKPOINT_STEPS=100,300 EVAL_CHECKPOINT_HOLES_LIST=53 \
SAVE_TRAIN_CHECKPOINT_EVERY=100 \
FULL_ROLLOUT_KS= ROLLOUT_LOOP_VALUES= FULL_LOG_EVERY=50 \
BLANK_LOSS_WEIGHT=8 LR=0.0015 WEIGHT_DECAY=0.001 \
FORWARD_DTYPE=bfloat16 FUTURE_SEED_SCALE=1 \
CASE_BANK_N=2 CASE_BANK_EVAL_N=128 CASE_BANK_LOOP_VALUES=1,2,3,5 \
RUN_NAME=<run_name> ./run.sh full
```

Kill criteria:

- If CUDA OOMs, stop exact PID and archive.
- If step50 or step100 is NaN, stop exact PID and archive.
- If step100 takes too long to leave time for archive, stop and classify as
  throughput-infeasible under current lease.
- If GPU lease expires, restart/queue GPU1 and resume from the latest saved
  train checkpoint.

Success criteria:

- Minimal: step100 finite and checkpoint eval written.
- Strong viability: step300 finite, hard-stage CE decreases from step150/200 to
  step300, and memory stays below 80GB.
- Long-scale trigger: no OOM/NaN and checkpoint metrics are not clearly worse
  than D192/L12 all-loop at comparable early budget.

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

Not a submission run.
