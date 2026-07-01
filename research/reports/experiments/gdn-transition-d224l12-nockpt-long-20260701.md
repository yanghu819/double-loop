# GDN FutureSeed D224/L12 No-Checkpoint Long Scale

## 1. Metainfo

- Plan ID: `P-DIAG-011`
- Status: approved
- Local branch: `codex/gpu1-experiment-tracking`
- Planned time: `2026-07-01 19:25 +0800`
- Machine: AIStation `GPU1` only
- Remote work dir: `/huyang2/double-loop`
- Source SHA: pending launch commit
- Parent evidence:
  - P-DIAG-005/P-DIAG-004 showed D192/L10 `GDN_EXPAND_V=4.0`
    has real slope on the 51-55 transition, reaching official `51-55`
    exact `0.0459` after resume scaling.
  - P-DIAG-010 showed D224/L12 no-checkpoint fits in 80GB, stays finite,
    and drops hard-stage CE from `1.4890` at step150 to `1.0738` at step300.
  - P-DIAG-006/P-DIAG-007 NaNs were tied to activation checkpointing, not
    necessarily to depth/width itself.

## 2. Hypothesis

The current bottleneck is not that FutureSeed/GDN cannot learn Sudoku at all;
it is that the model hits a global-consistency cliff once blanks cross roughly
50. The most bitter-lesson-compatible next axis is more effective recurrent
model capacity and longer hard-stage training, using the clean D224/L12
configuration that already passed the short viability gate.

Prediction:

- If capacity is the limiting factor, D224/L12 long resume should beat the
  D192/L10 transition result on holes53 or official `51-55` exact.
- If exact remains flat while blank accuracy improves, then larger same-family
  capacity is not enough; the next change should be a cleaner generic recurrent
  state formulation rather than another width/depth/step table.

This is a single scale test, not an ablation sweep.

## 3. Configuration

- Resume checkpoint:
  `/huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-e97469d-20260701T1050Z/runs/gdn-transition-d224l12-nockpt-viability-s300-20260701T1050Z-e97469d/checkpoints/train_state_step000300.pt`
- Data: official EqR Sudoku arrays, `sudoku-extreme-1k-aug-1000`
- Backbone: GDN Triton recurrent real backward + native terminal-state
  FutureSeed
- Architecture: `D_MODEL=224`, `LAYERS=12`, `HEADS=14`, `HEAD_DIM=16`,
  `CHANNEL_MULT=4`
- Memory/state: `GDN_EXPAND_V=4.0`
- Loop: `L_CYCLES=2`, `MAX_LOOPS=5`
- Loop supervision: `LOOP_LOSS=all`
- Curriculum: `HOLE_STAGES=46-50:100,51-55:2900`
- Train: total `FULL_STEPS=3000`, microbatch `FULL_BATCH=32`,
  `GRAD_ACCUM_STEPS=4`, effective batch `128`
- Eval: official test split, `FULL_EVAL_N=512`
- Checkpoint eval: steps `1000,2000,3000`, holes53
- Official blank-range eval: `46-50,51-55,56-64`
- CUDA path: `GDN_MODE=triton_recurrent`, `GDN_USE_SHORT_CONV=0`,
  `FORWARD_DTYPE=bfloat16`, no activation checkpoint
- Optimization: `LR=0.0015`, `WEIGHT_DECAY=0.001`, `BLANK_LOSS_WEIGHT=8`
- Forbidden: no CPU smoke, no GPU2, no selector/search/repair/oracle rollout,
  no Sudoku-specific rule, no seed/LR/loss/gate table.

## 4. Environment

Pending launch.

## 5. Commands

Remote launch shape:

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 \
SOURCE_SNAPSHOT_MODE=lean UPDATE_LEADERBOARD=0 \
PYTHON_BIN=/opt/conda/bin/python \
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
RESUME_TRAIN_CHECKPOINT=/huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-e97469d-20260701T1050Z/runs/gdn-transition-d224l12-nockpt-viability-s300-20260701T1050Z-e97469d/checkpoints/train_state_step000300.pt \
SUDOKU_SIZE=9 BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 \
GDN_EXPAND_V=4.0 D_MODEL=224 LAYERS=12 HEADS=14 HEAD_DIM=16 CHANNEL_MULT=4 \
L_CYCLES=2 MAX_LOOPS=5 LOOP_LOSS=all \
HOLE_STAGES=46-50:100,51-55:2900 \
FULL_STEPS=3000 FULL_BATCH=32 GRAD_ACCUM_STEPS=4 FULL_EVAL_N=512 \
HOLES_MIN=46 HOLES_MAX=55 EVAL_HOLES=53 EVAL_HOLES_LIST=50,53,56 \
OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64 \
EVAL_CHECKPOINT_STEPS=1000,2000,3000 EVAL_CHECKPOINT_HOLES_LIST=53 \
SAVE_TRAIN_CHECKPOINT_EVERY=500 \
FULL_ROLLOUT_KS= ROLLOUT_LOOP_VALUES= FULL_LOG_EVERY=100 \
BLANK_LOSS_WEIGHT=8 LR=0.0015 WEIGHT_DECAY=0.001 \
FORWARD_DTYPE=bfloat16 FUTURE_SEED_SCALE=1 \
CASE_BANK_N=3 CASE_BANK_EVAL_N=192 CASE_BANK_LOOP_VALUES=1,2,3,5 \
RUN_NAME=<run_name> ./run.sh full
```

Kill criteria:

- Stop exact PID on OOM or NaN.
- Stop if checkpoint1000 holes53 exact remains near zero and both CE and blank
  accuracy slope have clearly flattened.
- Stop if throughput is too slow to finish useful eval within the current
  GPU1 lease.

Success criteria:

- Minimal: no OOM/NaN through step1000 and checkpoint eval written.
- Positive: step3000 holes53 loop5 exact `>= 0.0615` or official `51-55`
  exact `> 0.0459`, exceeding the D192/L10 transition resume baseline.
- Strong: official `51-55` exact `>= 0.08` with visual cases showing fewer
  wrong blank cells through later loops.

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

Not a submission run.
