# GDN FutureSeed Conservative Depth Scaling

## 1. Metainfo

- Plan ID: `P-DIAG-007`
- Status: in-progress
- Local branch: `codex/gpu1-experiment-tracking`
- Scheduled time: `2026-07-01 15:45:00 +0800`
- Machine: AIStation `GPU1` only
- Remote work dir: `/huyang2/double-loop`
- Source SHA: pending launch commit
- Parent evidence:
  `gdn-transition-resume-expv4-lr15-s4500-20260701T0622Z-33b477e`
  and failed D224/L12 NaN boundary

## 2. Hypothesis

D224/L12 + expand_v4 failed at step100 with NaN under the proven D192 LR. That
does not answer whether capacity scaling works; it only says the aggressive
width+depth jump is not optimizer-stable.

Mechanism question: does a smaller capacity jump, specifically D192/L10 to
D192/L12 while keeping recurrent value/state expansion, improve the 51-55
transition without destabilizing training?

Prediction:

- If depth is a useful capacity axis, D192/L12 should beat D192/L10 expand_v4
  at matched steps and ideally push holes53 loop5 exact toward `>=0.06`.
- If it NaNs or underperforms D192/L10, then blind depth/state scaling is not
  the immediate upper-bound path. The next work should stabilize the generic
  recurrent state/optimizer rather than add more layers.

This is one conservative scaling gate, not a depth table.

## 3. Configuration

- Data: official EqR Sudoku arrays, `sudoku-extreme-1k-aug-1000`
- Backbone: GDN Triton recurrent real backward + native terminal-state
  FutureSeed
- Architecture: `D_MODEL=192`, `LAYERS=12`, `HEADS=12`, `HEAD_DIM=16`,
  `CHANNEL_MULT=4`
- Memory/state: `GDN_EXPAND_V=4.0`
- Loop: `L_CYCLES=2`, `MAX_LOOPS=5`, final-loop loss
- Curriculum: `HOLE_STAGES=46-50:500,51-55:2500`
- Train: `FULL_STEPS=3000`, microbatch `FULL_BATCH=64`,
  `GRAD_ACCUM_STEPS=2`, effective batch `128`
- Eval: official test split, `FULL_EVAL_N=1024`
- Checkpoint eval: steps `1000,2000,3000`, holes53
- Official blank-range eval: `46-50,51-55,56-64`
- Case-bank visualization: loops `1,2,3,5`
- CUDA path: `GDN_MODE=triton_recurrent`, `GDN_USE_SHORT_CONV=0`,
  `FORWARD_DTYPE=bfloat16`, `ACTIVATION_CHECKPOINT=1`
- Optimization: `LR=0.0015`, `WEIGHT_DECAY=0.001`, `BLANK_LOSS_WEIGHT=8`
- Forbidden: no CPU smoke, no GPU2, no selector/search/repair/oracle rollout,
  no Sudoku-specific rule, no LR/seed/width/depth table.

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
L_CYCLES=2 MAX_LOOPS=5 HOLE_STAGES=46-50:500,51-55:2500 \
FULL_STEPS=3000 FULL_BATCH=64 GRAD_ACCUM_STEPS=2 FULL_EVAL_N=1024 \
HOLES_MIN=46 HOLES_MAX=55 EVAL_HOLES=53 EVAL_HOLES_LIST=50,53,56 \
OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64 \
EVAL_CHECKPOINT_STEPS=1000,2000,3000 EVAL_CHECKPOINT_HOLES_LIST=53 \
SAVE_TRAIN_CHECKPOINT_EVERY=1000 \
FULL_ROLLOUT_KS= ROLLOUT_LOOP_VALUES= FULL_LOG_EVERY=100 \
BLANK_LOSS_WEIGHT=8 LR=0.0015 WEIGHT_DECAY=0.001 \
FORWARD_DTYPE=bfloat16 FUTURE_SEED_SCALE=1 ACTIVATION_CHECKPOINT=1 \
CASE_BANK_N=2 CASE_BANK_EVAL_N=256 CASE_BANK_LOOP_VALUES=1,2,3,5 \
RUN_NAME=<run_name> ./run.sh full
```

Kill criteria:

- If step100 is NaN, stop exact PID and archive. Do not lower LR repeatedly.
- If step100 takes more than `20` minutes, stop as throughput-low-ROI.
- If step1000 holes53 loop5 exact is below D192/L10 expand_v4 step1000
  (`0.0205`) and blank_acc/CE are not better, stop unless a checkpoint is
  already near.
- If step2000 remains below D192/L10 step2000 (`0.0215`) with worse blank_acc,
  stop and classify depth scaling as low ROI.

Success criteria:

- Strong: official `51-55` loop5 exact `>= 0.05`, or holes53 checkpoint loop5
  exact `>= 0.06` by step3000.
- Weak: better matched-step slope than D192/L10 and case-bank hard boards keep
  changing through loop5.
- Negative: instability or no matched-step improvement; move to a cleaner
  generic recurrent state formulation.

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

Not a submission run.
