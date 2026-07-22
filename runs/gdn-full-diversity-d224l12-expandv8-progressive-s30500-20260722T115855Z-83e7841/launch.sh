#!/usr/bin/env bash
set -euo pipefail

EXPECTED_SHA=83e7841f408c642b31a981bde43b710e98e3b22d
MODEL_CODE_SHA=c7d6d9977d61125810123ea52de5c5e3362a5d89
ROOT=/huyang2/double-loop
WT="$ROOT/.worktrees/pscale035-state-expandv8-83e7841-full-20260722"
PY=/opt/conda/bin/python
DATA="$ROOT/data/sudoku-extreme-full"
PARENT_CKPT="$ROOT/models/gdn-full-diversity-d224l12-s6000-20260711T1510Z-eeb38f5/checkpoints/train_state_step030000.pt"
CHECKPOINT_DIR="$ROOT/models/gdn-full-diversity-d224l12-expandv8-progressive-20260722/checkpoints"
ARTIFACT_DIR="$ROOT/artifacts/launch/pscale035-state-expandv8-20260722"
LAUNCH_ENV="$ARTIFACT_DIR/pscale035_s30500.launch.env"
SEGMENT_TS="$(date -u '+%Y%m%dT%H%M%SZ')"
RUN_NAME="gdn-full-diversity-d224l12-expandv8-progressive-s30500-${SEGMENT_TS}-83e7841"

test "$(git -C "$WT" rev-parse HEAD)" = "$EXPECTED_SHA"
test -z "$(git -C "$WT" status --porcelain --untracked-files=no)"
test -x "$PY"
test -f "$DATA/train/all__inputs.npy"
test -f "$DATA/train/all__labels.npy"
test -f "$DATA/test/all__inputs.npy"
test -f "$DATA/test/all__labels.npy"
test -s "$PARENT_CKPT"
test -s "$LAUNCH_ENV"

# The tracking-only commit after MODEL_CODE_SHA must not alter the executable path.
git -C "$WT" diff --quiet "$MODEL_CODE_SHA" "$EXPECTED_SHA" -- \
  run.sh \
  experiments/rwkv_fs_sudoku/study_rwkv_futureseed_loop.py \
  experiments/rwkv_fs_sudoku/gdn_triton.py

GPU_ROWS="$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader)"
test "$(printf '%s\n' "$GPU_ROWS" | wc -l | tr -d ' ')" = 1
printf '%s\n' "$GPU_ROWS" | grep -q 'NVIDIA A800-SXM4-80GB'
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"

mkdir -p "$CHECKPOINT_DIR"

export CUDA_VISIBLE_DEVICES=0
export PYTHON_BIN="$PY"
export PYTHON_EXTRA_PATH="$ROOT/.cache/python-extra-pylib"
export XDG_CACHE_HOME="$ROOT/.cache"
export UV_CACHE_DIR="$ROOT/.cache/uv"
export TRITON_CACHE_DIR="$ROOT/.cache/triton"
export TORCH_EXTENSIONS_DIR="$ROOT/.cache/torch_extensions"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SOURCE_SNAPSHOT_MODE=lean

cd "$WT"
RUN_DIR="$WT/runs/$RUN_NAME"
mkdir -p "$RUN_DIR"
cp "$LAUNCH_ENV" "$RUN_DIR/launch.env"
cp "${BASH_SOURCE[0]}" "$RUN_DIR/launch.sh"
printf 'actual_timestamp_utc=%s\nrun_name=%s\n' "$SEGMENT_TS" "$RUN_NAME" >>"$RUN_DIR/launch.env"

printf 'run_name=%s\nsource_sha=%s\nresume_step=30000\ntarget_step=30500\n' \
  "$RUN_NAME" "$EXPECTED_SHA"

env \
  SKIP_SETUP=1 SMOKE_DONE=1 \
  SUDOKU_SIZE=9 OFFICIAL_SUDOKU_DATA_DIR="$DATA" \
  OFFICIAL_SUDOKU_TRAIN_SPLIT=train OFFICIAL_SUDOKU_EVAL_SPLIT=test \
  BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 \
  GDN_EXPAND_V=8 GDN_PROGRESSIVE_BASE_EXPAND_V=4 \
  D_MODEL=224 LAYERS=12 HEADS=14 HEAD_DIM=16 CHANNEL_MULT=4 L_CYCLES=2 \
  FUTURE_SEED_SCALE=1 FUTURE_SEED_DECAY=0 FUTURE_SEED_UPDATE=fixed FUTURE_SEED_NORM_MODE=unit \
  MAX_LOOPS=5 LOOP_LOSS=all LOOP_UPDATE_MODE=fixed LOOP_UPDATE_GATE_INIT=0.95 \
  LOOP_FEEDBACK_SCALE=0 LOOP_TIME_SCALE=0 SCRATCH_MODE=none \
  NOISE_SCALE=0 ROLLOUT_NOISE_SCALE=0 HIDDEN_AGG_NOISE_SCALE=0 EXACT_MARGIN_WEIGHT=0 \
  FORWARD_DTYPE=bfloat16 BLANK_LOSS_WEIGHT=20 LR=0.002 WEIGHT_DECAY=0.001 \
  FULL_BATCH=16 GRAD_ACCUM_STEPS=8 FULL_EVAL_N=512 FULL_LOG_EVERY=25 FULL_ROLLOUT_KS="" \
  HOLES_MIN=46 HOLES_MAX=64 HOLE_STAGES=46-50:100,51-55:5900,51-64:24500 \
  EVAL_HOLES=60 EVAL_HOLES_LIST=53,60,64 \
  EVAL_CHECKPOINT_STEPS=30500 EVAL_CHECKPOINT_HOLES_LIST=53,60,64 \
  OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64 \
  CASE_BANK_HOLES=53,60,64 CASE_BANK_N=4 CASE_BANK_EVAL_N=256 CASE_BANK_LOOP_VALUES=1,3,5 \
  FULL_STEPS=30500 SAVE_TRAIN_CHECKPOINT_EVERY=25 \
  RESUME_TRAIN_CHECKPOINT="$PARENT_CKPT" TRAIN_CHECKPOINT_DIR="$CHECKPOINT_DIR" \
  RUN_NAME="$RUN_NAME" \
  ./run.sh full
