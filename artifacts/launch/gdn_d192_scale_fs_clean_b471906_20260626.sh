#!/usr/bin/env bash
set -euo pipefail

ROOT=/huyang2/double-loop
SHA=b47190644a97ace3d51283923d8a8806437ce628
WT="$ROOT/.worktrees/gdn-realbwd-official-sudoku-d192-fs-b471906-20260626"
DATA="$ROOT/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000"
LOG="$ROOT/artifacts/logs/gdn_d192_scale_fs_clean_b471906_20260626.log"
PID="$ROOT/artifacts/logs/gdn_d192_scale_fs_clean_b471906_20260626.pid"
RUN_NAME=gdn-d192-official-sudoku-fs-clean-s600-20260626T0825-b471906

mkdir -p "$ROOT/artifacts/logs"
test -d "$WT"
test "$(git -C "$WT" rev-parse HEAD)" = "$SHA"
test -d "$DATA"

FILTERED_STATUS="$(
  {
    git -C "$WT" diff --name-only
    git -C "$WT" diff --cached --name-only
    git -C "$WT" ls-files --others --exclude-standard
  } | grep -Ev '^(runs|artifacts|\.cache|\.venv)/' || true
)"
if [ -n "$FILTERED_STATUS" ]; then
  echo "Refusing to launch from dirty source tree:" >&2
  printf '%s\n' "$FILTERED_STATUS" >&2
  exit 2
fi

rm -f "$LOG" "$LOG.nohup" "$PID"
cd "$WT"
{
  echo "launch_sha=$(git rev-parse HEAD)"
  echo "launch_status_filtered_lines=0"
  echo "run_name=$RUN_NAME"
  echo "data=$DATA"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
  SMOKE_DONE=1 SKIP_SETUP=1 SOURCE_SNAPSHOT_MODE=lean \
  PYTHON_BIN=/opt/conda/bin/python CUDA_VISIBLE_DEVICES=0 \
  OFFICIAL_SUDOKU_DATA_DIR="$DATA" OFFICIAL_SUDOKU_TRAIN_SPLIT=train OFFICIAL_SUDOKU_EVAL_SPLIT=test \
  SUDOKU_SIZE=9 HOLE_PATTERN=random EVAL_HOLES=16 EVAL_HOLES_LIST=16 \
  BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 GDN_EXPAND_V=1.0 \
  D_MODEL=192 LAYERS=10 HEADS=12 HEAD_DIM=16 CHANNEL_MULT=4 L_CYCLES=2 MAX_LOOPS=5 \
  FULL_STEPS=600 FULL_BATCH=128 FULL_EVAL_N=2048 FULL_ROLLOUT_KS= FULL_LOG_EVERY=100 \
  FORWARD_DTYPE=bfloat16 LR=0.0015 WEIGHT_DECAY=0.001 BLANK_LOSS_WEIGHT=8 \
  FUTURE_SEED_SCALE=1 \
  RUN_NAME="$RUN_NAME" ./run.sh full
} 2>&1 | tee "$LOG"
