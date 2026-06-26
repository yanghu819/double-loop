#!/usr/bin/env bash
set -euo pipefail

ROOT=/huyang2/double-loop
WT="$ROOT/.worktrees/gdn-realbwd-official-sudoku-scale-fs-clean-be2815b-20260626"
DATA="$ROOT/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000"
LOG="$ROOT/artifacts/logs/gdn_realbwd_official_sudoku_scale_fs_clean_be2815b_20260626.log"
mkdir -p "$ROOT/artifacts/logs"

cd "$WT"
test "$(git rev-parse HEAD)" = "be2815bd27ece12df237ed01456e651d1dac5212"
test "$(git status --short | wc -l)" = "0"
test -d "$DATA"

{
  echo "launch_sha=$(git rev-parse HEAD)"
  echo "launch_dirty=$(git status --short | wc -l)"
  echo "data=$DATA"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
  SMOKE_DONE=1 SKIP_SETUP=1 SOURCE_SNAPSHOT_MODE=lean \
  PYTHON_BIN=/opt/conda/bin/python CUDA_VISIBLE_DEVICES=0 \
  OFFICIAL_SUDOKU_DATA_DIR="$DATA" OFFICIAL_SUDOKU_TRAIN_SPLIT=train OFFICIAL_SUDOKU_EVAL_SPLIT=test \
  SUDOKU_SIZE=9 HOLE_PATTERN=random EVAL_HOLES=16 EVAL_HOLES_LIST=16 \
  BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 GDN_EXPAND_V=1.0 \
  D_MODEL=128 LAYERS=6 HEADS=8 HEAD_DIM=16 CHANNEL_MULT=4 L_CYCLES=2 MAX_LOOPS=4 \
  FULL_STEPS=600 FULL_BATCH=128 FULL_EVAL_N=1024 FULL_ROLLOUT_KS= FULL_LOG_EVERY=100 \
  FORWARD_DTYPE=bfloat16 LR=0.0015 WEIGHT_DECAY=0.001 BLANK_LOSS_WEIGHT=8 \
  FUTURE_SEED_SCALE=1 \
  RUN_NAME=gdn-realbwd-official-sudoku-scale-fs-clean-20260626T0105-be2815b ./run.sh full
} 2>&1 | tee "$LOG"
