#!/usr/bin/env bash
set -euo pipefail
ROOT=/huyang2/double-loop
SHA=0f5771c8557ca757a3e7fd1bd2b16e5f70bf44fc
BASE_ENV=(
  SMOKE_DONE=1 SKIP_SETUP=1 SOURCE_SNAPSHOT_MODE=lean PYTHON_BIN=/opt/conda/bin/python CUDA_VISIBLE_DEVICES=0
  SUDOKU_SIZE=9 HOLE_PATTERN=random HOLE_STAGES=8-16:50,16-24:50 EVAL_HOLES=16 EVAL_HOLES_LIST=8,16,24
  BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 GDN_EXPAND_V=1.0
  D_MODEL=64 LAYERS=2 HEADS=4 HEAD_DIM=16 CHANNEL_MULT=2 L_CYCLES=1 MAX_LOOPS=2
  FULL_STEPS=100 FULL_BATCH=32 FULL_EVAL_N=256 FULL_ROLLOUT_KS= FULL_LOG_EVERY=25
  FORWARD_DTYPE=bfloat16 LR=0.002 BLANK_LOSS_WEIGHT=20
)
run_one() {
  local cond="$1" fs="$2"
  local wt="$ROOT/.worktrees/gdn-triton-${cond}-0f5771c-20260625T2035"
  rm -rf "$wt"
  git -C "$ROOT" worktree add --detach "$wt" "$SHA"
  cd "$wt"
  git status --short --branch
  env "${BASE_ENV[@]}" FUTURE_SEED_SCALE="$fs" RUN_NAME="gdn-triton-${cond}-s100-20260625T2035-0f5771c" ./run.sh full
}
run_one nofs 0.0
run_one fs 1.0
