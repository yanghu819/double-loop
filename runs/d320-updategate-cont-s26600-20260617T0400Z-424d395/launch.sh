#!/usr/bin/env bash
set -euo pipefail

cd "/huyang2/double-loop/.worktrees/d320-updategate-cont-s26600-20260617T0400Z-424d395"
RUN_NAME="d320-updategate-cont-s26600-20260617T0400Z-424d395"
RUN_DIR="runs/${RUN_NAME}"
mkdir -p "${RUN_DIR}/logs"

nohup env \
  SMOKE_DONE=1 \
  SKIP_SETUP=1 \
  PYTHON_BIN=/opt/conda/bin/python \
  PATH="/huyang2/double-loop/.worktrees/d320-updategate-cont-s26600-20260617T0400Z-424d395/.cache/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
  TORCH_EXTENSIONS_DIR="/huyang2/double-loop/.worktrees/d320-updategate-cont-s26600-20260617T0400Z-424d395/.cache/torch_extensions" \
  CUDA_VISIBLE_DEVICES=0 \
  SUDOKU_SIZE=12 \
  BOX_ROWS=3 \
  BOX_COLS=4 \
  HOLE_PATTERN=random \
  D_MODEL=320 \
  LAYERS=12 \
  HEADS=10 \
  HEAD_DIM=32 \
  CHANNEL_MULT=4 \
  L_CYCLES=2 \
  MAX_LOOPS=6 \
  HOLE_STAGES=16-36:200,36-60:300,60-72:500,72-84:1200,84-96:1800,96-108:1800,108-120:20800 \
  EVAL_HOLES=120 \
  EVAL_HOLES_LIST=96,108,120,132 \
  FULL_BATCH=48 \
  GRAD_ACCUM_STEPS=2 \
  FULL_EVAL_N=512 \
  FULL_ROLLOUT_KS=1 \
  ROLLOUT_LOOP_VALUES=1,3,4,5,6 \
  FULL_LOG_EVERY=100 \
  FULL_STEPS=26600 \
  EVAL_CHECKPOINT_STEPS=24600,25600,26600 \
  EVAL_CHECKPOINT_HOLES_LIST=96,108,120,132 \
  SAVE_TRAIN_CHECKPOINT_EVERY=200 \
  RWKV_KERNEL=statepassing \
  FORWARD_DTYPE=bfloat16 \
  LOOP_LOSS=all \
  LOOP_UPDATE_MODE=learned_gate \
  LOOP_UPDATE_GATE_INIT=0.95 \
  RESUME_TRAIN_CHECKPOINT="/huyang2/double-loop/.worktrees/d320-updategate-remap-s23600-20260616T1005Z-8e2e217/runs/d320-updategate-remap-s23600-20260616T1005Z-8e2e217/checkpoints/latest.pt" \
  RUN_NAME="${RUN_NAME}" \
  ./run.sh full \
  > "${RUN_DIR}/logs/nohup.log" 2>&1 &

echo "$!" > "${RUN_DIR}/pid"
echo "launched pid=$(cat "${RUN_DIR}/pid") run=${RUN_NAME}"
