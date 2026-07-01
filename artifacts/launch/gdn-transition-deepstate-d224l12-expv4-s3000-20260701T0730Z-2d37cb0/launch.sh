#!/usr/bin/env bash
set -euo pipefail

REPO=/huyang2/double-loop
SHA=2d37cb02f0f8ce9e6a038d164220a40544d5c944
RUN_NAME=gdn-transition-deepstate-d224l12-expv4-s3000-20260701T0730Z-2d37cb0
WT=/huyang2/double-loop/.worktrees/gdn-transition-deepstate-d224l12-2d37cb0-20260701T0730Z
LAUNCH_DIR=/huyang2/double-loop/artifacts/launch/${RUN_NAME}
mkdir -p "${LAUNCH_DIR}"

cd "${REPO}"
if [[ ! -d "${WT}/.git" ]]; then
  git worktree add --detach "${WT}" "${SHA}"
fi

cd "${WT}"
git checkout --detach "${SHA}"
git rev-parse HEAD > "${LAUNCH_DIR}/source_sha.txt"
git status --short > "${LAUNCH_DIR}/git_status.txt"

cat > "${LAUNCH_DIR}/launch.env" <<ENV
RUN_NAME=${RUN_NAME}
SOURCE_SHA=${SHA}
CUDA_VISIBLE_DEVICES=0
SMOKE_DONE=1
SKIP_SETUP=1
SOURCE_SNAPSHOT_MODE=lean
UPDATE_LEADERBOARD=0
PYTHON_BIN=/opt/conda/bin/python
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000
SUDOKU_SIZE=9
BACKBONE=gdn
GDN_MODE=triton_recurrent
GDN_USE_SHORT_CONV=0
GDN_EXPAND_V=4.0
D_MODEL=224
LAYERS=12
HEADS=14
HEAD_DIM=16
CHANNEL_MULT=4
L_CYCLES=2
MAX_LOOPS=5
HOLE_STAGES=46-50:500,51-55:2500
FULL_STEPS=3000
FULL_BATCH=48
GRAD_ACCUM_STEPS=3
FULL_EVAL_N=1024
HOLES_MIN=46
HOLES_MAX=55
EVAL_HOLES=53
EVAL_HOLES_LIST=50,53,56
OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64
EVAL_CHECKPOINT_STEPS=1000,2000,3000
EVAL_CHECKPOINT_HOLES_LIST=53
SAVE_TRAIN_CHECKPOINT_EVERY=1000
FULL_ROLLOUT_KS=
ROLLOUT_LOOP_VALUES=
FULL_LOG_EVERY=100
BLANK_LOSS_WEIGHT=8
LR=0.0015
WEIGHT_DECAY=0.001
FORWARD_DTYPE=bfloat16
FUTURE_SEED_SCALE=1
ACTIVATION_CHECKPOINT=1
CASE_BANK_N=2
CASE_BANK_EVAL_N=256
CASE_BANK_LOOP_VALUES=1,2,3,5
ENV

(
  set -a
  source "${LAUNCH_DIR}/launch.env"
  set +a
  export XDG_CACHE_HOME=/huyang2/double-loop/.cache
  export PIP_CACHE_DIR=/huyang2/double-loop/.cache/pip
  export HF_HOME=/huyang2/double-loop/.cache/huggingface
  export TORCH_HOME=/huyang2/double-loop/.cache/torch
  export TORCH_EXTENSIONS_DIR=/huyang2/double-loop/.cache/torch_extensions
  export CUDA_VISIBLE_DEVICES=0
  ./run.sh full
) > "${LAUNCH_DIR}/launcher.log" 2>&1 &

echo $! > "${LAUNCH_DIR}/wrapper.pid"
echo "${WT}/runs/${RUN_NAME}" > "${LAUNCH_DIR}/run_dir.txt"
echo "launched ${RUN_NAME} wrapper_pid=$(cat "${LAUNCH_DIR}/wrapper.pid")"
