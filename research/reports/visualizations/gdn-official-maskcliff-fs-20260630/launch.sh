#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="gdn-official-maskcliff-fs-d192l10-s1000-20260630T1357Z-74ed759"
REPO_ROOT="/huyang2/double-loop/.worktrees/gdn-maskcliff-74ed759-20260630T1357Z"
LAUNCH_DIR="/huyang2/double-loop/artifacts/launch/${RUN_NAME}"
mkdir -p "${LAUNCH_DIR}"

cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES=0
export SMOKE_DONE=1
export SKIP_SETUP=1
export SOURCE_SNAPSHOT_MODE=lean
export UPDATE_LEADERBOARD=0
export PYTHON_BIN=/opt/conda/bin/python

export XDG_CACHE_HOME=/huyang2/double-loop/.cache
export UV_CACHE_DIR=/huyang2/double-loop/.cache/uv
export UV_PYTHON_INSTALL_DIR=/huyang2/double-loop/.cache/uv/python
export PIP_CACHE_DIR=/huyang2/double-loop/.cache/pip
export HF_HOME=/huyang2/double-loop/.cache/huggingface
export TORCH_HOME=/huyang2/double-loop/.cache/torch
export TORCH_EXTENSIONS_DIR=/huyang2/double-loop/.cache/torch_extensions

export RUN_NAME
export SUDOKU_SIZE=9
export HOLE_PATTERN=random
export OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000
export OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64
export HOLES_MIN=46
export HOLES_MAX=64
export EVAL_HOLES=56
export EVAL_HOLES_LIST=56
export EVAL_CHECKPOINT_STEPS=500,1000

export BACKBONE=gdn
export GDN_MODE=triton_recurrent
export GDN_USE_SHORT_CONV=0
export D_MODEL=192
export LAYERS=10
export HEADS=12
export HEAD_DIM=16
export CHANNEL_MULT=4
export L_CYCLES=2
export MAX_LOOPS=5
export FUTURE_SEED_SCALE=1
export BLANK_LOSS_WEIGHT=8
export FORWARD_DTYPE=bfloat16

export FULL_STEPS=1000
export FULL_BATCH=128
export FULL_EVAL_N=512
export FULL_LOG_EVERY=100
export FULL_ROLLOUT_KS=
export ROLLOUT_LOOP_VALUES=

export CASE_BANK_N=1
export CASE_BANK_EVAL_N=256
export CASE_BANK_LOOP_VALUES=1,2,3,5

{
  printf 'run_name=%s\n' "${RUN_NAME}"
  printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'repo_root=%s\n' "${REPO_ROOT}"
  printf 'git_sha=%s\n' "$(git rev-parse HEAD)"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'hypothesis=%s\n' "GDN+native FutureSeed should reveal whether the high-mask cliff is generic FutureSeed-loop global consistency or RWKV-specific."
  env | sort | grep -E '^(BACKBONE|BLANK_LOSS_WEIGHT|CASE_BANK|CUDA_VISIBLE_DEVICES|D_MODEL|EVAL|FORWARD_DTYPE|FULL|FUTURE_SEED|GDN|HEAD|HOLE|LAYER|L_CYCLES|MAX_LOOPS|OFFICIAL|RUN_NAME|SMOKE_DONE|SKIP_SETUP|SOURCE|SUDOKU|TORCH|XDG|PYTHON_BIN)='
} > "${LAUNCH_DIR}/launch.env"

./run.sh full
