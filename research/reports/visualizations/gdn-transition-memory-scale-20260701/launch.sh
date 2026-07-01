#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="gdn-transition-memory-expv4-s3000-20260701T0301Z-29c8aea"
REPO_ROOT="/huyang2/double-loop/.worktrees/gdn-transition-memory-29c8aea-20260701T0301Z"
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
export HOLE_STAGES=46-50:500,51-55:2500
export HOLES_MIN=46
export HOLES_MAX=55
export EVAL_HOLES=53
export EVAL_HOLES_LIST=50,53,56
export EVAL_CHECKPOINT_STEPS=1000,2000,3000
export EVAL_CHECKPOINT_HOLES_LIST=53

export BACKBONE=gdn
export GDN_MODE=triton_recurrent
export GDN_USE_SHORT_CONV=0
export GDN_EXPAND_V=4.0
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
export LR=0.0015
export WEIGHT_DECAY=0.001

export FULL_STEPS=3000
export FULL_BATCH=64
export GRAD_ACCUM_STEPS=2
export FULL_EVAL_N=1024
export FULL_LOG_EVERY=100
export FULL_ROLLOUT_KS=
export ROLLOUT_LOOP_VALUES=

export CASE_BANK_N=2
export CASE_BANK_EVAL_N=256
export CASE_BANK_LOOP_VALUES=1,2,3,5

{
  printf 'run_name=%s\n' "${RUN_NAME}"
  printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'repo_root=%s\n' "${REPO_ROOT}"
  printf 'git_sha=%s\n' "$(git rev-parse HEAD)"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES}"
  printf 'hypothesis=%s\n' "Bigger GDN recurrent value/state memory can open the 51-55 blank transition if memory capacity is the cliff bottleneck."
  env | sort | grep -E '^(BACKBONE|BLANK_LOSS_WEIGHT|CASE_BANK|CUDA_VISIBLE_DEVICES|D_MODEL|EVAL|FORWARD_DTYPE|FULL|FUTURE_SEED|GDN|GRAD_ACCUM|HEAD|HOLE|LAYER|L_CYCLES|LR|MAX_LOOPS|OFFICIAL|RUN_NAME|SMOKE_DONE|SKIP_SETUP|SOURCE|SUDOKU|TORCH|WEIGHT|XDG|PYTHON_BIN)='
} > "${LAUNCH_DIR}/launch.env"

./run.sh full
