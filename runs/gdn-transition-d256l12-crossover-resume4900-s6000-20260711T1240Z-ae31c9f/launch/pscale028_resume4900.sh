#!/usr/bin/env bash
set -euo pipefail

SOURCE_SHA="ae31c9f54ffb0abaecf48e2102942431f57d36b3"
WORKTREE="/huyang2/double-loop/.worktrees/gdn-transition-d256l12-width-ae31c9f-20260702T1214Z"
RUN_NAME="gdn-transition-d256l12-crossover-resume4900-s6000-20260711T1240Z-ae31c9f"
CHECKPOINT="/huyang2/double-loop/models/gdn-transition-d256l12-crossover-s6000-20260711T0846Z-ae31c9f/checkpoints/train_state_step004900.pt"
CHECKPOINT_DIR="/huyang2/double-loop/models/gdn-transition-d256l12-crossover-s6000-20260711T0846Z-ae31c9f/checkpoints"
LAUNCH_DIR="/huyang2/double-loop/artifacts/launch/pscale028"

cd "${WORKTREE}"
test "$(git rev-parse HEAD)" = "${SOURCE_SHA}"
test -z "$(git status --porcelain --untracked-files=no)"
test -s "${CHECKPOINT}"
mkdir -p "${CHECKPOINT_DIR}" "${LAUNCH_DIR}"

{
  printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'run_name=%s\n' "${RUN_NAME}"
  printf 'source_sha=%s\n' "${SOURCE_SHA}"
  printf 'resume_checkpoint=%s\n' "${CHECKPOINT}"
  printf 'resume_global_step=4900\n'
  printf 'cuda_visible_devices=0\n'
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
} > "${LAUNCH_DIR}/${RUN_NAME}.launch.env"

export CUDA_VISIBLE_DEVICES=0
export SMOKE_DONE=1
export SKIP_SETUP=1
export SOURCE_SNAPSHOT_MODE=lean
export UPDATE_LEADERBOARD=0
export PYTHON_BIN=/opt/conda/bin/python
export SUDOKU_SIZE=9
export HOLE_PATTERN=random
export OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000
export BACKBONE=gdn
export GDN_MODE=triton_recurrent
export GDN_USE_SHORT_CONV=0
export GDN_EXPAND_V=4.0
export FORWARD_DTYPE=bfloat16
export D_MODEL=256
export LAYERS=12
export HEADS=16
export HEAD_DIM=16
export CHANNEL_MULT=4
export L_CYCLES=2
export FUTURE_SEED_SCALE=1
export FUTURE_SEED_UPDATE=fixed
export MAX_LOOPS=5
export LOOP_LOSS=all
export LOOP_UPDATE_MODE=fixed
export SCRATCH_MODE=none
export NOISE_SCALE=0
export ROLLOUT_NOISE_SCALE=0
export RESUME_TRAIN_CHECKPOINT="${CHECKPOINT}"
export TRAIN_CHECKPOINT_DIR="${CHECKPOINT_DIR}"
export SAVE_TRAIN_CHECKPOINT_EVERY=100
export HOLE_STAGES=46-50:100,51-55:900,51-64:1000,51-55:500,56-64:500,51-64:1000,51-55:500,56-64:500,51-64:1000
export EVAL_HOLES=60
export EVAL_HOLES_LIST=53,60,64
export EVAL_CHECKPOINT_STEPS=6000
export EVAL_CHECKPOINT_HOLES_LIST=53,60,64
export OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64
export FULL_BATCH=32
export GRAD_ACCUM_STEPS=4
export FULL_STEPS=6000
export FULL_EVAL_N=512
export FULL_ROLLOUT_KS=""
export FULL_LOG_EVERY=100
export CASE_BANK_HOLES=53,60,64
export CASE_BANK_N=4
export CASE_BANK_EVAL_N=256
export CASE_BANK_LOOP_VALUES=1,3,5
export RUN_NAME

exec ./run.sh full
