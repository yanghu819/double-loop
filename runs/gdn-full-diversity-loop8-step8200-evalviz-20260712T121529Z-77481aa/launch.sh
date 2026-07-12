#!/usr/bin/env bash
set -euo pipefail

SOURCE_SHA=77481aa3dca1a654b08b16006943806674fe6394
WORKTREE=/huyang2/double-loop/.worktrees/pscale031-loop8-77481aa-20260712T1943
DATA_DIR=/huyang2/double-loop/data/sudoku-extreme-full
CHECKPOINT=/huyang2/double-loop/models/gdn-full-diversity-loop8-s8600-20260712T115200Z-77481aa/checkpoints/train_state_step008200.pt
RUN_NAME=gdn-full-diversity-loop8-step8200-evalviz-20260712T121529Z-77481aa
LAUNCH_DIR=/huyang2/double-loop/artifacts/launch/pscale031

cd "${WORKTREE}"
test "$(git rev-parse HEAD)" = "${SOURCE_SHA}"
test -z "$(git status --porcelain --untracked-files=no)"
test -s "${CHECKPOINT}"
test -s "${DATA_DIR}/train/all__inputs.npy"
test -s "${DATA_DIR}/test/all__inputs.npy"
mkdir -p "${LAUNCH_DIR}"

{
  printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'plan_id=P-SCALE-031\n'
  printf 'purpose=post-abort full official eval and loop1-3-5-6-7-8 case visualization\n'
  printf 'run_name=%s\n' "${RUN_NAME}"
  printf 'source_sha=%s\n' "${SOURCE_SHA}"
  printf 'worktree=%s\n' "${WORKTREE}"
  printf 'resume_checkpoint=%s\n' "${CHECKPOINT}"
  printf 'resume_global_step=8200\n'
  printf 'training_steps_this_run=0\n'
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
export OFFICIAL_SUDOKU_DATA_DIR="${DATA_DIR}"
export OFFICIAL_SUDOKU_TRAIN_SPLIT=train
export OFFICIAL_SUDOKU_EVAL_SPLIT=test
export BACKBONE=gdn
export GDN_MODE=triton_recurrent
export GDN_USE_SHORT_CONV=0
export GDN_EXPAND_V=4.0
export FORWARD_DTYPE=bfloat16
export D_MODEL=224
export LAYERS=12
export HEADS=14
export HEAD_DIM=16
export CHANNEL_MULT=4
export L_CYCLES=2
export FUTURE_SEED_SCALE=1
export FUTURE_SEED_UPDATE=fixed
export MAX_LOOPS=8
export LOOP_LOSS=all
export LOOP_UPDATE_MODE=fixed
export SCRATCH_MODE=none
export NOISE_SCALE=0
export ROLLOUT_NOISE_SCALE=0
export HIDDEN_AGG_NOISE_MODE=gumbel
export RESUME_TRAIN_CHECKPOINT="${CHECKPOINT}"
export HOLE_STAGES=46-50:100,51-55:5900,51-64:2200
export EVAL_HOLES=60
export EVAL_HOLES_LIST=53,60,64
export OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64
export FULL_BATCH=32
export GRAD_ACCUM_STEPS=4
export FULL_STEPS=8200
export FULL_EVAL_N=512
export FULL_ROLLOUT_KS=""
export FULL_LOG_EVERY=100
export CASE_BANK_HOLES=53,60,64
export CASE_BANK_N=4
export CASE_BANK_EVAL_N=256
export CASE_BANK_LOOP_VALUES=1,3,5,6,7,8
export RUN_NAME

exec ./run.sh full
