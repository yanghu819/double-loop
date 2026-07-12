#!/usr/bin/env bash
set -euo pipefail

SOURCE_SHA="eeb38f5b6f9b6c1b1df0bbd3ca149daec06a171d"
WORKTREE="/huyang2/double-loop/.worktrees/pscale029-full-diversity-eeb38f5-20260711T1505Z"
DATA_DIR="/huyang2/double-loop/data/sudoku-extreme-full"
RUN_NAME="gdn-full-diversity-hard-resume6000-s8000-20260711T2150Z-eeb38f5"
CHECKPOINT_DIR="/huyang2/double-loop/models/gdn-full-diversity-d224l12-s6000-20260711T1510Z-eeb38f5/checkpoints"
CHECKPOINT="${CHECKPOINT_DIR}/train_state_step006000.pt"
PARENT_SCORE="/huyang2/double-loop/.worktrees/pscale029-full-diversity-formal-eeb38f5-20260711T1518Z/runs/gdn-full-diversity-d224l12-resume4500-s6000-20260711T1950Z-eeb38f5/score.json"
LAUNCH_DIR="/huyang2/double-loop/artifacts/launch/pscale030"

cd "${WORKTREE}"
test "$(git rev-parse HEAD)" = "${SOURCE_SHA}"
test "$(git status --porcelain --untracked-files=no)" = " M runs/visualization_index.html"
test -z "$(git status --porcelain --untracked-files=no -- . ':(exclude)runs/visualization_index.html')"
test -s "${CHECKPOINT}"
test -s "${PARENT_SCORE}"
test -s "${DATA_DIR}/train/all__inputs.npy"
test -s "${DATA_DIR}/test/all__inputs.npy"
mkdir -p "${CHECKPOINT_DIR}" "${LAUNCH_DIR}"

{
  printf 'timestamp_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'plan_id=P-SCALE-030\n'
  printf 'hypothesis=full-diversity hardest range remains data-compute limited\n'
  printf 'run_name=%s\n' "${RUN_NAME}"
  printf 'source_sha=%s\n' "${SOURCE_SHA}"
  printf 'worktree=%s\n' "${WORKTREE}"
  printf 'provenance_allowlist=runs/visualization_index.html generated-only\n'
  printf 'resume_checkpoint=%s\n' "${CHECKPOINT}"
  printf 'resume_global_step=6000\n'
  printf 'data_source_rows=3831994\n'
  printf 'raw_csv_sha256=64b46674db0148e0d73a16346dadeb2b1c00824d3fca3f85b2ae7037f6b4b38e\n'
  printf 'only_new_stage=51-64:2000\n'
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
export MAX_LOOPS=5
export LOOP_LOSS=all
export LOOP_UPDATE_MODE=fixed
export SCRATCH_MODE=none
export NOISE_SCALE=0
export ROLLOUT_NOISE_SCALE=0
export HIDDEN_AGG_NOISE_MODE=gumbel
export RESUME_TRAIN_CHECKPOINT="${CHECKPOINT}"
export TRAIN_CHECKPOINT_DIR="${CHECKPOINT_DIR}"
export SAVE_TRAIN_CHECKPOINT_EVERY=100
export HOLE_STAGES=46-50:100,51-55:5900,51-64:2000
export EVAL_HOLES=60
export EVAL_HOLES_LIST=53,60,64
export EVAL_CHECKPOINT_STEPS=7000,8000
export EVAL_CHECKPOINT_HOLES_LIST=53,60,64
export OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64
export FULL_BATCH=32
export GRAD_ACCUM_STEPS=4
export FULL_STEPS=8000
export FULL_EVAL_N=512
export FULL_ROLLOUT_KS=""
export FULL_LOG_EVERY=100
export CASE_BANK_HOLES=53,60,64
export CASE_BANK_N=4
export CASE_BANK_EVAL_N=256
export CASE_BANK_LOOP_VALUES=1,3,5
export RUN_NAME

exec ./run.sh full
