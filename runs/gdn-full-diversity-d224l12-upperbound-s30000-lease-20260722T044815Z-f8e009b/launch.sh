#!/usr/bin/env bash

set -euo pipefail

EXPECTED_SHA="f8e009bedcb0a6db88a49a79fc77b260cb3f3f30"
PARENT_SHA="eeb38f5b6f9b6c1b1df0bbd3ca149daec06a171d"
REPO="/huyang2/double-loop"
WT="${REPO}/.worktrees/pscale034-step30000-f8e009b-20260722"
PY="/opt/conda/bin/python"
PY_RECORD="${REPO}/.cache/python-bin"
DATA="${REPO}/data/sudoku-extreme-full"
CHECKPOINT_DIR="${REPO}/models/gdn-full-diversity-d224l12-s6000-20260711T1510Z-eeb38f5/checkpoints"
ARTIFACT_DIR="${REPO}/artifacts/launch/pscale034-step30000-lease-resumer-20260722"
LAUNCH_ENV="${ARTIFACT_DIR}/pscale034_step30000.launch.env"
SEGMENT_TS="$(date -u '+%Y%m%dT%H%M%SZ')"
RUN_NAME="gdn-full-diversity-d224l12-upperbound-s30000-lease-${SEGMENT_TS}-f8e009b"

test "$(git -C "${WT}" rev-parse HEAD)" = "${EXPECTED_SHA}"
test -z "$(git -C "${WT}" status --porcelain --untracked-files=no)"
test -x "${PY}"
test -s "${PY_RECORD}"
test "$(<"${PY_RECORD}")" = "${PY}"
test -f "${DATA}/train/all__inputs.npy"
test -f "${DATA}/train/all__labels.npy"
test -f "${DATA}/test/all__inputs.npy"
test -f "${DATA}/test/all__labels.npy"
test -f "${LAUNCH_ENV}"

# This final endpoint changes only the amount of broad 51-64 training compute.
git -C "${WT}" diff --quiet "${PARENT_SHA}" "${EXPECTED_SHA}" -- \
  run.sh \
  experiments/rwkv_fs_sudoku/study_rwkv_futureseed_loop.py \
  experiments/rwkv_fs_sudoku/gdn_triton.py

CHECKPOINT=""
RESUME_STEP=""
while IFS= read -r candidate; do
  filename="$(basename "${candidate}")"
  if [[ "${filename}" =~ ^train_state_step([0-9]+)\.pt$ ]]; then
    step=$((10#${BASH_REMATCH[1]}))
    if (( step >= 24000 && step <= 30000 )); then
      CHECKPOINT="${candidate}"
      RESUME_STEP="${step}"
    fi
  fi
done < <(find "${CHECKPOINT_DIR}" -maxdepth 1 -type f -name 'train_state_step*.pt' | sort)

test -n "${CHECKPOINT}"
test -n "${RESUME_STEP}"
test -s "${CHECKPOINT}"

GPU_ROWS="$(nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader)"
test "$(printf '%s\n' "${GPU_ROWS}" | wc -l | tr -d ' ')" = "1"
printf '%s\n' "${GPU_ROWS}" | grep -q 'NVIDIA A800-SXM4-80GB'

export CUDA_VISIBLE_DEVICES=0
export PYTHON_BIN="${PY}"
export PYTHON_EXTRA_PATH="${REPO}/.cache/python-extra-pylib"
export UV_CACHE_DIR="${REPO}/.cache/uv"
export TRITON_CACHE_DIR="${REPO}/.cache/triton"
export TORCH_EXTENSIONS_DIR="${REPO}/.cache/torch_extensions"
export SOURCE_SNAPSHOT_MODE=lean

cd "${WT}"

RUN_DIR="${WT}/runs/${RUN_NAME}"
mkdir -p "${RUN_DIR}"
cp "${LAUNCH_ENV}" "${RUN_DIR}/launch.env"
cp "${BASH_SOURCE[0]}" "${RUN_DIR}/launch.sh"
printf 'segment_timestamp_utc=%s\nactual_resume_checkpoint=%s\nactual_resume_global_step=%s\nrun_name=%s\n' \
  "${SEGMENT_TS}" "${CHECKPOINT}" "${RESUME_STEP}" "${RUN_NAME}" >> "${RUN_DIR}/launch.env"

printf 'segment_run_name=%s\nresume_step=%s\n' "${RUN_NAME}" "${RESUME_STEP}"

env \
  SKIP_SETUP=1 SMOKE_DONE=1 \
  SUDOKU_SIZE=9 OFFICIAL_SUDOKU_DATA_DIR="${DATA}" \
  OFFICIAL_SUDOKU_TRAIN_SPLIT=train OFFICIAL_SUDOKU_EVAL_SPLIT=test \
  BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 GDN_EXPAND_V=4.0 \
  D_MODEL=224 LAYERS=12 HEADS=14 HEAD_DIM=16 CHANNEL_MULT=4 L_CYCLES=2 \
  FUTURE_SEED_SCALE=1 FUTURE_SEED_DECAY=0 FUTURE_SEED_UPDATE=fixed FUTURE_SEED_NORM_MODE=unit \
  MAX_LOOPS=5 LOOP_LOSS=all LOOP_UPDATE_MODE=fixed LOOP_UPDATE_GATE_INIT=0.95 \
  LOOP_FEEDBACK_SCALE=0 LOOP_TIME_SCALE=0 SCRATCH_MODE=none \
  NOISE_SCALE=0 ROLLOUT_NOISE_SCALE=0 HIDDEN_AGG_NOISE_SCALE=0 EXACT_MARGIN_WEIGHT=0 \
  FORWARD_DTYPE=bfloat16 BLANK_LOSS_WEIGHT=20 LR=0.002 WEIGHT_DECAY=0.001 \
  FULL_BATCH=32 GRAD_ACCUM_STEPS=4 FULL_EVAL_N=512 FULL_LOG_EVERY=25 FULL_ROLLOUT_KS="" \
  HOLE_STAGES=46-50:100,51-55:5900,51-64:24000 \
  EVAL_HOLES=60 EVAL_HOLES_LIST=53,60,64 \
  EVAL_CHECKPOINT_STEPS=30000 EVAL_CHECKPOINT_HOLES_LIST=53,60,64 \
  OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64 \
  CASE_BANK_HOLES=53,60,64 CASE_BANK_N=4 CASE_BANK_EVAL_N=256 CASE_BANK_LOOP_VALUES=1,3,5 \
  FULL_STEPS=30000 SAVE_TRAIN_CHECKPOINT_EVERY=25 \
  RESUME_TRAIN_CHECKPOINT="${CHECKPOINT}" TRAIN_CHECKPOINT_DIR="${CHECKPOINT_DIR}" \
  RUN_NAME="${RUN_NAME}" \
  ./run.sh full
