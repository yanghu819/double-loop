#!/usr/bin/env bash
set -euo pipefail

BACKBONE_NAME="${1:?Usage: $0 <fla_gdn|kda|gdn2> [run-name]}"
case "$BACKBONE_NAME" in
  fla_gdn|kda|gdn2) ;;
  *)
    printf 'Expected fla_gdn, kda, or gdn2; got %s\n' "$BACKBONE_NAME" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="${2:-fla-official-${BACKBONE_NAME}-futureseed-${TIMESTAMP}-${GIT_SHA:0:7}}"
PERSIST_ROOT="${PERSIST_ROOT:-$REPO_ROOT}"
FULL_STEPS="${FULL_STEPS:-1500}"
EASY_STEPS="${EASY_STEPS:-100}"
if (( FULL_STEPS <= EASY_STEPS )); then
  printf 'FULL_STEPS must be greater than EASY_STEPS.\n' >&2
  exit 2
fi
HARD_STEPS="$((FULL_STEPS - EASY_STEPS))"

if [[ "${CUDA_VISIBLE_DEVICES:-0}" != "0" ]]; then
  printf 'This experiment is GPU1-only and requires CUDA_VISIBLE_DEVICES=0.\n' >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES=0
export FLA_DISABLE_BACKEND_DISPATCH=1
export FLA_CONV_BACKEND=triton
export FLA_STRICT_OFFICIAL=1
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PERSIST_ROOT/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PERSIST_ROOT/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PERSIST_ROOT/.cache/torchinductor}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PERSIST_ROOT/.cache/torch_extensions}"
export TMPDIR="${TMPDIR:-$PERSIST_ROOT/.cache/tmp}"
export PYTHONPATH="${PYTHONPATH:-$PERSIST_ROOT/.cache/python-extra-pylib}"
export SMOKE_DONE=1
export SKIP_SETUP=1
export SOURCE_SNAPSHOT_MODE=lean
export PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"

export OFFICIAL_SUDOKU_DATA_DIR="${OFFICIAL_SUDOKU_DATA_DIR:-$PERSIST_ROOT/data/sudoku-extreme-full}"
export OFFICIAL_SUDOKU_TRAIN_SPLIT=train
export OFFICIAL_SUDOKU_EVAL_SPLIT=test
export SUDOKU_SIZE=9
export HOLE_PATTERN=random
export HOLE_STAGES="${HOLE_STAGES:-46-50:${EASY_STEPS},51-55:${HARD_STEPS}}"
export HOLES_MIN=46
export HOLES_MAX=55
export EVAL_HOLES=53
export EVAL_HOLES_LIST=53
export OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64

export BACKBONE="$BACKBONE_NAME"
export GDN_MODE=chunk
export GDN_USE_SHORT_CONV=1
export GDN_EXPAND_V=2.0
export GDN_CONV_SIZE=4
export D_MODEL=192
export LAYERS=10
export HEADS=6
export HEAD_DIM=32
export CHANNEL_MULT=4
export L_CYCLES=2
export MAX_LOOPS=5
export LOOP_LOSS=all
export FULL_STEPS
export FULL_BATCH=32
export GRAD_ACCUM_STEPS=4
export FULL_EVAL_N=512
export FULL_ROLLOUT_KS=
export FULL_LOG_EVERY=100
export FORWARD_DTYPE=bfloat16
export LR=0.0015
export WEIGHT_DECAY=0.001
export BLANK_LOSS_WEIGHT=8
export FUTURE_SEED_SCALE=1
export FUTURE_SEED_UPDATE=fixed
export FUTURE_SEED_NORM_MODE=unit
export NOISE_SCALE=0
export EVAL_CHECKPOINT_STEPS="${EVAL_CHECKPOINT_STEPS:-500,1000,1500}"
export EVAL_CHECKPOINT_HOLES_LIST=53
export CASE_BANK_N=4
export CASE_BANK_EVAL_N=256
export CASE_BANK_LOOP_VALUES=1,2,3,4,5
export TRAIN_CHECKPOINT_DIR="$PERSIST_ROOT/models/$RUN_NAME/checkpoints"
export SAVE_TRAIN_CHECKPOINT_EVERY="${SAVE_TRAIN_CHECKPOINT_EVERY:-500}"
export RUN_NAME

mkdir -p \
  "$XDG_CACHE_HOME" \
  "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TORCH_EXTENSIONS_DIR" \
  "$TMPDIR" \
  "$TRAIN_CHECKPOINT_DIR"

cd "$REPO_ROOT"
exec ./run.sh full
