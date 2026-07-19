#!/usr/bin/env bash
set -euo pipefail

BACKBONE_NAME="${1:?Usage: $0 <gdn2|kda> [run-name]}"
case "$BACKBONE_NAME" in
  gdn2|kda) ;;
  *)
    printf 'Expected gdn2 or kda, got %s\n' "$BACKBONE_NAME" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="${2:-fla-${BACKBONE_NAME}-futureseed-sudoku-gate-${TIMESTAMP}-${GIT_SHA:0:7}}"
PERSIST_ROOT="${PERSIST_ROOT:-$REPO_ROOT}"

if [[ "${CUDA_VISIBLE_DEVICES:-0}" != "0" ]]; then
  printf 'This experiment is GPU1-only and requires CUDA_VISIBLE_DEVICES=0.\n' >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES=0
export SMOKE_DONE=1
export SKIP_SETUP=1
export SOURCE_SNAPSHOT_MODE=lean
export PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
export OFFICIAL_SUDOKU_DATA_DIR="${OFFICIAL_SUDOKU_DATA_DIR:-$PERSIST_ROOT/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000}"
export OFFICIAL_SUDOKU_TRAIN_SPLIT=train
export OFFICIAL_SUDOKU_EVAL_SPLIT=test
export SUDOKU_SIZE=9
export HOLE_PATTERN=random
# P-GDN-005 predates blank-range filtering and sampled uniformly from the full
# official train split. Its rows span 46-64 blanks, so this range preserves the
# same sampling semantics after the loader was made range-aware.
export HOLES_MIN=46
export HOLES_MAX=64
export EVAL_HOLES=16
export EVAL_HOLES_LIST=16
export OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64
export BACKBONE="$BACKBONE_NAME"
export GDN_MODE=chunk
export GDN_USE_SHORT_CONV=0
export GDN_EXPAND_V=1.0
export D_MODEL=128
export LAYERS=6
export HEADS=8
export HEAD_DIM=16
export CHANNEL_MULT=4
export L_CYCLES=2
export MAX_LOOPS=4
export LOOP_LOSS=all
export FULL_STEPS=600
export FULL_BATCH=128
export FULL_EVAL_N=1024
export FULL_ROLLOUT_KS=
export FULL_LOG_EVERY=100
export FORWARD_DTYPE=bfloat16
export LR=0.0015
export WEIGHT_DECAY=0.001
export BLANK_LOSS_WEIGHT=8
export FUTURE_SEED_SCALE=1
export EVAL_CHECKPOINT_STEPS=300,600
export EVAL_CHECKPOINT_HOLES_LIST=16
export CASE_BANK_N=4
export CASE_BANK_EVAL_N=256
export CASE_BANK_LOOP_VALUES=1,2,3,4
export TRAIN_CHECKPOINT_DIR="$PERSIST_ROOT/models/$RUN_NAME/checkpoints"
export SAVE_TRAIN_CHECKPOINT_EVERY=300
export RUN_NAME

cd "$REPO_ROOT"
exec ./run.sh full
