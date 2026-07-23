#!/usr/bin/env bash
set -euo pipefail

ARM="${1:?Usage: $0 <control|unseen>}"
case "$ARM" in
  control|unseen) ;;
  *)
    printf 'Expected control or unseen; got %s\n' "$ARM" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -a
# shellcheck disable=SC1091
source "$REPO_ROOT/configs/sudoku/gdn_scale.env"
set +a

if [[ "${CUDA_VISIBLE_DEVICES:-0}" != "0" ]]; then
  printf 'This experiment is GPU1-only and requires CUDA_VISIBLE_DEVICES=0.\n' >&2
  exit 3
fi
export CUDA_VISIBLE_DEVICES=0

PERSIST_ROOT="${PERSIST_ROOT:-/huyang2/double-loop}"
PARENT_CHECKPOINT="${PARENT_CHECKPOINT:-$PERSIST_ROOT/models/gdn-full-diversity-d224l12-s6000-20260711T1510Z-eeb38f5/checkpoints/train_state_step030000.pt}"
UNSEEN_INDEX_PATH="${UNSEEN_INDEX_PATH:-$PERSIST_ROOT/data/sudoku-extreme-full/indices/unseen-after-step030000-seed52-b128-51-64.npy}"
TARGET_STEPS="${TARGET_STEPS:-31500}"
if (( TARGET_STEPS <= 30000 )); then
  printf 'TARGET_STEPS must exceed the parent step 30000.\n' >&2
  exit 2
fi
if [[ ! -f "$PARENT_CHECKPOINT" ]]; then
  printf 'Missing parent checkpoint: %s\n' "$PARENT_CHECKPOINT" >&2
  exit 4
fi
if [[ "$ARM" == "unseen" && ! -f "$UNSEEN_INDEX_PATH" ]]; then
  printf 'Missing unseen-row index: %s\n' "$UNSEEN_INDEX_PATH" >&2
  exit 4
fi

export PERSIST_ROOT
export BACKBONE=gdn
export GDN_MODE=triton_recurrent
export RESUME_TRAIN_CHECKPOINT="$PARENT_CHECKPOINT"
export OFFICIAL_SUDOKU_TRAIN_INDICES=
if [[ "$ARM" == "unseen" ]]; then
  export OFFICIAL_SUDOKU_TRAIN_INDICES="$UNSEEN_INDEX_PATH"
fi
export FULL_STEPS="$TARGET_STEPS"
export HOLE_STAGES="46-50:100,51-55:5900,51-64:$((TARGET_STEPS - 6000))"
export EVAL_CHECKPOINT_STEPS="${COVERAGE_EVAL_STEPS:-30500,$TARGET_STEPS}"
export EVAL_CHECKPOINT_HOLES_LIST=53,60,64
export SAVE_TRAIN_CHECKPOINT_EVERY="${SAVE_TRAIN_CHECKPOINT_EVERY:-500}"
export FULL_ROLLOUT_KS=
export SOURCE_SNAPSHOT_MODE=lean
export SMOKE_DONE=1
export SKIP_SETUP=1
export PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PERSIST_ROOT/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PERSIST_ROOT/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PERSIST_ROOT/.cache/torchinductor}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PERSIST_ROOT/.cache/torch_extensions}"
export TMPDIR="${TMPDIR:-$PERSIST_ROOT/.cache/tmp}"
export PATH="$PERSIST_ROOT/.cache/bin:$PATH"
export PYTHONPATH="$PERSIST_ROOT/.cache/python-extra-pylib${PYTHONPATH:+:$PYTHONPATH}"

GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
export RUN_NAME="${RUN_NAME:-gdn-data-coverage-${ARM}-s${TARGET_STEPS}-${TIMESTAMP}-${GIT_SHA:0:7}}"
export TRAIN_CHECKPOINT_DIR="${TRAIN_CHECKPOINT_DIR:-$PERSIST_ROOT/models/$RUN_NAME/checkpoints}"
mkdir -p "$TRAIN_CHECKPOINT_DIR"

printf 'arm=%s\nparent=%s\ntrain_indices=%s\ntarget_steps=%s\nrun_name=%s\n' \
  "$ARM" "$PARENT_CHECKPOINT" "${OFFICIAL_SUDOKU_TRAIN_INDICES:-full_with_replacement}" \
  "$TARGET_STEPS" "$RUN_NAME"

cd "$REPO_ROOT"
exec ./run.sh full
