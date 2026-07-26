#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${BASELINE_CONFIG:-$REPO_ROOT/configs/sudoku/gdn2_scale.env}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  printf 'GDN2 scale config does not exist: %s\n' "$CONFIG_PATH" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$CONFIG_PATH"
set +a

if [[ "${CUDA_VISIBLE_DEVICES:-0}" != "0" ]]; then
  printf 'This GDN2 scale run is GPU1-only and requires CUDA_VISIBLE_DEVICES=0.\n' >&2
  exit 3
fi
if [[ "$BACKBONE" != "gdn2" || "${FLA_STRICT_OFFICIAL:-0}" != "1" ]]; then
  printf 'The scale candidate requires strict official FLA GDN2.\n' >&2
  exit 4
fi
if (( FULL_BATCH * GRAD_ACCUM_STEPS != 128 )); then
  printf 'The scale candidate requires effective batch 128.\n' >&2
  exit 4
fi

export CUDA_VISIBLE_DEVICES=0
export PERSIST_ROOT="${PERSIST_ROOT:-$REPO_ROOT}"
export RUNS_ROOT="${RUNS_ROOT:-$PERSIST_ROOT/runs}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PERSIST_ROOT/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PERSIST_ROOT/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PERSIST_ROOT/.cache/torchinductor}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PERSIST_ROOT/.cache/torch_extensions}"
export TMPDIR="${TMPDIR:-$PERSIST_ROOT/.cache/tmp}"
export PATH="$PERSIST_ROOT/.cache/bin:$PATH"
PYTHON_EXTRA_PATH="${PYTHON_EXTRA_PATH:-$PERSIST_ROOT/.cache/python-extra-pylib}"
export PYTHONPATH="$PYTHON_EXTRA_PATH${PYTHONPATH:+:$PYTHONPATH}"
export PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
export SOURCE_SNAPSHOT_MODE=lean
export REQUIRE_CLEAN_SOURCE=1
export SMOKE_DONE=1
export SKIP_SETUP=1
ulimit -c 0

if [[ "${GDN2_SCALE_PROBE:-0}" == "1" ]]; then
  export FULL_STEPS=2
  export HOLE_STAGES=51-55:2
  export FULL_EVAL_N=8
  export OFFICIAL_EVAL_BLANK_RANGES=51-55
  export EVAL_HOLES_LIST=53
  export EVAL_CHECKPOINT_STEPS=2
  export EVAL_CHECKPOINT_HOLES_LIST=53
  export SAVE_TRAIN_CHECKPOINT_EVERY=
  export CASE_BANK_N=0
  export FULL_LOG_EVERY=1
fi

GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ "${GDN2_SCALE_PROBE:-0}" == "1" ]]; then
  DEFAULT_NAME="gdn2-futureseed-scale-probe-${TIMESTAMP}-${GIT_SHA:0:7}"
else
  DEFAULT_NAME="gdn2-futureseed-d256l12-s${FULL_STEPS}-${TIMESTAMP}-${GIT_SHA:0:7}"
fi
export RUN_NAME="${RUN_NAME:-$DEFAULT_NAME}"
export TRAIN_CHECKPOINT_DIR="${TRAIN_CHECKPOINT_DIR:-$PERSIST_ROOT/models/$RUN_NAME/checkpoints}"

mkdir -p \
  "$XDG_CACHE_HOME" \
  "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TORCH_EXTENSIONS_DIR" \
  "$TMPDIR" \
  "$RUNS_ROOT" \
  "$TRAIN_CHECKPOINT_DIR"

cd "$REPO_ROOT"
exec ./run.sh full
