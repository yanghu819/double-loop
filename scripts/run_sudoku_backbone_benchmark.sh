#!/usr/bin/env bash
set -euo pipefail

BACKBONE_NAME="${1:?Usage: $0 <rwkv|gdn|gdn2|kda> [run-name]}"
case "$BACKBONE_NAME" in
  rwkv|gdn|gdn2|kda) ;;
  *)
    printf 'Expected rwkv, gdn, gdn2, or kda; got %s\n' "$BACKBONE_NAME" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${BENCHMARK_CONFIG:-$REPO_ROOT/configs/sudoku/backbone_benchmark.env}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  printf 'Benchmark config does not exist: %s\n' "$CONFIG_PATH" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$CONFIG_PATH"
set +a

if [[ "${CUDA_VISIBLE_DEVICES:-0}" != "0" ]]; then
  printf 'This benchmark is GPU1-only and requires CUDA_VISIBLE_DEVICES=0.\n' >&2
  exit 3
fi
export CUDA_VISIBLE_DEVICES=0

PERSIST_ROOT="${PERSIST_ROOT:-$REPO_ROOT}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PERSIST_ROOT/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PERSIST_ROOT/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PERSIST_ROOT/.cache/torchinductor}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PERSIST_ROOT/.cache/torch_extensions}"
export TMPDIR="${TMPDIR:-$PERSIST_ROOT/.cache/tmp}"
PYTHON_EXTRA_PATH="${PYTHON_EXTRA_PATH:-$PERSIST_ROOT/.cache/python-extra-pylib}"
export PYTHONPATH="$PYTHON_EXTRA_PATH${PYTHONPATH:+:$PYTHONPATH}"
export PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
export SOURCE_SNAPSHOT_MODE=lean
export SMOKE_DONE=1
export SKIP_SETUP=1

FULL_STEPS="${BENCHMARK_STEPS:-500}"
EASY_STEPS="${BENCHMARK_EASY_STEPS:-100}"
if (( EASY_STEPS < 1 || FULL_STEPS <= EASY_STEPS )); then
  printf 'Require BENCHMARK_STEPS > BENCHMARK_EASY_STEPS >= 1.\n' >&2
  exit 2
fi
HARD_STEPS="$((FULL_STEPS - EASY_STEPS))"
export FULL_STEPS
export HOLE_STAGES="46-50:${EASY_STEPS},51-55:${HARD_STEPS}"
export HOLES_MIN=46
export HOLES_MAX=55
export EVAL_CHECKPOINT_STEPS="$FULL_STEPS"
export EVAL_CHECKPOINT_HOLES_LIST=53
export SAVE_TRAIN_CHECKPOINT_EVERY="$FULL_STEPS"

case "$BACKBONE_NAME" in
  rwkv)
    export BACKBONE=rwkv
    export RWKV_KERNEL=statepassing
    export FLA_STRICT_OFFICIAL=0
    ;;
  gdn)
    export BACKBONE=fla_gdn
    export RWKV_KERNEL=statepassing
    export FLA_STRICT_OFFICIAL=1
    ;;
  gdn2)
    export BACKBONE=gdn2
    export RWKV_KERNEL=statepassing
    export FLA_STRICT_OFFICIAL=1
    ;;
  kda)
    export BACKBONE=kda
    export RWKV_KERNEL=statepassing
    export FLA_STRICT_OFFICIAL=1
    ;;
esac

export FLA_DISABLE_BACKEND_DISPATCH=1
export FLA_CONV_BACKEND=triton

GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="${2:-sudoku-backbone-${BACKBONE_NAME}-s${FULL_STEPS}-${TIMESTAMP}-${GIT_SHA:0:7}}"
export RUN_NAME
export TRAIN_CHECKPOINT_DIR="$PERSIST_ROOT/models/$RUN_NAME/checkpoints"

printf 'benchmark_backbone=%s\ninternal_backbone=%s\nsteps=%s\nrun_name=%s\n' \
  "$BACKBONE_NAME" "$BACKBONE" "$FULL_STEPS" "$RUN_NAME"

if [[ "${BENCHMARK_DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

for split in train test; do
  for name in all__inputs.npy all__labels.npy; do
    path="$OFFICIAL_SUDOKU_DATA_DIR/$split/$name"
    if [[ ! -f "$path" ]]; then
      printf 'Missing official Sudoku array: %s\n' "$path" >&2
      exit 4
    fi
  done
done

if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'CUDA Python is not executable: %s\n' "$PYTHON_BIN" >&2
  exit 4
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf 'nvidia-smi is required; CPU smoke is forbidden.\n' >&2
  exit 4
fi

mkdir -p \
  "$XDG_CACHE_HOME" \
  "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TORCH_EXTENSIONS_DIR" \
  "$TMPDIR" \
  "$TRAIN_CHECKPOINT_DIR"

cd "$REPO_ROOT"
exec ./run.sh full
