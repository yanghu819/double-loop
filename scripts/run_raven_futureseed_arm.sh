#!/usr/bin/env bash
set -euo pipefail

ARM="${1:?Usage: $0 <gdn2|raven> [run-name]}"
case "$ARM" in
  gdn2|raven) ;;
  *) printf 'Expected gdn2 or raven, got %s\n' "$ARM" >&2; exit 2 ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-/huyang2/double-loop}"
CONFIG_PATH="${RAVEN_COMPARE_CONFIG:-$REPO_ROOT/configs/sudoku/raven_futureseed_compare.env}"
FLA_SHA="31d15f7554bd5df05d3da6f75e09146279d2b1a8"
FLA_SOURCE_ROOT="${RAVEN_FLA_SOURCE_ROOT:-$PERSIST_ROOT/.cache/fla-upstream-31d15f7}"

set -a
# shellcheck disable=SC1090
source "$CONFIG_PATH"
set +a

[[ "${CUDA_VISIBLE_DEVICES:-0}" == "0" ]] || {
  printf 'Raven comparison is GPU1-only and requires CUDA_VISIBLE_DEVICES=0.\n' >&2
  exit 3
}
export CUDA_VISIBLE_DEVICES=0
export FLA_DISABLE_BACKEND_DISPATCH=1
export FLA_CONV_BACKEND=triton
export FLA_EXPECTED_SOURCE_SHA="$FLA_SHA"
export FLA_SOURCE_ROOT
export PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
export PYTHON_EXTRA_PATH="$FLA_SOURCE_ROOT:$PERSIST_ROOT/.cache/python-extra-pylib"
export PYTHONPATH="$PYTHON_EXTRA_PATH${PYTHONPATH:+:$PYTHONPATH}"
export PERSIST_ROOT
export RUNS_ROOT="${RUNS_ROOT:-$PERSIST_ROOT/runs}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PERSIST_ROOT/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PERSIST_ROOT/.cache/triton-raven-$FLA_SHA}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PERSIST_ROOT/.cache/torchinductor}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PERSIST_ROOT/.cache/torch_extensions}"
export TMPDIR="${TMPDIR:-$PERSIST_ROOT/.cache/tmp}"
export SOURCE_SNAPSHOT_MODE=lean
export REQUIRE_CLEAN_SOURCE=1
export SMOKE_DONE=1
export SKIP_SETUP=1
export FLA_STRICT_OFFICIAL=1
export FULL_STEPS="$BENCHMARK_STEPS"
export HOLE_STAGES="46-50:${BENCHMARK_EASY_STEPS},51-55:$((BENCHMARK_STEPS - BENCHMARK_EASY_STEPS))"
export HOLES_MIN=46
export HOLES_MAX=55
export EVAL_CHECKPOINT_STEPS="$BENCHMARK_STEPS"
export EVAL_CHECKPOINT_HOLES_LIST=53
export SAVE_TRAIN_CHECKPOINT_EVERY="$BENCHMARK_STEPS"
export GDN2_GAIN_BUDGET_MODE=none
export GDN2_FAST_SLOW_DECAY_MODE=none
export GDN2_ADDRESS_MODE=none
export ACTIVATION_CHECKPOINT=0

if [[ "$ARM" == raven ]]; then
  export BACKBONE=raven
  export GDN_USE_SHORT_CONV=0
else
  export BACKBONE=gdn2
  # Keep the established strong GDN2 baseline, including its official short convolution.
  export GDN_USE_SHORT_CONV=1
  export RAVEN_NUM_SLOTS=0
  export RAVEN_TOPK=0
fi

[[ -d "$FLA_SOURCE_ROOT/.git" ]] || {
  printf 'Pinned Raven FLA checkout is missing: %s\n' "$FLA_SOURCE_ROOT" >&2
  exit 4
}
ACTUAL_FLA_SHA="$(git -C "$FLA_SOURCE_ROOT" rev-parse HEAD)"
[[ "$ACTUAL_FLA_SHA" == "$FLA_SHA" ]] || {
  printf 'Pinned Raven FLA SHA mismatch: %s != %s\n' "$ACTUAL_FLA_SHA" "$FLA_SHA" >&2
  exit 4
}
[[ -x "$PYTHON_BIN" ]] || { printf 'CUDA Python is missing: %s\n' "$PYTHON_BIN" >&2; exit 4; }

for split in train test; do
  for name in all__inputs.npy all__labels.npy; do
    [[ -f "$OFFICIAL_SUDOKU_DATA_DIR/$split/$name" ]] || {
      printf 'Missing official Sudoku array: %s\n' "$OFFICIAL_SUDOKU_DATA_DIR/$split/$name" >&2
      exit 4
    }
  done
done

GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="${2:-raven-fs-compare-${ARM}-s${BENCHMARK_STEPS}-${TIMESTAMP}-${GIT_SHA:0:7}}"
export RUN_NAME
export TRAIN_CHECKPOINT_DIR="$PERSIST_ROOT/models/$RUN_NAME/checkpoints"
export EXACT_TRAIN_PID_FILE="$PERSIST_ROOT/artifacts/launch/$RUN_NAME.train.pid"
RUN_DIR="$RUNS_ROOT/$RUN_NAME"
mkdir -p "$RUN_DIR" "$TRAIN_CHECKPOINT_DIR" "$(dirname "$EXACT_TRAIN_PID_FILE")" \
  "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$TMPDIR"

GPU_ROW="$(nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader,nounits -i 0)"
{
  printf 'timestamp_utc=%s\n' "$TIMESTAMP"
  printf 'arm=%s\nbackbone=%s\n' "$ARM" "$BACKBONE"
  printf 'git_sha=%s\nfla_sha=%s\nfla_source_root=%s\n' "$GIT_SHA" "$FLA_SHA" "$FLA_SOURCE_ROOT"
  printf 'gpu=%s\n' "$GPU_ROW"
  printf 'state_elements_per_head=1024\n'
  printf 'raven_slots=%s\nraven_topk=%s\n' "${RAVEN_NUM_SLOTS:-0}" "${RAVEN_TOPK:-0}"
  printf 'hypothesis=Sparse content-routed slots preserve FutureSeed context with less interference than dense GDN2 memory.\n'
  printf 'success=Raven improves hard-range exact or blank accuracy without extra state elements, and loop5 does not regress loop1.\n'
  printf 'kill=NaN, official-kernel/source mismatch, or no CE descent by step300.\n'
} > "$RUN_DIR/launch.env"

cd "$REPO_ROOT"
exec ./run.sh full
