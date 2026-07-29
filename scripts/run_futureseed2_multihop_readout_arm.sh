#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-formal}"
if [[ "$MODE" != "contract" && "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
  printf 'usage: %s contract|smoke|formal\n' "$0" >&2
  exit 2
fi
if [[ "${CUDA_VISIBLE_DEVICES:-0}" != "0" ]]; then
  printf 'P-FS2-005 is GPU1-only and requires CUDA_VISIBLE_DEVICES=0.\n' >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES=0
export PERSIST_ROOT="${PERSIST_ROOT:-/huyang2/double-loop}"
export RUNS_ROOT="${RUNS_ROOT:-$PERSIST_ROOT/runs}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PERSIST_ROOT/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PERSIST_ROOT/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PERSIST_ROOT/.cache/torchinductor}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PERSIST_ROOT/.cache/torch_extensions}"
export TMPDIR="${TMPDIR:-$PERSIST_ROOT/.cache/tmp}"
export PATH="$PERSIST_ROOT/.cache/bin:$PATH"
export PYTHONPATH="$PERSIST_ROOT/.cache/python-extra-pylib${PYTHONPATH:+:$PYTHONPATH}"
export PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
export SOURCE_SNAPSHOT_MODE=lean
export REQUIRE_CLEAN_SOURCE=1
export SMOKE_DONE=1
export SKIP_SETUP=1

set -a
# shellcheck disable=SC1091
source "$REPO_ROOT/configs/sudoku/futureseed2_multihop_readout_probe.env"
set +a

GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ "$MODE" == "contract" ]]; then
  RUN_NAME="${RUN_NAME:-futureseed2-readout-contract-${TIMESTAMP}-${GIT_SHA:0:7}}"
elif [[ "$MODE" == "smoke" ]]; then
  RUN_NAME="${RUN_NAME:-futureseed2-readout-smoke2-${TIMESTAMP}-${GIT_SHA:0:7}}"
else
  RUN_NAME="${RUN_NAME:-futureseed2-readout-s9100-${TIMESTAMP}-${GIT_SHA:0:7}}"
fi
export RUN_NAME
export TRAIN_CHECKPOINT_DIR="${TRAIN_CHECKPOINT_DIR:-$PERSIST_ROOT/models/$RUN_NAME/checkpoints}"
RUN_DIR="$RUNS_ROOT/$RUN_NAME"

mkdir -p \
  "$XDG_CACHE_HOME" \
  "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TORCH_EXTENSIONS_DIR" \
  "$TMPDIR" \
  "$RUN_DIR" \
  "$TRAIN_CHECKPOINT_DIR"

"$PYTHON_BIN" "$REPO_ROOT/experiments/rwkv_fs_sudoku/check_futureseed2_multihop_readout.py" \
  --repo "$REPO_ROOT" \
  --out "$RUN_DIR/multihop_readout_contract.json"

if [[ "$MODE" == "contract" ]]; then
  printf 'contract completed run_dir=%s\n' "$RUN_DIR"
  exit 0
fi

if [[ "$MODE" == "smoke" ]]; then
  export HOLE_STAGES=46-50:500,51-55:3500,51-60:4000,51-64:1002
  export FULL_STEPS=9002
  export FULL_EVAL_N=8
  export EVAL_HOLES_LIST=53
  export OFFICIAL_EVAL_BLANK_RANGES=51-55
  export EVAL_CHECKPOINT_STEPS=9002
  export EVAL_CHECKPOINT_HOLES_LIST=53
  export SAVE_TRAIN_CHECKPOINT_EVERY=
  export CASE_BANK_N=0
  export FULL_LOG_EVERY=1
fi

cd "$REPO_ROOT"
exec ./run.sh full
