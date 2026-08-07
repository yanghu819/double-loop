#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-/huyang2/double-loop}"
EXPECTED_UUID="GPU-53e9f3b4-2966-65d3-6614-09c540921519"

if [[ "${CUDA_VISIBLE_DEVICES:-0}" != "0" ]]; then
  printf 'P-GDN3-012 requires CUDA_VISIBLE_DEVICES=0.\n' >&2
  exit 3
fi
export CUDA_VISIBLE_DEVICES=0

VISIBLE_GPU="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)"
if [[ "$VISIBLE_GPU" != "0, $EXPECTED_UUID" ]]; then
  printf 'Unexpected visible GPU contract: %s\n' "$VISIBLE_GPU" >&2
  exit 4
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; then
  printf 'P-GDN3-012 refuses to overlap an existing GPU compute process.\n' >&2
  exit 5
fi
if git -C "$REPO_ROOT" symbolic-ref -q HEAD >/dev/null; then
  printf 'P-GDN3-012 requires a detached source worktree.\n' >&2
  exit 6
fi
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
  printf 'P-GDN3-012 requires a clean source worktree.\n' >&2
  exit 7
fi

export PERSIST_ROOT
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
export UPDATE_LEADERBOARD=0

set -a
# shellcheck disable=SC1091
source "$REPO_ROOT/configs/sudoku/gdn3_adaptive_signed_erase_probe.env"
set +a

ACTUAL_PARENT_SHA="$(sha256sum "$RESUME_TRAIN_CHECKPOINT" | awk '{print $1}')"
if [[ "$ACTUAL_PARENT_SHA" != "$RESUME_TRAIN_CHECKPOINT_SHA256" ]]; then
  printf 'Parent checkpoint SHA mismatch: %s != %s\n' \
    "$ACTUAL_PARENT_SHA" "$RESUME_TRAIN_CHECKPOINT_SHA256" >&2
  exit 8
fi

GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ "${GDN3_FULL_STACK_PROBE:-0}" == "1" ]]; then
  export HOLE_STAGES=46-50:500,51-55:2501
  export FULL_STEPS=3001
  export FULL_EVAL_N=8
  export EVAL_HOLES_LIST=53
  export OFFICIAL_EVAL_BLANK_RANGES=51-55
  export EVAL_CHECKPOINT_STEPS=3001
  export EVAL_CHECKPOINT_HOLES_LIST=53
  export SAVE_TRAIN_CHECKPOINT_EVERY=
  export CASE_BANK_N=0
  export FULL_LOG_EVERY=1
  DEFAULT_NAME="p-gdn3-012-adaptive-signed-erase-probe-s3001-${TIMESTAMP}-${GIT_SHA:0:7}"
else
  DEFAULT_NAME="p-gdn3-012-adaptive-signed-erase-s3100-${TIMESTAMP}-${GIT_SHA:0:7}"
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
