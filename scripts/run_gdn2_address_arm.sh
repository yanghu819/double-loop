#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARM="${1:-}"
PHASE="${2:-formal}"
if [[ "$ARM" != "control" && "$ARM" != "position_qk" && \
      "$ARM" != "anchor_rotary" && "$ARM" != "anchor_phase" && \
      "$ARM" != "anchor_residual" && "$ARM" != "anchor_qk_residual" ]]; then
  printf 'usage: %s control|position_qk|anchor_rotary|anchor_phase|anchor_residual|anchor_qk_residual [smoke|formal]\n' "$0" >&2
  exit 2
fi
if [[ "$PHASE" != "smoke" && "$PHASE" != "formal" ]]; then
  printf 'phase must be smoke or formal\n' >&2
  exit 2
fi
if [[ "${CUDA_VISIBLE_DEVICES:-0}" != "0" ]]; then
  printf 'GDN2 address probes are GPU1-only and require CUDA_VISIBLE_DEVICES=0.\n' >&2
  exit 3
fi
if [[ -z "${GPU1_UUID:-}" ]]; then
  printf 'GPU1_UUID must be supplied from the current AIStation GPU1 probe.\n' >&2
  exit 4
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
export PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
export FLA_DISABLE_BACKEND_DISPATCH=1
export FLA_CONV_BACKEND=triton
export SOURCE_SNAPSHOT_MODE=lean
export REQUIRE_CLEAN_SOURCE=1
export SMOKE_DONE=1
export SKIP_SETUP=1
export UPDATE_LEADERBOARD=0
if [[ "$ARM" == "control" ]]; then
  export GDN2_ADDRESS_MODE=none
else
  export GDN2_ADDRESS_MODE="$ARM"
fi

if [[ ! -s "$PERSIST_ROOT/.cache/python-extra-path" ]]; then
  printf 'Persistent Python path marker is missing; run setup.sh first.\n' >&2
  exit 5
fi
PYTHON_EXTRA_PATH="$(<"$PERSIST_ROOT/.cache/python-extra-path")"
export PYTHONPATH="$PYTHON_EXTRA_PATH${PYTHONPATH:+:$PYTHONPATH}"

set -a
# shellcheck disable=SC1091
source "$REPO_ROOT/configs/sudoku/gdn2_address_payload_probe.env"
set +a

P_ADDR_TARGET_STEP="${P_ADDR_TARGET_STEP:-9100}"
if [[ "$PHASE" == "formal" && "$P_ADDR_TARGET_STEP" != "9100" ]]; then
  if [[ "$ARM" == "control" ]]; then
    if [[ "${P_ADDR_MATCHED_CONTROL_CONTINUATION:-0}" != "1" || \
          "$P_ADDR_TARGET_STEP" != "9300" ]]; then
      printf '%s\n' \
        'Control continuation is restricted to the explicit matched step9300 falsifier.' >&2
      exit 6
    fi
  elif [[ "$ARM" != "position_qk" && "$ARM" != "anchor_rotary" && \
          "$ARM" != "anchor_phase" && "$ARM" != "anchor_residual" && \
          "$ARM" != "anchor_qk_residual" ]]; then
    printf 'Unsupported continuation arm: %s.\n' "$ARM" >&2
    exit 6
  fi
  if [[ "$P_ADDR_TARGET_STEP" -le 9100 ]]; then
    printf 'P_ADDR_TARGET_STEP must be greater than 9100, got %s.\n' \
      "$P_ADDR_TARGET_STEP" >&2
    exit 6
  fi
  if [[ -z "${P_ADDR_RESUME_CHECKPOINT:-}" || \
        -z "${P_ADDR_RESUME_CHECKPOINT_SHA256:-}" || \
        -z "${P_ADDR_RESUME_SOURCE_SHA:-}" ]]; then
    printf 'Continuation requires checkpoint path, SHA256, and source SHA.\n' >&2
    exit 6
  fi
  LAST_STAGE_STEPS="$((P_ADDR_TARGET_STEP - 8000))"
  export HOLE_STAGES="46-50:500,51-55:3500,51-60:4000,51-64:${LAST_STAGE_STEPS}"
  export FULL_STEPS="$P_ADDR_TARGET_STEP"
  export EVAL_CHECKPOINT_STEPS="$P_ADDR_TARGET_STEP"
  export RESUME_TRAIN_CHECKPOINT="$P_ADDR_RESUME_CHECKPOINT"
  export RESUME_TRAIN_CHECKPOINT_SHA256="$P_ADDR_RESUME_CHECKPOINT_SHA256"
  export RESUME_TRAIN_SOURCE_SHA="$P_ADDR_RESUME_SOURCE_SHA"
fi

VISIBLE_UUID="$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader,nounits | tr -d '[:space:]')"
if [[ "$VISIBLE_UUID" != "$GPU1_UUID" ]]; then
  printf 'GPU UUID mismatch: visible=%s expected GPU1=%s\n' "$VISIBLE_UUID" "$GPU1_UUID" >&2
  exit 7
fi

EXPECTED_CHECKPOINT="$RESUME_TRAIN_CHECKPOINT"
EXPECTED_HASH="$RESUME_TRAIN_CHECKPOINT_SHA256"
ACTUAL_HASH="$(sha256sum "$EXPECTED_CHECKPOINT" | awk '{print $1}')"
if [[ "$ACTUAL_HASH" != "$EXPECTED_HASH" ]]; then
  printf 'checkpoint hash mismatch: %s != %s\n' "$ACTUAL_HASH" "$EXPECTED_HASH" >&2
  exit 8
fi

if [[ "$PHASE" == "smoke" ]]; then
  export HOLE_STAGES=46-50:500,51-55:3500,51-60:4000,51-64:1001
  export FULL_STEPS=9001
  export FULL_BATCH=4
  export GRAD_ACCUM_STEPS=1
  export FULL_EVAL_N=8
  export EVAL_HOLES_LIST=53
  export OFFICIAL_EVAL_BLANK_RANGES=51-55
  export EVAL_CHECKPOINT_STEPS=9001
  export EVAL_CHECKPOINT_HOLES_LIST=53
  export SAVE_TRAIN_CHECKPOINT_EVERY=
  export CASE_BANK_N=0
  export FULL_LOG_EVERY=1
fi

GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ "$PHASE" == "smoke" ]]; then
  DEFAULT_NAME="gdn2-address-${ARM}-smoke-${TIMESTAMP}-${GIT_SHA:0:7}"
else
  DEFAULT_NAME="gdn2-address-${ARM}-s${P_ADDR_TARGET_STEP}-${TIMESTAMP}-${GIT_SHA:0:7}"
fi
export RUN_NAME="${RUN_NAME:-$DEFAULT_NAME}"
export TRAIN_CHECKPOINT_DIR="${TRAIN_CHECKPOINT_DIR:-$PERSIST_ROOT/models/$RUN_NAME/checkpoints}"
export EXACT_TRAIN_PID_FILE="${EXACT_TRAIN_PID_FILE:-$PERSIST_ROOT/artifacts/launch/$RUN_NAME.pid}"

mkdir -p \
  "$XDG_CACHE_HOME" \
  "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TORCH_EXTENSIONS_DIR" \
  "$TMPDIR" \
  "$RUNS_ROOT" \
  "$TRAIN_CHECKPOINT_DIR" \
  "$(dirname "$EXACT_TRAIN_PID_FILE")"

cd "$REPO_ROOT"
exec ./run.sh full
