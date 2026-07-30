#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-formal}"
if [[ "$MODE" != "smoke" && "$MODE" != "formal" ]]; then
  printf 'usage: %s smoke|formal\n' "$0" >&2
  exit 2
fi
if [[ "${CUDA_VISIBLE_DEVICES:-0}" != "0" ]]; then
  printf 'P-OCC-001 is GPU1-only and requires CUDA_VISIBLE_DEVICES=0.\n' >&2
  exit 3
fi
export CUDA_VISIBLE_DEVICES=0

if [[ -z "${GPU1_UUID:-}" ]]; then
  printf 'GPU1_UUID must be supplied from the current AIStation GPU1 probe.\n' >&2
  exit 4
fi
VISIBLE_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | sed -n '1p')"
VISIBLE_COUNT="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l | tr -d ' ')"
if [[ "$VISIBLE_COUNT" != "1" || "$VISIBLE_UUID" != "$GPU1_UUID" ]]; then
  printf 'GPU contract mismatch: count=%s uuid=%s expected=%s\n' \
    "$VISIBLE_COUNT" "$VISIBLE_UUID" "$GPU1_UUID" >&2
  exit 5
fi

SOURCE_STATUS="$(git -C "$REPO_ROOT" status --short -- . \
  ':(exclude).cache' \
  ':(exclude).venv' \
  ':(exclude)artifacts' \
  ':(exclude)models' \
  ':(exclude)runs')"
if [[ -n "$SOURCE_STATUS" ]]; then
  printf 'Refusing probe with a dirty source tree:\n%s\n' "$SOURCE_STATUS" >&2
  exit 6
fi

SOURCE_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
export SOURCE_SHA
export FLA_DISABLE_BACKEND_DISPATCH=1
export FLA_CONV_BACKEND=triton
export XDG_CACHE_HOME="$REPO_ROOT/.cache"
export TRITON_CACHE_DIR="$REPO_ROOT/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$REPO_ROOT/.cache/torchinductor"
export TORCH_EXTENSIONS_DIR="$REPO_ROOT/.cache/torch_extensions"
export TMPDIR="$REPO_ROOT/.cache/tmp"
export PATH="$REPO_ROOT/.cache/bin:$PATH"
if [[ -z "${PYTHON_EXTRA_PATH:-}" && -s "$REPO_ROOT/.cache/python-extra-path" ]]; then
  PYTHON_EXTRA_PATH="$(<"$REPO_ROOT/.cache/python-extra-path")"
fi
PYTHON_EXTRA_PATH="${PYTHON_EXTRA_PATH:-$REPO_ROOT/.cache/python-extra-pylib}"
export PYTHONPATH="$PYTHON_EXTRA_PATH${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'CUDA Python is not executable: %s\n' "$PYTHON_BIN" >&2
  exit 7
fi

mkdir -p \
  "$XDG_CACHE_HOME" \
  "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TORCH_EXTENSIONS_DIR" \
  "$TMPDIR"

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="${RUN_NAME:-kda-occurrence-${MODE}-${TIMESTAMP}-${SOURCE_SHA:0:7}}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/runs/$RUN_NAME}"
mkdir -p "$OUT_DIR"
printf '%s\n' "$SOURCE_SHA" > "$OUT_DIR/source_HEAD.txt"
printf '%s\n' "$VISIBLE_UUID" > "$OUT_DIR/gpu_uuid.txt"
{
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
  printf 'FLA_CONV_BACKEND=%s\n' "$FLA_CONV_BACKEND"
  printf 'FLA_DISABLE_BACKEND_DISPATCH=%s\n' "$FLA_DISABLE_BACKEND_DISPATCH"
  printf 'GPU1_UUID=%s\n' "$GPU1_UUID"
  printf 'PYTHON_BIN=%s\n' "$PYTHON_BIN"
  printf 'RUN_NAME=%s\n' "$RUN_NAME"
  printf 'SOURCE_SHA=%s\n' "$SOURCE_SHA"
  printf 'TORCHINDUCTOR_CACHE_DIR=%s\n' "$TORCHINDUCTOR_CACHE_DIR"
  printf 'TORCH_EXTENSIONS_DIR=%s\n' "$TORCH_EXTENSIONS_DIR"
  printf 'TRITON_CACHE_DIR=%s\n' "$TRITON_CACHE_DIR"
  printf 'XDG_CACHE_HOME=%s\n' "$XDG_CACHE_HOME"
} > "$OUT_DIR/launch.env"

ARGS=(
  --out-dir "$OUT_DIR"
  --steps 300
  --batch-size 128
  --eval-batch-size 128
  --eval-batches 4
  --train-length 128
  --train-repeats 8
  --eval-specs 128:8,512:16
  --d-model 128
  --heads 4
  --head-dim 32
  --expand-v 2
  --log-every 25
)
if [[ "$MODE" == "smoke" ]]; then
  ARGS=(
    --out-dir "$OUT_DIR"
    --steps 2
    --batch-size 8
    --eval-batch-size 8
    --eval-batches 1
    --train-length 64
    --train-repeats 4
    --eval-specs 64:4
    --d-model 64
    --heads 2
    --head-dim 32
    --expand-v 1
    --log-every 1
    --success-accuracy 0
  )
fi

"$PYTHON_BIN" \
  "$REPO_ROOT/experiments/repeated_key_retrieval/kda_occurrence_probe.py" \
  "${ARGS[@]}" \
  2>&1 | tee "$OUT_DIR/run.log"
