#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export BASELINE_CONFIG="$REPO_ROOT/configs/sudoku/gdn3_position_qk_scale.env"
export PERSIST_ROOT="${PERSIST_ROOT:-/huyang2/double-loop}"

if [[ "${CUDA_VISIBLE_DEVICES:-0}" != "0" ]]; then
  printf 'P-GDN3-004 requires CUDA_VISIBLE_DEVICES=0.\n' >&2
  exit 3
fi
VISIBLE_GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d '[:space:]')"
if [[ "$VISIBLE_GPU_COUNT" != "1" ]]; then
  printf 'P-GDN3-004 requires exactly one visible GPU.\n' >&2
  exit 4
fi
EXPECTED_GPU_UUID="${GPU1_UUID:-GPU-53e9f3b4-2966-65d3-6614-09c540921519}"
VISIBLE_GPU_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader,nounits | tr -d '[:space:]')"
if [[ "$VISIBLE_GPU_UUID" != "$EXPECTED_GPU_UUID" ]]; then
  printf 'GPU1 UUID mismatch: visible=%s expected=%s\n' \
    "$VISIBLE_GPU_UUID" "$EXPECTED_GPU_UUID" >&2
  exit 5
fi
if [[ ! -s "$PERSIST_ROOT/.cache/python-extra-path" ]]; then
  printf 'Persistent Python path marker is missing: %s\n' \
    "$PERSIST_ROOT/.cache/python-extra-path" >&2
  exit 6
fi
export PYTHON_EXTRA_PATH="$(<"$PERSIST_ROOT/.cache/python-extra-path")"

GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ "${GDN2_SCALE_PROBE:-0}" == "1" ]]; then
  export RUN_NAME="gdn3-position-qk-d256l12-fit-${TIMESTAMP}-${GIT_SHA:0:7}"
else
  export RUN_NAME="gdn3-position-qk-d256l12-s12000-${TIMESTAMP}-${GIT_SHA:0:7}"
fi

exec "$REPO_ROOT/scripts/run_canonical_gdn2_scale.sh"
