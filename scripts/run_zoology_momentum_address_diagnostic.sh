#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG="${PMOMADDR001_CONFIG:-$REPO_ROOT/configs/retrieval/zoology_momentum_address_diagnostic.env}"
source "$CONFIG"

: "${EXPECTED_GPU_NAME:?set EXPECTED_GPU_NAME}"
: "${EXPECTED_GPU_UUID:?set EXPECTED_GPU_UUID}"
: "${EXPECTED_SOURCE_SHA:?set EXPECTED_SOURCE_SHA}"
: "${MOMENTUM_CHECKPOINT:?set MOMENTUM_CHECKPOINT}"
: "${MOMENTUM_SCORE:?set MOMENTUM_SCORE}"
: "${MOMENTUM_CASES:?set MOMENTUM_CASES}"

export CUDA_VISIBLE_DEVICES=0
export XDG_CACHE_HOME="$PERSIST_ROOT/.cache"
export TORCH_HOME="$PERSIST_ROOT/.cache/torch"
export TORCH_EXTENSIONS_DIR="$PERSIST_ROOT/.cache/torch_extensions"
export TRITON_CACHE_DIR="$PERSIST_ROOT/.cache/triton"
export PYTHONPATH="$REPO_ROOT:$ZOOLOGY_ROOT:$FLA_SOURCE_ROOT"
export PERSIST_ROOT PYTHON_BIN ZOOLOGY_ROOT ZOOLOGY_SHA FLA_SOURCE_ROOT
export FLA_EXPECTED_SOURCE_SHA FLA_DISABLE_BACKEND_DISPATCH FLA_CONV_BACKEND
export MDN_REPO_ROOT MDN_FLA_ROOT MDN_EXPECTED_SHA
export EXPECTED_GPU_NAME EXPECTED_GPU_UUID EXPECTED_SOURCE_SHA
export WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1

VISIBLE_GPU="$(nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader)"
[[ "$VISIBLE_GPU" == "0, $EXPECTED_GPU_UUID, $EXPECTED_GPU_NAME" ]]
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]
[[ "$(git -C "$ZOOLOGY_ROOT" rev-parse HEAD)" == "$ZOOLOGY_SHA" ]]
[[ -z "$(git -C "$ZOOLOGY_ROOT" status --porcelain)" ]]
[[ "$(git -C "$MDN_REPO_ROOT" rev-parse HEAD)" == "$MDN_EXPECTED_SHA" ]]
[[ -z "$(git -C "$MDN_REPO_ROOT" status --porcelain --untracked-files=no)" ]]
if git -C "$REPO_ROOT" symbolic-ref -q HEAD >/dev/null; then
  printf 'P-DIAG-MOMADDR-001 requires a detached source worktree.\n' >&2
  exit 8
fi
GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ "$GIT_SHA" == "$EXPECTED_SOURCE_SHA" ]]
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]]
REMOTE_SHA="$(
  curl -fsSL --connect-timeout 10 --max-time 30 \
    "https://api.github.com/repos/yanghu819/double-loop/git/ref/${SOURCE_REMOTE_REF#refs/}" \
    | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)["object"]["sha"])'
)"
[[ "$REMOTE_SHA" == "$GIT_SHA" ]]
[[ -f "$MOMENTUM_CHECKPOINT" && -f "$MOMENTUM_SCORE" && -f "$MOMENTUM_CASES" ]]

LOCK_DIR="$PERSIST_ROOT/artifacts/locks"
mkdir -p "$LOCK_DIR"
exec 9>"$LOCK_DIR/p-diag-momaddr-001.lock"
flock -n 9

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="${RUN_NAME:-p-diag-momaddr-001-${TIMESTAMP}-${GIT_SHA:0:7}}"
RUN_DIR="$PERSIST_ROOT/runs/$RUN_NAME"
mkdir -p "$RUN_DIR"
STATUS=0
GPU_SAMPLER_PID=""

sample_gpu() {
  while true; do
    printf '%s,' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    nvidia-smi \
      --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw \
      --format=csv,noheader,nounits
    sleep 2
  done
}

finalize() {
  local shell_status="$?"
  trap - EXIT
  set +e
  [[ "$STATUS" != 0 ]] || STATUS="$shell_status"
  if [[ -n "$GPU_SAMPLER_PID" ]]; then
    kill "$GPU_SAMPLER_PID" 2>/dev/null || true
    wait "$GPU_SAMPLER_PID" 2>/dev/null || true
  fi
  if [[ "$STATUS" != 0 && ! -f "$RUN_DIR/abort.json" ]]; then
    "$PYTHON_BIN" - "$RUN_DIR/abort.json" "$STATUS" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "exit_status": int(sys.argv[2]),
    "phase": "momentum_address_diagnostic",
    "reason": "P-DIAG-MOMADDR-001 integrity or infrastructure failure",
    "scientific_failure": False,
    "rescue_authorized": False,
}, indent=2, sort_keys=True) + "\n")
PY
  fi
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/completed_at.txt"
  printf '%s\n' "$STATUS" > "$RUN_DIR/exit_status.txt"
  nvidia-smi --query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader > "$RUN_DIR/gpu_after.txt"
  find "$RUN_DIR" -type f ! -name artifacts.sha256 -print0 | sort -z \
    | xargs -0 sha256sum > "$RUN_DIR/artifacts.sha256"
  printf 'completed run_dir=%s status=%s\n' "$RUN_DIR" "$STATUS"
  exit "$STATUS"
}
trap finalize EXIT
trap 'STATUS=130; exit 130' INT
trap 'STATUS=143; exit 143' TERM

LAUNCHER_PID="$$"
LAUNCHER_PGID="$(ps -o pgid= -p "$$" | tr -d ' ')"
[[ "$LAUNCHER_PID" == "$LAUNCHER_PGID" ]]
printf '%s\n' "$LAUNCHER_PID" > "$RUN_DIR/pid.txt"
printf '%s\n' "$LAUNCHER_PGID" > "$RUN_DIR/pgid.txt"
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_DIR/git_sha.txt"
git -C "$REPO_ROOT" status --short > "$RUN_DIR/git_status.txt"
printf '%s\n' "$REMOTE_SHA" > "$RUN_DIR/github_readback_sha.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/started_at.txt"
git -C "$REPO_ROOT" archive --format=tar.gz \
  --output="$RUN_DIR/source_snapshot.tar.gz" HEAD
sample_gpu >> "$RUN_DIR/gpu_samples.csv" &
GPU_SAMPLER_PID="$!"

set +e
timeout --signal=TERM --kill-after=30 1200 \
  "$PYTHON_BIN" -u -m experiments.zoology_mqar.momentum_address_diagnostic \
  --checkpoint "$MOMENTUM_CHECKPOINT" \
  --formal-score "$MOMENTUM_SCORE" \
  --formal-cases "$MOMENTUM_CASES" \
  --output "$RUN_DIR/diagnostic.json" \
  2>&1 | tee "$RUN_DIR/diagnostic.log"
PIPE=("${PIPESTATUS[@]}")
STATUS="${PIPE[0]}"
[[ "$STATUS" != 0 || "${PIPE[1]}" == 0 ]] || STATUS=74
set -e
printf 'python=%s\ntee=%s\ncombined=%s\n' \
  "${PIPE[0]}" "${PIPE[1]}" "$STATUS" > "$RUN_DIR/diagnostic_status.txt"
[[ "$STATUS" != 0 || -s "$RUN_DIR/diagnostic.json" ]] || STATUS=66
exit "$STATUS"
