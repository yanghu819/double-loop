#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$REPO_ROOT/configs/retrieval/zoology_contractive_dplr.env"
set -a
# shellcheck disable=SC1090
source "$CONFIG"
set +a

: "${EXPECTED_GPU_NAME:?Set the exact task-mode GPU name}"
: "${EXPECTED_GPU_UUID:?Set the exact task-mode GPU UUID}"
: "${EXPECTED_SOURCE_SHA:?Set the exact pushed source SHA}"

export CUDA_VISIBLE_DEVICES=0
export XDG_CACHE_HOME="$PERSIST_ROOT/.cache"
export UV_CACHE_DIR="$PERSIST_ROOT/.cache/uv"
export TORCH_HOME="$PERSIST_ROOT/.cache/torch"
export TORCH_EXTENSIONS_DIR="$PERSIST_ROOT/.cache/torch_extensions"
export TRITON_CACHE_DIR="$PERSIST_ROOT/.cache/triton"
export PYTHONPATH="$REPO_ROOT:$ZOOLOGY_ROOT:$FLA_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/opt/conda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

VISIBLE_GPU="$(nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader)"
if [[ "$VISIBLE_GPU" != "0, $EXPECTED_GPU_UUID, $EXPECTED_GPU_NAME" ]]; then
  printf 'Unexpected visible GPU: %s\n' "$VISIBLE_GPU" >&2
  exit 3
fi
if git -C "$REPO_ROOT" symbolic-ref -q HEAD >/dev/null; then
  printf 'P-GDN3-030 requires a detached source worktree.\n' >&2
  exit 5
fi
GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
if [[ "$GIT_SHA" != "$EXPECTED_SOURCE_SHA" ]] || [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
  printf 'Source is not the exact clean pushed SHA.\n' >&2
  exit 6
fi
REMOTE_SHA="$(env -u LD_LIBRARY_PATH timeout 30 git -C "$REPO_ROOT" ls-remote --refs origin "$SOURCE_REMOTE_REF" | awk '{print $1}')"
if [[ "$REMOTE_SHA" != "$GIT_SHA" ]]; then
  printf 'GitHub readback differs from detached source: %s vs %s\n' "$REMOTE_SHA" "$GIT_SHA" >&2
  exit 7
fi

exec 9>"$PERSIST_ROOT/.gpu0-model.lock"
if ! flock -n 9; then
  printf 'Another model process holds the global GPU0 lock.\n' >&2
  exit 8
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; then
  printf 'P-GDN3-030 refuses to overlap an existing GPU compute process.\n' >&2
  exit 4
fi

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="${RUN_NAME:-p-gdn3-030-contractive-dplr-l1024-${TIMESTAMP}-${GIT_SHA:0:7}}"
RUN_DIR="$PERSIST_ROOT/runs/$RUN_NAME"
OUT_DIR="$RUN_DIR/output"
mkdir -p "$OUT_DIR"
printf '%s\n' "$$" > "$RUN_DIR/pid.txt"
printf '%s\n' "$(ps -o pgid= -p $$ | tr -d ' ')" > "$RUN_DIR/pgid.txt"
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_DIR/git_sha.txt"
git -C "$REPO_ROOT" status --porcelain=v1 > "$RUN_DIR/source_status.txt"
printf '%s\n' "$REMOTE_SHA" > "$RUN_DIR/github_readback_sha.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/started_at.txt"
cp "$CONFIG" "$RUN_DIR/launch.env"
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader > "$RUN_DIR/gpu.txt"
git -C "$REPO_ROOT" ls-files -z -- . \
  ':(exclude).cache/**' ':(exclude).venv/**' ':(exclude)artifacts/**' \
  ':(exclude)models/**' ':(exclude)repos/**' ':(exclude)runs/**' \
  | tar --null -czf "$RUN_DIR/source_snapshot.tar.gz" \
      -C "$REPO_ROOT" --files-from -

STATUS=0
PHASE=contract
finalize() {
  local shell_status="$?"
  trap - EXIT
  set +e
  [[ "$STATUS" != 0 ]] || STATUS="$shell_status"
  if [[ "$STATUS" != 0 && "$STATUS" != 3 && ! -f "$RUN_DIR/abort.json" ]]; then
    "$PYTHON_BIN" - "$RUN_DIR/abort.json" "$STATUS" "$PHASE" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "reason": "P-GDN3-030 contract, training, integrity, or wall-budget failure",
    "exit_status": int(sys.argv[2]),
    "phase": sys.argv[3],
    "scientific_failure": False,
    "rescue_authorized": False,
}, indent=2, sort_keys=True) + "\n")
PY
  fi
  printf '%s\n' "$STATUS" > "$RUN_DIR/exit_status.txt"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/completed_at.txt"
  nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv,noheader \
    > "$RUN_DIR/gpu_final.txt"
  find "$RUN_DIR" -type f ! -name artifacts.sha256 -print0 \
    | sort -z | xargs -0 sha256sum > "$RUN_DIR/artifacts.sha256"
  exit "$STATUS"
}
trap finalize EXIT
trap 'STATUS=130; exit 130' INT
trap 'STATUS=143; exit 143' TERM

set +e
"$PYTHON_BIN" "$REPO_ROOT/scripts/check_zoology_contractive_dplr.py" \
  --output "$RUN_DIR/contract.json" \
  --expected-gpu-name "$EXPECTED_GPU_NAME" \
  --expected-gpu-uuid "$EXPECTED_GPU_UUID" \
  2>&1 | tee "$RUN_DIR/contract.log"
PIPE=("${PIPESTATUS[@]}")
STATUS="${PIPE[0]}"
[[ "$STATUS" != 0 || "${PIPE[1]}" == 0 ]] || STATUS=74
set -e
printf 'python=%s\ntee=%s\ncombined=%s\n' "${PIPE[0]}" "${PIPE[1]}" "$STATUS" \
  > "$RUN_DIR/contract_status.txt"

if [[ "$STATUS" == 0 ]]; then
  PHASE=formal
  set +e
  timeout --signal=TERM --kill-after=30 "$WALL_BUDGET_SEC" \
    "$PYTHON_BIN" -u -m experiments.zoology_mqar.contractive_dplr_endpoint \
    --output-dir "$OUT_DIR" \
    --historical-reference-run "$HISTORICAL_REFERENCE_RUN" \
    --runtime-reference-run "$RUNTIME_REFERENCE_RUN" \
    --max-epochs "$MAX_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    2>&1 | tee "$RUN_DIR/formal.log"
  PIPE=("${PIPESTATUS[@]}")
  STATUS="${PIPE[0]}"
  [[ "$STATUS" != 0 || "${PIPE[1]}" == 0 ]] || STATUS=74
  set -e
  printf 'endpoint=%s\ntee=%s\ncombined=%s\n' "${PIPE[0]}" "${PIPE[1]}" "$STATUS" \
    > "$RUN_DIR/formal_status.txt"
fi
if [[ -f "$OUT_DIR/decision.json" ]]; then
  cp "$OUT_DIR/decision.json" "$RUN_DIR/score.json"
  PHASE=decision
fi
exit "$STATUS"
