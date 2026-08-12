#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$REPO_ROOT/configs/retrieval/zoology_gdn2_dual_hash.env"
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
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; then
  printf 'P-GDN3-024 refuses to overlap an existing GPU compute process.\n' >&2
  exit 4
fi
if git -C "$REPO_ROOT" symbolic-ref -q HEAD >/dev/null; then
  printf 'P-GDN3-024 requires a detached source worktree.\n' >&2
  exit 5
fi
GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
if [[ "$GIT_SHA" != "$EXPECTED_SOURCE_SHA" ]] || [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
  printf 'Source is not the exact clean pushed SHA.\n' >&2
  exit 6
fi

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="${RUN_NAME:-p-gdn3-024-dual-hash-l1024-${TIMESTAMP}-${GIT_SHA:0:7}}"
RUN_DIR="$PERSIST_ROOT/runs/$RUN_NAME"
OUT_DIR="$RUN_DIR/output"
mkdir -p "$OUT_DIR"
printf '%s\n' "$$" > "$RUN_DIR/pid.txt"
printf '%s\n' "$(ps -o pgid= -p $$ | tr -d ' ')" > "$RUN_DIR/pgid.txt"
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_DIR/git_sha.txt"
cp "$CONFIG" "$RUN_DIR/launch.env"
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader > "$RUN_DIR/gpu.txt"

STATUS=0
set +e
"$PYTHON_BIN" "$REPO_ROOT/scripts/check_zoology_gdn2_dual_hash.py" \
  --output "$RUN_DIR/contract.json" \
  --expected-gpu-name "$EXPECTED_GPU_NAME" \
  --expected-gpu-uuid "$EXPECTED_GPU_UUID" \
  2>&1 | tee "$RUN_DIR/contract.log"
STATUS="${PIPESTATUS[0]}"
set -e
if [[ "$STATUS" == "0" ]]; then
  set +e
  timeout --signal=TERM --kill-after=30 "$WALL_BUDGET_SEC" \
    "$PYTHON_BIN" -u -m experiments.zoology_mqar.gdn2_dual_hash_endpoint \
    --output-dir "$OUT_DIR" \
    --max-epochs "$MAX_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    2>&1 | tee "$RUN_DIR/formal.log"
  STATUS="${PIPESTATUS[0]}"
  set -e
fi
if [[ -f "$OUT_DIR/comparison.json" ]]; then
  cp "$OUT_DIR/comparison.json" "$RUN_DIR/score.json"
  if [[ "$STATUS" == "0" ]]; then
    set +e
    "$PYTHON_BIN" - "$RUN_DIR/score.json" <<'PY'
import json
import sys

score = json.load(open(sys.argv[1]))
raise SystemExit(0 if score["registered_gate"]["passed"] else 2)
PY
    STATUS="$?"
    set -e
  fi
fi
if [[ "$STATUS" != "0" ]]; then
  "$PYTHON_BIN" - "$RUN_DIR/abort.json" "$STATUS" "$RUN_DIR/score.json" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

score_path = Path(sys.argv[3])
status = int(sys.argv[2])
Path(sys.argv[1]).write_text(json.dumps({
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "reason": (
        "P-GDN3-024 registered science/cost gate failed"
        if status == 2 and score_path.exists()
        else "P-GDN3-024 contract, training, or integrity failed"
    ),
    "exit_status": status,
    "scientific_failure": status == 2 and score_path.exists(),
}, indent=2, sort_keys=True) + "\n")
PY
fi
printf '%s\n' "$STATUS" > "$RUN_DIR/exit_status.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/completed_at.txt"
find "$RUN_DIR" -type f ! -name artifacts.sha256 -print0 \
  | sort -z | xargs -0 sha256sum > "$RUN_DIR/artifacts.sha256"
exit "$STATUS"
