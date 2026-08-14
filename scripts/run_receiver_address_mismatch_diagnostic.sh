#!/usr/bin/env bash
set -euo pipefail

: "${EXPECTED_SOURCE_SHA:?required}"
: "${EXPECTED_GPU_UUID:?required}"
: "${CHECKPOINT:?required}"
: "${EXPECTED_CHECKPOINT_SHA256:?required}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-/huyang2/double-loop}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
ZOOLOGY_ROOT="${ZOOLOGY_ROOT:-$PERSIST_ROOT/repos/zoology-official}"
EXPECTED_ZOOLOGY_SHA="${EXPECTED_ZOOLOGY_SHA:-1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb}"
RUN_NAME="${RUN_NAME:-p-diag-addr-001-$(date -u +%Y%m%dT%H%M%SZ)-${EXPECTED_SOURCE_SHA:0:7}}"
RUN_DIR="$PERSIST_ROOT/runs/$RUN_NAME"
mkdir -p "$RUN_DIR/output"

[[ "$(git -C "$REPO_ROOT" rev-parse HEAD)" == "$EXPECTED_SOURCE_SHA" ]]
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]]
[[ "$(git -C "$ZOOLOGY_ROOT" rev-parse HEAD)" == "$EXPECTED_ZOOLOGY_SHA" ]]
VISIBLE="$(nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader)"
[[ "$(printf '%s\n' "$VISIBLE" | wc -l | tr -d ' ')" == "1" ]]
printf '%s\n' "$VISIBLE" | grep -F "0, $EXPECTED_GPU_UUID, NVIDIA A100-SXM4-80GB"
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]

printf '%s\n' "$EXPECTED_SOURCE_SHA" > "$RUN_DIR/git_sha.txt"
printf '%s\n' "$VISIBLE" > "$RUN_DIR/gpu.txt"
printf '%s\n' "$$" > "$RUN_DIR/pid.txt"
printf '%s\n' "$(ps -o pgid= -p $$ | tr -d ' ')" > "$RUN_DIR/pgid.txt"

export CUDA_VISIBLE_DEVICES=0
export FLA_EXPECTED_SOURCE_SHA=9c8e42e762fce087c27b673af4922795d9edb85e
export PYTHONPATH="$REPO_ROOT:$ZOOLOGY_ROOT:$PERSIST_ROOT/.cache/fla-active:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

set +e
timeout 1200 "$PYTHON_BIN" "$REPO_ROOT/experiments/zoology_mqar/diagnose_receiver_address_mismatch.py" \
  --checkpoint "$CHECKPOINT" \
  --expected-checkpoint-sha256 "$EXPECTED_CHECKPOINT_SHA256" \
  --output "$RUN_DIR/output/diagnostic.json" \
  2>&1 | tee "$RUN_DIR/run.log"
status=${PIPESTATUS[0]}
set -e
printf '%s\n' "$status" > "$RUN_DIR/exit_status.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv,noheader > "$RUN_DIR/gpu_final.txt"
if [[ "$status" -ne 0 && "$status" -ne 3 ]]; then
  python - "$RUN_DIR" "$status" <<'PY'
import json, sys
from pathlib import Path
run, status = Path(sys.argv[1]), int(sys.argv[2])
(run / "abort.json").write_text(json.dumps({"status": "failed", "exit_status": status}, indent=2) + "\n")
PY
fi
exit "$status"
