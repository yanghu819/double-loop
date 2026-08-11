#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
P020_CONFIG="$REPO_ROOT/configs/retrieval/zoology_gdn2_log_spd.env"
set -a
# shellcheck disable=SC1090
source "$P020_CONFIG"
set +a

: "${EXPECTED_GPU_NAME:?Set the exact admitted task-mode GPU name}"
: "${EXPECTED_GPU_UUID:?Set the exact admitted task-mode GPU UUID}"
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
EXPECTED_VISIBLE_GPU="0, $EXPECTED_GPU_UUID, $EXPECTED_GPU_NAME"
if [[ "$VISIBLE_GPU" != "$EXPECTED_VISIBLE_GPU" ]]; then
  printf 'Unexpected visible GPU: %s\n' "$VISIBLE_GPU" >&2
  exit 3
fi
if [[ "$EXPECTED_GPU_NAME" == *A800* ]]; then
  printf 'The previously invalidated A800 runtime may run contract only, not formal replay.\n' >&2
  exit 4
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; then
  printf 'P-GDN3-020 refuses to overlap an existing GPU compute process.\n' >&2
  exit 5
fi
if [[ "$(git -C "$ZOOLOGY_ROOT" rev-parse HEAD)" != "$ZOOLOGY_SHA" ]]; then
  printf 'Unexpected Zoology SHA.\n' >&2
  exit 6
fi
if [[ -n "$(git -C "$ZOOLOGY_ROOT" status --porcelain)" ]]; then
  printf 'P-GDN3-020 requires a clean Zoology checkout.\n' >&2
  exit 7
fi
if git -C "$REPO_ROOT" symbolic-ref -q HEAD >/dev/null; then
  printf 'P-GDN3-020 requires a detached source worktree.\n' >&2
  exit 8
fi
GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
if [[ "$GIT_SHA" != "$EXPECTED_SOURCE_SHA" ]]; then
  printf 'Unexpected source SHA: %s != %s\n' "$GIT_SHA" "$EXPECTED_SOURCE_SHA" >&2
  exit 9
fi
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
  printf 'P-GDN3-020 requires a clean source worktree.\n' >&2
  exit 10
fi
REMOTE_SHA="$(git -C "$REPO_ROOT" ls-remote --refs origin "$SOURCE_REMOTE_REF" | awk '{print $1}')"
if [[ "$REMOTE_SHA" != "$GIT_SHA" ]]; then
  printf 'GitHub source readback mismatch: %s != %s\n' "$REMOTE_SHA" "$GIT_SHA" >&2
  exit 11
fi

REFERENCE_RUN="$PERSIST_ROOT/$HISTORICAL_REFERENCE_RUN"
for required in git_sha.txt score.json output/length_1024/future_seed_gdn2/cases.json; do
  if [[ ! -f "$REFERENCE_RUN/$required" ]]; then
    printf 'Historical directional MQAR evidence is missing: %s\n' "$REFERENCE_RUN/$required" >&2
    exit 12
  fi
done

LOCK_DIR="$PERSIST_ROOT/artifacts/locks"
mkdir -p "$LOCK_DIR"
exec 9>"$LOCK_DIR/p-gdn3-020.lock"
if ! flock -n 9; then
  printf 'Another P-GDN3-020 resource already owns the formal lease.\n' >&2
  exit 13
fi

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="${RUN_NAME:-p-gdn3-020-log-spd-l1024-${TIMESTAMP}-${GIT_SHA:0:7}}"
RUN_DIR="$PERSIST_ROOT/runs/$RUN_NAME"
OUT_DIR="$RUN_DIR/output"
if [[ -e "$RUN_DIR" ]]; then
  printf 'Refusing to reuse an existing run directory: %s\n' "$RUN_DIR" >&2
  exit 14
fi
mkdir -p "$OUT_DIR"

STATUS=""
FINALIZED=0
finalize() {
  local shell_status="$?"
  if [[ "$FINALIZED" == "1" ]]; then
    return
  fi
  FINALIZED=1
  set +e
  local final_status="${STATUS:-$shell_status}"
  if [[ "$shell_status" != "0" && "$final_status" == "0" ]]; then
    final_status="$shell_status"
  fi
  if [[ "$final_status" != "0" && ! -f "$RUN_DIR/abort.json" ]]; then
    "$PYTHON_BIN" -c 'import json,sys; from datetime import datetime,timezone; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"timestamp_utc":datetime.now(timezone.utc).isoformat(),"reason":"launcher exited before classified terminal state","exit_status":int(sys.argv[2]),"scientific_failure":False},indent=2,sort_keys=True)+"\n")' "$RUN_DIR/abort.json" "$final_status"
  fi
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/completed_at.txt"
  nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu \
    --format=csv,noheader > "$RUN_DIR/gpu_after.txt"
  printf '%s\n' "$final_status" > "$RUN_DIR/exit_status.txt"
  find "$RUN_DIR" -type f ! -name artifacts.sha256 -print0 \
    | sort -z | xargs -0 sha256sum > "$RUN_DIR/artifacts.sha256"
  printf 'completed run_dir=%s status=%s\n' "$RUN_DIR" "$final_status"
}
trap finalize EXIT

cp "$P020_CONFIG" "$RUN_DIR/launch.env"
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_DIR/git_sha.txt"
git -C "$REPO_ROOT" status --short > "$RUN_DIR/git_status.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/started_at.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader > "$RUN_DIR/gpu_before.txt"
git -C "$REPO_ROOT" ls-files -z -- . \
  ':(exclude).cache/**' ':(exclude).venv/**' ':(exclude)artifacts/**' \
  ':(exclude)models/**' ':(exclude)repos/**' ':(exclude)runs/**' \
  | tar --null -czf "$RUN_DIR/source_snapshot.tar.gz" \
      -C "$REPO_ROOT" --files-from -

"$PYTHON_BIN" - "$RUN_DIR/config.json" <<PY
import json
from pathlib import Path

Path("$RUN_DIR/config.json").write_text(json.dumps({
    "plan": "P-GDN3-020",
    "git_sha": "$GIT_SHA",
    "run_name": "$RUN_NAME",
    "gpu_name": "$EXPECTED_GPU_NAME",
    "gpu_uuid": "$EXPECTED_GPU_UUID",
    "sequence_length": 1024,
    "num_kv_pairs": 4,
    "max_epochs": int("$MAX_EPOCHS"),
    "batch_size": int("$BATCH_SIZE"),
    "seed": 123,
    "model": "D128/L2/H4/K32 pinned-official GDN2 plus native FutureSeed",
    "mechanism": "bounded trace-free Log-SPD joint Q/K address metric",
    "candidate_new_parameters": 4216,
    "candidate_new_state": 0,
    "candidate_new_scans": 0,
    "candidate_starts_only_after_runtime_control_carrier_admission": True,
    "candidate_starts_only_after_runtime_geometry_branch_admission": True,
}, indent=2, sort_keys=True) + "\n")
PY

set +e
"$PYTHON_BIN" "$REPO_ROOT/scripts/check_zoology_gdn2_log_spd.py" \
  --output "$RUN_DIR/contract.json" \
  --expected-gpu-name "$EXPECTED_GPU_NAME" \
  --expected-gpu-uuid "$EXPECTED_GPU_UUID" \
  2>&1 | tee "$RUN_DIR/contract.log"
CONTRACT_PIPE_STATUS=("${PIPESTATUS[@]}")
CONTRACT_STATUS="${CONTRACT_PIPE_STATUS[0]}"
CONTRACT_TEE_STATUS="${CONTRACT_PIPE_STATUS[1]}"
if [[ "$CONTRACT_STATUS" == "0" && "$CONTRACT_TEE_STATUS" != "0" ]]; then
  CONTRACT_STATUS=74
fi
set -e

STATUS="$CONTRACT_STATUS"
if [[ "$CONTRACT_STATUS" == "0" ]]; then
  set +e
  timeout --signal=TERM --kill-after=30 "$WALL_BUDGET_SEC" \
    "$PYTHON_BIN" -u -m experiments.zoology_mqar.gdn2_log_spd_endpoint \
    --output-dir "$OUT_DIR" \
    --historical-reference "$REFERENCE_RUN" \
    --max-epochs "$MAX_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    2>&1 | tee "$RUN_DIR/formal.log"
  FORMAL_PIPE_STATUS=("${PIPESTATUS[@]}")
  STATUS="${FORMAL_PIPE_STATUS[0]}"
  FORMAL_TEE_STATUS="${FORMAL_PIPE_STATUS[1]}"
  if [[ "$STATUS" == "0" && "$FORMAL_TEE_STATUS" != "0" ]]; then
    STATUS=74
  fi
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
  "$PYTHON_BIN" - "$RUN_DIR/abort.json" "$STATUS" "$OUT_DIR/carrier_admission.json" "$OUT_DIR/branch_admission.json" "$OUT_DIR/comparison.json" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

status = int(sys.argv[2])
carrier_path = Path(sys.argv[3])
branch_path = Path(sys.argv[4])
comparison_path = Path(sys.argv[5])
carrier = json.loads(carrier_path.read_text()) if carrier_path.exists() else None
branch = json.loads(branch_path.read_text()) if branch_path.exists() else None
candidate_started = comparison_path.exists() or (branch is not None and branch["passed"])
if carrier is not None and not carrier["passed"]:
    reason = "matched directional MQAR carrier admission failed; candidate did not start"
    scientific_failure = False
elif branch is not None and not branch["passed"]:
    reason = "matched runtime geometry did not open the Log-SPD branch; candidate did not start"
    scientific_failure = False
elif status == 2 and comparison_path.exists():
    reason = "P-GDN3-020 registered mechanism gate failed"
    scientific_failure = True
elif status == 124:
    reason = "wall budget exceeded"
    scientific_failure = False
else:
    reason = "strict contract or formal integrity returned nonzero"
    scientific_failure = False
Path(sys.argv[1]).write_text(json.dumps({
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "reason": reason,
    "exit_status": status,
    "scientific_failure": scientific_failure,
    "candidate_started": candidate_started,
}, indent=2, sort_keys=True) + "\n")
PY
fi

exit "$STATUS"
