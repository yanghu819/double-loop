#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$REPO_ROOT/configs/retrieval/zoology_surprise_regression_seed.env"
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
  exit 20
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; then
  printf 'P-FS2-008 refuses to overlap an existing GPU compute process.\n' >&2
  exit 4
fi
if [[ "$(git -C "$ZOOLOGY_ROOT" rev-parse HEAD)" != "$ZOOLOGY_SHA" ]] \
  || [[ -n "$(git -C "$ZOOLOGY_ROOT" status --porcelain)" ]]; then
  printf 'P-FS2-008 requires the exact clean Zoology checkout.\n' >&2
  exit 5
fi
if git -C "$REPO_ROOT" symbolic-ref -q HEAD >/dev/null; then
  printf 'P-FS2-008 requires a detached source worktree.\n' >&2
  exit 6
fi
GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
if [[ "$GIT_SHA" != "$EXPECTED_SOURCE_SHA" ]] \
  || [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
  printf 'Source is not the exact clean pushed SHA.\n' >&2
  exit 7
fi
REMOTE_SHA="$(env -u LD_LIBRARY_PATH git -C "$REPO_ROOT" ls-remote origin "$SOURCE_REMOTE_REF" | awk '{print $1}')"
if [[ "$REMOTE_SHA" != "$GIT_SHA" ]]; then
  printf 'GitHub readback differs from detached source: %s vs %s\n' "$REMOTE_SHA" "$GIT_SHA" >&2
  exit 8
fi

REFERENCE_RUN="$HISTORICAL_REFERENCE_RUN"
for required in git_sha.txt score.json output/length_1024/future_seed_gdn2/cases.json; do
  if [[ ! -f "$REFERENCE_RUN/$required" ]]; then
    printf 'Historical directional MQAR evidence is missing: %s\n' "$REFERENCE_RUN/$required" >&2
    exit 9
  fi
done

LOCK_DIR="$PERSIST_ROOT/artifacts/locks"
mkdir -p "$LOCK_DIR"
exec 9>"$LOCK_DIR/p-fs2-008.lock"
if ! flock -n 9; then
  printf 'Another P-FS2-008 process owns the formal lease.\n' >&2
  exit 10
fi

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="${RUN_NAME:-p-fs2-008-surprise-regression-l1024-${TIMESTAMP}-${GIT_SHA:0:7}}"
RUN_DIR="$PERSIST_ROOT/runs/$RUN_NAME"
OUT_DIR="$RUN_DIR/output"
if [[ -e "$RUN_DIR" ]]; then
  printf 'Refusing to reuse an existing run directory: %s\n' "$RUN_DIR" >&2
  exit 11
fi
mkdir -p "$OUT_DIR"

STATUS=0
PHASE=preflight
FINALIZED=0
set_phase() {
  PHASE="$1"
  printf '%s\n' "$PHASE" > "$RUN_DIR/phase.txt"
}
finalize() {
  local shell_status="$?"
  local final_status="$STATUS"
  local archive_status=0
  if [[ "$FINALIZED" == "1" ]]; then
    exit "$shell_status"
  fi
  FINALIZED=1
  trap - EXIT INT TERM
  if [[ "$shell_status" != "0" && "$final_status" == "0" ]]; then
    final_status="$shell_status"
  fi
  if [[ "$final_status" != "0" && "$final_status" != "3" && ! -f "$RUN_DIR/abort.json" ]]; then
    "$PYTHON_BIN" - "$RUN_DIR/abort.json" "$final_status" "$PHASE" <<'PY' || archive_status=1
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "reason": "P-FS2-008 contract, training, integrity, or wall-budget failure",
    "exit_status": int(sys.argv[2]),
    "phase": sys.argv[3],
    "scientific_failure": False,
    "rescue_authorized": False,
}, indent=2, sort_keys=True) + "\n")
PY
  fi
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/completed_at.txt" || archive_status=1
  nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu \
    --format=csv,noheader > "$RUN_DIR/gpu_final.txt" || archive_status=1
  printf '%s\n' "$final_status" > "$RUN_DIR/exit_status.txt" || archive_status=1
  if ! find "$RUN_DIR" -type f ! -name artifacts.sha256 ! -name artifacts.sha256.tmp -print0 \
    | sort -z | xargs -0 sha256sum > "$RUN_DIR/artifacts.sha256.tmp"; then
    archive_status=1
  elif ! mv "$RUN_DIR/artifacts.sha256.tmp" "$RUN_DIR/artifacts.sha256"; then
    archive_status=1
  fi
  if [[ "$archive_status" != "0" && "$final_status" == "0" ]]; then
    final_status=70
    printf '%s\n' "$final_status" > "$RUN_DIR/exit_status.txt"
  fi
  exit "$final_status"
}
trap finalize EXIT
trap 'STATUS=130; exit 130' INT
trap 'STATUS=143; exit 143' TERM

LAUNCHER_PID="$$"
LAUNCHER_PGID="$(ps -o pgid= -p $$ | tr -d ' ')"
printf '%s\n' "$LAUNCHER_PID" > "$RUN_DIR/pid.txt"
printf '%s\n' "$LAUNCHER_PGID" > "$RUN_DIR/pgid.txt"
if [[ "$LAUNCHER_PID" != "$LAUNCHER_PGID" ]]; then
  STATUS=12
  printf 'P-FS2-008 must be launched under its own setsid process group.\n' >&2
  exit "$STATUS"
fi
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_DIR/git_sha.txt"
git -C "$REPO_ROOT" status --porcelain=v1 > "$RUN_DIR/source_status.txt"
cp "$CONFIG" "$RUN_DIR/launch.env"
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
    "plan": "P-FS2-008",
    "git_sha": "$GIT_SHA",
    "run_name": "$RUN_NAME",
    "gpu_name": "$EXPECTED_GPU_NAME",
    "gpu_uuid": "$EXPECTED_GPU_UUID",
    "sequence_length": 1024,
    "num_kv_pairs": 4,
    "max_epochs": int("$MAX_EPOCHS"),
    "batch_size": int("$BATCH_SIZE"),
    "seed": 123,
    "model": "D128/L2/H4/K32/V32 pinned official GDN2",
    "arms_in_fixed_order": [
        "future_seed_gdn2_runtime_control",
        "future_seed_gdn2_surprise_regression",
    ],
    "mechanism": "receiver-native exact-surprise weighted regression FutureSeed",
    "new_parameters": 0,
    "new_persistent_state": 0,
    "additional_official_scans": 0,
    "rescue_authorized": False,
}, indent=2, sort_keys=True) + "\n")
PY

set_phase contract
set +e
"$PYTHON_BIN" "$REPO_ROOT/scripts/check_zoology_surprise_regression_seed.py" \
  --output "$RUN_DIR/contract.json" \
  --expected-gpu-name "$EXPECTED_GPU_NAME" \
  --expected-gpu-uuid "$EXPECTED_GPU_UUID" \
  2>&1 | tee "$RUN_DIR/contract.log"
CONTRACT_PIPE_STATUS=("${PIPESTATUS[@]}")
STATUS="${CONTRACT_PIPE_STATUS[0]}"
if [[ "$STATUS" == "0" && "${CONTRACT_PIPE_STATUS[1]}" != "0" ]]; then
  STATUS=74
fi
set -e
printf 'python=%s\ntee=%s\ncombined=%s\n' \
  "${CONTRACT_PIPE_STATUS[0]}" "${CONTRACT_PIPE_STATUS[1]}" "$STATUS" \
  > "$RUN_DIR/contract_status.txt"

if [[ "$STATUS" == "0" ]]; then
  if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; then
    STATUS=13
    printf 'GPU remained occupied after strict contract.\n' >&2
    exit "$STATUS"
  fi
  set_phase formal_endpoint
  set +e
  timeout --signal=TERM --kill-after=30 "$WALL_BUDGET_SEC" \
    "$PYTHON_BIN" -u -m experiments.zoology_mqar.surprise_regression_seed_endpoint \
    --output-dir "$OUT_DIR" \
    --historical-reference-run "$REFERENCE_RUN" \
    --max-epochs "$MAX_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    2>&1 | tee "$RUN_DIR/formal.log"
  FORMAL_PIPE_STATUS=("${PIPESTATUS[@]}")
  if [[ "${FORMAL_PIPE_STATUS[1]}" != "0" ]]; then
    STATUS=74
  else
    STATUS="${FORMAL_PIPE_STATUS[0]}"
  fi
  set -e
  printf 'endpoint=%s\ntee=%s\ncombined=%s\n' \
    "${FORMAL_PIPE_STATUS[0]}" "${FORMAL_PIPE_STATUS[1]}" "$STATUS" \
    > "$RUN_DIR/formal_status.txt"
fi

set_phase decision
if [[ "$STATUS" == "0" || "$STATUS" == "3" ]]; then
  EXPECTED_DECISION_STATUS=completed_passed
  if [[ "$STATUS" == "3" ]]; then
    EXPECTED_DECISION_STATUS=completed_rejected
  fi
  if [[ ! -f "$OUT_DIR/decision.json" ]]; then
    STATUS=75
    printf 'Endpoint returned a science status without decision.json.\n' >&2
  elif ! "$PYTHON_BIN" - "$OUT_DIR/decision.json" "$EXPECTED_DECISION_STATUS" <<'PY'
import json
import sys
from pathlib import Path

decision = json.loads(Path(sys.argv[1]).read_text())
if decision.get("status") != sys.argv[2]:
    raise SystemExit(
        f"decision status mismatch: {decision.get('status')} != {sys.argv[2]}"
    )
if decision.get("plan") != "P-FS2-008":
    raise SystemExit("decision plan mismatch")
PY
  then
    STATUS=75
    printf 'Endpoint decision artifact failed validation.\n' >&2
  fi
fi
if [[ -f "$OUT_DIR/decision.json" ]]; then
  cp "$OUT_DIR/decision.json" "$RUN_DIR/score.json"
fi
set_phase archived
exit "$STATUS"
