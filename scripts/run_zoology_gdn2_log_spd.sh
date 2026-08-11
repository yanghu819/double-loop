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
PHASE="preflight"
LAUNCHER_PID="$$"
LAUNCHER_PGID="$(ps -o pgid= -p "$$" | tr -d ' ')"
set_phase() {
  PHASE="$1"
  printf '%s\n' "$PHASE" > "$RUN_DIR/phase.txt"
}
gpu_is_idle() {
  [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]
}
finalize() {
  local shell_status="$?"
  if [[ "$FINALIZED" == "1" ]]; then
    return
  fi
  FINALIZED=1
  trap - EXIT
  set +e
  local final_status="${STATUS:-$shell_status}"
  local archive_status=0
  if [[ "$shell_status" != "0" && "$final_status" == "0" ]]; then
    final_status="$shell_status"
  fi
  if [[ "$final_status" != "0" && ! -f "$RUN_DIR/abort.json" ]]; then
    "$PYTHON_BIN" -c 'import json,sys; from datetime import datetime,timezone; from pathlib import Path; Path(sys.argv[1]).write_text(json.dumps({"timestamp_utc":datetime.now(timezone.utc).isoformat(),"reason":"launcher exited before classified terminal state","exit_status":int(sys.argv[2]),"phase":sys.argv[3],"scientific_failure":False},indent=2,sort_keys=True)+"\n")' "$RUN_DIR/abort.json" "$final_status" "$PHASE" || archive_status=1
  fi
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/completed_at.txt" || archive_status=1
  nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu \
    --format=csv,noheader > "$RUN_DIR/gpu_after.txt" || archive_status=1
  if [[ "$archive_status" != "0" && "$final_status" == "0" ]]; then
    final_status=70
  fi
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
    find "$RUN_DIR" -type f ! -name artifacts.sha256 ! -name artifacts.sha256.tmp -print0 \
      | sort -z | xargs -0 sha256sum > "$RUN_DIR/artifacts.sha256.tmp"
    mv "$RUN_DIR/artifacts.sha256.tmp" "$RUN_DIR/artifacts.sha256"
  fi
  printf 'completed run_dir=%s status=%s\n' "$RUN_DIR" "$final_status"
  exit "$final_status"
}
on_signal() {
  STATUS="$1"
  exit "$1"
}
trap finalize EXIT
trap 'on_signal 130' INT
trap 'on_signal 143' TERM

printf '%s\n' "$LAUNCHER_PID" > "$RUN_DIR/pid.txt"
printf '%s\n' "$LAUNCHER_PGID" > "$RUN_DIR/pgid.txt"
set_phase preflight
if [[ "$LAUNCHER_PID" != "$LAUNCHER_PGID" ]]; then
  STATUS=15
  printf 'P-GDN3-020 must be launched under its own setsid process group.\n' >&2
  exit "$STATUS"
fi
cp "$P020_CONFIG" "$RUN_DIR/launch.env"
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_DIR/git_sha.txt"
git -C "$REPO_ROOT" status --short > "$RUN_DIR/git_status.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/started_at.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader > "$RUN_DIR/gpu_before.txt"
if ! gpu_is_idle; then
  STATUS=16
  printf 'GPU became occupied before the formal lease completed.\n' >&2
  exit "$STATUS"
fi

EXPECTED_ORIGIN_URL="https://github.com/yanghu819/double-loop.git"
ORIGIN_URL="$(git -C "$REPO_ROOT" remote get-url origin)"
if [[ "$ORIGIN_URL" != "$EXPECTED_ORIGIN_URL" ]]; then
  STATUS=17
  printf 'Unexpected origin URL: %s\n' "$ORIGIN_URL" >&2
  exit "$STATUS"
fi
set_phase github_readback
set +e
timeout --signal=TERM --kill-after=5 30 \
  git -C "$REPO_ROOT" ls-remote --refs origin "$SOURCE_REMOTE_REF" \
  > "$RUN_DIR/github_ls_remote.txt" 2> "$RUN_DIR/github_ls_remote.stderr"
REMOTE_STATUS="$?"
set -e
if [[ "$REMOTE_STATUS" != "0" ]]; then
  STATUS=69
  printf 'GitHub source readback failed with status %s.\n' "$REMOTE_STATUS" >&2
  exit "$STATUS"
fi
REMOTE_SHA="$(awk '{print $1}' "$RUN_DIR/github_ls_remote.txt")"
if [[ "$REMOTE_SHA" != "$GIT_SHA" ]]; then
  STATUS=11
  printf 'GitHub source readback mismatch: %s != %s\n' "$REMOTE_SHA" "$GIT_SHA" >&2
  exit "$STATUS"
fi
printf 'origin=%s\nref=%s\nsha=%s\nreadback_utc=%s\n' \
  "$ORIGIN_URL" "$SOURCE_REMOTE_REF" "$REMOTE_SHA" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$RUN_DIR/github_provenance.txt"
set_phase snapshot
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

if ! gpu_is_idle; then
  STATUS=18
  printf 'GPU became occupied before strict contract.\n' >&2
  exit "$STATUS"
fi
set_phase contract
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/contract_started_at.txt"
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
printf 'python=%s\ntee=%s\ncombined=%s\n' \
  "${CONTRACT_PIPE_STATUS[0]}" "$CONTRACT_TEE_STATUS" "$CONTRACT_STATUS" \
  > "$RUN_DIR/contract_status.txt"

STATUS="$CONTRACT_STATUS"
if [[ "$CONTRACT_STATUS" == "0" ]]; then
  if ! gpu_is_idle; then
    STATUS=19
    printf 'GPU remained occupied after strict contract.\n' >&2
    exit "$STATUS"
  fi
  set_phase formal_endpoint
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/formal_started_at.txt"
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
  printf 'endpoint=%s\ntee=%s\ncombined=%s\n' \
    "${FORMAL_PIPE_STATUS[0]}" "$FORMAL_TEE_STATUS" "$STATUS" \
    > "$RUN_DIR/formal_status.txt"
fi

set_phase decision
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
  "$PYTHON_BIN" - "$RUN_DIR/abort.json" "$STATUS" "$OUT_DIR/carrier_admission.json" "$OUT_DIR/branch_admission.json" "$OUT_DIR/candidate_started.json" "$OUT_DIR/comparison.json" "$RUN_DIR/phase.txt" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

status = int(sys.argv[2])
carrier_path = Path(sys.argv[3])
branch_path = Path(sys.argv[4])
candidate_marker = Path(sys.argv[5])
comparison_path = Path(sys.argv[6])
phase = Path(sys.argv[7]).read_text().strip()
carrier = json.loads(carrier_path.read_text()) if carrier_path.exists() else None
branch = json.loads(branch_path.read_text()) if branch_path.exists() else None
candidate_started = candidate_marker.exists()
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
    reason = "wall budget exceeded after candidate start" if candidate_started else "wall budget exceeded before candidate start"
    scientific_failure = candidate_started
else:
    reason = "strict contract or formal integrity returned nonzero"
    scientific_failure = False
Path(sys.argv[1]).write_text(json.dumps({
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "reason": reason,
    "exit_status": status,
    "scientific_failure": scientific_failure,
    "candidate_started": candidate_started,
    "phase": phase,
}, indent=2, sort_keys=True) + "\n")
PY
fi

set_phase archived
exit "$STATUS"
