#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$REPO_ROOT/configs/retrieval/zoology_gdn2_oig.env"
set -a
# shellcheck disable=SC1090
source "$CONFIG"
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
export ZOOLOGY_ROOT FLA_SOURCE_ROOT FLA_EXPECTED_SOURCE_SHA
export FLA_GDN2_SOURCE_SHA256 FLA_GDN2_OPS_SHA256
export FLA_DISABLE_BACKEND_DISPATCH FLA_CONV_BACKEND

PINNED_FLA_SHA=9c8e42e762fce087c27b673af4922795d9edb85e
CHECKER="$REPO_ROOT/scripts/check_zoology_gdn2_oig.py"
ENDPOINT_MODULE=experiments.zoology_mqar.gdn2_oig_endpoint
P020_SCORE="$P020_REFERENCE_DIR/score.json"
P020_CASES="$P020_REFERENCE_DIR/cases.json"
GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="${RUN_NAME:-p-gdn3-027-oig-l1024-${TIMESTAMP}-${GIT_SHA:0:7}}"
RUN_DIR="$PERSIST_ROOT/runs/$RUN_NAME"
OUT_DIR="$RUN_DIR/output"

if [[ -e "$RUN_DIR" ]]; then
  printf 'Refusing to reuse existing run directory: %s\n' "$RUN_DIR" >&2
  exit 64
fi
mkdir -p "$OUT_DIR"

STATUS=0
PHASE=preflight
CLASSIFICATION=running
FINALIZED=0

set_phase() {
  PHASE="$1"
  printf '%s\n' "$PHASE" > "$RUN_DIR/phase.txt"
}

gpu_compute_pids() {
  nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d'
}

fail() {
  STATUS="$1"
  shift
  printf '%s\n' "$*" >&2
  exit "$STATUS"
}

finalize() {
  local shell_status="$?"
  local archive_status=0
  local final_status
  if [[ "$FINALIZED" == 1 ]]; then
    return
  fi
  FINALIZED=1
  trap - EXIT
  set +e

  final_status="$STATUS"
  if [[ "$shell_status" != 0 && "$final_status" == 0 ]]; then
    final_status="$shell_status"
  fi
  if [[ "$CLASSIFICATION" == running ]]; then
    if [[ "$final_status" == 0 ]]; then
      CLASSIFICATION=passed
    else
      CLASSIFICATION=error
    fi
  fi

  if [[ "$final_status" != 0 && "$CLASSIFICATION" != registered_science_close \
        && ! -f "$RUN_DIR/abort.json" ]]; then
    "$PYTHON_BIN" - "$RUN_DIR/abort.json" "$final_status" "$PHASE" <<'PY' || archive_status=1
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "exit_status": int(sys.argv[2]),
    "phase": sys.argv[3],
    "reason": "P-GDN3-027 contract, integrity, infrastructure, or endpoint execution error",
    "scientific_failure": False,
}, indent=2, sort_keys=True) + "\n")
PY
  fi

  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/completed_at.txt" || archive_status=1
  nvidia-smi --query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader > "$RUN_DIR/gpu_after.txt" 2> "$RUN_DIR/gpu_after.stderr" \
    || archive_status=1
  printf '%s\n' "$final_status" > "$RUN_DIR/exit_status.txt" || archive_status=1
  "$PYTHON_BIN" - "$RUN_DIR/run_status.json" "$final_status" "$PHASE" "$CLASSIFICATION" <<'PY' \
    || archive_status=1
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "exit_status": int(sys.argv[2]),
    "terminal_phase": sys.argv[3],
    "classification": sys.argv[4],
}, indent=2, sort_keys=True) + "\n")
PY

  if ! find "$RUN_DIR" -type f \
      ! -name artifacts.sha256 ! -name artifacts.sha256.tmp -print0 \
      | sort -z | xargs -0 sha256sum > "$RUN_DIR/artifacts.sha256.tmp"; then
    archive_status=1
  elif ! mv "$RUN_DIR/artifacts.sha256.tmp" "$RUN_DIR/artifacts.sha256"; then
    archive_status=1
  fi
  if [[ "$archive_status" != 0 ]]; then
    final_status=70
    CLASSIFICATION=error
    printf '%s\n' "$final_status" > "$RUN_DIR/exit_status.txt"
    "$PYTHON_BIN" - "$RUN_DIR/abort.json" <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path

Path(__import__("sys").argv[1]).write_text(json.dumps({
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "exit_status": 70,
    "phase": "archive",
    "reason": "P-GDN3-027 artifact archival failed",
    "scientific_failure": False,
}, indent=2, sort_keys=True) + "\n")
PY
    "$PYTHON_BIN" - "$RUN_DIR/run_status.json" <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

Path(sys.argv[1]).write_text(json.dumps({
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "exit_status": 70,
    "terminal_phase": "archive",
    "classification": "error",
}, indent=2, sort_keys=True) + "\n")
PY
    find "$RUN_DIR" -type f \
      ! -name artifacts.sha256 ! -name artifacts.sha256.tmp -print0 \
      | sort -z | xargs -0 sha256sum > "$RUN_DIR/artifacts.sha256.tmp"
    mv "$RUN_DIR/artifacts.sha256.tmp" "$RUN_DIR/artifacts.sha256"
  fi
  printf 'completed run_dir=%s status=%s classification=%s\n' \
    "$RUN_DIR" "$final_status" "$CLASSIFICATION"
  exit "$final_status"
}

trap finalize EXIT
trap 'STATUS=130; CLASSIFICATION=error; exit 130' INT
trap 'STATUS=143; CLASSIFICATION=error; exit 143' TERM

printf '%s\n' "$$" > "$RUN_DIR/pid.txt"
ps -o pgid= -p "$$" | tr -d ' ' > "$RUN_DIR/pgid.txt"
cp "$CONFIG" "$RUN_DIR/launch.env"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/started_at.txt"
set_phase preflight

VISIBLE_GPU="$(nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader)"
EXPECTED_VISIBLE_GPU="0, $EXPECTED_GPU_NAME, $EXPECTED_GPU_UUID"
printf '%s\n' "$VISIBLE_GPU" > "$RUN_DIR/gpu_identity.txt"
[[ "$VISIBLE_GPU" == "$EXPECTED_VISIBLE_GPU" ]] \
  || fail 11 "Unexpected visible GPU: $VISIBLE_GPU"
[[ -z "$(gpu_compute_pids)" ]] \
  || fail 12 "P-GDN3-027 refuses to overlap an existing GPU compute process."
[[ "$FLA_EXPECTED_SOURCE_SHA" == "$PINNED_FLA_SHA" ]] \
  || fail 13 "Pinned FLA SHA changed: $FLA_EXPECTED_SOURCE_SHA"
[[ -d "$FLA_SOURCE_ROOT/fla/ops/gdn2" ]] \
  || fail 14 "Pinned FLA GDN2 source tree is missing."
[[ "$(git -C "$ZOOLOGY_ROOT" rev-parse HEAD)" == "$ZOOLOGY_SHA" ]] \
  || fail 15 "Unexpected Zoology SHA."
[[ -z "$(git -C "$ZOOLOGY_ROOT" status --porcelain)" ]] \
  || fail 16 "Zoology checkout is dirty."
if git -C "$REPO_ROOT" symbolic-ref -q HEAD >/dev/null; then
  fail 17 "P-GDN3-027 requires a detached source worktree."
fi
[[ "$GIT_SHA" == "$EXPECTED_SOURCE_SHA" ]] \
  || fail 18 "Unexpected source SHA: $GIT_SHA != $EXPECTED_SOURCE_SHA"
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] \
  || fail 19 "P-GDN3-027 requires a clean source worktree."
[[ -f "$CHECKER" ]] || fail 20 "Missing strict CUDA checker: $CHECKER"
[[ -f "$P020_SCORE" && -f "$P020_CASES" ]] \
  || fail 21 "Frozen P020 reference artifacts are missing."
[[ "$(sha256sum "$P020_SCORE" | awk '{print $1}')" == "$P020_SCORE_SHA256" ]] \
  || fail 22 "Frozen P020 score hash changed."
[[ "$(sha256sum "$P020_CASES" | awk '{print $1}')" == "$P020_CASES_SHA256" ]] \
  || fail 23 "Frozen P020 cases hash changed."
[[ "$SEQUENCE_LENGTH" == 1024 && "$NUM_KV_PAIRS" == 4 \
    && "$TRAIN_EXAMPLES" == 10000 && "$VALID_EXAMPLES" == 1000 \
    && "$MAX_EPOCHS" == 10 && "$BATCH_SIZE" == 32 && "$SEED" == 123 \
    && "$LEARNING_RATE" == 0.001 && "$WEIGHT_DECAY" == 0.1 ]] \
  || fail 24 "The fixed matched directional MQAR protocol was modified."

git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_DIR/source_HEAD.txt"
git -C "$REPO_ROOT" status --short > "$RUN_DIR/source_status.txt"
git -C "$REPO_ROOT" diff --binary HEAD > "$RUN_DIR/source.patch"
nvidia-smi --query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader > "$RUN_DIR/gpu_before.txt"

LOCK_DIR="$PERSIST_ROOT/artifacts/locks"
mkdir -p "$LOCK_DIR"
exec 9> "$LOCK_DIR/p-gdn3-027.lock"
flock -n 9 || fail 25 "Another P-GDN3-027 launcher owns the formal lease."

set_phase github_readback
SOURCE_BRANCH="${SOURCE_REMOTE_REF#refs/heads/}"
set +e
timeout --signal=TERM --kill-after=5 30 curl -fsSL \
  --connect-timeout 10 --max-time 25 \
  "https://api.github.com/repos/$GITHUB_API_REPOSITORY/git/ref/heads/$SOURCE_BRANCH" \
  > "$RUN_DIR/github_ref.json" 2> "$RUN_DIR/github_ref.stderr"
REMOTE_STATUS="$?"
set -e
[[ "$REMOTE_STATUS" == 0 ]] || fail 69 "GitHub source readback failed."
REMOTE_SHA="$("$PYTHON_BIN" - "$RUN_DIR/github_ref.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    print(json.load(handle)["object"]["sha"])
PY
)"
[[ "$REMOTE_SHA" == "$GIT_SHA" ]] \
  || fail 26 "GitHub source readback mismatch: $REMOTE_SHA != $GIT_SHA"
printf 'remote=%s\nref=%s\nsha=%s\nreadback_utc=%s\n' \
  "https://api.github.com/repos/$GITHUB_API_REPOSITORY" "$SOURCE_REMOTE_REF" \
  "$REMOTE_SHA" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$RUN_DIR/github_provenance.txt"

set_phase snapshot
git -C "$REPO_ROOT" ls-files -z -- \
  configs experiments scripts research plans.md \
  | tar --null -czf "$RUN_DIR/source_snapshot.tar.gz" \
      -C "$REPO_ROOT" --files-from -

"$PYTHON_BIN" - "$RUN_DIR/config.json" <<PY
import json
from pathlib import Path

Path("$RUN_DIR/config.json").write_text(json.dumps({
    "plan": "P-GDN3-027",
    "git_sha": "$GIT_SHA",
    "run_name": "$RUN_NAME",
    "gpu_name": "$EXPECTED_GPU_NAME",
    "gpu_uuid": "$EXPECTED_GPU_UUID",
    "model": "D128/L2/H4/K32/V32 pinned-FLA GDN2 plus native FutureSeed",
    "mechanism": "constrained online inverse-Gram preconditioned committed delta",
    "protocol": {
        "arms": ["contemporaneous_control", "oig_candidate"],
        "sequence_length": int("$SEQUENCE_LENGTH"),
        "num_kv_pairs": int("$NUM_KV_PAIRS"),
        "train_examples": int("$TRAIN_EXAMPLES"),
        "validation_examples": int("$VALID_EXAMPLES"),
        "epochs": int("$MAX_EPOCHS"),
        "batch_size": int("$BATCH_SIZE"),
        "seed": int("$SEED"),
        "learning_rate": float("$LEARNING_RATE"),
        "weight_decay": float("$WEIGHT_DECAY"),
    },
    "candidate_new_parameters": 8,
    "candidate_extra_state_values_per_layer": 4096,
    "candidate_logical_scans_per_layer": 1,
    "p020_score_sha256": "$P020_SCORE_SHA256",
    "p020_cases_sha256": "$P020_CASES_SHA256",
}, indent=2, sort_keys=True) + "\n")
PY

[[ -z "$(gpu_compute_pids)" ]] \
  || fail 27 "GPU became occupied before the strict CUDA contract."
set_phase contract
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/contract_started_at.txt"
set +e
"$PYTHON_BIN" "$CHECKER" \
  --output "$RUN_DIR/contract.json" \
  --expected-gpu-name "$EXPECTED_GPU_NAME" \
  --expected-gpu-uuid "$EXPECTED_GPU_UUID" \
  2>&1 | tee "$RUN_DIR/contract.log"
PIPE=("${PIPESTATUS[@]}")
CHECKER_STATUS="${PIPE[0]}"
[[ "$CHECKER_STATUS" != 0 || "${PIPE[1]}" == 0 ]] || CHECKER_STATUS=74
set -e
printf 'checker=%s\ntee=%s\ncombined=%s\n' \
  "${PIPE[0]}" "${PIPE[1]}" "$CHECKER_STATUS" \
  > "$RUN_DIR/contract_status.txt"
[[ "$CHECKER_STATUS" == 0 ]] \
  || fail "$CHECKER_STATUS" "P-GDN3-027 strict CUDA contract failed."
[[ -z "$(gpu_compute_pids)" ]] \
  || fail 28 "GPU remained occupied after the strict CUDA contract."

set_phase formal_endpoint
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/formal_started_at.txt"
set +e
timeout --signal=TERM --kill-after=30 "$WALL_BUDGET_SEC" \
  "$PYTHON_BIN" -u -m "$ENDPOINT_MODULE" \
  --output-dir "$OUT_DIR" \
  2>&1 | tee "$RUN_DIR/formal.log"
PIPE=("${PIPESTATUS[@]}")
ENDPOINT_STATUS="${PIPE[0]}"
[[ "$ENDPOINT_STATUS" != 0 || "${PIPE[1]}" == 0 ]] || ENDPOINT_STATUS=74
set -e
printf 'endpoint=%s\ntee=%s\ncombined=%s\n' \
  "${PIPE[0]}" "${PIPE[1]}" "$ENDPOINT_STATUS" \
  > "$RUN_DIR/formal_status.txt"

set_phase decision
if [[ "$ENDPOINT_STATUS" == 0 || "$ENDPOINT_STATUS" == 3 ]]; then
  [[ -f "$OUT_DIR/comparison.json" ]] \
    || fail 29 "Endpoint returned a terminal science status without comparison.json."
  cp "$OUT_DIR/comparison.json" "$RUN_DIR/score.json"
fi
if [[ "$ENDPOINT_STATUS" == 3 ]]; then
  STATUS=3
  CLASSIFICATION=registered_science_close
  if [[ -f "$OUT_DIR/abort.json" ]]; then
    mv "$OUT_DIR/abort.json" "$RUN_DIR/endpoint_science_close.json"
  fi
  "$PYTHON_BIN" - "$RUN_DIR/science_close.json" <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

Path(sys.argv[1]).write_text(json.dumps({
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "exit_status": 3,
    "reason": "P-GDN3-027 completed but missed at least one preregistered science or cost gate",
    "scientific_failure": True,
    "rescue_authorized": False,
}, indent=2, sort_keys=True) + "\n")
PY
elif [[ "$ENDPOINT_STATUS" == 0 ]]; then
  STATUS=0
  CLASSIFICATION=passed
else
  fail "$ENDPOINT_STATUS" "P-GDN3-027 endpoint execution failed."
fi

set_phase archived
exit "$STATUS"
