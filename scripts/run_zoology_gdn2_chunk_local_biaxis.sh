#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONFIG="${P035_CONFIG:-$REPO_ROOT/configs/retrieval/zoology_gdn2_chunk_local_biaxis.env}"
source "$CONFIG"

: "${EXPECTED_GPU_NAME:?set EXPECTED_GPU_NAME}"
: "${EXPECTED_GPU_UUID:?set EXPECTED_GPU_UUID}"
: "${EXPECTED_SOURCE_SHA:?set EXPECTED_SOURCE_SHA}"

export CUDA_VISIBLE_DEVICES=0
export XDG_CACHE_HOME="$PERSIST_ROOT/.cache"
export TORCH_HOME="$PERSIST_ROOT/.cache/torch"
export TORCH_EXTENSIONS_DIR="$PERSIST_ROOT/.cache/torch_extensions"
export TRITON_CACHE_DIR="$PERSIST_ROOT/.cache/triton"
export PYTHONPATH="$REPO_ROOT:$ZOOLOGY_ROOT:$FLA_SOURCE_ROOT"
export FLA_EXPECTED_SOURCE_SHA FLA_DISABLE_BACKEND_DISPATCH FLA_CONV_BACKEND
export WANDB_MODE=disabled

VISIBLE_GPU="$(nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader)"
[[ "$VISIBLE_GPU" == "0, $EXPECTED_GPU_UUID, $EXPECTED_GPU_NAME" ]]
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]
[[ "$(git -C "$ZOOLOGY_ROOT" rev-parse HEAD)" == "$ZOOLOGY_SHA" ]]
[[ -z "$(git -C "$ZOOLOGY_ROOT" status --porcelain)" ]]
if git -C "$REPO_ROOT" symbolic-ref -q HEAD >/dev/null; then
  printf 'P-GDN3-035 requires a detached source worktree.\n' >&2
  exit 8
fi
GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ "$GIT_SHA" == "$EXPECTED_SOURCE_SHA" ]]
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]]
[[ "$(git -C "$REPO_ROOT" remote get-url origin)" == "https://github.com/yanghu819/double-loop.git" ]]

LOCK_DIR="$PERSIST_ROOT/artifacts/locks"
mkdir -p "$LOCK_DIR"
exec 9>"$LOCK_DIR/p-gdn3-035.lock"
flock -n 9

TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="${RUN_NAME:-p-gdn3-035-chunk-local-biaxis-l1024-${TIMESTAMP}-${GIT_SHA:0:7}}"
RUN_DIR="$PERSIST_ROOT/runs/$RUN_NAME"
OUT_DIR="$RUN_DIR/output"
mkdir -p "$OUT_DIR"
STATUS=0
PHASE=preflight

finalize() {
  local shell_status="$?"
  trap - EXIT
  set +e
  [[ "$STATUS" != 0 ]] || STATUS="$shell_status"
  if [[ "$STATUS" != 0 && ! -f "$RUN_DIR/abort.json" ]]; then
    "$PYTHON_BIN" - "$RUN_DIR/abort.json" "$STATUS" "$PHASE" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "exit_status": int(sys.argv[2]),
    "phase": sys.argv[3],
    "reason": "P-GDN3-035 contract, integrity, science, or cost gate returned nonzero",
    "scientific_failure": sys.argv[3] in {"formal", "decision"},
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
cp "$CONFIG" "$RUN_DIR/launch.env"
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_DIR/git_sha.txt"
git -C "$REPO_ROOT" status --short > "$RUN_DIR/git_status.txt"
printf 'https://github.com/yanghu819/double-loop/commit/%s\n' "$GIT_SHA" \
  > "$RUN_DIR/github_commit_url.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/started_at.txt"
nvidia-smi --query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total \
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
    "plan": "P-GDN3-035",
    "git_sha": "$GIT_SHA",
    "gpu_name": "$EXPECTED_GPU_NAME",
    "gpu_uuid": "$EXPECTED_GPU_UUID",
    "model": "D128/L2/H4/K32/V32 pinned-official GDN2 plus native FutureSeed",
    "mechanism": "bounded chunk-local persistent value-axis decay",
    "sequence_length": 1024,
    "num_kv_pairs": 4,
    "max_epochs": int("$MAX_EPOCHS"),
    "batch_size": int("$BATCH_SIZE"),
    "seed": 123,
    "new_parameters": 8192,
    "new_recurrent_state": 0,
    "physical_anchor_size": 64,
    "value_groups": 8,
}, indent=2, sort_keys=True) + "\n")
PY

PHASE=contract
set +e
"$PYTHON_BIN" "$REPO_ROOT/experiments/zoology_mqar/check_chunk_local_biaxis_cuda.py" \
  --expected-gpu-uuid "$EXPECTED_GPU_UUID" \
  --output "$RUN_DIR/contract.json" \
  2>&1 | tee "$RUN_DIR/contract.log"
PIPE=("${PIPESTATUS[@]}")
STATUS="${PIPE[0]}"
[[ "$STATUS" != 0 || "${PIPE[1]}" == 0 ]] || STATUS=74
set -e
printf 'python=%s\ntee=%s\ncombined=%s\n' "${PIPE[0]}" "${PIPE[1]}" "$STATUS" \
  > "$RUN_DIR/contract_status.txt"

if [[ "$STATUS" == 0 ]]; then
  [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]
  PHASE=formal
  set +e
  timeout --signal=TERM --kill-after=30 "$WALL_BUDGET_SEC" \
    "$PYTHON_BIN" -u -m experiments.zoology_mqar.gdn2_chunk_local_biaxis_endpoint \
    --output-dir "$OUT_DIR" \
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

if [[ -f "$OUT_DIR/comparison.json" ]]; then
  PHASE=decision
  cp "$OUT_DIR/comparison.json" "$RUN_DIR/score.json"
  if [[ "$STATUS" == 0 ]]; then
    set +e
    "$PYTHON_BIN" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1]))["registered_gate"]["passed"] else 2)' "$RUN_DIR/score.json"
    STATUS="$?"
    set -e
  fi
fi

exit "$STATUS"
