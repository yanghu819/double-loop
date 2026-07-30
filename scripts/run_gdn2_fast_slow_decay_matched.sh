#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_VISIBLE_DEVICES=0
export PERSIST_ROOT="${PERSIST_ROOT:-/huyang2/double-loop}"
GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
ARTIFACT_DIR="$PERSIST_ROOT/artifacts/gdn2-fast-slow-decay/$GIT_SHA"
READY_JSON="$ARTIFACT_DIR/formal_ready.json"
CONTRACT_JSON="$ARTIFACT_DIR/cuda_contract.json"
if [[ ! -f "$READY_JSON" ]]; then
  printf 'formal-ready manifest missing: %s\n' "$READY_JSON" >&2
  exit 4
fi

MATCHED_ID="${MATCHED_ID:-$(date -u +%Y%m%dT%H%M%SZ)-${GIT_SHA:0:7}}"
CONTROL_NAME="gdn2-fast-slow-matched-${MATCHED_ID}-external-identity"
CANDIDATE_NAME="gdn2-fast-slow-matched-${MATCHED_ID}-positive-causal"
CONTROL_LOG="$ARTIFACT_DIR/${MATCHED_ID}-control.log"
CANDIDATE_LOG="$ARTIFACT_DIR/${MATCHED_ID}-candidate.log"
mkdir -p "$ARTIFACT_DIR"

RUN_NAME="$CONTROL_NAME" FAST_SLOW_CONTRACT_JSON="$CONTRACT_JSON" \
  "$REPO_ROOT/scripts/run_gdn2_fast_slow_decay_arm.sh" external_identity formal \
  2>&1 | tee "$CONTROL_LOG"

CONTROL_CE="$(
  python3 - "$CONTROL_LOG" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
matches = re.findall(r"step=9050 ce=([0-9.]+)", text)
if not matches:
    raise SystemExit("control step9050 CE is missing")
print(matches[-1])
PY
)"

RUN_NAME="$CANDIDATE_NAME" FAST_SLOW_CONTRACT_JSON="$CONTRACT_JSON" \
  "$REPO_ROOT/scripts/run_gdn2_fast_slow_decay_arm.sh" positive_causal formal \
  >"$CANDIDATE_LOG" 2>&1 &
WRAPPER_PID=$!
tail -n +1 -F "$CANDIDATE_LOG" &
TAIL_PID=$!
trap 'kill "$TAIL_PID" 2>/dev/null || true' EXIT

KILLED=0
while kill -0 "$WRAPPER_PID" 2>/dev/null; do
  if grep -q 'step=9050 ce=' "$CANDIDATE_LOG"; then
    CANDIDATE_CE="$(
      python3 - "$CANDIDATE_LOG" <<'PY'
import pathlib
import re
import sys

text = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
print(re.findall(r"step=9050 ce=([0-9.]+)", text)[-1])
PY
    )"
    if python3 - "$CONTROL_CE" "$CANDIDATE_CE" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[2]) > float(sys.argv[1]) + 0.10 else 1)
PY
    then
      PID_FILE="$PERSIST_ROOT/artifacts/launch/$CANDIDATE_NAME.pid"
      if [[ -s "$PID_FILE" ]]; then
        TRAIN_PID="$(<"$PID_FILE")"
        CMDLINE="$(tr '\0' ' ' <"/proc/$TRAIN_PID/cmdline" 2>/dev/null || true)"
        if [[ "$CMDLINE" == *"study_rwkv_futureseed_loop.py"* ]]; then
          kill -TERM "$TRAIN_PID"
          KILLED=1
          python3 - "$ARTIFACT_DIR/${MATCHED_ID}-abort.json" \
            "$TRAIN_PID" "$CONTROL_CE" "$CANDIDATE_CE" <<'PY'
import datetime
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "status": "aborted",
    "reason": "candidate_step9050_ce_exceeds_control_by_more_than_0.10",
    "exact_pid": int(sys.argv[2]),
    "control_step9050_ce": float(sys.argv[3]),
    "candidate_step9050_ce": float(sys.argv[4]),
    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
        fi
      fi
    fi
    break
  fi
  sleep 5
done

set +e
wait "$WRAPPER_PID"
STATUS=$?
set -e
kill "$TAIL_PID" 2>/dev/null || true
wait "$TAIL_PID" 2>/dev/null || true
trap - EXIT
if [[ "$KILLED" == "1" ]]; then
  exit 42
fi
if [[ "$STATUS" != "0" ]]; then
  exit "$STATUS"
fi

CONTROL_RESULT="$PERSIST_ROOT/runs/$CONTROL_NAME/output/futureseed_loop_seed52.json"
CANDIDATE_RESULT="$PERSIST_ROOT/runs/$CANDIDATE_NAME/output/futureseed_loop_seed52.json"
python3 "$REPO_ROOT/scripts/compare_gdn2_fast_slow_decay.py" \
  --control "$CONTROL_RESULT" \
  --candidate "$CANDIDATE_RESULT" \
  --out-dir "$PERSIST_ROOT/runs/gdn2-fast-slow-comparison-$MATCHED_ID"
