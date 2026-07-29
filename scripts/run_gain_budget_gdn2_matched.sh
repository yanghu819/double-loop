#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARM_SCRIPT="$REPO_ROOT/scripts/run_gain_budget_gdn2_arm.sh"
PERSIST_ROOT="${PERSIST_ROOT:-/huyang2/double-loop}"
RUNS_ROOT="${RUNS_ROOT:-$PERSIST_ROOT/runs}"
REAL_PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
MONITOR_INTERVAL_SECONDS="${MONITOR_INTERVAL_SECONDS:-2}"
PID_WAIT_SECONDS="${PID_WAIT_SECONDS:-180}"
TERM_GRACE_SECONDS="${TERM_GRACE_SECONDS:-20}"

if [[ "${CUDA_VISIBLE_DEVICES:-0}" != "0" ]]; then
  printf 'P-GAIN-001 matched run requires CUDA_VISIBLE_DEVICES=0.\n' >&2
  exit 2
fi
if [[ ! -x "$REAL_PYTHON_BIN" ]]; then
  printf 'CUDA Python is not executable: %s\n' "$REAL_PYTHON_BIN" >&2
  exit 3
fi
if [[ ! -x "$ARM_SCRIPT" ]]; then
  printf 'Gain-Budget arm launcher is not executable: %s\n' "$ARM_SCRIPT" >&2
  exit 4
fi
if [[ ! "$MONITOR_INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]] || \
  [[ ! "$PID_WAIT_SECONDS" =~ ^[1-9][0-9]*$ ]] || \
  [[ ! "$TERM_GRACE_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  printf 'Monitor, PID wait, and TERM grace intervals must be positive integers.\n' >&2
  exit 5
fi

export CUDA_VISIBLE_DEVICES=0
export PERSIST_ROOT
export RUNS_ROOT

GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
MATCH_TIMESTAMP="${MATCH_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
MATCH_ID="${MATCH_ID:-gain-budget-gdn2-matched-$MATCH_TIMESTAMP-${GIT_SHA:0:7}}"
CONTROL_RUN_NAME="${CONTROL_RUN_NAME:-${MATCH_ID}-external-identity}"
CANDIDATE_RUN_NAME="${CANDIDATE_RUN_NAME:-${MATCH_ID}-decay-funded}"
CONTROL_RUN_DIR="$RUNS_ROOT/$CONTROL_RUN_NAME"
CANDIDATE_RUN_DIR="$RUNS_ROOT/$CANDIDATE_RUN_NAME"
COMPARISON_DIR="${GAIN_BUDGET_COMPARISON_DIR:-$RUNS_ROOT/$MATCH_ID-comparison}"
EXACT_TRAIN_PID_FILE="${EXACT_TRAIN_PID_FILE:-$CANDIDATE_RUN_DIR/exact_train.pid}"
CONTROL_LOG="$CONTROL_RUN_DIR/logs/run.log"
CANDIDATE_LOG="$CANDIDATE_RUN_DIR/logs/run.log"

for path in "$CONTROL_RUN_DIR" "$CANDIDATE_RUN_DIR" "$COMPARISON_DIR"; do
  if [[ -e "$path" ]]; then
    printf 'Refusing to reuse matched-run path: %s\n' "$path" >&2
    exit 6
  fi
done
if [[ -e "$EXACT_TRAIN_PID_FILE" ]]; then
  printf 'Refusing to reuse exact training PID file: %s\n' \
    "$EXACT_TRAIN_PID_FILE" >&2
  exit 6
fi

extract_metric_at_step() {
  local log_path="$1"
  local wanted_step="$2"
  local wanted_key="$3"
  [[ -f "$log_path" ]] || return 0
  awk -v wanted_step="$wanted_step" -v wanted_key="$wanted_key" '
    {
      row_step = ""
      row_value = ""
      for (field_i = 1; field_i <= NF; field_i++) {
        if ($field_i ~ /^step=/) {
          row_step = $field_i
          sub(/^step=/, "", row_step)
        }
        prefix = wanted_key "="
        if (index($field_i, prefix) == 1) {
          row_value = substr($field_i, length(prefix) + 1)
        }
      }
      if (row_step == wanted_step && row_value != "") {
        value = row_value
      }
    }
    END {
      if (value != "") {
        print value
      }
    }
  ' "$log_path"
}

first_diagnostic_violation() {
  local log_path="$1"
  [[ -f "$log_path" ]] || return 0
  awk '
    function token_value(key,    field_i, prefix, value) {
      prefix = key "="
      for (field_i = 1; field_i <= NF; field_i++) {
        if (index($field_i, prefix) == 1) {
          value = substr($field_i, length(prefix) + 1)
          return value
        }
      }
      return ""
    }
    function invalid_number(value, lowered) {
      lowered = tolower(value)
      return lowered ~ /nan/ || lowered ~ /inf/
    }
    {
      step = token_value("step")
      if (step == "") {
        next
      }
      infeasible = token_value("gb_infeas")
      if (infeasible != "" && (invalid_number(infeasible) || infeasible + 0 > 0)) {
        printf "gb_infeas\t%s\t%s\n", step, infeasible
        exit
      }
      bound = token_value("gb_bound")
      if (bound != "" && (invalid_number(bound) || bound + 0 > 1.001)) {
        printf "gb_bound\t%s\t%s\n", step, bound
        exit
      }
      delta = token_value("gb_delta")
      if (delta != "" && (invalid_number(delta) || delta + 0 > 0.003)) {
        printf "gb_delta\t%s\t%s\n", step, delta
        exit
      }
      clipped = token_value("gb_clip")
      if (clipped != "" && invalid_number(clipped)) {
        printf "gb_clip\t%s\t%s\n", step, clipped
        exit
      }
    }
  ' "$log_path"
}

training_pid_matches() {
  local pid="$1"
  local expected_start_ticks="$2"
  local current_pid=""
  local current_start_ticks=""
  local command_line=""

  [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ -r "$EXACT_TRAIN_PID_FILE" ]] || return 1
  read -r current_pid < "$EXACT_TRAIN_PID_FILE" || return 1
  [[ "$current_pid" == "$pid" ]] || return 1
  [[ -r "/proc/$pid/stat" && -r "/proc/$pid/cmdline" ]] || return 1
  current_start_ticks="$(awk '{print $22}' "/proc/$pid/stat" 2>/dev/null || true)"
  [[ -n "$current_start_ticks" && "$current_start_ticks" == "$expected_start_ticks" ]] || return 1
  command_line="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
  [[ "$command_line" == *"study_rwkv_futureseed_loop.py"* ]]
}

terminate_exact_training_pid() {
  local pid="$1"
  local start_ticks="$2"
  local waited=0

  if ! training_pid_matches "$pid" "$start_ticks"; then
    printf 'Refusing non-exact kill: PID %s no longer matches %s.\n' \
      "$pid" "$EXACT_TRAIN_PID_FILE" >&2
    return 1
  fi
  kill -TERM "$pid"
  while (( waited < TERM_GRACE_SECONDS )); do
    if ! training_pid_matches "$pid" "$start_ticks"; then
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
  if training_pid_matches "$pid" "$start_ticks"; then
    kill -KILL "$pid"
  fi
}

write_abort_json() {
  local reason="$1"
  local trigger_field="$2"
  local trigger_step="$3"
  local trigger_value="$4"
  local exact_pid="$5"
  local control_ce_9025="$6"
  local control_ce_9050="$7"
  local candidate_ce_9025="$8"
  local candidate_ce_9050="$9"
  local abort_path="$CANDIDATE_RUN_DIR/abort.json"

  "$REAL_PYTHON_BIN" - \
    "$abort_path" \
    "$reason" \
    "$trigger_field" \
    "$trigger_step" \
    "$trigger_value" \
    "$exact_pid" \
    "$EXACT_TRAIN_PID_FILE" \
    "$CONTROL_RUN_DIR" \
    "$CANDIDATE_RUN_DIR" \
    "$control_ce_9025" \
    "$control_ce_9050" \
    "$candidate_ce_9025" \
    "$candidate_ce_9050" <<'PY'
import json
import math
import pathlib
import sys
from datetime import datetime, timezone


def optional_float(raw: str):
    if not raw:
        return None
    value = float(raw)
    return value if math.isfinite(value) else raw


def optional_int(raw: str):
    return int(raw) if raw else None


(
    output,
    reason,
    trigger_field,
    trigger_step,
    trigger_value,
    exact_pid,
    exact_pid_file,
    control_run,
    candidate_run,
    control_9025,
    control_9050,
    candidate_9025,
    candidate_9050,
) = sys.argv[1:]
payload = {
    "status": "aborted",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "reason": reason,
    "termination": {
        "method": "exact_pid",
        "pid": optional_int(exact_pid),
        "pid_file": exact_pid_file,
    },
    "trigger": {
        "field": trigger_field or None,
        "step": optional_int(trigger_step),
        "value": optional_float(trigger_value),
    },
    "control_run": control_run,
    "candidate_run": candidate_run,
    "ce": {
        "control": {
            "9025": optional_float(control_9025),
            "9050": optional_float(control_9050),
        },
        "candidate": {
            "9025": optional_float(candidate_9025),
            "9050": optional_float(candidate_9050),
        },
    },
}
control_25 = payload["ce"]["control"]["9025"]
control_50 = payload["ce"]["control"]["9050"]
candidate_25 = payload["ce"]["candidate"]["9025"]
candidate_50 = payload["ce"]["candidate"]["9050"]
if all(isinstance(value, float) for value in (control_25, control_50, candidate_25, candidate_50)):
    payload["ce"]["candidate_minus_control"] = {
        "9025": candidate_25 - control_25,
        "9050": candidate_50 - control_50,
    }
path = pathlib.Path(output)
path.parent.mkdir(parents=True, exist_ok=True)
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
tmp.replace(path)
PY
}

find_comparator() {
  local candidate=""
  if [[ -n "${GAIN_BUDGET_COMPARATOR:-}" ]]; then
    printf '%s\n' "$GAIN_BUDGET_COMPARATOR"
    return 0
  fi
  for candidate in \
    "$REPO_ROOT/scripts/compare_gain_budget_gdn2.py" \
    "$REPO_ROOT/scripts/build_gain_budget_gdn2_comparison.py" \
    "$REPO_ROOT/scripts/compare_gain_budget_gdn2.sh"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

run_comparator_if_present() {
  local comparator=""
  local control_results=()
  local candidate_results=()
  if ! comparator="$(find_comparator)"; then
    printf 'Required Gain-Budget comparator is missing.\n' >&2
    return 22
  fi
  mapfile -t control_results < <(
    find "$CONTROL_RUN_DIR/output" -maxdepth 1 -type f \
      -name 'futureseed_loop_seed*.json' | sort
  )
  mapfile -t candidate_results < <(
    find "$CANDIDATE_RUN_DIR/output" -maxdepth 1 -type f \
      -name 'futureseed_loop_seed*.json' | sort
  )
  if (( ${#control_results[@]} != 1 || ${#candidate_results[@]} != 1 )); then
    printf 'Expected exactly one result JSON per arm; control=%s candidate=%s\n' \
      "${#control_results[@]}" "${#candidate_results[@]}" >&2
    return 21
  fi
  mkdir -p "$COMPARISON_DIR"
  export GAIN_BUDGET_CONTROL_RUN_DIR="$CONTROL_RUN_DIR"
  export GAIN_BUDGET_CANDIDATE_RUN_DIR="$CANDIDATE_RUN_DIR"
  export GAIN_BUDGET_COMPARISON_DIR="$COMPARISON_DIR"
  if [[ "$comparator" == *.py ]]; then
    "$REAL_PYTHON_BIN" "$comparator" \
      --control-result "${control_results[0]}" \
      --candidate-result "${candidate_results[0]}" \
      --out-dir "$COMPARISON_DIR"
  else
    "$comparator" \
      --control-result "${control_results[0]}" \
      --candidate-result "${candidate_results[0]}" \
      --out-dir "$COMPARISON_DIR"
  fi
}

printf 'Starting matched control: %s\n' "$CONTROL_RUN_NAME"
EXACT_TRAIN_PID_FILE= \
RUN_NAME="$CONTROL_RUN_NAME" \
  "$ARM_SCRIPT" external_identity

control_ce_9025="$(extract_metric_at_step "$CONTROL_LOG" 9025 ce)"
control_ce_9050="$(extract_metric_at_step "$CONTROL_LOG" 9050 ce)"
"$REAL_PYTHON_BIN" - "$control_ce_9025" "$control_ce_9050" <<'PY'
import math
import sys

if len(sys.argv) != 3 or not all(sys.argv[1:]):
    raise SystemExit("Control log is missing CE at step 9025 or 9050")
values = [float(value) for value in sys.argv[1:]]
if not all(math.isfinite(value) for value in values):
    raise SystemExit(f"Control CE is non-finite: {values}")
PY

printf 'Starting matched candidate: %s\n' "$CANDIDATE_RUN_NAME"
EXACT_TRAIN_PID_FILE="$EXACT_TRAIN_PID_FILE" \
RUN_NAME="$CANDIDATE_RUN_NAME" \
  "$ARM_SCRIPT" decay_funded &
candidate_launcher_pid=$!

exact_train_pid=""
exact_train_start_ticks=""
waited_for_pid=0
while (( waited_for_pid < PID_WAIT_SECONDS )); do
  if [[ -s "$EXACT_TRAIN_PID_FILE" ]]; then
    read -r exact_train_pid < "$EXACT_TRAIN_PID_FILE"
    if [[ "$exact_train_pid" =~ ^[1-9][0-9]*$ ]] && \
      [[ -r "/proc/$exact_train_pid/stat" ]]; then
      exact_train_start_ticks="$(
        awk '{print $22}' "/proc/$exact_train_pid/stat" 2>/dev/null || true
      )"
      if [[ -n "$exact_train_start_ticks" ]] && \
        training_pid_matches "$exact_train_pid" "$exact_train_start_ticks"; then
        break
      fi
    fi
  fi
  if ! kill -0 "$candidate_launcher_pid" 2>/dev/null; then
    break
  fi
  sleep 1
  waited_for_pid=$((waited_for_pid + 1))
done

if [[ -z "$exact_train_pid" ]] || \
  [[ -z "$exact_train_start_ticks" ]] || \
  ! training_pid_matches "$exact_train_pid" "$exact_train_start_ticks"; then
  if wait "$candidate_launcher_pid"; then
    candidate_status=0
  else
    candidate_status=$?
  fi
  printf 'Candidate never published a live exact training PID; status=%s.\n' \
    "$candidate_status" >&2
  write_abort_json \
    "candidate_exited_before_exact_training_pid" \
    "exit_status" \
    "" \
    "$candidate_status" \
    "" \
    "$control_ce_9025" \
    "$control_ce_9050" \
    "$(extract_metric_at_step "$CANDIDATE_LOG" 9025 ce)" \
    "$(extract_metric_at_step "$CANDIDATE_LOG" 9050 ce)"
  exit 7
fi

printf 'Monitoring candidate training PID %s via %s\n' \
  "$exact_train_pid" "$EXACT_TRAIN_PID_FILE"

cleanup_live_candidate() {
  local status=$?
  trap - EXIT INT TERM
  if training_pid_matches "$exact_train_pid" "$exact_train_start_ticks"; then
    terminate_exact_training_pid \
      "$exact_train_pid" "$exact_train_start_ticks" || true
  fi
  exit "$status"
}
trap cleanup_live_candidate EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

aborted=0
candidate_ce_9025=""
candidate_ce_9050=""
ce_gate_checked=0
while training_pid_matches "$exact_train_pid" "$exact_train_start_ticks"; do
  violation="$(first_diagnostic_violation "$CANDIDATE_LOG")"
  if [[ -n "$violation" ]]; then
    IFS=$'\t' read -r trigger_field trigger_step trigger_value <<< "$violation"
    candidate_ce_9025="$(extract_metric_at_step "$CANDIDATE_LOG" 9025 ce)"
    candidate_ce_9050="$(extract_metric_at_step "$CANDIDATE_LOG" 9050 ce)"
    terminate_exact_training_pid "$exact_train_pid" "$exact_train_start_ticks"
    write_abort_json \
      "gain_budget_numerical_guard" \
      "$trigger_field" \
      "$trigger_step" \
      "$trigger_value" \
      "$exact_train_pid" \
      "$control_ce_9025" \
      "$control_ce_9050" \
      "$candidate_ce_9025" \
      "$candidate_ce_9050"
    aborted=1
    break
  fi

  if (( ce_gate_checked == 0 )); then
    candidate_ce_9050="$(extract_metric_at_step "$CANDIDATE_LOG" 9050 ce)"
    if [[ -n "$candidate_ce_9050" ]]; then
      candidate_ce_9025="$(extract_metric_at_step "$CANDIDATE_LOG" 9025 ce)"
      if [[ -z "$candidate_ce_9025" ]]; then
        terminate_exact_training_pid "$exact_train_pid" "$exact_train_start_ticks"
        write_abort_json \
          "missing_candidate_ce_checkpoint" \
          "ce" \
          "9050" \
          "$candidate_ce_9050" \
          "$exact_train_pid" \
          "$control_ce_9025" \
          "$control_ce_9050" \
          "" \
          "$candidate_ce_9050"
        aborted=1
        break
      fi
      candidate_clip_9025="$(
        extract_metric_at_step "$CANDIDATE_LOG" 9025 gb_clip
      )"
      candidate_clip_9050="$(
        extract_metric_at_step "$CANDIDATE_LOG" 9050 gb_clip
      )"
      if [[ -z "$candidate_clip_9025" || -z "$candidate_clip_9050" ]]; then
        terminate_exact_training_pid "$exact_train_pid" "$exact_train_start_ticks"
        write_abort_json \
          "missing_candidate_clip_checkpoint" \
          "gb_clip" \
          "9050" \
          "$candidate_clip_9050" \
          "$exact_train_pid" \
          "$control_ce_9025" \
          "$control_ce_9050" \
          "$candidate_ce_9025" \
          "$candidate_ce_9050"
        aborted=1
        break
      fi
      clip_inactive="$(
        "$REAL_PYTHON_BIN" - \
          "$candidate_clip_9025" \
          "$candidate_clip_9050" <<'PY'
import math
import sys

values = [float(value) for value in sys.argv[1:]]
print(int(
    all(math.isfinite(value) for value in values)
    and all(value < 0.01 for value in values)
))
PY
      )"
      if [[ "$clip_inactive" == "1" ]]; then
        terminate_exact_training_pid "$exact_train_pid" "$exact_train_start_ticks"
        write_abort_json \
          "gain_budget_projection_inactive_below_one_percent" \
          "gb_clip" \
          "9050" \
          "$candidate_clip_9050" \
          "$exact_train_pid" \
          "$control_ce_9025" \
          "$control_ce_9050" \
          "$candidate_ce_9025" \
          "$candidate_ce_9050"
        aborted=1
        break
      fi
      read -r delta_9025 delta_9050 should_abort < <(
        "$REAL_PYTHON_BIN" - \
          "$control_ce_9025" \
          "$control_ce_9050" \
          "$candidate_ce_9025" \
          "$candidate_ce_9050" <<'PY'
import math
import sys

control_25, control_50, candidate_25, candidate_50 = map(float, sys.argv[1:])
values = (control_25, control_50, candidate_25, candidate_50)
if not all(math.isfinite(value) for value in values):
    print("nan nan 1")
else:
    delta_25 = candidate_25 - control_25
    delta_50 = candidate_50 - control_50
    print(f"{delta_25:.12g} {delta_50:.12g} {int(delta_25 > 0.15 and delta_50 > 0.15)}")
PY
      )
      ce_gate_checked=1
      if [[ "$should_abort" == "1" ]]; then
        terminate_exact_training_pid "$exact_train_pid" "$exact_train_start_ticks"
        write_abort_json \
          "candidate_ce_worse_by_more_than_0.15_at_9025_and_9050" \
          "ce_delta_candidate_minus_control" \
          "9050" \
          "$delta_9050" \
          "$exact_train_pid" \
          "$control_ce_9025" \
          "$control_ce_9050" \
          "$candidate_ce_9025" \
          "$candidate_ce_9050"
        aborted=1
        break
      fi
    fi
  fi
  sleep "$MONITOR_INTERVAL_SECONDS"
done

if wait "$candidate_launcher_pid"; then
  candidate_status=0
else
  candidate_status=$?
fi
trap - EXIT INT TERM

if (( aborted == 1 )); then
  exit 20
fi
if (( candidate_status != 0 )); then
  write_abort_json \
    "candidate_process_failed" \
    "exit_status" \
    "" \
    "$candidate_status" \
    "$exact_train_pid" \
    "$control_ce_9025" \
    "$control_ce_9050" \
    "$candidate_ce_9025" \
    "$candidate_ce_9050"
  exit "$candidate_status"
fi

comparison_status=0
if run_comparator_if_present; then
  comparison_status=0
else
  comparison_status=$?
fi
exit "$comparison_status"
