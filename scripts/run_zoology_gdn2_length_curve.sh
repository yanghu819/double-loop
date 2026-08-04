#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
P016_CONFIG="${P016_CONFIG:-$REPO_ROOT/configs/retrieval/zoology_gdn2_futureseed_length_curve.env}"
set -a
source "$P016_CONFIG"
set +a

export CUDA_VISIBLE_DEVICES=0
export XDG_CACHE_HOME="$PERSIST_ROOT/.cache"
export UV_CACHE_DIR="$PERSIST_ROOT/.cache/uv"
export TORCH_HOME="$PERSIST_ROOT/.cache/torch"
export TORCH_EXTENSIONS_DIR="$PERSIST_ROOT/.cache/torch_extensions"
export PYTHONPATH="$REPO_ROOT:$ZOOLOGY_ROOT:$FLA_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/opt/conda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

EXPECTED_GPU_UUID="GPU-53e9f3b4-2966-65d3-6614-09c540921519"
GPU_COUNT="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l | tr -d ' ')"
GPU_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
if [[ "$GPU_COUNT" != "1" || "$GPU_UUID" != "$EXPECTED_GPU_UUID" ]]; then
  printf 'Unexpected visible GPU set: count=%s uuid=%s\n' "$GPU_COUNT" "$GPU_UUID" >&2
  exit 3
fi

ACTUAL_ZOOLOGY_SHA="$(git -C "$ZOOLOGY_ROOT" rev-parse HEAD)"
ZOOLOGY_STATUS="$(git -C "$ZOOLOGY_ROOT" status --short)"
if [[ "$ACTUAL_ZOOLOGY_SHA" != "$ZOOLOGY_SHA" || -n "$ZOOLOGY_STATUS" ]]; then
  printf 'Pinned Zoology checkout is not exact and clean.\n' >&2
  exit 4
fi
printf '%s  %s\n' "$ZOOLOGY_MODEL_SHA256" \
  "$ZOOLOGY_ROOT/zoology/model.py" | sha256sum --check --status
printf '%s  %s\n' "$ZOOLOGY_TRAIN_SHA256" \
  "$ZOOLOGY_ROOT/zoology/train.py" | sha256sum --check --status
printf '%s  %s\n' "$FLA_GDN2_SOURCE_SHA256" \
  "$FLA_SOURCE_ROOT/fla/layers/gdn2.py" | sha256sum --check --status

LENGTH64_REFERENCE="$PERSIST_ROOT/$LENGTH64_REFERENCE_RUN"
LENGTH1024_REFERENCE="$PERSIST_ROOT/$LENGTH1024_REFERENCE_RUN"
if [[ ! -f "$LENGTH64_REFERENCE/score.json" || ! -f "$LENGTH1024_REFERENCE/score.json" ]]; then
  printf 'Frozen P-CAUSAL-010/P-CAUSAL-012 references are missing.\n' >&2
  exit 5
fi

GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
SOURCE_STATUS="$(git -C "$REPO_ROOT" status --short -- . \
  ':(exclude).cache' ':(exclude).venv' ':(exclude)artifacts' \
  ':(exclude)models' ':(exclude)runs')"
if [[ -n "$SOURCE_STATUS" ]]; then
  printf 'Refusing to run dirty source:\n%s\n' "$SOURCE_STATUS" >&2
  exit 6
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="${RUN_NAME:-zoology-gdn2-fs-length-curve-${TS}-${GIT_SHA:0:7}}"
RUN_DIR="$PERSIST_ROOT/runs/$RUN_NAME"
OUT_DIR="$RUN_DIR/output"
VISUAL_DIR="$RUN_DIR/visualizations"
mkdir -p "$PERSIST_ROOT/.cache" "$PERSIST_ROOT/artifacts" \
  "$PERSIST_ROOT/models" "$PERSIST_ROOT/runs" "$OUT_DIR" "$VISUAL_DIR"

cp "$P016_CONFIG" "$RUN_DIR/launch.env"
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_DIR/git_sha.txt"
git -C "$REPO_ROOT" status --short > "$RUN_DIR/git_status.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/started_at.txt"
printf '%s\n' "$$" > "$RUN_DIR/launcher_pid.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader > "$RUN_DIR/gpu_before.txt"
git -C "$REPO_ROOT" ls-files -z -- . \
  ':(exclude).cache/**' ':(exclude).venv/**' ':(exclude)artifacts/**' \
  ':(exclude)models/**' ':(exclude)repos/**' ':(exclude)runs/**' \
  | tar --null -czf "$RUN_DIR/source_snapshot.tar.gz" \
      -C "$REPO_ROOT" --files-from -
sha256sum "$RUN_DIR/source_snapshot.tar.gz" > "$RUN_DIR/source_snapshot.sha256"

"$PYTHON_BIN" - "$RUN_DIR/config.json" <<PY
import json
from pathlib import Path

Path("$RUN_DIR/config.json").write_text(json.dumps({
    "plan": "P-CAUSAL-016",
    "git_sha": "$GIT_SHA",
    "run_name": "$RUN_NAME",
    "hypothesis": "With four associations fixed, FutureSeed quality should degrade smoothly as irrelevant context dilutes one terminal recurrent state, while the causal future direction stays at chance.",
    "new_sequence_lengths": [128, 256, 512],
    "frozen_sequence_lengths": [64, 1024],
    "num_kv_pairs": 4,
    "model": "strict official-FLA GDN2 D128/L2/H4/D32 chunk/Triton",
    "training": "10k/1k per new length, batch32, ten epochs, AdamW1e-3, WD0.1, cosine, seed123",
    "arms": ["causal_gdn2", "future_seed_gdn2"],
    "process_isolation": "one fresh Python process per newly trained arm",
    "cost_measurement": "every arm and length reconstructed, warmed, and measured in its own fresh Python process",
    "registered_floors": {"L128": 0.95, "L256": 0.90, "L512": 0.80},
    "registered_delta": "FutureSeed future accuracy minus causal >= 0.70 at every length",
    "budget_seconds": int("$WALL_BUDGET_SEC"),
    "kill": "stop if L128 FutureSeed future accuracy <0.80 or on source/data/kernel/identity/dependency/finite-backward failure; no rescue",
    "success_claim": "native terminal-state seeding remains an effective future-context route through 512 tokens and degrades gradually through the frozen 1024-token endpoint",
    "forbidden": ["attention rescue", "reverse scan", "selector", "search", "repair", "task rule", "seed/LR/width/depth/loss sweep"],
}, indent=2, sort_keys=True) + "\n")
PY

set +e
"$PYTHON_BIN" "$REPO_ROOT/scripts/check_zoology_gdn2_length_curve.py" \
  --output "$RUN_DIR/preflight.json" 2>&1 | tee "$RUN_DIR/preflight.log"
PREFLIGHT_STATUS="${PIPESTATUS[0]}"
set -e
if [[ "$PREFLIGHT_STATUS" != "0" ]]; then
  "$PYTHON_BIN" - "$RUN_DIR/abort.json" "$PREFLIGHT_STATUS" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "reason": "strict GPU/source/data/identity/directionality preflight returned nonzero",
    "exit_status": int(sys.argv[2]),
    "scientific_failure": False,
}, indent=2, sort_keys=True) + "\n")
PY
  date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/completed_at.txt"
  nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu \
    --format=csv,noheader > "$RUN_DIR/gpu_after.txt"
  printf '%s\n' "$PREFLIGHT_STATUS" > "$RUN_DIR/exit_status.txt"
  exit "$PREFLIGHT_STATUS"
fi

START_SECONDS="$(date +%s)"
run_arm() {
  local length="$1"
  local arm="$2"
  local now remaining
  now="$(date +%s)"
  remaining=$((WALL_BUDGET_SEC - (now - START_SECONDS)))
  if (( remaining <= 0 )); then
    return 124
  fi
  printf '\n=== fresh-process arm length=%s arm=%s remaining=%ss ===\n' \
    "$length" "$arm" "$remaining" | tee -a "$RUN_DIR/formal.log"
  set +e
  timeout --signal=TERM --kill-after=30 "$remaining" \
    "$PYTHON_BIN" -u -m experiments.zoology_mqar.gdn2_length_curve arm \
    --output-dir "$OUT_DIR" --sequence-length "$length" --arm "$arm" \
    --max-epochs "$MAX_EPOCHS" --batch-size "$BATCH_SIZE" \
    2>&1 | tee -a "$RUN_DIR/formal.log"
  local status="${PIPESTATUS[0]}"
  set -e
  return "$status"
}

STATUS=0
for spec in "128 causal_gdn2" "128 future_seed_gdn2"; do
  read -r length arm <<<"$spec"
  if run_arm "$length" "$arm"; then
    :
  else
    STATUS="$?"
    break
  fi
done

if [[ "$STATUS" == "0" ]]; then
  set +e
  "$PYTHON_BIN" - "$OUT_DIR/length_128/future_seed_gdn2/score.json" <<'PY'
import json
import sys

score = json.load(open(sys.argv[1]))
raise SystemExit(0 if score["metrics"]["future"]["accuracy"] >= 0.80 else 2)
PY
  LENGTH128_STATUS="$?"
  set -e
  if [[ "$LENGTH128_STATUS" != "0" ]]; then
    STATUS=2
    cp "$OUT_DIR/length_128/future_seed_gdn2/score.json" "$RUN_DIR/score.json"
    "$PYTHON_BIN" - "$RUN_DIR/abort.json" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "reason": "L128 FutureSeed future accuracy fell below the preregistered 0.80 early kill",
    "exit_status": 2,
    "scientific_failure": True,
}, indent=2, sort_keys=True) + "\n")
PY
  fi
fi

if [[ "$STATUS" == "0" ]]; then
  for spec in "256 future_seed_gdn2" "256 causal_gdn2" \
              "512 causal_gdn2" "512 future_seed_gdn2"; do
    read -r length arm <<<"$spec"
    if run_arm "$length" "$arm"; then
      :
    else
      STATUS="$?"
      break
    fi
  done
fi

if [[ "$STATUS" == "0" ]]; then
  for spec in "64 causal_gdn2" "64 future_seed_gdn2" \
              "128 future_seed_gdn2" "128 causal_gdn2" \
              "256 causal_gdn2" "256 future_seed_gdn2" \
              "512 future_seed_gdn2" "512 causal_gdn2" \
              "1024 causal_gdn2" "1024 future_seed_gdn2"; do
    read -r length arm <<<"$spec"
    now="$(date +%s)"
    remaining=$((WALL_BUDGET_SEC - (now - START_SECONDS)))
    if (( remaining <= 0 )); then
      STATUS=124
      break
    fi
    benchmark_output="$OUT_DIR/fresh_process_benchmarks/length_${length}/${arm}.json"
    printf '\n=== fresh-process benchmark length=%s arm=%s remaining=%ss ===\n' \
      "$length" "$arm" "$remaining" | tee -a "$RUN_DIR/formal.log"
    set +e
    timeout --signal=TERM --kill-after=30 "$remaining" \
      "$PYTHON_BIN" -u -m experiments.zoology_mqar.gdn2_length_curve benchmark \
      --output "$benchmark_output" --sequence-length "$length" --arm "$arm" \
      2>&1 | tee -a "$RUN_DIR/formal.log"
    BENCHMARK_STATUS="${PIPESTATUS[0]}"
    set -e
    if [[ "$BENCHMARK_STATUS" != "0" ]]; then
      STATUS="$BENCHMARK_STATUS"
      break
    fi
  done
fi

if [[ "$STATUS" == "0" ]]; then
  set +e
  "$PYTHON_BIN" -u -m experiments.zoology_mqar.gdn2_length_curve aggregate \
    --output-dir "$OUT_DIR" \
    --length64-reference-run "$LENGTH64_REFERENCE" \
    --length1024-reference-run "$LENGTH1024_REFERENCE" \
    --max-epochs "$MAX_EPOCHS" --batch-size "$BATCH_SIZE" \
    2>&1 | tee -a "$RUN_DIR/formal.log"
  STATUS="${PIPESTATUS[0]}"
  set -e
fi

if [[ -f "$OUT_DIR/comparison.json" ]]; then
  cp "$OUT_DIR/comparison.json" "$RUN_DIR/score.json"
  "$PYTHON_BIN" "$REPO_ROOT/scripts/render_zoology_gdn2_length_curve.py" \
    --output-dir "$OUT_DIR" --visual-dir "$VISUAL_DIR" \
    2>&1 | tee "$RUN_DIR/visualizations.log"
fi

if [[ "$STATUS" == "0" ]]; then
  set +e
  "$PYTHON_BIN" - "$RUN_DIR/score.json" <<'PY'
import json
import sys

score = json.load(open(sys.argv[1]))
raise SystemExit(0 if score["registered_gate"]["passed"] else 2)
PY
  GATE_STATUS="$?"
  set -e
  if [[ "$GATE_STATUS" != "0" ]]; then
    STATUS="$GATE_STATUS"
    "$PYTHON_BIN" - "$RUN_DIR/abort.json" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "reason": "FutureSeed context-length curve missed one or more registered quality/delta floors",
    "exit_status": 2,
    "scientific_failure": True,
}, indent=2, sort_keys=True) + "\n")
PY
  fi
fi
if [[ "$STATUS" != "0" && ! -f "$RUN_DIR/abort.json" ]]; then
  "$PYTHON_BIN" - "$RUN_DIR/abort.json" "$STATUS" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

status = int(sys.argv[2])
Path(sys.argv[1]).write_text(json.dumps({
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "reason": "wall budget exceeded" if status == 124 else "formal arm returned nonzero",
    "exit_status": status,
    "scientific_failure": status not in (124,),
}, indent=2, sort_keys=True) + "\n")
PY
fi
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/completed_at.txt"
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu \
  --format=csv,noheader > "$RUN_DIR/gpu_after.txt"
printf '%s\n' "$STATUS" > "$RUN_DIR/exit_status.txt"
printf 'completed run_dir=%s status=%s\n' "$RUN_DIR" "$STATUS"
exit "$STATUS"
