#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
P017_CONFIG="${P017_CONFIG:-$REPO_ROOT/configs/retrieval/zoology_bidirectional_gdn2_ceiling.env}"
set -a
source "$P017_CONFIG"
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

if [[ "$(git -C "$ZOOLOGY_ROOT" rev-parse HEAD)" != "$ZOOLOGY_SHA" ]] || \
   [[ -n "$(git -C "$ZOOLOGY_ROOT" status --short)" ]]; then
  printf 'Pinned Zoology checkout is not exact and clean.\n' >&2
  exit 4
fi
printf '%s  %s\n' "$ZOOLOGY_MODEL_SHA256" "$ZOOLOGY_ROOT/zoology/model.py" | sha256sum --check --status
printf '%s  %s\n' "$ZOOLOGY_TRAIN_SHA256" "$ZOOLOGY_ROOT/zoology/train.py" | sha256sum --check --status
printf '%s  %s\n' "$FLA_GDN2_SOURCE_SHA256" "$FLA_SOURCE_ROOT/fla/layers/gdn2.py" | sha256sum --check --status

REFERENCE_PATH="$PERSIST_ROOT/$REFERENCE_RUN"
if [[ ! -f "$REFERENCE_PATH/score.json" ]]; then
  printf 'P-CAUSAL-016 reference run is missing.\n' >&2
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
RUN_NAME="${RUN_NAME:-zoology-bidirectional-gdn2-ceiling-${TS}-${GIT_SHA:0:7}}"
RUN_DIR="$PERSIST_ROOT/runs/$RUN_NAME"
OUT_DIR="$RUN_DIR/output"
VISUAL_DIR="$RUN_DIR/visualizations"
mkdir -p "$PERSIST_ROOT/.cache" "$PERSIST_ROOT/artifacts" "$PERSIST_ROOT/models" "$PERSIST_ROOT/runs" "$OUT_DIR" "$VISUAL_DIR"

cp "$P017_CONFIG" "$RUN_DIR/launch.env"
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_DIR/git_sha.txt"
git -C "$REPO_ROOT" status --short > "$RUN_DIR/git_status.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/started_at.txt"
printf '%s\n' "$$" > "$RUN_DIR/launcher_pid.txt"
ps -o pgid= -p "$$" | tr -d ' ' > "$RUN_DIR/launcher_pgid.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader > "$RUN_DIR/gpu_before.txt"
git -C "$REPO_ROOT" ls-files -z -- . \
  ':(exclude).cache/**' ':(exclude).venv/**' ':(exclude)artifacts/**' \
  ':(exclude)models/**' ':(exclude)repos/**' ':(exclude)runs/**' \
  | tar --null -czf "$RUN_DIR/source_snapshot.tar.gz" -C "$REPO_ROOT" --files-from -
sha256sum "$RUN_DIR/source_snapshot.tar.gz" > "$RUN_DIR/source_snapshot.sha256"

"$PYTHON_BIN" - "$RUN_DIR/config.json" <<PY
import json
from pathlib import Path
Path("$RUN_DIR/config.json").write_text(json.dumps({
    "plan": "P-CAUSAL-017",
    "git_sha": "$GIT_SHA",
    "run_name": "$RUN_NAME",
    "hypothesis": "An explicit forward-plus-reverse GDN2 should open the same L512 future-retrieval carrier; native FutureSeed should match its quality with fewer recurrent streams and lower systems cost.",
    "prediction": "bidirectional past/future >=0.95 and joint exact >=0.90; FutureSeed future within0.03 and joint within0.05; FS throughput >=1.25x and peak memory <=0.80x",
    "data": "exact frozen P-CAUSAL-016 L512 directional MQAR, K4, 10k/1k",
    "model": "D128/L2/H4/D32 official-FLA GDN2",
    "baseline": "two independent forward/reverse official GDN2 streams per layer with concatenate-linear fusion",
    "training": "batch32, ten epochs, AdamW1e-3, WD0.1, cosine, seed123",
    "benchmark": "three fresh processes per arm, rotated order, three compile warmups plus five benchmark warmups and 200 measured forward-backward steps",
    "budget_seconds": int("$WALL_BUDGET_SEC"),
    "kill": "stop after fixed endpoint if the bidirectional carrier misses 0.95 past/future or 0.90 joint; no rescue",
    "success_claim": "FutureSeed approximates explicit bidirectional recurrence at lower recurrent compute and memory, without reverse scanning",
    "forbidden": ["call reverse scan FutureSeed", "task rule", "selector", "search", "repair", "seed/LR/epoch/width/depth/loss rescue"],
}, indent=2, sort_keys=True) + "\n")
PY

set +e
"$PYTHON_BIN" "$REPO_ROOT/scripts/check_zoology_bidirectional_gdn2_ceiling.py" \
  --output "$RUN_DIR/preflight.json" --reference-run "$REFERENCE_PATH" \
  2>&1 | tee "$RUN_DIR/preflight.log"
STATUS="${PIPESTATUS[0]}"
set -e

START_SECONDS="$(date +%s)"
run_timed() {
  local label="$1"
  shift
  local elapsed remaining
  elapsed=$(( $(date +%s) - START_SECONDS ))
  remaining=$(( WALL_BUDGET_SEC - elapsed ))
  if (( remaining <= 0 )); then
    return 124
  fi
  printf '\n=== %s remaining=%ss ===\n' "$label" "$remaining" | tee -a "$RUN_DIR/formal.log"
  set +e
  timeout --signal=TERM --kill-after=30 "$remaining" "$@" 2>&1 | tee -a "$RUN_DIR/formal.log"
  local command_status="${PIPESTATUS[0]}"
  set -e
  return "$command_status"
}

if [[ "$STATUS" == "0" ]]; then
  run_timed "train explicit_bidirectional_gdn2" \
    "$PYTHON_BIN" -u -m experiments.zoology_mqar.bidirectional_gdn2_ceiling train \
    --output-dir "$OUT_DIR" || STATUS="$?"
fi

if [[ "$STATUS" == "0" ]]; then
  schedule=(
    "1 causal_gdn2" "1 future_seed_gdn2" "1 explicit_bidirectional_gdn2"
    "2 future_seed_gdn2" "2 explicit_bidirectional_gdn2" "2 causal_gdn2"
    "3 explicit_bidirectional_gdn2" "3 causal_gdn2" "3 future_seed_gdn2"
  )
  for spec in "${schedule[@]}"; do
    read -r replicate arm <<<"$spec"
    output="$OUT_DIR/benchmarks/$arm/rep${replicate}.json"
    if run_timed "benchmark rep=$replicate arm=$arm" \
      "$PYTHON_BIN" -u -m experiments.zoology_mqar.bidirectional_gdn2_ceiling benchmark \
      --output "$output" --arm "$arm" --replicate "$replicate" \
      --warmup-steps "$BENCHMARK_WARMUP_STEPS" \
      --measured-steps "$BENCHMARK_MEASURED_STEPS"; then
      :
    else
      STATUS="$?"
      break
    fi
  done
fi

if [[ "$STATUS" == "0" ]]; then
  run_timed "aggregate" \
    "$PYTHON_BIN" -u -m experiments.zoology_mqar.bidirectional_gdn2_ceiling aggregate \
    --output-dir "$OUT_DIR" --reference-run "$REFERENCE_PATH" || STATUS="$?"
fi

if [[ -f "$OUT_DIR/comparison.json" ]]; then
  cp "$OUT_DIR/comparison.json" "$RUN_DIR/score.json"
  "$PYTHON_BIN" "$REPO_ROOT/scripts/render_zoology_bidirectional_gdn2_ceiling.py" \
    --output-dir "$OUT_DIR" --visual-dir "$VISUAL_DIR" \
    2>&1 | tee "$RUN_DIR/visualizations.log"
fi

if [[ "$STATUS" == "0" && -f "$RUN_DIR/score.json" ]]; then
  set +e
  "$PYTHON_BIN" - "$RUN_DIR/score.json" <<'PY'
import json, sys
score = json.load(open(sys.argv[1]))
raise SystemExit(0 if score["registered_gate"]["passed"] else 2)
PY
  STATUS="$?"
  set -e
fi

if [[ "$STATUS" != "0" ]]; then
  "$PYTHON_BIN" - "$RUN_DIR/abort.json" "$STATUS" <<'PY'
import json, sys
from datetime import datetime, timezone
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "reason": "P-CAUSAL-017 integrity, runtime, or preregistered scientific gate returned nonzero",
    "exit_status": int(sys.argv[2]),
    "scientific_failure": int(sys.argv[2]) == 2,
}, indent=2, sort_keys=True) + "\n")
PY
fi

date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/completed_at.txt"
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader > "$RUN_DIR/gpu_after.txt"
printf '%s\n' "$STATUS" > "$RUN_DIR/exit_status.txt"
printf 'completed run_dir=%s status=%s\n' "$RUN_DIR" "$STATUS"
exit "$STATUS"
