#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
P010_CONFIG="${P010_CONFIG:-$REPO_ROOT/configs/retrieval/zoology_gdn2_futureseed_length_scaling.env}"
set -a
source "$P010_CONFIG"
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
if [[ "$ACTUAL_ZOOLOGY_SHA" != "$ZOOLOGY_SHA" ]]; then
  printf 'Unexpected Zoology SHA: %s != %s\n' "$ACTUAL_ZOOLOGY_SHA" "$ZOOLOGY_SHA" >&2
  exit 4
fi
ACTUAL_FLA_SHA="$(git -C "$FLA_SOURCE_ROOT" rev-parse HEAD)"
if [[ "$ACTUAL_FLA_SHA" != "$FLA_EXPECTED_SOURCE_SHA" ]]; then
  printf 'Unexpected FLA SHA: %s != %s\n' "$ACTUAL_FLA_SHA" "$FLA_EXPECTED_SOURCE_SHA" >&2
  exit 4
fi

GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
SOURCE_STATUS="$(git -C "$REPO_ROOT" status --short -- . \
  ':(exclude).cache' ':(exclude).venv' ':(exclude)artifacts' \
  ':(exclude)models' ':(exclude)runs')"
if [[ -n "$SOURCE_STATUS" ]]; then
  printf 'Refusing to run dirty source:\n%s\n' "$SOURCE_STATUS" >&2
  exit 5
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="${RUN_NAME:-zoology-mqar-length-endpoints-${TS}-${GIT_SHA:0:7}}"
RUN_DIR="$PERSIST_ROOT/runs/$RUN_NAME"
OUT_DIR="$RUN_DIR/output"
VISUAL_DIR="$RUN_DIR/visualizations"
mkdir -p "$PERSIST_ROOT/.cache" "$PERSIST_ROOT/artifacts" \
  "$PERSIST_ROOT/models" "$PERSIST_ROOT/runs" "$OUT_DIR" "$VISUAL_DIR"

cp "$P010_CONFIG" "$RUN_DIR/launch.env"
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
sha256sum "$RUN_DIR/source_snapshot.tar.gz" > "$RUN_DIR/source_snapshot.sha256"

IFS=',' read -r -a LENGTH_ARGS <<< "$SEQUENCE_LENGTHS"
"$PYTHON_BIN" - "$RUN_DIR/config.json" "$GIT_SHA" "$RUN_NAME" <<PY
import json
from pathlib import Path

Path("$RUN_DIR/config.json").write_text(json.dumps({
    "plan": "P-CAUSAL-010",
    "git_sha": "$GIT_SHA",
    "run_name": "$RUN_NAME",
    "sequence_lengths": [int(value) for value in "$SEQUENCE_LENGTHS".split(",")],
    "num_kv_pairs": int("$NUM_KV_PAIRS"),
    "max_epochs": int("$MAX_EPOCHS"),
    "batch_size": int("$BATCH_SIZE"),
    "train_examples_per_arm": 10000,
    "validation_examples_per_arm": 1000,
    "seed": 123,
    "arms": ["causal_gdn2", "future_seed_gdn2", "bidirectional_attention"],
    "future_seed": "cross-layer terminal recurrent state seeding; no reverse scan",
    "loops": "not used",
}, indent=2, sort_keys=True) + "\n")
PY

if [[ "${P010_SKIP_PREFLIGHT:-0}" != "1" ]]; then
  set +e
  "$PYTHON_BIN" "$REPO_ROOT/scripts/check_zoology_mqar_length_scaling.py" \
    --output "$RUN_DIR/preflight.json" \
    2>&1 | tee "$RUN_DIR/preflight.log"
  PREFLIGHT_STATUS="${PIPESTATUS[0]}"
  set -e
  if [[ "$PREFLIGHT_STATUS" != "0" ]]; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/completed_at.txt"
    nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu \
      --format=csv,noheader > "$RUN_DIR/gpu_after.txt"
    printf '%s\n' "$PREFLIGHT_STATUS" > "$RUN_DIR/exit_status.txt"
    "$PYTHON_BIN" - "$RUN_DIR/abort.json" "$PREFLIGHT_STATUS" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "reason": "strict CUDA preflight returned nonzero",
    "exit_status": int(sys.argv[2]),
    "scientific_failure": False,
}, indent=2, sort_keys=True) + "\n")
PY
    exit "$PREFLIGHT_STATUS"
  fi
fi

set +e
timeout --signal=TERM --kill-after=30 "$WALL_BUDGET_SEC" \
  "$PYTHON_BIN" -u -m experiments.zoology_mqar.length_scaling \
  --output-dir "$OUT_DIR" \
  --sequence-lengths "${LENGTH_ARGS[@]}" \
  --num-kv-pairs "$NUM_KV_PAIRS" \
  --max-epochs "$MAX_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --enforce-length64-gate \
  2>&1 | tee "$RUN_DIR/formal.log"
STATUS="${PIPESTATUS[0]}"
set -e

date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/completed_at.txt"
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu \
  --format=csv,noheader > "$RUN_DIR/gpu_after.txt"
printf '%s\n' "$STATUS" > "$RUN_DIR/exit_status.txt"
if [[ "$STATUS" != "0" ]]; then
  "$PYTHON_BIN" - "$RUN_DIR/abort.json" "$STATUS" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

status = int(sys.argv[2])
Path(sys.argv[1]).write_text(json.dumps({
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "reason": (
        "registered wall budget exceeded"
        if status == 124
        else "preflighted endpoint experiment returned nonzero"
    ),
    "exit_status": status,
    "scientific_failure": status != 124,
}, indent=2, sort_keys=True) + "\n")
PY
  exit "$STATUS"
fi

cp "$OUT_DIR/comparison.json" "$RUN_DIR/score.json"
"$PYTHON_BIN" "$REPO_ROOT/scripts/render_zoology_mqar_length_scaling.py" \
  --output-dir "$OUT_DIR" \
  --visual-dir "$VISUAL_DIR" \
  2>&1 | tee "$RUN_DIR/visualizations.log"
printf 'completed run_dir=%s\n' "$RUN_DIR"
