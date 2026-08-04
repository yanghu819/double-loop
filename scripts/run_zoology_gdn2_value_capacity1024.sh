#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
P014_CONFIG="${P014_CONFIG:-$REPO_ROOT/configs/retrieval/zoology_gdn2_value_capacity1024.env}"
set -a
source "$P014_CONFIG"
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
if [[ "$(git -C "$ZOOLOGY_ROOT" rev-parse HEAD)" != "$ZOOLOGY_SHA" ]]; then
  printf 'Unexpected Zoology SHA.\n' >&2
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
REFERENCE_PATH="$REPO_ROOT/$REFERENCE_RUN"
if [[ ! -f "$REFERENCE_PATH/score.json" ]]; then
  printf 'Frozen D128 reference is missing: %s\n' "$REFERENCE_PATH" >&2
  exit 6
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="${RUN_NAME:-zoology-gdn2-value-capacity1024-${TS}-${GIT_SHA:0:7}}"
RUN_DIR="$PERSIST_ROOT/runs/$RUN_NAME"
OUT_DIR="$RUN_DIR/output"
VISUAL_DIR="$RUN_DIR/visualizations"
mkdir -p "$PERSIST_ROOT/.cache" "$PERSIST_ROOT/artifacts" \
  "$PERSIST_ROOT/models" "$PERSIST_ROOT/runs" "$OUT_DIR" "$VISUAL_DIR"

cp "$P014_CONFIG" "$RUN_DIR/launch.env"
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

"$PYTHON_BIN" - "$RUN_DIR/config.json" <<PY
import json
from pathlib import Path

Path("$RUN_DIR/config.json").write_text(json.dumps({
    "plan": "P-CAUSAL-014",
    "git_sha": "$GIT_SHA",
    "run_name": "$RUN_NAME",
    "sequence_length": 1024,
    "num_kv_pairs": 4,
    "max_epochs": int("$MAX_EPOCHS"),
    "batch_size": int("$BATCH_SIZE"),
    "train_examples_per_candidate_arm": 10000,
    "validation_examples_per_arm": 1000,
    "seed": 123,
    "reference_model": "official-FLA GDN2 D128/L2/H4/D32 expand-v1",
    "candidate_model": "official-FLA GDN2 D128/L2/H4/D32 expand-v2",
    "candidate_arms": ["causal_gdn2_expandv2", "future_seed_gdn2_expandv2"],
    "state_values_per_layer": {"reference": 4096, "candidate": 8192},
    "future_seed": "native cross-layer terminal-state seeding; no reverse scan",
}, indent=2, sort_keys=True) + "\n")
PY

set +e
"$PYTHON_BIN" "$REPO_ROOT/scripts/check_zoology_gdn2_value_capacity.py" \
  --output "$RUN_DIR/preflight.json" \
  2>&1 | tee "$RUN_DIR/preflight.log"
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
    "reason": "strict expand-v2 CUDA/source/data/state preflight returned nonzero",
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

set +e
timeout --signal=TERM --kill-after=30 "$WALL_BUDGET_SEC" \
  "$PYTHON_BIN" -u -m experiments.zoology_mqar.gdn2_value_capacity_endpoint \
  --output-dir "$OUT_DIR" --reference-run "$REFERENCE_PATH" \
  --max-epochs "$MAX_EPOCHS" --batch-size "$BATCH_SIZE" \
  2>&1 | tee "$RUN_DIR/formal.log"
STATUS="${PIPESTATUS[0]}"
set -e
if [[ -f "$OUT_DIR/comparison.json" ]]; then
  cp "$OUT_DIR/comparison.json" "$RUN_DIR/score.json"
  "$PYTHON_BIN" "$REPO_ROOT/scripts/render_zoology_gdn2_value_capacity.py" \
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
    "reason": "D128 expand-v2 state-only strong gate missed",
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
    "reason": "wall budget exceeded" if status == 124 else "formal run returned nonzero",
    "exit_status": status,
    "scientific_failure": status != 124,
}, indent=2, sort_keys=True) + "\n")
PY
fi
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/completed_at.txt"
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu \
  --format=csv,noheader > "$RUN_DIR/gpu_after.txt"
printf '%s\n' "$STATUS" > "$RUN_DIR/exit_status.txt"
printf 'completed run_dir=%s status=%s\n' "$RUN_DIR" "$STATUS"
exit "$STATUS"
