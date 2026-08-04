#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
P015_CONFIG="${P015_CONFIG:-$REPO_ROOT/configs/retrieval/zoology_rope_bidirectional_mqar.env}"
set -a
source "$P015_CONFIG"
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

ENDPOINT_REFERENCE="$PERSIST_ROOT/$ENDPOINT_REFERENCE_RUN"
OFFICIAL_REFERENCE="$PERSIST_ROOT/$OFFICIAL_REFERENCE_RUN"
PLAIN_SCORE="$ENDPOINT_REFERENCE/output/length_64/bidirectional_attention/score.json"
if [[ ! -f "$PLAIN_SCORE" || ! -f "$OFFICIAL_REFERENCE/output/score.json" ]]; then
  printf 'Frozen P-CAUSAL-010/P-CAUSAL-011 references are missing.\n' >&2
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
RUN_NAME="${RUN_NAME:-zoology-rope-bidir-mqar-${TS}-${GIT_SHA:0:7}}"
RUN_DIR="$PERSIST_ROOT/runs/$RUN_NAME"
OUT_DIR="$RUN_DIR/output"
VISUAL_DIR="$RUN_DIR/visualizations"
mkdir -p "$PERSIST_ROOT/.cache" "$PERSIST_ROOT/artifacts" \
  "$PERSIST_ROOT/models" "$PERSIST_ROOT/runs" "$OUT_DIR" "$VISUAL_DIR"

cp "$P015_CONFIG" "$RUN_DIR/launch.env"
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
    "plan": "P-CAUSAL-015",
    "git_sha": "$GIT_SHA",
    "run_name": "$RUN_NAME",
    "hypothesis": "Full attention had visibility but lacked a translation-invariant relative address for adjacent key/value binding.",
    "intervention": "parameter-free standard RoPE on Q/K of the frozen parameter-matched full noncausal SDPA",
    "data": "fixed mixed-direction MQAR, length64, four associations, exact P-CAUSAL-007 hashes",
    "model": "D128/L2/H4/head57, same tensors and initialization as frozen plain SDPA",
    "training": "batch32, 10k/1k, 30 epochs, AdamW1e-3, WD0.1, seed123",
    "registered_pass": "past>=0.90, future>=0.90, joint_exact>=0.80",
    "budget_seconds": int("$WALL_BUDGET_SEC"),
    "kill": "one arm only; no RoPE base/scale, LR, width, depth, epoch, loss, or seed rescue",
    "success_claim": "relative-address attention is a valid bidirectional ceiling for the formal FutureSeed length/cost comparison",
    "forbidden": ["FutureSeed change", "reverse scan", "selector", "search", "repair", "task rule", "sweep"],
}, indent=2, sort_keys=True) + "\n")
PY

set +e
"$PYTHON_BIN" "$REPO_ROOT/scripts/check_zoology_rope_bidirectional_mqar.py" \
  --output "$RUN_DIR/preflight.json" \
  --reference-score "$PLAIN_SCORE" \
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
    "reason": "strict CUDA/source/data/identity/directionality preflight returned nonzero",
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
  "$PYTHON_BIN" -u -m experiments.zoology_mqar.rope_bidirectional_carrier \
  --output-dir "$OUT_DIR" \
  --endpoint-reference "$ENDPOINT_REFERENCE" \
  --official-reference "$OFFICIAL_REFERENCE" \
  --max-epochs "$MAX_EPOCHS" \
  2>&1 | tee "$RUN_DIR/formal.log"
STATUS="${PIPESTATUS[0]}"
set -e

if [[ -f "$OUT_DIR/comparison.json" ]]; then
  cp "$OUT_DIR/comparison.json" "$RUN_DIR/score.json"
  "$PYTHON_BIN" "$REPO_ROOT/scripts/render_zoology_rope_bidirectional_mqar.py" \
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
    "reason": "RoPE attention missed the registered past/future/joint carrier gate",
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
