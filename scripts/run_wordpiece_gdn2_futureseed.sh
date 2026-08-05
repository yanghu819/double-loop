#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORDPIECE_CONFIG="${WORDPIECE_CONFIG:-$REPO_ROOT/configs/retrieval/wordpiece_gdn2_futureseed.env}"
P019_CONFIG="$REPO_ROOT/configs/retrieval/wordpiece_gdn2_futureseed.env"
P020_CONFIG="$REPO_ROOT/configs/retrieval/wordpiece_gdn2_depth_futureseed.env"
P021_CONFIG="$REPO_ROOT/configs/retrieval/wordpiece_gdn2_data_diversity.env"
P022_CONFIG="$REPO_ROOT/configs/retrieval/wordpiece_gdn2_joint_scale.env"
RESOLVED_CONFIG="$(readlink -f "$WORDPIECE_CONFIG")"
if [[ "$RESOLVED_CONFIG" != "$(readlink -f "$P019_CONFIG")" && \
      "$RESOLVED_CONFIG" != "$(readlink -f "$P020_CONFIG")" && \
      "$RESOLVED_CONFIG" != "$(readlink -f "$P021_CONFIG")" && \
      "$RESOLVED_CONFIG" != "$(readlink -f "$P022_CONFIG")" ]]; then
  printf 'WordPiece FutureSeed forbids an unregistered launch config: %s\n' "$WORDPIECE_CONFIG" >&2
  exit 4
fi
set -a
source "$WORDPIECE_CONFIG"
set +a

TRAIN_SOURCE="${TRAIN_SOURCE:-$DATA_DIR/train.json}"
VALIDATION_SOURCE="${VALIDATION_SOURCE:-$DATA_DIR/validation.json}"
DATA_MANIFEST="${DATA_MANIFEST:-$DATA_DIR/manifest.json}"

export CUDA_VISIBLE_DEVICES=0
export XDG_CACHE_HOME="$PERSIST_ROOT/.cache"
export UV_CACHE_DIR="$PERSIST_ROOT/.cache/uv"
export HF_HOME="$PERSIST_ROOT/.cache/huggingface"
export HF_DATASETS_CACHE="$PERSIST_ROOT/.cache/huggingface/datasets"
export TORCH_HOME="$PERSIST_ROOT/.cache/torch"
export TORCH_EXTENSIONS_DIR="$PERSIST_ROOT/.cache/torch_extensions"
export TRITON_CACHE_DIR="$PERSIST_ROOT/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PERSIST_ROOT/.cache/torchinductor"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export DISABLE_MLFLOW_INTEGRATION=true
export PYTHONPATH="$REPO_ROOT:$ZOOLOGY_ROOT:$FLA_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/opt/conda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

EXPECTED_GPU_UUID="GPU-53e9f3b4-2966-65d3-6614-09c540921519"
GPU_COUNT="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l | tr -d ' ')"
GPU_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
if [[ "$GPU_COUNT" != "1" || "$GPU_UUID" != "$EXPECTED_GPU_UUID" ]]; then
  printf 'Unexpected visible GPU set: count=%s uuid=%s\n' "$GPU_COUNT" "$GPU_UUID" >&2
  exit 3
fi

sha256_file() {
  sha256sum "$1" | cut -d' ' -f1
}

assert_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256_file "$path")"
  if [[ "$actual" != "$expected" ]]; then
    printf 'SHA256 mismatch for %s: %s != %s\n' "$path" "$actual" "$expected" >&2
    exit 4
  fi
}

assert_sha256 "$CHECKPOINT_FILE" "$CHECKPOINT_SHA256"
assert_sha256 "$MODEL_DIR/config.json" "$MODEL_CONFIG_SHA256"
assert_sha256 "$MODEL_DIR/vocab.txt" "$VOCAB_SHA256"
assert_sha256 "$FLA_WHEEL" "$FLA_WHEEL_SHA256"
assert_sha256 "$TRAIN_SOURCE" "$TRAIN_SHA256"
assert_sha256 "$VALIDATION_SOURCE" "$VALIDATION_SHA256"
assert_sha256 "$DATA_MANIFEST" "$DATA_MANIFEST_SHA256"

ACTUAL_ZOOLOGY_SHA="$(git -C "$ZOOLOGY_ROOT" rev-parse HEAD)"
ZOOLOGY_STATUS="$(git -C "$ZOOLOGY_ROOT" status --short)"
if [[ "$ACTUAL_ZOOLOGY_SHA" != "$ZOOLOGY_SHA" || -n "$ZOOLOGY_STATUS" ]]; then
  printf 'Pinned Zoology checkout is not exact and clean.\n' >&2
  exit 4
fi

FLA_MARKER="$PERSIST_ROOT/.cache/fla-source-sha"
if [[ ! -d "$FLA_SOURCE_ROOT" || ! -f "$FLA_MARKER" ]]; then
  printf 'Pinned FLA source or provenance marker is missing.\n' >&2
  exit 4
fi
FLA_RESOLVED_ROOT="$(readlink -f "$FLA_SOURCE_ROOT")"
FLA_MARKER_SHA="$(tr -d '\r\n' < "$FLA_MARKER")"
if [[ "$FLA_MARKER_SHA" != "$FLA_EXPECTED_SOURCE_SHA" ]] || \
   [[ "$FLA_RESOLVED_ROOT" != "$PERSIST_ROOT/.cache/fla-versions/$FLA_EXPECTED_SOURCE_SHA-"* ]]; then
  printf 'Pinned FLA activation is not exact: marker=%s root=%s\n' \
    "$FLA_MARKER_SHA" "$FLA_RESOLVED_ROOT" >&2
  exit 4
fi
assert_sha256 "$FLA_RESOLVED_ROOT/fla/layers/gdn2.py" "$FLA_GDN2_SOURCE_SHA256"
if [[ "$FLA_DISABLE_BACKEND_DISPATCH" != "1" || "$FLA_CONV_BACKEND" != "triton" ]]; then
  printf 'FLA backend contract is not pinned to dispatch-disabled Triton.\n' >&2
  exit 4
fi

RUNNER="$REPO_ROOT/experiments/zoology_mlm/wordpiece_futureseed_mlm.py"
RENDER_SCRIPT="$REPO_ROOT/scripts/render_wordpiece_futureseed_mlm.py"
if [[ ! -f "$RUNNER" || ! -f "$RENDER_SCRIPT" ]]; then
  printf 'Pinned runner or render script is missing.\n' >&2
  exit 4
fi
if ! command -v timeout >/dev/null 2>&1; then
  printf 'GNU timeout is required.\n' >&2
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
RUN_NAME="${RUN_NAME:-wordpiece-gdn2-futureseed-${TS}-${GIT_SHA:0:7}}"
RUN_DIR="$PERSIST_ROOT/runs/$RUN_NAME"
OUT_DIR="$RUN_DIR/output"
CHECKPOINT_DIR="$PERSIST_ROOT/models/$RUN_NAME"
VISUAL_DIR="$RUN_DIR/visualizations"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
if [[ -e "$RUN_DIR" ]]; then
  printf 'Refusing to reuse run directory: %s\n' "$RUN_DIR" >&2
  exit 6
fi
mkdir -p "$PERSIST_ROOT/.cache" "$PERSIST_ROOT/artifacts" \
  "$PERSIST_ROOT/models" "$PERSIST_ROOT/runs" "$OUT_DIR" \
  "$CHECKPOINT_DIR" "$VISUAL_DIR"

cp "$WORDPIECE_CONFIG" "$RUN_DIR/launch.env"
sha256sum "$RUN_DIR/launch.env" > "$RUN_DIR/launch.env.sha256"
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_DIR/git_sha.txt"
git -C "$REPO_ROOT" status --short > "$RUN_DIR/git_status.txt"
printf '%s\n' "$ACTUAL_ZOOLOGY_SHA" > "$RUN_DIR/zoology_sha.txt"
printf '%s\n' "$FLA_RESOLVED_ROOT" > "$RUN_DIR/fla_source_root.txt"
printf '%s\n' "$$" > "$RUN_DIR/launcher_pid.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/started_at.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader > "$RUN_DIR/gpu_before.txt"
sha256sum \
  "$CHECKPOINT_FILE" \
  "$MODEL_DIR/config.json" \
  "$MODEL_DIR/vocab.txt" \
  "$TRAIN_SOURCE" \
  "$VALIDATION_SOURCE" \
  "$DATA_MANIFEST" \
  > "$RUN_DIR/input_assets.sha256"
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
    "plan": "$WORDPIECE_EXPERIMENT_PLAN",
    "git_sha": "$GIT_SHA",
    "run_name": "$RUN_NAME",
    "model": "official-FLA GDN2 D128/L${WORDPIECE_MODEL_LAYERS}/H4/D32 with common BERT lexical endpoints",
    "model_layers": int("$WORDPIECE_MODEL_LAYERS"),
    "active_future_seed_routes": int("$WORDPIECE_MODEL_LAYERS") - 1,
    "arms": ["causal_gdn2", "future_seed_gdn2"],
    "sequence_length": int("$SEQUENCE_LENGTH"),
    "train_windows": int("$TRAIN_WINDOWS"),
    "validation_windows": int("$VALIDATION_WINDOWS"),
    "mask_probability": float("$MASK_PROBABILITY"),
    "train_epochs": int("$TRAIN_EPOCHS"),
    "max_steps": int("$MAX_STEPS"),
    "train_batch": int("$TRAIN_BATCH"),
    "eval_batch": int("$EVAL_BATCH"),
    "learning_rate": float("$CORE_LR"),
    "weight_decay": float("$WEIGHT_DECAY"),
    "seed": int("$SEED"),
    "train_source": "$TRAIN_SOURCE",
    "validation_source": "$VALIDATION_SOURCE",
    "data_manifest": "$DATA_MANIFEST",
    "wall_budget_sec": int("$WALL_BUDGET_SEC"),
    "preflight_only": bool(int("$PREFLIGHT_ONLY")),
    "train_input_tokens_per_arm": (
        int("$MAX_STEPS") * int("$TRAIN_BATCH") * int("$SEQUENCE_LENGTH")
    ),
    "offline_only": True,
    "reverse_scan": False,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

EXTRA_ARGS=()
if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  EXTRA_ARGS+=(--preflight-only)
elif [[ "$PREFLIGHT_ONLY" != "0" ]]; then
  printf 'PREFLIGHT_ONLY must be 0 or 1, got %s\n' "$PREFLIGHT_ONLY" >&2
  exit 4
fi

set +e
timeout --signal=TERM --kill-after=30 "$WALL_BUDGET_SEC" \
  "$PYTHON_BIN" -u -m experiments.zoology_mlm.wordpiece_futureseed_mlm \
  --output-dir "$OUT_DIR" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --model-dir "$MODEL_DIR" \
  --checkpoint-file "$CHECKPOINT_FILE" \
  --train-json "$TRAIN_SOURCE" \
  --validation-json "$VALIDATION_SOURCE" \
  --sequence-length "$SEQUENCE_LENGTH" \
  --train-windows "$TRAIN_WINDOWS" \
  --validation-windows "$VALIDATION_WINDOWS" \
  --mask-probability "$MASK_PROBABILITY" \
  --train-epochs "$TRAIN_EPOCHS" \
  --max-steps "$MAX_STEPS" \
  --train-batch "$TRAIN_BATCH" \
  --eval-batch "$EVAL_BATCH" \
  --learning-rate "$CORE_LR" \
  --weight-decay "$WEIGHT_DECAY" \
  --seed "$SEED" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$RUN_DIR/formal.log"
STATUS="${PIPESTATUS[0]}"
set -e

FAILURE_REASON=""
if [[ "$STATUS" == "124" ]]; then
  FAILURE_REASON="wall budget exceeded"
elif [[ "$STATUS" == "2" ]]; then
  FAILURE_REASON="preregistered scientific gate did not pass"
elif [[ "$STATUS" != "0" ]]; then
  FAILURE_REASON="formal run returned nonzero"
fi

if [[ -f "$OUT_DIR/comparison.json" ]]; then
  cp "$OUT_DIR/comparison.json" "$RUN_DIR/score.json"
  set +e
  "$PYTHON_BIN" "$RENDER_SCRIPT" \
    --comparison "$OUT_DIR/comparison.json" \
    --causal-cases "$OUT_DIR/causal_gdn2/cases.json" \
    --future-seed-cases "$OUT_DIR/future_seed_gdn2/cases.json" \
    --output-dir "$VISUAL_DIR" \
    2>&1 | tee "$RUN_DIR/visualizations.log"
  RENDER_STATUS="${PIPESTATUS[0]}"
  set -e
  printf '%s\n' "$RENDER_STATUS" > "$RUN_DIR/render_exit_status.txt"
  if [[ "$RENDER_STATUS" != "0" ]]; then
    STATUS=7
    FAILURE_REASON="render script returned nonzero"
  fi
elif [[ "$STATUS" == "0" && "$PREFLIGHT_ONLY" != "1" ]]; then
  STATUS=6
  FAILURE_REASON="formal run completed without comparison.json"
fi

if [[ "$STATUS" != "0" ]]; then
  "$PYTHON_BIN" - "$RUN_DIR/abort.json" "$STATUS" "$FAILURE_REASON" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

status = int(sys.argv[2])
Path(sys.argv[1]).write_text(json.dumps({
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "reason": sys.argv[3],
    "exit_status": status,
    "scientific_failure": status == 2,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
fi

date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/completed_at.txt"
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu \
  --format=csv,noheader > "$RUN_DIR/gpu_after.txt"
printf '%s\n' "$STATUS" > "$RUN_DIR/exit_status.txt"
printf 'completed run_dir=%s status=%s\n' "$RUN_DIR" "$STATUS"
exit "$STATUS"
