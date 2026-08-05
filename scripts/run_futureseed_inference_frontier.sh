#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
P023_CONFIG="${P023_CONFIG:-$REPO_ROOT/configs/retrieval/futureseed_inference_frontier.env}"
if [[ "$(readlink -f "$P023_CONFIG")" != \
      "$(readlink -f "$REPO_ROOT/configs/retrieval/futureseed_inference_frontier.env")" ]]; then
  printf 'P-CAUSAL-023 forbids an unregistered config: %s\n' "$P023_CONFIG" >&2
  exit 4
fi
set -a
source "$P023_CONFIG"
set +a

export CUDA_VISIBLE_DEVICES=0
export XDG_CACHE_HOME="$PERSIST_ROOT/.cache"
export UV_CACHE_DIR="$PERSIST_ROOT/.cache/uv"
export HF_HOME="$PERSIST_ROOT/.cache/huggingface"
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

assert_sha256 "$BERT_CHECKPOINT" "$BERT_CHECKPOINT_SHA256"
assert_sha256 "$BERT_MODEL_DIR/config.json" "$BERT_CONFIG_SHA256"
assert_sha256 "$BERT_MODEL_DIR/vocab.txt" "$BERT_VOCAB_SHA256"
assert_sha256 "$VALIDATION_JSON" "$VALIDATION_SHA256"
assert_sha256 "$CAUSAL_GDN2_CHECKPOINT" "$CAUSAL_GDN2_CHECKPOINT_SHA256"
assert_sha256 "$FUTURE_SEED_GDN2_CHECKPOINT" "$FUTURE_SEED_GDN2_CHECKPOINT_SHA256"
assert_sha256 "$FLA_WHEEL" "$FLA_WHEEL_SHA256"

ACTUAL_ZOOLOGY_SHA="$(git -C "$ZOOLOGY_ROOT" rev-parse HEAD)"
ZOOLOGY_STATUS="$(git -C "$ZOOLOGY_ROOT" status --short)"
if [[ "$ACTUAL_ZOOLOGY_SHA" != "$ZOOLOGY_SHA" || -n "$ZOOLOGY_STATUS" ]]; then
  printf 'Pinned Zoology checkout is not exact and clean.\n' >&2
  exit 4
fi
FLA_MARKER="$PERSIST_ROOT/.cache/fla-source-sha"
FLA_RESOLVED_ROOT="$(readlink -f "$FLA_SOURCE_ROOT")"
if [[ ! -f "$FLA_MARKER" ]] || \
   [[ "$(tr -d '\r\n' < "$FLA_MARKER")" != "$FLA_EXPECTED_SOURCE_SHA" ]] || \
   [[ "$FLA_RESOLVED_ROOT" != "$PERSIST_ROOT/.cache/fla-versions/$FLA_EXPECTED_SOURCE_SHA-"* ]]; then
  printf 'Pinned FLA activation is not exact.\n' >&2
  exit 4
fi
assert_sha256 "$FLA_RESOLVED_ROOT/fla/layers/gdn2.py" "$FLA_GDN2_SOURCE_SHA256"
if [[ "$FLA_DISABLE_BACKEND_DISPATCH" != "1" || "$FLA_CONV_BACKEND" != "triton" ]]; then
  printf 'FLA backend contract drifted.\n' >&2
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
if ! command -v timeout >/dev/null 2>&1; then
  printf 'GNU timeout is required.\n' >&2
  exit 4
fi

PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
if [[ "$PREFLIGHT_ONLY" != "0" && "$PREFLIGHT_ONLY" != "1" ]]; then
  printf 'PREFLIGHT_ONLY must be 0 or 1.\n' >&2
  exit 4
fi
if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  EFFECTIVE_REPETITIONS=1
  EFFECTIVE_WARMUP=2
  EFFECTIVE_MEASURED=2
else
  EFFECTIVE_REPETITIONS="$REPETITIONS"
  EFFECTIVE_WARMUP="$WARMUP_STEPS"
  EFFECTIVE_MEASURED="$MEASURED_STEPS"
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="${RUN_NAME:-futureseed-inference-frontier-${TS}-${GIT_SHA:0:7}}"
RUN_DIR="$PERSIST_ROOT/runs/$RUN_NAME"
RAW_DIR="$RUN_DIR/raw"
VISUAL_DIR="$RUN_DIR/visualizations"
if [[ -e "$RUN_DIR" ]]; then
  printf 'Refusing to reuse run directory: %s\n' "$RUN_DIR" >&2
  exit 6
fi
mkdir -p "$RAW_DIR" "$VISUAL_DIR"
cp "$P023_CONFIG" "$RUN_DIR/launch.env"
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_DIR/git_sha.txt"
git -C "$REPO_ROOT" status --short > "$RUN_DIR/git_status.txt"
printf '%s\n' "$ACTUAL_ZOOLOGY_SHA" > "$RUN_DIR/zoology_sha.txt"
printf '%s\n' "$FLA_RESOLVED_ROOT" > "$RUN_DIR/fla_source_root.txt"
printf '%s\n' "$$" > "$RUN_DIR/launcher_pid.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/started_at.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader > "$RUN_DIR/gpu_before.txt"
sha256sum "$BERT_CHECKPOINT" "$BERT_MODEL_DIR/config.json" \
  "$BERT_MODEL_DIR/vocab.txt" "$VALIDATION_JSON" \
  "$CAUSAL_GDN2_CHECKPOINT" "$FUTURE_SEED_GDN2_CHECKPOINT" "$FLA_WHEEL" \
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
    "plan": "P-CAUSAL-023",
    "git_sha": "$GIT_SHA",
    "run_name": "$RUN_NAME",
    "arms": ["bidirectional_bert", "causal_gdn2", "future_seed_gdn2"],
    "sequence_length": 128,
    "validation_windows": 256,
    "repetitions": int("$EFFECTIVE_REPETITIONS"),
    "warmup_steps": int("$EFFECTIVE_WARMUP"),
    "measured_steps_batch64": int("$EFFECTIVE_MEASURED"),
    "measured_steps_batch1": int("$EFFECTIVE_MEASURED") * 3,
    "preflight_only": bool(int("$PREFLIGHT_ONLY")),
    "fresh_process_per_arm_and_repetition": True,
    "retraining": False,
    "length_extrapolation": False,
    "offline_only": True,
    "reverse_scan": False,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

WORKER="$REPO_ROOT/experiments/zoology_mlm/futureseed_inference_frontier.py"
SUMMARIZER="$REPO_ROOT/scripts/summarize_futureseed_inference_frontier.py"
ORDERS=(
  "bidirectional_bert causal_gdn2 future_seed_gdn2"
  "future_seed_gdn2 bidirectional_bert causal_gdn2"
  "causal_gdn2 future_seed_gdn2 bidirectional_bert"
  "bidirectional_bert future_seed_gdn2 causal_gdn2"
  "causal_gdn2 bidirectional_bert future_seed_gdn2"
)

run_worker() {
  local arm="$1"
  local repetition="$2"
  local output="$RAW_DIR/rep${repetition}-${arm}.json"
  local checkpoint=""
  local checkpoint_sha=""
  if [[ "$arm" == "causal_gdn2" ]]; then
    checkpoint="$CAUSAL_GDN2_CHECKPOINT"
    checkpoint_sha="$CAUSAL_GDN2_CHECKPOINT_SHA256"
  elif [[ "$arm" == "future_seed_gdn2" ]]; then
    checkpoint="$FUTURE_SEED_GDN2_CHECKPOINT"
    checkpoint_sha="$FUTURE_SEED_GDN2_CHECKPOINT_SHA256"
  fi
  local args=(
    "$PYTHON_BIN" -u "$WORKER"
    --arm "$arm"
    --model-dir "$BERT_MODEL_DIR"
    --bert-checkpoint "$BERT_CHECKPOINT"
    --validation-json "$VALIDATION_JSON"
    --fla-wheel "$FLA_WHEEL"
    --output "$output"
    --repetition "$repetition"
    --warmup-steps "$EFFECTIVE_WARMUP"
    --measured-steps "$EFFECTIVE_MEASURED"
  )
  if [[ -n "$checkpoint" ]]; then
    args+=(--gdn-checkpoint "$checkpoint" --gdn-checkpoint-sha256 "$checkpoint_sha")
  fi
  if [[ "$repetition" == "0" ]]; then
    args+=(--quality)
  fi
  local elapsed=$((SECONDS - START_SECONDS))
  local remaining=$((WALL_BUDGET_SEC - elapsed))
  if ((remaining <= 0)); then
    return 124
  fi
  timeout --signal=TERM --kill-after=30 "$remaining" \
    "${args[@]}" >> "$RUN_DIR/formal.log" 2>&1
}

set +e
START_SECONDS=$SECONDS
STATUS=0
for ((repetition=0; repetition<EFFECTIVE_REPETITIONS; repetition++)); do
  read -r -a order <<< "${ORDERS[$repetition]}"
  for arm in "${order[@]}"; do
    printf 'repetition=%s arm=%s started=%s\n' \
      "$repetition" "$arm" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      | tee -a "$RUN_DIR/progress.log"
    run_worker "$arm" "$repetition"
    STATUS=$?
    if [[ "$STATUS" != "0" ]]; then
      break 2
    fi
    nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu \
      --format=csv,noheader >> "$RUN_DIR/gpu_between_workers.txt"
  done
done
if [[ "$STATUS" == "0" ]]; then
  "$PYTHON_BIN" "$SUMMARIZER" \
    --raw-dir "$RAW_DIR" \
    --output-dir "$VISUAL_DIR" \
    --expected-repetitions "$EFFECTIVE_REPETITIONS" \
    > "$RUN_DIR/summary.log" 2>&1
  STATUS=$?
fi
set -e

if [[ -f "$VISUAL_DIR/score.json" ]]; then
  cp "$VISUAL_DIR/score.json" "$RUN_DIR/score.json"
fi
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
Path(sys.argv[1]).write_text(json.dumps({
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "reason": "P-CAUSAL-023 systems run failed or exceeded its wall budget",
    "exit_status": int(sys.argv[2]),
    "scientific_failure": False,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  exit "$STATUS"
fi
printf 'completed run_dir=%s\n' "$RUN_DIR"
