#!/usr/bin/env bash
set -euo pipefail

ARM="${1:?Usage: run_established_bert_mlm.sh <bidirectional|causal>}"
if [[ "$ARM" != "bidirectional" && "$ARM" != "causal" ]]; then
  printf 'Unknown arm: %s\n' "$ARM" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
P009_CONFIG="${P009_CONFIG:-$REPO_ROOT/configs/retrieval/established_bert_mlm.env}"
set -a
source "$P009_CONFIG"
set +a

export CUDA_VISIBLE_DEVICES=0
export XDG_CACHE_HOME="$PERSIST_ROOT/.cache"
export HF_HOME="$PERSIST_ROOT/.cache/huggingface"
export HF_DATASETS_CACHE="$PERSIST_ROOT/.cache/huggingface/datasets"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export DISABLE_MLFLOW_INTEGRATION=true

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

assert_sha256 "$P009_UPSTREAM_RUN_MLM" "$P009_UPSTREAM_RUN_MLM_SHA256"
assert_sha256 "$P009_ACCURACY_METRIC_PATH" "$P009_ACCURACY_METRIC_SHA256"
assert_sha256 "$P009_TOKENIZER_DIR/vocab.txt" "$P009_TOKENIZER_VOCAB_SHA256"
assert_sha256 "$P009_TOKENIZER_DIR/config.json" "$P009_TOKENIZER_CONFIG_SHA256"
assert_sha256 "$P009_ACCELERATE_WHEEL" "$P009_ACCELERATE_WHEEL_SHA256"
assert_sha256 "$P009_DATA_DIR/train.json" "$P009_TRAIN_DATA_SHA256"
assert_sha256 "$P009_DATA_DIR/validation.json" "$P009_VALIDATION_DATA_SHA256"
assert_sha256 "$P009_DATA_DIR/manifest.json" "$P009_DATA_MANIFEST_SHA256"
assert_sha256 \
  "$REPO_ROOT/configs/retrieval/bert_mini_bidirectional.json" \
  "$P009_BIDIRECTIONAL_CONFIG_SHA256"
assert_sha256 \
  "$REPO_ROOT/configs/retrieval/bert_mini_causal.json" \
  "$P009_CAUSAL_CONFIG_SHA256"
UPSTREAM_COMMIT="$(git -C "$P009_TRANSFORMERS_REPO" rev-parse HEAD)"
if [[ "$UPSTREAM_COMMIT" != "$P009_UPSTREAM_COMMIT" ]]; then
  printf 'Unexpected Transformers commit: %s != %s\n' \
    "$UPSTREAM_COMMIT" "$P009_UPSTREAM_COMMIT" >&2
  exit 4
fi

UV_BIN="$PERSIST_ROOT/.cache/uv-bootstrap/bin/uv"
PYTHON_EXTRA="$PERSIST_ROOT/.cache/python-extra-pylib"
mkdir -p "$PYTHON_EXTRA" "$PERSIST_ROOT/.cache/huggingface" \
  "$PERSIST_ROOT/artifacts" "$PERSIST_ROOT/models" "$PERSIST_ROOT/runs"
"$UV_BIN" pip install \
  --python "$PYTHON_BIN" \
  --target "$PYTHON_EXTRA" \
  --no-deps \
  --no-index \
  --reinstall \
  "$P009_ACCELERATE_WHEEL"
export PYTHONPATH="$PYTHON_EXTRA${PYTHONPATH:+:$PYTHONPATH}"

GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
SOURCE_STATUS="$(git -C "$REPO_ROOT" status --short -- . \
  ':(exclude).cache' ':(exclude).venv' ':(exclude)artifacts' \
  ':(exclude)models' ':(exclude)runs')"
if [[ -n "$SOURCE_STATUS" ]]; then
  printf 'Refusing to run dirty source:\n%s\n' "$SOURCE_STATUS" >&2
  exit 5
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="${RUN_NAME:-established-bert-mlm-${ARM}-${TS}-${GIT_SHA:0:7}}"
RUN_DIR="$PERSIST_ROOT/runs/$RUN_NAME"
MODEL_DIR="$PERSIST_ROOT/models/$RUN_NAME"
mkdir -p "$RUN_DIR/upstream" "$MODEL_DIR"

cp "$P009_CONFIG" "$RUN_DIR/launch.env"
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_DIR/git_sha.txt"
git -C "$REPO_ROOT" status --short > "$RUN_DIR/git_status.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/started_at.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu \
  --format=csv,noheader > "$RUN_DIR/gpu_before.txt"
git -C "$REPO_ROOT" ls-files -z -- . \
  ':(exclude).cache/**' ':(exclude).venv/**' ':(exclude)artifacts/**' \
  ':(exclude)models/**' ':(exclude)repos/**' ':(exclude)runs/**' \
  | tar --null -czf "$RUN_DIR/source_snapshot.tar.gz" \
      -C "$REPO_ROOT" --files-from -

"$PYTHON_BIN" "$REPO_ROOT/scripts/materialize_offline_run_mlm.py" \
  --upstream "$P009_UPSTREAM_RUN_MLM" \
  --output "$RUN_DIR/upstream/run_mlm.py" \
  --manifest "$RUN_DIR/upstream/manifest.json"

if [[ "$ARM" == "bidirectional" ]]; then
  ARM_CONFIG="$REPO_ROOT/configs/retrieval/bert_mini_bidirectional.json"
else
  ARM_CONFIG="$REPO_ROOT/configs/retrieval/bert_mini_causal.json"
fi

if [[ "${P009_SKIP_PREFLIGHT:-0}" != "1" ]]; then
  "$PYTHON_BIN" "$REPO_ROOT/scripts/check_established_bert_mlm.py" \
    --bidirectional-config "$REPO_ROOT/configs/retrieval/bert_mini_bidirectional.json" \
    --causal-config "$REPO_ROOT/configs/retrieval/bert_mini_causal.json" \
    --official-config "$P009_TOKENIZER_DIR/config.json" \
    --tokenizer "$P009_TOKENIZER_DIR" \
    --data-manifest "$P009_DATA_DIR/manifest.json" \
    --output "$RUN_DIR/preflight.json" \
    2>&1 | tee "$RUN_DIR/preflight.log"
fi

"$PYTHON_BIN" - "$RUN_DIR/config.json" <<PY
import json
from pathlib import Path

Path("$RUN_DIR/config.json").write_text(json.dumps({
    "plan": "P-CAUSAL-009",
    "arm": "$ARM",
    "git_sha": "$GIT_SHA",
    "run_name": "$RUN_NAME",
    "model_dir": "$MODEL_DIR",
    "source": "huggingface/transformers v4.46.3 run_mlm.py",
    "model": "google/bert_uncased_L-2_H-128_A-2 architecture from scratch",
    "max_steps": int("$P009_MAX_STEPS"),
    "train_batch": int("$P009_TRAIN_BATCH"),
    "eval_batch": int("$P009_EVAL_BATCH"),
    "sequence_length": int("$P009_SEQUENCE_LENGTH"),
    "mlm_probability": float("$P009_MLM_PROBABILITY"),
    "learning_rate": float("$P009_LEARNING_RATE"),
    "seed": int("$P009_SEED"),
    "registered_input_tokens": int("$P009_MAX_STEPS") * int("$P009_TRAIN_BATCH") * int("$P009_SEQUENCE_LENGTH"),
}, indent=2, sort_keys=True) + "\n")
PY

set +e
"$PYTHON_BIN" "$RUN_DIR/upstream/run_mlm.py" \
  --config_name "$ARM_CONFIG" \
  --tokenizer_name "$P009_TOKENIZER_DIR" \
  --train_file "$P009_DATA_DIR/train.json" \
  --validation_file "$P009_DATA_DIR/validation.json" \
  --cache_dir "$PERSIST_ROOT/.cache/huggingface/p009" \
  --do_train \
  --do_eval \
  --max_seq_length "$P009_SEQUENCE_LENGTH" \
  --mlm_probability "$P009_MLM_PROBABILITY" \
  --max_steps "$P009_MAX_STEPS" \
  --per_device_train_batch_size "$P009_TRAIN_BATCH" \
  --per_device_eval_batch_size "$P009_EVAL_BATCH" \
  --learning_rate "$P009_LEARNING_RATE" \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --lr_scheduler_type linear \
  --adam_beta1 0.9 \
  --adam_beta2 0.999 \
  --adam_epsilon 1e-6 \
  --bf16 \
  --seed "$P009_SEED" \
  --data_seed "$P009_SEED" \
  --eval_strategy steps \
  --eval_steps 250 \
  --logging_strategy steps \
  --logging_steps 50 \
  --save_strategy no \
  --report_to none \
  --dataloader_num_workers 0 \
  --overwrite_output_dir \
  --output_dir "$MODEL_DIR" \
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

Path(sys.argv[1]).write_text(json.dumps({
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "reason": "official run_mlm process returned nonzero",
    "exit_status": int(sys.argv[2]),
    "scientific_failure": False,
}, indent=2, sort_keys=True) + "\n")
PY
  exit "$STATUS"
fi

cp "$MODEL_DIR/all_results.json" "$RUN_DIR/all_results.json"
cp "$MODEL_DIR/train_results.json" "$RUN_DIR/train_results.json"
cp "$MODEL_DIR/eval_results.json" "$RUN_DIR/eval_results.json"
cp "$MODEL_DIR/trainer_state.json" "$RUN_DIR/trainer_state.json"
printf 'completed run_dir=%s model_dir=%s\n' "$RUN_DIR" "$MODEL_DIR"
