#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
P018_CONFIG="${P018_CONFIG:-$REPO_ROOT/configs/retrieval/pretrained_bert_carrier.env}"
set -a
source "$P018_CONFIG"
set +a

export CUDA_VISIBLE_DEVICES=0
export XDG_CACHE_HOME="$PERSIST_ROOT/.cache"
export HF_HOME="$PERSIST_ROOT/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

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

if [[ "$P018_CHECKPOINT_SHA256" == TO_BE_FILLED_AFTER_LOCAL_DOWNLOAD ]]; then
  printf 'Checkpoint hash is not registered; download locally, hash, upload, then commit it.\n' >&2
  exit 4
fi
assert_sha256 "$P018_CHECKPOINT_FILE" "$P018_CHECKPOINT_SHA256"
assert_sha256 "$P018_MODEL_DIR/config.json" "$P018_CONFIG_SHA256"
assert_sha256 "$P018_MODEL_DIR/vocab.txt" "$P018_VOCAB_SHA256"
assert_sha256 "$P018_VALIDATION_JSON" "$P018_VALIDATION_SHA256"
assert_sha256 "$P018_DATA_MANIFEST" "$P018_DATA_MANIFEST_SHA256"

GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
SOURCE_STATUS="$(git -C "$REPO_ROOT" status --short -- . \
  ':(exclude).cache' ':(exclude).venv' ':(exclude)artifacts' \
  ':(exclude)models' ':(exclude)runs')"
if [[ -n "$SOURCE_STATUS" ]]; then
  printf 'Refusing to run dirty source:\n%s\n' "$SOURCE_STATUS" >&2
  exit 5
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="${RUN_NAME:-pretrained-bert-carrier-${TS}-${GIT_SHA:0:7}}"
RUN_DIR="$PERSIST_ROOT/runs/$RUN_NAME"
mkdir -p "$RUN_DIR/visualizations"

cp "$P018_CONFIG" "$RUN_DIR/launch.env"
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

set +e
"$PYTHON_BIN" "$REPO_ROOT/scripts/evaluate_pretrained_bert_carrier.py" \
  --model-dir "$P018_MODEL_DIR" \
  --checkpoint-file "$P018_CHECKPOINT_FILE" \
  --validation-json "$P018_VALIDATION_JSON" \
  --data-manifest "$P018_DATA_MANIFEST" \
  --output-dir "$RUN_DIR/visualizations" \
  --sequence-length "$P018_SEQUENCE_LENGTH" \
  --candidate-windows "$P018_CANDIDATE_WINDOWS" \
  --batch-size "$P018_BATCH_SIZE" \
  --cases "$P018_CASES" \
  --seed "$P018_SEED" \
  2>&1 | tee "$RUN_DIR/formal.log"
STATUS="${PIPESTATUS[0]}"
set -e

date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_DIR/completed_at.txt"
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu \
  --format=csv,noheader > "$RUN_DIR/gpu_after.txt"
printf '%s\n' "$STATUS" > "$RUN_DIR/exit_status.txt"
if [[ -f "$RUN_DIR/visualizations/score.json" ]]; then
  cp "$RUN_DIR/visualizations/score.json" "$RUN_DIR/score.json"
fi
if [[ "$STATUS" != "0" ]]; then
  "$PYTHON_BIN" - "$RUN_DIR/abort.json" "$STATUS" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "reason": "pretrained bidirectional carrier gate did not pass",
    "exit_status": int(sys.argv[2]),
    "scientific_failure": int(sys.argv[2]) == 2,
}, indent=2, sort_keys=True) + "\n")
PY
  exit "$STATUS"
fi

printf 'completed run_dir=%s\n' "$RUN_DIR"
