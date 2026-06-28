#!/usr/bin/env bash
set -euo pipefail

ROOT=/huyang2/double-loop
SHA="${SOURCE_SHA:?set SOURCE_SHA to the GitHub-truth commit}"
SHORT="${SHA:0:7}"
WT="${WT_OVERRIDE:-$ROOT/.worktrees/gdn-realbwd-official-sudoku-d192-fs-hardcurr-s4000-${SHORT}-20260628}"
DATA="$ROOT/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_NAME="${RUN_NAME:-gdn-d192-official-sudoku-fs-hardcurr-s4000-${RUN_STAMP}-${SHORT}}"
LOG="$ROOT/artifacts/logs/${RUN_NAME}.log"
PID="$ROOT/artifacts/logs/${RUN_NAME}.pid"

mkdir -p "$ROOT/.worktrees" "$ROOT/artifacts/logs"

cd "$ROOT"
if [ -n "${GITHUB_TOKEN:-}" ]; then
  AUTH="$(printf 'x-access-token:%s' "$GITHUB_TOKEN" | base64 -w0)"
  git -c http.extraHeader="Authorization: Basic $AUTH" fetch origin codex/gpu1-experiment-tracking
  unset AUTH GITHUB_TOKEN
else
  git fetch origin codex/gpu1-experiment-tracking
fi

if [ -d "$WT" ]; then
  git -C "$WT" restore --staged . || true
  git -C "$WT" restore . || true
  git -C "$WT" checkout --detach "$SHA"
else
  git -c advice.detachedHead=false worktree add --quiet --detach "$WT" "$SHA"
fi
test "$(git -C "$WT" rev-parse HEAD)" = "$SHA"
test -d "$DATA"

FILTERED_STATUS="$(
  {
    git -C "$WT" diff --name-only
    git -C "$WT" diff --cached --name-only
    git -C "$WT" ls-files --others --exclude-standard
  } | grep -Ev '^(runs|artifacts|\.cache|\.venv)/' || true
)"
if [ -n "$FILTERED_STATUS" ]; then
  echo "Refusing to launch from dirty source tree:" >&2
  printf '%s\n' "$FILTERED_STATUS" >&2
  exit 2
fi

rm -f "$LOG" "$LOG.nohup" "$PID"
cd "$WT"
{
  echo "launch_sha=$(git rev-parse HEAD)"
  echo "launch_status_filtered_lines=0"
  echo "run_name=$RUN_NAME"
  echo "data=$DATA"
  echo "official_blank_curriculum=46-55:800,56-64:3200"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
  SMOKE_DONE=1 SKIP_SETUP=1 SOURCE_SNAPSHOT_MODE=lean \
  PYTHON_BIN=/opt/conda/bin/python CUDA_VISIBLE_DEVICES=0 \
  OFFICIAL_SUDOKU_DATA_DIR="$DATA" OFFICIAL_SUDOKU_TRAIN_SPLIT=train OFFICIAL_SUDOKU_EVAL_SPLIT=test \
  SUDOKU_SIZE=9 HOLE_PATTERN=random EVAL_HOLES=16 EVAL_HOLES_LIST=16 \
  HOLE_STAGES=46-55:800,56-64:3200 \
  BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 GDN_EXPAND_V=1.0 \
  D_MODEL=192 LAYERS=10 HEADS=12 HEAD_DIM=16 CHANNEL_MULT=4 L_CYCLES=2 MAX_LOOPS=5 \
  FULL_STEPS=4000 FULL_BATCH=128 FULL_EVAL_N=2048 FULL_ROLLOUT_KS= FULL_LOG_EVERY=100 \
  EVAL_CHECKPOINT_STEPS=1500,2500 EVAL_CHECKPOINT_HOLES_LIST=16 SAVE_TRAIN_CHECKPOINT_EVERY=-1 \
  FORWARD_DTYPE=bfloat16 LR=0.0015 WEIGHT_DECAY=0.001 BLANK_LOSS_WEIGHT=8 \
  FUTURE_SEED_SCALE=1 FUTURE_SEED_UPDATE=fixed \
  RUN_NAME="$RUN_NAME" ./run.sh full
} 2>&1 | tee "$LOG"
