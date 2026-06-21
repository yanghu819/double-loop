#!/usr/bin/env bash
set -euo pipefail

WT="${WT:-/huyang2/double-loop/.worktrees/futureseed-bidir-955e266-20260621T0850Z}"
SHA_SHORT="${SHA_SHORT:-955e266}"
TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
LAUNCH_DIR="${LAUNCH_DIR:-/huyang2/double-loop/artifacts/launch}"
RUN_NOFS="bidir-rwkv9-nofs-s800-${TS}-${SHA_SHORT}"
RUN_FS="bidir-rwkv9-fs-s800-${TS}-${SHA_SHORT}"
MASTER_LOG="${LAUNCH_DIR}/futureseed-bidir-rwkv9-${TS}.log"
PIDFILE="${LAUNCH_DIR}/futureseed-bidir-rwkv9-${TS}.pid"

if [[ "${BACKGROUND:-1}" == "1" ]]; then
  mkdir -p "$LAUNCH_DIR"
  BACKGROUND=0 nohup "$0" > "$MASTER_LOG" 2>&1 &
  echo $! > "$PIDFILE"
  printf 'pid=%s\nlog=%s\npidfile=%s\nrun_nofs=%s\nrun_fs=%s\nworktree=%s\n' \
    "$(cat "$PIDFILE")" "$MASTER_LOG" "$PIDFILE" "$RUN_NOFS" "$RUN_FS" "$WT"
  exit 0
fi

cd "$WT"

printf 'hypothesis=FutureSeed cheaply injects future/noncausal context into causal RWKV; expected lower early-late blank-acc gap and faster opening versus no-FS.\n'
printf 'sha=%s\nrun_nofs=%s\nrun_fs=%s\nworktree=%s\n' "$(git rev-parse HEAD)" "$RUN_NOFS" "$RUN_FS" "$WT"

run_one() {
  local fs_scale="$1"
  local run_name="$2"

  env \
    PATH="/huyang2/double-loop/.cache/ninja-prefix/bin:$PATH" \
    CUDA_VISIBLE_DEVICES=0 \
    SMOKE_DONE=1 \
    SKIP_SETUP=1 \
    SOURCE_SNAPSHOT_MODE=lean \
    PYTHON_BIN=/opt/conda/bin/python \
    XDG_CACHE_HOME=/huyang2/double-loop/.cache \
    UV_CACHE_DIR=/huyang2/double-loop/.cache/uv \
    PIP_CACHE_DIR=/huyang2/double-loop/.cache/pip \
    HF_HOME=/huyang2/double-loop/.cache/huggingface \
    TORCH_HOME=/huyang2/double-loop/.cache/torch \
    TORCH_EXTENSIONS_DIR=/huyang2/double-loop/.cache/torch_extensions \
    SUDOKU_SIZE=9 \
    HOLE_PATTERN=random \
    D_MODEL=96 \
    LAYERS=8 \
    HEADS=8 \
    HEAD_DIM=12 \
    CHANNEL_MULT=4 \
    L_CYCLES=2 \
    MAX_LOOPS=5 \
    HOLE_STAGES=4-8:300,8-12:500 \
    EVAL_HOLES=12 \
    EVAL_HOLES_LIST=8,12,16 \
    FULL_STEPS=800 \
    FULL_BATCH=192 \
    FULL_EVAL_N=512 \
    FULL_ROLLOUT_KS=1 \
    ROLLOUT_LOOP_VALUES=1,3,5 \
    FULL_LOG_EVERY=100 \
    RWKV_KERNEL=statepassing \
    FORWARD_DTYPE=bfloat16 \
    FUTURE_SEED_SCALE="$fs_scale" \
    SEED=52 \
    RUN_NAME="$run_name" \
    ./run.sh full
}

run_one 0 "$RUN_NOFS"
run_one 1 "$RUN_FS"

printf 'completed_pair run_nofs=%s run_fs=%s\n' "$RUN_NOFS" "$RUN_FS"
