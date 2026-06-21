#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/huyang2/double-loop/official_eqr_compare}"
REPO_ROOT="${REPO_ROOT:-/huyang2/double-loop}"
LAUNCH_DIR="${LAUNCH_DIR:-/huyang2/double-loop/artifacts/launch}"
TS="${TS:-20260621T1110Z}"
SHA="${SHA:-aba94e9}"
LOG="${LAUNCH_DIR}/official-eqr-e512-pair-${TS}.log"
PIDFILE="${LAUNCH_DIR}/official-eqr-e512-pair-${TS}.pid"

mkdir -p "$LAUNCH_DIR"

if [[ "${BACKGROUND:-1}" == "1" ]]; then
  BACKGROUND=0 nohup "$0" > "$LOG" 2>&1 &
  echo $! > "$PIDFILE"
  printf 'pid=%s\nlog=%s\npidfile=%s\n' "$(cat "$PIDFILE")" "$LOG" "$PIDFILE"
  exit 0
fi

cd "$REPO_ROOT"
export OFFICIAL_EQR_BASE="$BASE"
export PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled
export OUTPUT_ROOT="$BASE/outputs"
export XDG_CACHE_HOME="$BASE/.cache"
export PIP_CACHE_DIR="$BASE/.cache/pip"
export HF_HOME="$BASE/.cache/huggingface"
export TORCH_EXTENSIONS_DIR="$BASE/.cache/torch_extensions"

printf 'hypothesis=Official e256 token accuracy was misleading because PATH F1 stayed near zero; e512 tests whether longer official training opens true path prediction and whether FutureSeed accelerates that opening.\n'
printf 'prediction=If FutureSeed is an opening aid, it should improve step500/1000 and maybe final path metrics; if it only biases early dynamics, final e512 should again match or trail clean EqR.\n'
printf 'budget=GPU1 only, two official EqR training jobs, epochs=512, global_batch_size=128, checkpoint every 1000 steps, eval every 500 steps.\n'
printf 'kill_criteria=Stop exact PID if GPU memory stays high with utilization near zero for >10m, if AdamAtan2/official data gate fails, or if logs show repeated crash before step500.\n'
printf 'claim_if_success=Official EqR baseline is reproduced on Maze with path-aware visual evidence; FutureSeed claim only if it improves PATH/exact under matched official compute.\n'
printf 'base=%s ts=%s sha=%s\n' "$BASE" "$TS" "$SHA"

run_one() {
  local kind="$1"
  local run_name="official-eqr-${kind}-sdpa-e512-${TS}-${SHA}"
  printf '\n=== launch kind=%s run_name=%s ===\n' "$kind" "$run_name"
  EPOCHS=512 \
  TRAIN_EPOCHS_PER_ITER=512 \
  GLOBAL_BATCH_SIZE=128 \
  EVAL_INTERVAL_STEPS=500 \
  CHECKPOINT_INTERVAL_STEPS=1000 \
  HEAVY_METRICS_LOG_INTERVAL=100 \
  STEPS_HIST_LOG_INTERVAL_STEPS=100 \
  RUN_NAME="$run_name" \
  "$REPO_ROOT/scripts/official_eqr_compare/run_official_eqr_compare.sh" "train-${kind}"
  local pidfile
  pidfile="$(ls -t "$BASE"/artifacts/"$run_name".pid | head -1)"
  local pid
  pid="$(cat "$pidfile")"
  printf 'waiting kind=%s pid=%s pidfile=%s\n' "$kind" "$pid" "$pidfile"
  while kill -0 "$pid" 2>/dev/null; do
    tail -n 8 "$BASE/logs/${run_name}.log" || true
    nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
    sleep 60
  done
  wait "$pid" || true
  printf 'finished kind=%s pid=%s\n' "$kind" "$pid"
  tail -n 40 "$BASE/logs/${run_name}.log" || true
}

run_one base
run_one futureseed

printf 'completed official e512 pair ts=%s\n' "$TS"
