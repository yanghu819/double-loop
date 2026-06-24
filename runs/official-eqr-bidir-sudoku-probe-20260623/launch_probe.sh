#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/huyang2/double-loop}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPARE="${COMPARE:-${ROOT}/official_eqr_compare}"
QUICK5="${QUICK5:-${ROOT}/official_eqr_sudoku_repro_20260623/artifacts/sudoku_quick5_20260623}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_ROOT="${LOG_ROOT:-${COMPARE}/artifacts/bidir_sudoku_probe_${RUN_STAMP}}"
COMPARE_SCRIPT="${COMPARE_SCRIPT:-${REPO_ROOT}/scripts/official_eqr_compare/run_official_eqr_compare.sh}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/official_eqr_sudoku_repro_20260623/eqr-official/data/sudoku-extreme-1k-aug-1000}"

mkdir -p "${LOG_ROOT}"

QUICK_BASELINE_MIN_METRICS="${QUICK_BASELINE_MIN_METRICS:-2}"
quick_metric_count="$(find "${QUICK5}/metrics" -maxdepth 1 -type f -name 'seed*.json' 2>/dev/null | wc -l | tr -d ' ')"
if (( quick_metric_count < QUICK_BASELINE_MIN_METRICS )); then
  echo "[abort] official Sudoku quick baseline has only ${quick_metric_count} metric file(s); need ${QUICK_BASELINE_MIN_METRICS}: ${QUICK5}/metrics" >&2
  exit 10
fi

if pgrep -af '[e]valuate.py|[p]retrain.py|[t]orchrun' >"${LOG_ROOT}/active_gpu_tasks.txt"; then
  echo "[abort] active GPU task found; refusing to launch on GPU1" >&2
  cat "${LOG_ROOT}/active_gpu_tasks.txt" >&2
  exit 11
fi

if [[ ! -f "${COMPARE_SCRIPT}" ]]; then
  echo "[abort] compare script not found: ${COMPARE_SCRIPT}" >&2
  exit 12
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "[abort] official Sudoku data not found: ${DATA_ROOT}" >&2
  exit 13
fi

git -C "${REPO_ROOT}" rev-parse HEAD >"${LOG_ROOT}/source.sha" 2>/dev/null || true

export OFFICIAL_EQR_BASE="${COMPARE}"
export CUDA_VISIBLE_DEVICES=0
export EQR_TRAIN_CONFIG="train/eqr_sudoku"
export EQR_DATA_LINKS="sudoku-extreme-1k-aug-1000=${DATA_ROOT}"
export ENABLE_PATH_TOKEN_LOSS_OVERRIDES=0
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
export EPOCHS="${EPOCHS:-64}"
export TRAIN_EPOCHS_PER_ITER="${TRAIN_EPOCHS_PER_ITER:-${EPOCHS}}"
export EVAL_INTERVAL_STEPS="${EVAL_INTERVAL_STEPS:-250}"
export CHECKPOINT_INTERVAL_STEPS="${CHECKPOINT_INTERVAL_STEPS:-250}"
export HEAVY_METRICS_LOG_INTERVAL="${HEAVY_METRICS_LOG_INTERVAL:-100}"
export STEPS_HIST_LOG_INTERVAL_STEPS="${STEPS_HIST_LOG_INTERVAL_STEPS:-100}"
export FUTURE_SEED_SCALE="${FUTURE_SEED_SCALE:-1.0}"
export FUTURE_SEED_GATE_BIAS="${FUTURE_SEED_GATE_BIAS:--2.0}"
export EQR_EXTRA_OVERRIDES="${EQR_EXTRA_OVERRIDES:-arch.mlp_t=false arch.hidden_size=192 arch.num_heads=6 arch.halt_max_steps=16 arch.noise_scale=0.01}"

echo "[config] RUN_STAMP=${RUN_STAMP}" | tee "${LOG_ROOT}/launch.log"
echo "[config] OFFICIAL_EQR_BASE=${OFFICIAL_EQR_BASE}" | tee -a "${LOG_ROOT}/launch.log"
echo "[config] QUICK_BASELINE_METRICS=${quick_metric_count} min=${QUICK_BASELINE_MIN_METRICS}" | tee -a "${LOG_ROOT}/launch.log"
echo "[config] EQR_EXTRA_OVERRIDES=${EQR_EXTRA_OVERRIDES}" | tee -a "${LOG_ROOT}/launch.log"
echo "[config] EPOCHS=${EPOCHS} GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}" | tee -a "${LOG_ROOT}/launch.log"

bash "${COMPARE_SCRIPT}" prepare-bidir | tee -a "${LOG_ROOT}/launch.log"

run_one() {
  local kind="$1"
  local run_name="official-eqr-${kind}-sudoku-${RUN_STAMP}"
  local marker="${LOG_ROOT}/${kind}.marker"
  touch "${marker}"
  RUN_NAME="${run_name}" bash "${COMPARE_SCRIPT}" "train-${kind}" | tee "${LOG_ROOT}/${kind}.launch.log"
  local pidfile="${COMPARE}/artifacts/${run_name}.pid"
  if [[ ! -f "${pidfile}" ]]; then
    echo "[error] missing pidfile for ${kind}: ${pidfile}" >&2
    exit 20
  fi
  local pid
  pid="$(cat "${pidfile}")"
  echo "[${kind}] pid=${pid}" | tee -a "${LOG_ROOT}/launch.log"
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 60
    tail -n 20 "${COMPARE}/logs/${run_name}.log" >"${LOG_ROOT}/${kind}.tail.log" 2>/dev/null || true
  done
  echo "[${kind}] finished $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG_ROOT}/launch.log"
  local ckpt
  ckpt="$(find "${COMPARE}/outputs/${kind}" -type f -name 'step_*.pth' -newer "${marker}" -print | sort | tail -n 1)"
  if [[ -z "${ckpt}" ]]; then
    echo "[error] no checkpoint found for ${kind}" >&2
    exit 21
  fi
  echo "${ckpt}" >"${LOG_ROOT}/${kind}.checkpoint"
  echo "[${kind}] checkpoint=${ckpt}" | tee -a "${LOG_ROOT}/launch.log"
}

run_one causal-cheap
run_one causal-bidir-futureseed

echo "[done] training pair completed; run eval manually after inspecting checkpoints" | tee -a "${LOG_ROOT}/launch.log"
