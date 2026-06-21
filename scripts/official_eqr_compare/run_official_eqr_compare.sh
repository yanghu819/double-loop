#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE="${OFFICIAL_EQR_BASE:-/huyang2/double-loop/official_eqr_compare}"
EQR_URL="${EQR_URL:-https://github.com/locuslab/eqr.git}"
EQR_SHA="${EQR_SHA:-aba94e9cde0f273ce644db5261cd6915ba6561f0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ACTION="${1:-status}"

export XDG_CACHE_HOME="${BASE}/.cache"
export PIP_CACHE_DIR="${BASE}/.cache/pip"
export HF_HOME="${BASE}/.cache/huggingface"
export WANDB_MODE="${WANDB_MODE:-offline}"
export OUTPUT_ROOT="${BASE}/outputs"
export TORCH_EXTENSIONS_DIR="${BASE}/.cache/torch_extensions"

mkdir -p "${BASE}"/{logs,artifacts,outputs,.cache/torch_extensions}

clone_one() {
  local dst="$1"
  if [[ ! -d "${dst}/.git" ]]; then
    git clone "${EQR_URL}" "${dst}"
  fi
  git -C "${dst}" fetch --quiet origin "${EQR_SHA}" || true
  git -C "${dst}" checkout --detach "${EQR_SHA}"
  git -C "${dst}" reset --hard "${EQR_SHA}"
  git -C "${dst}" clean -fdx
}

prepare_repos() {
  clone_one "${BASE}/eqr-clean"
  clone_one "${BASE}/eqr-futureseed"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/official_eqr_compare/apply_futureseed_patch.py" "${BASE}/eqr-futureseed"
  git -C "${BASE}/eqr-clean" rev-parse HEAD > "${BASE}/artifacts/eqr-clean.sha"
  git -C "${BASE}/eqr-futureseed" rev-parse HEAD > "${BASE}/artifacts/eqr-futureseed-base.sha"
  git -C "${BASE}/eqr-futureseed" diff > "${BASE}/artifacts/futureseed.patch"
}

prepare_env() {
  if [[ ! -x "${BASE}/.venv/bin/python" ]]; then
    python -m venv --system-site-packages "${BASE}/.venv"
  fi
  # Keep this minimal. FlashAttention is usually supplied by the AIStation image.
  "${BASE}/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
  "${BASE}/.venv/bin/python" -m pip install hydra-core omegaconf coolname argdantic colorama huggingface_hub adam-atan2
}

check_official_optimizer() {
  "${BASE}/.venv/bin/python" - <<'PY'
from adam_atan2 import AdamATan2
import adam_atan2_backend
print("AdamATan2", AdamATan2)
print("adam_atan2_backend", adam_atan2_backend.__file__)
PY
}

download_data() {
  cd "${BASE}/eqr-clean"
  "${BASE}/.venv/bin/python" -m pip install huggingface_hub >/dev/null
  bash scripts/download_artifacts.sh
}

run_train() {
  local kind="$1"
  local repo="${BASE}/eqr-clean"
  local extra=()
  if [[ "${kind}" == "futureseed" ]]; then
    repo="${BASE}/eqr-futureseed"
    extra+=(arch.future_seed_scale="${FUTURE_SEED_SCALE:-1.0}" arch.future_seed_gate_bias="${FUTURE_SEED_GATE_BIAS:--2.0}")
  elif [[ "${kind}" != "base" ]]; then
    echo "unknown run kind: ${kind}" >&2
    exit 2
  fi

  check_official_optimizer

  local run_name="${RUN_NAME:-official-eqr-${kind}-$(date -u +%Y%m%dT%H%M%SZ)}"
  local log="${BASE}/logs/${run_name}.log"
  local pidfile="${BASE}/artifacts/${run_name}.pid"
  cd "${repo}"
  mkdir -p data
  if [[ ! -e data/maze-30x30-unique-1k ]]; then
    ln -s "${BASE}/eqr-clean/data/maze-30x30-unique-1k" data/maze-30x30-unique-1k
  fi
  (
    export WANDB_MODE=offline
    export OUTPUT_ROOT="${BASE}/outputs/${kind}"
    export CUDA_VISIBLE_DEVICES=0
    . "${BASE}/.venv/bin/activate"
    NPROC_PER_NODE=1 bash scripts/train.sh eqr_maze_unique \
      max_steps="${MAX_STEPS:-500}" \
      global_batch_size="${GLOBAL_BATCH_SIZE:-128}" \
      eval_interval_steps="${EVAL_INTERVAL_STEPS:-250}" \
      checkpoint_interval_steps="${CHECKPOINT_INTERVAL_STEPS:-500}" \
      heavy_metrics_log_interval="${HEAVY_METRICS_LOG_INTERVAL:-100}" \
      steps_hist_log_interval_steps="${STEPS_HIST_LOG_INTERVAL_STEPS:-100}" \
      wandb_mode=offline \
      run_name="${run_name}" \
      "${extra[@]}"
  ) >"${log}" 2>&1 &
  echo $! > "${pidfile}"
  echo "run_name=${run_name}"
  echo "pid=$(cat "${pidfile}")"
  echo "log=${log}"
}

case "${ACTION}" in
  prepare)
    prepare_repos
    prepare_env
    ;;
  check)
    check_official_optimizer
    ;;
  download-data)
    download_data
    ;;
  train-base)
    run_train base
    ;;
  train-futureseed)
    run_train futureseed
    ;;
  status)
    echo "BASE=${BASE}"
    find "${BASE}" -maxdepth 2 -type f \( -name "*.pid" -o -name "*.sha" -o -name "*.patch" \) -print 2>/dev/null | sort || true
    pgrep -af '[p]retrain.py|[t]orchrun|[o]fficial-eqr' || true
    ;;
  *)
    cat >&2 <<EOF
usage: $0 prepare|check|download-data|train-base|train-futureseed|status
EOF
    exit 2
    ;;
esac
