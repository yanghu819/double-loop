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
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/official_eqr_compare/apply_path_loss_patch.py" "${BASE}/eqr-clean"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/official_eqr_compare/apply_path_loss_patch.py" "${BASE}/eqr-futureseed"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/official_eqr_compare/apply_futureseed_patch.py" "${BASE}/eqr-futureseed"
  git -C "${BASE}/eqr-clean" rev-parse HEAD > "${BASE}/artifacts/eqr-clean.sha"
  git -C "${BASE}/eqr-futureseed" rev-parse HEAD > "${BASE}/artifacts/eqr-futureseed-base.sha"
  git -C "${BASE}/eqr-clean" diff > "${BASE}/artifacts/path_loss.patch"
  git -C "${BASE}/eqr-futureseed" diff > "${BASE}/artifacts/futureseed.patch"
}

prepare_causal_repos() {
  clone_one "${BASE}/eqr-causal"
  clone_one "${BASE}/eqr-causal-futureseed"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/official_eqr_compare/apply_path_loss_patch.py" "${BASE}/eqr-causal"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/official_eqr_compare/apply_causal_attention_patch.py" "${BASE}/eqr-causal"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/official_eqr_compare/apply_path_loss_patch.py" "${BASE}/eqr-causal-futureseed"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/official_eqr_compare/apply_causal_attention_patch.py" "${BASE}/eqr-causal-futureseed"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/official_eqr_compare/apply_futureseed_patch.py" "${BASE}/eqr-causal-futureseed"
  git -C "${BASE}/eqr-causal" rev-parse HEAD > "${BASE}/artifacts/eqr-causal.sha"
  git -C "${BASE}/eqr-causal-futureseed" rev-parse HEAD > "${BASE}/artifacts/eqr-causal-futureseed-base.sha"
  git -C "${BASE}/eqr-causal" diff > "${BASE}/artifacts/causal.patch"
  git -C "${BASE}/eqr-causal-futureseed" diff > "${BASE}/artifacts/causal_futureseed.patch"
}

prepare_bidir_repos() {
  clone_one "${BASE}/eqr-causal-cheap"
  clone_one "${BASE}/eqr-causal-bidir-futureseed"
  if [[ "${APPLY_PATH_LOSS_PATCH:-0}" == "1" ]]; then
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/official_eqr_compare/apply_path_loss_patch.py" "${BASE}/eqr-causal-cheap"
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/official_eqr_compare/apply_path_loss_patch.py" "${BASE}/eqr-causal-bidir-futureseed"
  fi
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/official_eqr_compare/apply_causal_attention_patch.py" "${BASE}/eqr-causal-cheap"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/official_eqr_compare/apply_causal_attention_patch.py" "${BASE}/eqr-causal-bidir-futureseed"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/official_eqr_compare/apply_bidir_futureseed_patch.py" "${BASE}/eqr-causal-bidir-futureseed"
  git -C "${BASE}/eqr-causal-cheap" rev-parse HEAD > "${BASE}/artifacts/eqr-causal-cheap.sha"
  git -C "${BASE}/eqr-causal-bidir-futureseed" rev-parse HEAD > "${BASE}/artifacts/eqr-causal-bidir-futureseed-base.sha"
  git -C "${BASE}/eqr-causal-cheap" diff > "${BASE}/artifacts/causal_cheap.patch"
  git -C "${BASE}/eqr-causal-bidir-futureseed" diff > "${BASE}/artifacts/causal_bidir_futureseed.patch"
}

prepare_mixer_replacement_repos() {
  clone_one "${BASE}/eqr-mixer-base"
  clone_one "${BASE}/eqr-futureseed-mixer"
  "${PYTHON_BIN}" "${REPO_ROOT}/scripts/official_eqr_compare/apply_futureseed_mixer_replacement_patch.py" "${BASE}/eqr-futureseed-mixer"
  git -C "${BASE}/eqr-mixer-base" rev-parse HEAD > "${BASE}/artifacts/eqr-mixer-base.sha"
  git -C "${BASE}/eqr-futureseed-mixer" rev-parse HEAD > "${BASE}/artifacts/eqr-futureseed-mixer-base.sha"
  git -C "${BASE}/eqr-mixer-base" diff > "${BASE}/artifacts/mixer_base.patch"
  git -C "${BASE}/eqr-futureseed-mixer" diff > "${BASE}/artifacts/futureseed_mixer_replacement.patch"
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

apply_attention_runtime_fallback() {
  local repo="$1"
  "${BASE}/.venv/bin/python" - "${repo}" <<'PY'
from pathlib import Path
import sys

repo = Path(sys.argv[1])
path = repo / "models" / "layers.py"
text = path.read_text(encoding="utf-8")
marker = "Runtime compatibility fallback: use PyTorch SDPA"
if marker in text:
    print(f"attention fallback already present in {path}")
    raise SystemExit(0)

old = '''        if q.is_cuda:
            if flash_attn_func is None:
                colored_exception(RuntimeError, "flash_attn is not installed but CUDA attention was requested.")
            y = flash_attn_func(q=q, k=k, v=v, causal=self.causal)
            if isinstance(y, tuple):
                y = y[0]
        else:
            y = F.scaled_dot_product_attention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), is_causal=self.causal).permute(0, 2, 1, 3)
'''
new = '''        if q.is_cuda and flash_attn_func is not None:
            y = flash_attn_func(q=q, k=k, v=v, causal=self.causal)
            if isinstance(y, tuple):
                y = y[0]
        else:
            # Runtime compatibility fallback: use PyTorch SDPA when the installed
            # flash-attn package is absent or ABI-incompatible with torch.
            y = F.scaled_dot_product_attention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), is_causal=self.causal).permute(0, 2, 1, 3)
'''
if old not in text:
    raise RuntimeError(f"attention block did not match in {path}")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
print(f"attention fallback applied to {path}")
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
  case "${kind}" in
    base)
      repo="${BASE}/eqr-clean"
      ;;
    futureseed)
      repo="${BASE}/eqr-futureseed"
      extra+=(arch.future_seed_scale="${FUTURE_SEED_SCALE:-1.0}" arch.future_seed_gate_bias="${FUTURE_SEED_GATE_BIAS:--2.0}")
      ;;
    causal)
      repo="${BASE}/eqr-causal"
      extra+=(arch.attention_causal=true)
      ;;
    causal-futureseed)
      repo="${BASE}/eqr-causal-futureseed"
      extra+=(arch.attention_causal=true arch.future_seed_scale="${FUTURE_SEED_SCALE:-1.0}" arch.future_seed_gate_bias="${FUTURE_SEED_GATE_BIAS:--2.0}")
      ;;
    causal-cheap)
      repo="${BASE}/eqr-causal-cheap"
      extra+=(arch.attention_causal=true)
      ;;
    causal-bidir-futureseed)
      repo="${BASE}/eqr-causal-bidir-futureseed"
      extra+=(arch.attention_causal=true arch.future_seed_mode=reverse_causal arch.future_seed_scale="${FUTURE_SEED_SCALE:-1.0}" arch.future_seed_gate_bias="${FUTURE_SEED_GATE_BIAS:--2.0}")
      ;;
    mixer-base)
      repo="${BASE}/eqr-mixer-base"
      ;;
    futureseed-mixer)
      repo="${BASE}/eqr-futureseed-mixer"
      extra+=(arch.mixer_replacement_mode=future_seed_scan)
      ;;
    *)
      echo "unknown run kind: ${kind}" >&2
      exit 2
      ;;
  esac
  if [[ -n "${EQR_EXTRA_OVERRIDES:-}" ]]; then
    # Space-separated Hydra overrides, e.g.
    # EQR_EXTRA_OVERRIDES='arch.hidden_size=96 arch.num_heads=8'.
    # Keep values shell-quoted by the caller if they contain spaces.
    read -r -a user_overrides <<< "${EQR_EXTRA_OVERRIDES}"
    extra+=("${user_overrides[@]}")
  fi

  check_official_optimizer
  apply_attention_runtime_fallback "${repo}"

  local run_name="${RUN_NAME:-official-eqr-${kind}-$(date -u +%Y%m%dT%H%M%SZ)}"
  local log="${BASE}/logs/${run_name}.log"
  local pidfile="${BASE}/artifacts/${run_name}.pid"
  local train_config="${EQR_TRAIN_CONFIG:-train/eqr_maze_unique}"
  local default_path_loss_overrides=1
  if [[ "${kind}" == "causal-cheap" || "${kind}" == "causal-bidir-futureseed" || "${kind}" == "mixer-base" || "${kind}" == "futureseed-mixer" ]]; then
    default_path_loss_overrides="${APPLY_PATH_LOSS_PATCH:-0}"
  fi
  local loss_overrides=()
  if [[ "${ENABLE_PATH_TOKEN_LOSS_OVERRIDES:-${default_path_loss_overrides}}" != "0" ]]; then
    loss_overrides+=(arch.loss.path_token_weight="${PATH_TOKEN_WEIGHT:-1.0}" arch.loss.path_token_id="${PATH_TOKEN_ID:-5}")
  fi
  cd "${repo}"
  mkdir -p data
  if [[ ! -e data/maze-30x30-unique-1k ]]; then
    ln -s "${BASE}/eqr-clean/data/maze-30x30-unique-1k" data/maze-30x30-unique-1k
  fi
  if [[ -n "${EQR_DATA_LINKS:-}" ]]; then
    for spec in ${EQR_DATA_LINKS}; do
      local name="${spec%%=*}"
      local path="${spec#*=}"
      ln -sfn "${path}" "data/${name}"
    done
  fi
  (
    export WANDB_MODE=disabled
    export OUTPUT_ROOT="${BASE}/outputs/${kind}"
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONPATH="${BASE}/.venv/lib/python3.10/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
    . "${BASE}/.venv/bin/activate"
    echo "[Info] EQR_EXTRA_OVERRIDES=${EQR_EXTRA_OVERRIDES:-}"
    echo "[Info] EQR_TRAIN_CONFIG=${train_config}"
    "${BASE}/.venv/bin/python" pretrain.py --config-name "${train_config}" \
      epochs="${EPOCHS:-64}" \
      train_epochs_per_iter="${TRAIN_EPOCHS_PER_ITER:-${EPOCHS:-64}}" \
      global_batch_size="${GLOBAL_BATCH_SIZE:-128}" \
      eval_interval_steps="${EVAL_INTERVAL_STEPS:-250}" \
      checkpoint_interval_steps="${CHECKPOINT_INTERVAL_STEPS:-500}" \
      heavy_metrics_log_interval="${HEAVY_METRICS_LOG_INTERVAL:-100}" \
      steps_hist_log_interval_steps="${STEPS_HIST_LOG_INTERVAL_STEPS:-100}" \
      +wandb_mode=disabled \
      +run_name="${run_name}" \
      "${loss_overrides[@]}" \
      "${extra[@]}"
  ) >"${log}" 2>&1 &
  echo $! > "${pidfile}"
  echo "run_name=${run_name}"
  echo "pid=$(cat "${pidfile}")"
  echo "log=${log}"
}

run_eval() {
  local kind="$1"
  local checkpoint="${2:-}"
  if [[ -z "${checkpoint}" ]]; then
    echo "usage: $0 eval-base|eval-futureseed|eval-causal|eval-causal-futureseed <checkpoint-path>" >&2
    exit 2
  fi

  local repo="${BASE}/eqr-clean"
  case "${kind}" in
    base)
      repo="${BASE}/eqr-clean"
      ;;
    futureseed)
      repo="${BASE}/eqr-futureseed"
      ;;
    causal)
      repo="${BASE}/eqr-causal"
      ;;
    causal-futureseed)
      repo="${BASE}/eqr-causal-futureseed"
      ;;
    causal-cheap)
      repo="${BASE}/eqr-causal-cheap"
      ;;
    causal-bidir-futureseed)
      repo="${BASE}/eqr-causal-bidir-futureseed"
      ;;
    mixer-base)
      repo="${BASE}/eqr-mixer-base"
      ;;
    futureseed-mixer)
      repo="${BASE}/eqr-futureseed-mixer"
      ;;
    *)
      echo "unknown eval kind: ${kind}" >&2
      exit 2
      ;;
  esac

  apply_attention_runtime_fallback "${repo}"
  cd "${repo}"
  (
    export WANDB_MODE=disabled
    export OUTPUT_ROOT="${BASE}/outputs/${kind}"
    export CUDA_VISIBLE_DEVICES=0
    export PYTHONPATH="${BASE}/.venv/lib/python3.10/site-packages${PYTHONPATH:+:${PYTHONPATH}}"
    . "${BASE}/.venv/bin/activate"
    "${BASE}/.venv/bin/python" evaluate.py \
      eval_yaml="${EQR_EVAL_YAML:-config/eval/depth_breadth.yaml}" \
      checkpoint="${checkpoint}" \
      global_batch_size="${EVAL_GLOBAL_BATCH_SIZE:-128}" \
      suffix="${EVAL_SUFFIX:-final_D16_B1_N0.5_S1.0}"
  )
}

case "${ACTION}" in
  prepare)
    prepare_repos
    prepare_env
    ;;
  prepare-causal)
    prepare_causal_repos
    ;;
  prepare-bidir)
    prepare_bidir_repos
    prepare_env
    ;;
  prepare-mixer-replacement)
    prepare_mixer_replacement_repos
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
  train-causal)
    run_train causal
    ;;
  train-causal-futureseed)
    run_train causal-futureseed
    ;;
  train-causal-cheap)
    run_train causal-cheap
    ;;
  train-causal-bidir-futureseed)
    run_train causal-bidir-futureseed
    ;;
  train-mixer-base)
    run_train mixer-base
    ;;
  train-futureseed-mixer)
    run_train futureseed-mixer
    ;;
  eval-base)
    run_eval base "${2:-}"
    ;;
  eval-futureseed)
    run_eval futureseed "${2:-}"
    ;;
  eval-causal)
    run_eval causal "${2:-}"
    ;;
  eval-causal-futureseed)
    run_eval causal-futureseed "${2:-}"
    ;;
  eval-causal-cheap)
    run_eval causal-cheap "${2:-}"
    ;;
  eval-causal-bidir-futureseed)
    run_eval causal-bidir-futureseed "${2:-}"
    ;;
  eval-mixer-base)
    run_eval mixer-base "${2:-}"
    ;;
  eval-futureseed-mixer)
    run_eval futureseed-mixer "${2:-}"
    ;;
  status)
    echo "BASE=${BASE}"
    find "${BASE}" -maxdepth 2 -type f \( -name "*.pid" -o -name "*.sha" -o -name "*.patch" \) -print 2>/dev/null | sort || true
    pgrep -af '[p]retrain.py|[t]orchrun|[o]fficial-eqr' || true
    ;;
  *)
    cat >&2 <<EOF
usage: $0 prepare|prepare-causal|prepare-bidir|prepare-mixer-replacement|check|download-data|train-base|train-futureseed|train-causal|train-causal-futureseed|train-causal-cheap|train-causal-bidir-futureseed|train-mixer-base|train-futureseed-mixer|eval-base|eval-futureseed|eval-causal|eval-causal-futureseed|eval-causal-cheap|eval-causal-bidir-futureseed|eval-mixer-base|eval-futureseed-mixer|status
EOF
    exit 2
    ;;
esac
