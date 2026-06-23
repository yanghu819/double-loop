#!/usr/bin/env bash
set -euo pipefail

BASE="${BASE:-/huyang2/double-loop/official_eqr_repro_20260623}"
REPO="${BASE}/eqr-official"
VENV="${VENV:-/huyang2/double-loop/official_eqr_compare/.venv}"
CKPT="${BASE}/downloaded_checkpoints/maze-unique/eqr.pth"
LOG_DIR="${BASE}/logs"
ACTION="${1:-status}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export XDG_CACHE_HOME="${BASE}/.cache"
export HF_HOME="${BASE}/.cache/huggingface"
export TORCH_EXTENSIONS_DIR="${BASE}/.cache/torch_extensions"

mkdir -p "${LOG_DIR}" "${BASE}/artifacts" "${BASE}/.cache"

activate_env() {
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  cd "${REPO}"
}

write_env() {
  activate_env
  {
    echo "time_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "base=${BASE}"
    echo "repo=${REPO}"
    echo "repo_head=$(git rev-parse HEAD)"
    echo "repo_status_lines=$(git status --short | wc -l)"
    echo "checkpoint=${CKPT}"
    sha256sum "${CKPT}"
    python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))'
    python -c 'from adam_atan2 import AdamATan2; import adam_atan2_backend; print("adam_atan2_backend", adam_atan2_backend.__file__)'
    python -c 'import sys; print("python", sys.version)'
  } | tee "${LOG_DIR}/env.txt"
}

apply_sdpa_fallback() {
  activate_env
  python - <<'PY'
from pathlib import Path

path = Path("models/layers.py")
text = path.read_text(encoding="utf-8")
marker = "Runtime compatibility fallback: use PyTorch SDPA"
if marker in text:
    print("sdpa fallback already present")
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
            # FlashAttention wheel is unavailable for this host glibc/toolchain.
            y = F.scaled_dot_product_attention(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3), is_causal=self.causal).permute(0, 2, 1, 3)
'''
if old not in text:
    raise RuntimeError("attention block did not match pristine official source")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
print("applied sdpa fallback")
PY
  git --no-pager diff > "${BASE}/artifacts/sdpa_fallback.patch"
}

run_eval() {
  local label="$1"
  shift
  activate_env
  local log="${LOG_DIR}/eval_${label}.log"
  echo "[run_eval] label=${label} args=$*" | tee "${log}"
  python evaluate.py \
    eval_yaml=config/eval/depth_breadth.yaml \
    checkpoint="${CKPT}" \
    "$@" 2>&1 | tee -a "${log}"
}

run_all_sdpa() {
  local maze_noise="${MAZE_EVAL_NOISE:-0.01}"
  write_env
  apply_sdpa_fallback
  run_eval "D16_B1_N${maze_noise}" global_batch_size=128 noise_scale="${maze_noise}"
  run_eval "D64_B1_N${maze_noise}" halt_max_steps=64 global_batch_size=64 noise_scale="${maze_noise}"
  run_eval "D64_B128_N${maze_noise}" halt_max_steps=64 different_init=128 convergence_top_k=1 global_batch_size=16 noise_scale="${maze_noise}"
}

run_nofallback_probe() {
  write_env
  activate_env
  local log="${LOG_DIR}/eval_nofallback_probe.log"
  echo "[nofallback] Expect failure if FlashAttention is unavailable." | tee "${log}"
  python evaluate.py \
    eval_yaml=config/eval/depth_breadth.yaml \
    checkpoint="${CKPT}" \
    global_batch_size=16 \
    max_eval_steps=1 2>&1 | tee -a "${log}"
}

case "${ACTION}" in
  status)
    write_env
    ;;
  nofallback-probe)
    run_nofallback_probe
    ;;
  sdpa-eval)
    run_all_sdpa
    ;;
  *)
    echo "usage: $0 status|nofallback-probe|sdpa-eval" >&2
    exit 2
    ;;
esac
