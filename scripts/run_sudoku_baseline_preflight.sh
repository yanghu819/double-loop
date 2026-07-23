#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$REPO_ROOT/runs/sudoku-baseline-preflight-$(date -u +%Y%m%dT%H%M%SZ)}"
PERSIST_ROOT="${PERSIST_ROOT:-$REPO_ROOT}"

if [[ "${CUDA_VISIBLE_DEVICES:-0}" != "0" ]]; then
  printf 'This preflight is GPU1-only and requires CUDA_VISIBLE_DEVICES=0.\n' >&2
  exit 3
fi
export CUDA_VISIBLE_DEVICES=0
export FLA_DISABLE_BACKEND_DISPATCH=1
export FLA_CONV_BACKEND=triton
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PERSIST_ROOT/.cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PERSIST_ROOT/.cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$PERSIST_ROOT/.cache/torchinductor}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PERSIST_ROOT/.cache/torch_extensions}"
export TMPDIR="${TMPDIR:-$PERSIST_ROOT/.cache/tmp}"
export PATH="$PERSIST_ROOT/.cache/bin:$PATH"
PYTHON_EXTRA_PATH="${PYTHON_EXTRA_PATH:-$PERSIST_ROOT/.cache/python-extra-pylib}"
export PYTHONPATH="$PYTHON_EXTRA_PATH${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'CUDA Python is not executable: %s\n' "$PYTHON_BIN" >&2
  exit 4
fi

mkdir -p \
  "$OUT_DIR" \
  "$XDG_CACHE_HOME" \
  "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TORCH_EXTENSIONS_DIR" \
  "$TMPDIR"
"$PYTHON_BIN" "$REPO_ROOT/experiments/rwkv_fs_sudoku/check_fla_delta_backbones.py" \
  --backbone all \
  --check all \
  --wheel "$REPO_ROOT/wheelhouse/flash_linear_attention-0.5.2-py3-none-any.whl" \
  --out "$OUT_DIR/fla_kernel_gate.json" \
  2>&1 | tee "$OUT_DIR/fla_kernel_gate.log"

"$PYTHON_BIN" "$REPO_ROOT/scripts/check_sudoku_backbone_contract.py" \
  --repo "$REPO_ROOT" \
  --out "$OUT_DIR/backbone_contract.json" \
  2>&1 | tee "$OUT_DIR/backbone_contract.log"

printf 'preflight_dir=%s\n' "$OUT_DIR"
