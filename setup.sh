#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$REPO_ROOT/experiments/rwkv_fs_sudoku"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_ROOT/.cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_ROOT/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$REPO_ROOT/.cache/uv/python}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$REPO_ROOT/.cache/pip}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$REPO_ROOT/.cache/torch}"

mkdir -p "$REPO_ROOT/.cache" "$REPO_ROOT/artifacts" "$REPO_ROOT/models" "$REPO_ROOT/runs" "$XDG_CACHE_HOME" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME"

if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
else
  BOOTSTRAP_DIR="$REPO_ROOT/.cache/uv-bootstrap"
  if [[ ! -x "$BOOTSTRAP_DIR/bin/uv" ]]; then
    python3 -m venv "$BOOTSTRAP_DIR"
    "$BOOTSTRAP_DIR/bin/python" -m pip install --upgrade pip
    if compgen -G "$REPO_ROOT/wheelhouse/uv*.whl" >/dev/null; then
      "$BOOTSTRAP_DIR/bin/pip" install --no-index --find-links "$REPO_ROOT/wheelhouse" uv
    else
      "$BOOTSTRAP_DIR/bin/pip" install uv
    fi
  fi
  UV_BIN="$BOOTSTRAP_DIR/bin/uv"
fi

"$UV_BIN" --version

(
  cd "$EXP_DIR"
  UV_PROJECT_ENVIRONMENT="$REPO_ROOT/.venv" "$UV_BIN" sync --frozen --python 3.11
)

printf 'Environment ready: %s\n' "$REPO_ROOT/.venv"
