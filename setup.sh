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
export PATH="$REPO_ROOT/.cache/bin:$PATH"

mkdir -p "$REPO_ROOT/.cache" "$REPO_ROOT/.cache/bin" "$REPO_ROOT/artifacts" "$REPO_ROOT/models" "$REPO_ROOT/runs" "$XDG_CACHE_HOME" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME"

ensure_ninja() {
  local py_bin="$1"
  if command -v ninja >/dev/null 2>&1; then
    return 0
  fi
  local target="$REPO_ROOT/.cache/ninja-pylib"
  mkdir -p "$target" "$REPO_ROOT/.cache/bin"
  if compgen -G "$REPO_ROOT/wheelhouse/ninja*.whl" >/dev/null; then
    "$py_bin" -m pip install --target "$target" --upgrade --no-index --find-links "$REPO_ROOT/wheelhouse" ninja
  else
    "$py_bin" -m pip install --target "$target" --upgrade ninja
  fi
  local ninja_bin
  ninja_bin="$(PYTHONPATH="$target${PYTHONPATH:+:$PYTHONPATH}" "$py_bin" - "$target" <<'PY'
import importlib.util
import pathlib
import sys

target = pathlib.Path(sys.argv[1])
spec = importlib.util.find_spec("ninja")
if spec is None or spec.origin is None:
    raise SystemExit(1)
root = pathlib.Path(spec.origin).parent
for candidate in (
    target / "bin" / "ninja",
    target / "bin" / "ninja.exe",
    root / "data" / "bin" / "ninja",
    root / "data" / "bin" / "ninja.exe",
):
    if candidate.exists():
        print(candidate)
        raise SystemExit(0)
raise SystemExit(2)
PY
)"
  ln -sf "$ninja_bin" "$REPO_ROOT/.cache/bin/ninja"
}

SYSTEM_TORCH_PYTHON="${SYSTEM_TORCH_PYTHON:-/opt/conda/bin/python}"
if [[ "${REUSE_SYSTEM_TORCH:-1}" == "1" && -x "$SYSTEM_TORCH_PYTHON" ]]; then
  if "$SYSTEM_TORCH_PYTHON" - <<'PY'
import sys

try:
    import torch
except Exception:
    raise SystemExit(1)

if sys.version_info < (3, 10):
    raise SystemExit(1)
if not torch.cuda.is_available():
    raise SystemExit(1)
print(f"reusing {sys.executable}: torch={torch.__version__} cuda={torch.cuda.get_device_name(0)}")
PY
  then
    ensure_ninja "$SYSTEM_TORCH_PYTHON"
    printf '%s\n' "$SYSTEM_TORCH_PYTHON" > "$REPO_ROOT/.cache/python-bin"
    printf 'Environment ready: %s\n' "$SYSTEM_TORCH_PYTHON"
    exit 0
  fi
fi

if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
else
  BOOTSTRAP_DIR="$REPO_ROOT/.cache/uv-bootstrap"
  if [[ ! -x "$BOOTSTRAP_DIR/bin/uv" ]]; then
    mkdir -p "$BOOTSTRAP_DIR/bin"
    if compgen -G "$REPO_ROOT/wheelhouse/uv*.whl" >/dev/null; then
      python3 -m pip install --target "$BOOTSTRAP_DIR/pylib" --no-index --find-links "$REPO_ROOT/wheelhouse" uv
      ln -sf "$BOOTSTRAP_DIR/pylib/bin/uv" "$BOOTSTRAP_DIR/bin/uv"
    elif python3 -m pip --version >/dev/null 2>&1; then
      python3 -m pip install --target "$BOOTSTRAP_DIR/pylib" uv
      ln -sf "$BOOTSTRAP_DIR/pylib/bin/uv" "$BOOTSTRAP_DIR/bin/uv"
    else
      curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$BOOTSTRAP_DIR/bin" sh
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
