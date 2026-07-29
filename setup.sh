#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-$REPO_ROOT}"
EXP_DIR="$REPO_ROOT/experiments/rwkv_fs_sudoku"
FLA_COMMIT="9c8e42e762fce087c27b673af4922795d9edb85e"
FLA_WHEEL="$REPO_ROOT/wheelhouse/flash_linear_attention-0.5.2-9c8e42e-py3-none-any.whl"
FLA_WHEEL_SHA256="0280db310981915eb048ece99d7bedca8b5caa9be65c99835a0f912ada977d6a"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PERSIST_ROOT/.cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PERSIST_ROOT/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$PERSIST_ROOT/.cache/uv/python}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PERSIST_ROOT/.cache/pip}"
export HF_HOME="${HF_HOME:-$PERSIST_ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PERSIST_ROOT/.cache/torch}"
export PATH="$PERSIST_ROOT/.cache/bin:$PATH"

mkdir -p "$PERSIST_ROOT/.cache" "$PERSIST_ROOT/.cache/bin" "$PERSIST_ROOT/artifacts" "$PERSIST_ROOT/models" "$PERSIST_ROOT/runs" "$XDG_CACHE_HOME" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME"

ensure_ninja() {
  local py_bin="$1"
  if command -v ninja >/dev/null 2>&1; then
    return 0
  fi
  local target="$PERSIST_ROOT/.cache/ninja-pylib"
  mkdir -p "$target" "$PERSIST_ROOT/.cache/bin"
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
  ln -sf "$ninja_bin" "$PERSIST_ROOT/.cache/bin/ninja"
}

ensure_runtime_deps() {
  local py_bin="$1"
  local target="$PERSIST_ROOT/.cache/python-extra-pylib"
  local fla_versions_root="$PERSIST_ROOT/.cache/fla-versions"
  local fla_version_dir="$fla_versions_root/$FLA_COMMIT-${FLA_WHEEL_SHA256:0:16}"
  local fla_active="$PERSIST_ROOT/.cache/fla-active"
  local fla_pythonpath="$fla_active:$target"
  mkdir -p "$target" "$fla_versions_root"
  if ! command -v flock >/dev/null 2>&1; then
    printf 'flock is required for atomic FLA activation.\n' >&2
    return 1
  fi
  exec 9>"$PERSIST_ROOT/.cache/fla-install.lock"
  flock 9

  if [[ ! -f "$FLA_WHEEL" ]]; then
    printf 'Pinned FLA wheel is missing: %s\n' "$FLA_WHEEL" >&2
    return 1
  fi
  "$py_bin" - "$FLA_WHEEL" "$FLA_WHEEL_SHA256" <<'PY'
import hashlib
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
actual = hashlib.sha256(path.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit(f"Pinned FLA wheel SHA256 mismatch: {actual} != {expected}")
PY

  local missing
  missing="$(
    PYTHONPATH="$target${PYTHONPATH:+:$PYTHONPATH}" "$py_bin" - <<'PY'
import importlib.util

modules = {
    "colorama": "colorama>=0.4.6",
    "einops": "einops>=0.8.0",
    "numpy": "numpy>=2.0",
    "pydantic": "pydantic>=2.0",
    "transformers": "transformers>=4.45.0",
}
for module, package in modules.items():
    if importlib.util.find_spec(module) is None:
        print(package)
PY
  )"

  if [[ -n "$missing" ]]; then
    local pip_args=(--target "$target" --upgrade)
    # shellcheck disable=SC2086
    "$py_bin" -m pip install "${pip_args[@]}" $missing
  fi

  local fla_needs_install=1
  if [[ -L "$fla_active" ]] && \
    [[ "$(readlink -f "$fla_active")" == "$fla_version_dir" ]] && \
    [[ -s "$PERSIST_ROOT/.cache/fla-source-sha" ]] && \
    [[ "$(<"$PERSIST_ROOT/.cache/fla-source-sha")" == "$FLA_COMMIT" ]] && \
    PYTHONPATH="$fla_pythonpath${PYTHONPATH:+:$PYTHONPATH}" "$py_bin" - <<'PY'
try:
    from fla.ops.gdn2 import chunk_gdn2  # noqa: F401
    from fla.ops.kda import chunk_kda  # noqa: F401
except Exception:
    raise SystemExit(1)
PY
  then
    fla_needs_install=0
  fi

  if [[ "$fla_needs_install" == "1" ]]; then
    local fla_stage
    fla_stage="$(mktemp -d "$fla_versions_root/.install.XXXXXX")"
    "$py_bin" -m pip install --target "$fla_stage" --no-deps --no-index "$FLA_WHEEL"
    PYTHONPATH="$fla_stage:$target${PYTHONPATH:+:$PYTHONPATH}" "$py_bin" - <<'PY'
import fla
from fla.ops.gdn2 import chunk_gdn2  # noqa: F401
from fla.ops.kda import chunk_kda  # noqa: F401

print(f"flash-linear-attention={fla.__version__}")
PY
    if [[ -d "$fla_version_dir" ]] && \
      PYTHONPATH="$fla_version_dir:$target${PYTHONPATH:+:$PYTHONPATH}" "$py_bin" - <<'PY'
from fla.ops.gdn2 import chunk_gdn2  # noqa: F401
from fla.ops.kda import chunk_kda  # noqa: F401
PY
    then
      rm -rf "$fla_stage"
    elif [[ -e "$fla_version_dir" ]]; then
      printf 'Refusing to replace corrupt immutable FLA version: %s\n' "$fla_version_dir" >&2
      rm -rf "$fla_stage"
      return 1
    else
      mv "$fla_stage" "$fla_version_dir"
    fi
    if [[ -e "$fla_active" && ! -L "$fla_active" ]]; then
      printf 'Refusing to replace non-symlink FLA active path: %s\n' "$fla_active" >&2
      return 1
    fi
    local active_tmp="$PERSIST_ROOT/.cache/.fla-active.$$"
    rm -f "$active_tmp"
    ln -s "$fla_version_dir" "$active_tmp"
    mv -Tf "$active_tmp" "$fla_active"
  fi

  PYTHONPATH="$fla_pythonpath${PYTHONPATH:+:$PYTHONPATH}" "$py_bin" - <<'PY'
import pathlib

import fla
from fla.ops.gdn2 import chunk_gdn2  # noqa: F401
from fla.ops.kda import chunk_kda  # noqa: F401

root = pathlib.Path(fla.__file__).resolve().parent
print(f"active flash-linear-attention={fla.__version__} root={root}")
PY
  rm -rf "$target/fla" "$target"/flash_linear_attention-*.dist-info
  local marker_tmp="$PERSIST_ROOT/.cache/.fla-source-sha.$$"
  local path_tmp="$PERSIST_ROOT/.cache/.python-extra-path.$$"
  printf '%s\n' "$FLA_COMMIT" > "$marker_tmp"
  printf '%s\n' "$fla_pythonpath" > "$path_tmp"
  mv -f "$marker_tmp" "$PERSIST_ROOT/.cache/fla-source-sha"
  mv -f "$path_tmp" "$PERSIST_ROOT/.cache/python-extra-path"
  flock -u 9
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
    ensure_runtime_deps "$SYSTEM_TORCH_PYTHON"
    printf '%s\n' "$SYSTEM_TORCH_PYTHON" > "$PERSIST_ROOT/.cache/python-bin"
    printf 'Environment ready: %s\n' "$SYSTEM_TORCH_PYTHON"
    exit 0
  fi
fi

if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
else
  BOOTSTRAP_DIR="$PERSIST_ROOT/.cache/uv-bootstrap"
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
  UV_PROJECT_ENVIRONMENT="$PERSIST_ROOT/.venv" "$UV_BIN" sync --frozen --python 3.11
)

printf 'Environment ready: %s\n' "$PERSIST_ROOT/.venv"
