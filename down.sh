#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-$REPO_ROOT}"
DATA_DIR="${OFFICIAL_SUDOKU_DATA_DIR:-$PERSIST_ROOT/data/sudoku-extreme-full}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PERSIST_ROOT/.cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PERSIST_ROOT/.cache/uv}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PERSIST_ROOT/.cache/pip}"
export HF_HOME="${HF_HOME:-$PERSIST_ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PERSIST_ROOT/.cache/torch}"

mkdir -p \
  "$PERSIST_ROOT/.cache" \
  "$PERSIST_ROOT/artifacts" \
  "$PERSIST_ROOT/models" \
  "$PERSIST_ROOT/runs" \
  "$PERSIST_ROOT/data" \
  "$XDG_CACHE_HOME" \
  "$UV_CACHE_DIR" \
  "$PIP_CACHE_DIR" \
  "$HF_HOME" \
  "$TORCH_HOME"

if [[ -n "${SUDOKU_DATA_ARCHIVE:-}" ]]; then
  if [[ ! -f "$SUDOKU_DATA_ARCHIVE" ]]; then
    printf 'SUDOKU_DATA_ARCHIVE does not exist: %s\n' "$SUDOKU_DATA_ARCHIVE" >&2
    exit 2
  fi
  mkdir -p "$DATA_DIR"
  tar -xf "$SUDOKU_DATA_ARCHIVE" -C "$DATA_DIR"
fi

missing=0
for split in train test; do
  for name in all__inputs.npy all__labels.npy; do
    path="$DATA_DIR/$split/$name"
    if [[ ! -f "$path" ]]; then
      printf 'Missing dataset file: %s\n' "$path" >&2
      missing=1
    fi
  done
done

if [[ "$missing" == "1" ]]; then
  printf '%s\n' \
    'Stage the official Sudoku arrays locally, upload one archive, then run:' \
    '  SUDOKU_DATA_ARCHIVE=/huyang2/double-loop/artifacts/sudoku-extreme-full.tar ./down.sh' \
    'The archive root must contain train/ and test/. No server-side Hugging Face fallback is used.'
  exit 4
fi

printf 'Official Sudoku data ready: %s\n' "$DATA_DIR"
FLA_WHEEL="$REPO_ROOT/wheelhouse/flash_linear_attention-0.5.2-9c8e42e-py3-none-any.whl"
FLA_WHEEL_SHA256="0280db310981915eb048ece99d7bedca8b5caa9be65c99835a0f912ada977d6a"
python3 - "$FLA_WHEEL" "$FLA_WHEEL_SHA256" <<'PY'
import hashlib
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"Pinned FLA wheel is missing: {path}")
actual = hashlib.sha256(path.read_bytes()).hexdigest()
if actual != sys.argv[2]:
    raise SystemExit(f"Pinned FLA wheel SHA256 mismatch: {actual} != {sys.argv[2]}")
PY
printf 'Pinned FLA wheel ready: %s\n' "$FLA_WHEEL"

RAVEN_FLA_COMMIT="31d15f7554bd5df05d3da6f75e09146279d2b1a8"
RAVEN_FLA_ROOT="${RAVEN_FLA_SOURCE_ROOT:-$PERSIST_ROOT/.cache/fla-upstream-31d15f7}"
if [[ "${DOWNLOAD_RAVEN_FLA:-0}" == "1" && ! -d "$RAVEN_FLA_ROOT/.git" ]]; then
  git clone --filter=blob:none https://github.com/fla-org/flash-linear-attention.git "$RAVEN_FLA_ROOT"
  git -C "$RAVEN_FLA_ROOT" checkout --detach "$RAVEN_FLA_COMMIT"
fi
if [[ "${REQUIRE_RAVEN_FLA:-0}" == "1" || "${DOWNLOAD_RAVEN_FLA:-0}" == "1" ]]; then
  if [[ ! -d "$RAVEN_FLA_ROOT/.git" ]]; then
    printf 'Pinned Raven FLA checkout is missing: %s\n' "$RAVEN_FLA_ROOT" >&2
    exit 5
  fi
  RAVEN_FLA_ACTUAL="$(git -C "$RAVEN_FLA_ROOT" rev-parse HEAD)"
  if [[ "$RAVEN_FLA_ACTUAL" != "$RAVEN_FLA_COMMIT" ]]; then
    printf 'Pinned Raven FLA SHA mismatch: %s != %s\n' "$RAVEN_FLA_ACTUAL" "$RAVEN_FLA_COMMIT" >&2
    exit 5
  fi
  printf 'Pinned Raven FLA source ready: %s @ %s\n' "$RAVEN_FLA_ROOT" "$RAVEN_FLA_ACTUAL"
fi

ZOOLOGY_COMMIT="1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb"
ZOOLOGY_ROOT="${ZOOLOGY_ROOT:-$PERSIST_ROOT/repos/zoology-official}"
if [[ "${DOWNLOAD_ZOOLOGY:-0}" == "1" && ! -d "$ZOOLOGY_ROOT/.git" ]]; then
  mkdir -p "$(dirname "$ZOOLOGY_ROOT")"
  git clone --filter=blob:none https://github.com/HazyResearch/zoology.git "$ZOOLOGY_ROOT"
  git -C "$ZOOLOGY_ROOT" checkout --detach "$ZOOLOGY_COMMIT"
fi
if [[ "${REQUIRE_ZOOLOGY:-0}" == "1" || "${DOWNLOAD_ZOOLOGY:-0}" == "1" ]]; then
  if [[ ! -d "$ZOOLOGY_ROOT/.git" ]]; then
    printf 'Pinned Zoology checkout is missing: %s\n' "$ZOOLOGY_ROOT" >&2
    exit 6
  fi
  ZOOLOGY_ACTUAL="$(git -C "$ZOOLOGY_ROOT" rev-parse HEAD)"
  if [[ "$ZOOLOGY_ACTUAL" != "$ZOOLOGY_COMMIT" ]]; then
    printf 'Pinned Zoology SHA mismatch: %s != %s\n' "$ZOOLOGY_ACTUAL" "$ZOOLOGY_COMMIT" >&2
    exit 6
  fi
  printf 'Pinned Zoology source ready: %s @ %s\n' "$ZOOLOGY_ROOT" "$ZOOLOGY_ACTUAL"
fi
