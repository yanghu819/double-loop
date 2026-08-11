#!/usr/bin/env bash
set -euo pipefail

MODE="${1:?usage: run_gdn2_binding_diagnostic.sh contract|full [out_dir]}"
PERSIST_ROOT="${PERSIST_ROOT:-/huyang2/double-loop}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
FLA_SHA="9c8e42e762fce087c27b673af4922795d9edb85e"
FLA_ROOT="$PERSIST_ROOT/.cache/fla-versions/${FLA_SHA}-0280db310981915e"
ZOOLOGY_ROOT="${ZOOLOGY_ROOT:-$PERSIST_ROOT/repos/zoology-official}"
ZOOLOGY_SHA="1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb"
EXPECTED_GPU_UUID="${EXPECTED_GPU_UUID:-GPU-e7f175ea-1d38-93c4-2d07-fc1a938dc9d2}"
SUDOKU_CHECKPOINT="${SUDOKU_CHECKPOINT:-$PERSIST_ROOT/models/gdn3-position-qk-d256l12-s12000-20260806T131527Z-9f2ee8d/checkpoints/train_state_step003000.pt}"
SUDOKU_DATA_DIR="${SUDOKU_DATA_DIR:-$PERSIST_ROOT/data/sudoku-extreme-full}"
OUT_DIR="${2:-$PERSIST_ROOT/runs/p-bind-001-$(date -u +%Y%m%dT%H%M%SZ)-${SOURCE_SHA:0:7}}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"

[[ "$MODE" == "contract" || "$MODE" == "full" ]] || {
  printf 'mode must be contract or full\n' >&2
  exit 2
}
[[ "${CUDA_VISIBLE_DEVICES:-0}" == "0" ]] || {
  printf 'P-BIND-001 requires CUDA_VISIBLE_DEVICES=0\n' >&2
  exit 2
}
[[ -d "$FLA_ROOT/fla" ]] || {
  printf 'pinned FLA tree missing: %s\n' "$FLA_ROOT" >&2
  exit 2
}
[[ -d "$ZOOLOGY_ROOT/zoology" ]] || {
  printf 'pinned Zoology tree missing: %s\n' "$ZOOLOGY_ROOT" >&2
  exit 2
}
[[ "$(git -C "$ZOOLOGY_ROOT" rev-parse HEAD)" == "$ZOOLOGY_SHA" ]] || {
  printf 'pinned Zoology SHA mismatch\n' >&2
  exit 2
}
[[ -z "$(git -C "$ZOOLOGY_ROOT" status --porcelain)" ]] || {
  printf 'pinned Zoology worktree is dirty\n' >&2
  exit 2
}
[[ -f "$SUDOKU_CHECKPOINT" ]] || {
  printf 'Sudoku parent missing: %s\n' "$SUDOKU_CHECKPOINT" >&2
  exit 2
}
[[ -d "$SUDOKU_DATA_DIR/test" ]] || {
  printf 'official Sudoku test split missing: %s\n' "$SUDOKU_DATA_DIR" >&2
  exit 2
}
[[ -z "$(git -C "$REPO_ROOT" status --porcelain)" ]] || {
  printf 'formal source worktree is dirty\n' >&2
  exit 2
}

export CUDA_VISIBLE_DEVICES=0
export PERSIST_ROOT
export ZOOLOGY_ROOT
export FLA_EXPECTED_SOURCE_SHA="$FLA_SHA"
export FLA_DISABLE_BACKEND_DISPATCH=1
export FLA_CONV_BACKEND=triton
export FLA_STRICT_OFFICIAL=1
export XDG_CACHE_HOME="$PERSIST_ROOT/.cache"
export TRITON_CACHE_DIR="$PERSIST_ROOT/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PERSIST_ROOT/.cache/torchinductor"
export TORCH_EXTENSIONS_DIR="$PERSIST_ROOT/.cache/torch_extensions"
export TMPDIR="$PERSIST_ROOT/.cache/tmp"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/experiments/rwkv_fs_sudoku:$ZOOLOGY_ROOT:$FLA_ROOT:$PERSIST_ROOT/.cache/python-extra-pylib${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUT_DIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
  "$TORCH_EXTENSIONS_DIR" "$TMPDIR"

exec "$PYTHON_BIN" -m experiments.gdn2_diagnostics.run_binding_interference \
  --mode "$MODE" \
  --repo "$REPO_ROOT" \
  --source-sha "$SOURCE_SHA" \
  --expected-gpu-uuid "$EXPECTED_GPU_UUID" \
  --sudoku-checkpoint "$SUDOKU_CHECKPOINT" \
  --sudoku-data-dir "$SUDOKU_DATA_DIR" \
  --sudoku-batch-size 32 \
  --mqar-diagnostic-examples 128 \
  --out-dir "$OUT_DIR"
