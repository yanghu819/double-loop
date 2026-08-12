#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-/huyang2/double-loop}"
EXPECTED_UUID="${EXPECTED_UUID:?Set EXPECTED_UUID to the admitted single GPU UUID}"
EXPECTED_FLA_SHA="9c8e42e762fce087c27b673af4922795d9edb85e"
PARENT_CHECKPOINT="${PARENT_CHECKPOINT:-$PERSIST_ROOT/models/gdn3-position-qk-d256l12-s12000-20260806T131527Z-9f2ee8d/checkpoints/train_state_step003000.pt}"
PARENT_CHECKPOINT_SHA256="6339c3cb2b5fc5230a581d6633716483e35ff8e4522f06a9d7aaf26512f023da"
PARENT_SOURCE_SHA="9f2ee8d1738032bc5f09b55db0b81d507780b376"
DATA_DIR="${DATA_DIR:-$PERSIST_ROOT/data/sudoku-extreme-full}"
RUN_NAME="${RUN_NAME:-p-loop-003-phase-credit-$(date -u +%Y%m%dT%H%M%SZ)-$(git -C "$REPO_ROOT" rev-parse --short=7 HEAD)}"
RUN_DIR="$PERSIST_ROOT/runs/$RUN_NAME"
CACHE_ROOT="$PERSIST_ROOT/.cache/p-loop-003"
COMPLETED=0

record_abort() {
  local status=$?
  trap - EXIT
  if [[ "$COMPLETED" != "1" ]]; then
    mkdir -p "$RUN_DIR"
    printf '{"plan_id":"P-LOOP-003","run_name":"%s","exit_code":%d,"timestamp_utc":"%s","scientific_failure":false}\n' \
      "$RUN_NAME" "$status" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      > "$RUN_DIR/abort.json"
  fi
  exit "$status"
}
trap record_abort EXIT

if [[ "${CUDA_VISIBLE_DEVICES:-0}" != "0" ]]; then
  printf 'P-LOOP-003 requires CUDA_VISIBLE_DEVICES=0.\n' >&2
  exit 3
fi
export CUDA_VISIBLE_DEVICES=0
VISIBLE_GPU="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)"
if [[ "$VISIBLE_GPU" != "0, $EXPECTED_UUID" ]]; then
  printf 'Unexpected visible GPU contract: %s\n' "$VISIBLE_GPU" >&2
  exit 4
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; then
  printf 'P-LOOP-003 refuses to overlap an existing GPU compute process.\n' >&2
  exit 5
fi
if git -C "$REPO_ROOT" symbolic-ref -q HEAD >/dev/null; then
  printf 'P-LOOP-003 requires a detached source worktree.\n' >&2
  exit 6
fi
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
  printf 'P-LOOP-003 requires a clean source worktree.\n' >&2
  exit 7
fi
if [[ "$(sha256sum "$PARENT_CHECKPOINT" | awk '{print $1}')" != "$PARENT_CHECKPOINT_SHA256" ]]; then
  printf 'Parent checkpoint SHA mismatch.\n' >&2
  exit 8
fi

export PERSIST_ROOT
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export TORCH_EXTENSIONS_DIR="$CACHE_ROOT/torch_extensions"
export TMPDIR="$CACHE_ROOT/tmp"
export TORCH_HOME="$CACHE_ROOT/torch"
export HF_HOME="$CACHE_ROOT/huggingface"
export PATH="$PERSIST_ROOT/.cache/bin:$PATH"
export PYTHONPATH="$REPO_ROOT/experiments/rwkv_fs_sudoku:$PERSIST_ROOT/.cache/fla-active:$PERSIST_ROOT/.cache/python-extra-pylib${PYTHONPATH:+:$PYTHONPATH}"
export FLA_EXPECTED_SOURCE_SHA="$EXPECTED_FLA_SHA"
export FLA_SOURCE_SHA_MARKER="$PERSIST_ROOT/.cache/fla-source-sha"
export FLA_DISABLE_BACKEND_DISPATCH=1
export FLA_CONV_BACKEND=triton
export FLA_STRICT_OFFICIAL=1
unset FLA_SOURCE_ROOT

mkdir -p "$RUN_DIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$TMPDIR" \
  "$TORCH_HOME" "$HF_HOME"
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_DIR/source_HEAD.txt"
git -C "$REPO_ROOT" diff --binary > "$RUN_DIR/source.patch"
printf '%s\n' "$VISIBLE_GPU" > "$RUN_DIR/gpu.txt"

PYTHON_BIN="${PYTHON_BIN:-$PERSIST_ROOT/official_eqr_compare/.venv/bin/python}"
cd "$REPO_ROOT"
"$PYTHON_BIN" experiments/rwkv_fs_sudoku/diagnose_futureseed_phase_credit_cuda.py \
  --checkpoint "$PARENT_CHECKPOINT" \
  --checkpoint-sha256 "$PARENT_CHECKPOINT_SHA256" \
  --parent-source-sha "$PARENT_SOURCE_SHA" \
  --data-dir "$DATA_DIR" \
  --out "$RUN_DIR/futureseed_phase_credit.json" \
  --expected-gpu-uuid "$EXPECTED_UUID" \
  --batch-size 8 \
  --eval-seed 52051 \
  --blank-ranges 51-55 56-60 61-64 \
  --blank-weight 8 \
  2>&1 | tee "$RUN_DIR/run.log"
sha256sum "$RUN_DIR"/futureseed_phase_credit.json "$RUN_DIR"/run.log \
  "$RUN_DIR"/source_HEAD.txt "$RUN_DIR"/source.patch > "$RUN_DIR/hashes.sha256"
printf '0\n' > "$RUN_DIR/status"
COMPLETED=1
