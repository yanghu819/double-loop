#!/usr/bin/env bash
set -euo pipefail

PERSIST_ROOT=/huyang2/double-loop
WORKTREE="$PERSIST_ROOT/.worktrees/fla-seed-survival-d9b4963-20260721"
SOURCE_SHA=d9b496378ff2748d86a59e5c7c2930b836d33cd4
RUN_NAME=fla-futureseed-state-survival-20260721T001116Z-d9b4963
RUN_ROOT="$PERSIST_ROOT/runs/$RUN_NAME"
PYTHON_BIN=/opt/conda/bin/python

export CUDA_VISIBLE_DEVICES=0
export FLA_DISABLE_BACKEND_DISPATCH=1
export FLA_CONV_BACKEND=triton
export FLA_STRICT_OFFICIAL=1
export XDG_CACHE_HOME="$PERSIST_ROOT/.cache"
export TRITON_CACHE_DIR="$PERSIST_ROOT/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PERSIST_ROOT/.cache/torchinductor"
export TORCH_EXTENSIONS_DIR="$PERSIST_ROOT/.cache/torch_extensions"
export TMPDIR="$PERSIST_ROOT/.cache/tmp"
export PYTHONPATH="$PERSIST_ROOT/.cache/python-extra-pylib"

mkdir -p \
  "$RUN_ROOT" \
  "$XDG_CACHE_HOME" \
  "$TRITON_CACHE_DIR" \
  "$TORCHINDUCTOR_CACHE_DIR" \
  "$TORCH_EXTENSIONS_DIR" \
  "$TMPDIR"

if [[ "$(git -C "$WORKTREE" rev-parse HEAD)" != "$SOURCE_SHA" ]]; then
  echo "wrong detached source SHA" >&2
  exit 10
fi
if [[ -n "$(git -C "$WORKTREE" status --short --untracked-files=no)" ]]; then
  echo "tracked worktree is dirty" >&2
  exit 11
fi

{
  printf 'source_sha=%s\n' "$SOURCE_SHA"
  printf 'run_name=%s\n' "$RUN_NAME"
  printf 'started_at_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'cuda_visible_devices=%s\n' "$CUDA_VISIBLE_DEVICES"
  printf 'python=%s\n' "$PYTHON_BIN"
  printf 'fla_disable_backend_dispatch=%s\n' "$FLA_DISABLE_BACKEND_DISPATCH"
  printf 'fla_conv_backend=%s\n' "$FLA_CONV_BACKEND"
  printf 'fla_strict_official=%s\n' "$FLA_STRICT_OFFICIAL"
} > "$RUN_ROOT/launch.env"
printf '%s\n' "$SOURCE_SHA" > "$RUN_ROOT/source_HEAD.txt"
nvidia-smi --query-gpu=name,index,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$RUN_ROOT/gpu_before.csv"

on_exit() {
  status=$?
  completed_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ $status -eq 0 ]]; then
    printf '{"status":"complete","exit_code":0,"completed_at_utc":"%s"}\n' "$completed_at" > "$RUN_ROOT/status.json"
  else
    printf '{"status":"failed","exit_code":%d,"completed_at_utc":"%s"}\n' "$status" "$completed_at" > "$RUN_ROOT/status.json"
  fi
  nvidia-smi --query-gpu=name,index,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$RUN_ROOT/gpu_after.csv" || true
}
trap on_exit EXIT

cd "$WORKTREE/experiments/rwkv_fs_sudoku"
"$PYTHON_BIN" check_fla_delta_backbones.py \
  --backbone all \
  --wheel "$PERSIST_ROOT/wheelhouse/flash_linear_attention-0.5.2-py3-none-any.whl" \
  --out "$RUN_ROOT/strict_gate.json"

"$PYTHON_BIN" diagnose_fla_futureseed_survival.py \
  --wheel "$PERSIST_ROOT/wheelhouse/flash_linear_attention-0.5.2-py3-none-any.whl" \
  --data-dir "$PERSIST_ROOT/data/sudoku-extreme-full" \
  --split test \
  --batch-size 4 \
  --input-seed 52003 \
  --holes-min 51 \
  --holes-max 55 \
  --prefixes 1,8,32,81 \
  --checkpoint "fla_gdn=$PERSIST_ROOT/models/fla-official-gdn-fs-d192l10-s1500-20260720T0825Z-c3342b8/checkpoints/train_state_step000500.pt" \
  --checkpoint "kda=$PERSIST_ROOT/models/fla-official-kda-fs-d192l10-s500-20260720T0908Z-c3342b8/checkpoints/train_state_step000500.pt" \
  --checkpoint "gdn2=$PERSIST_ROOT/models/fla-official-gdn2-fs-d192l10-s500-clean-20260720T1034Z-c3342b8/checkpoints/train_state_step000500.pt" \
  --plan-id P-LA-003 \
  --out-dir "$RUN_ROOT"
