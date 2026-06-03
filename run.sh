#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$REPO_ROOT/experiments/rwkv_fs_sudoku"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_ROOT/.cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_ROOT/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$REPO_ROOT/.cache/uv/python}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$REPO_ROOT/.cache/pip}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$REPO_ROOT/.cache/torch}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p "$REPO_ROOT/.cache" "$REPO_ROOT/artifacts" "$REPO_ROOT/models" "$REPO_ROOT/runs"

case "$MODE" in
  smoke|full) ;;
  *)
    printf 'Usage: %s [smoke|full]\n' "$0" >&2
    exit 2
    ;;
esac

if [[ "${SKIP_SETUP:-0}" != "1" ]]; then
  "$REPO_ROOT/setup.sh"
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" && -s "$REPO_ROOT/.cache/python-bin" ]]; then
  PYTHON_BIN="$(<"$REPO_ROOT/.cache/python-bin")"
fi
if [[ -n "$PYTHON_BIN" && ! -x "$PYTHON_BIN" ]]; then
  printf 'Configured PYTHON_BIN is not executable: %s\n' "$PYTHON_BIN" >&2
  exit 1
fi

UV_BIN="${UV_BIN:-}"
if [[ -z "$PYTHON_BIN" && -z "$UV_BIN" ]]; then
  if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
  else
    UV_BIN="$REPO_ROOT/.cache/uv-bootstrap/bin/uv"
  fi
fi

if [[ -z "$PYTHON_BIN" && ! -x "$UV_BIN" ]]; then
  printf 'uv is not available; run ./setup.sh first.\n' >&2
  exit 1
fi

if [[ "$MODE" == "full" && "${SMOKE_DONE:-0}" != "1" ]]; then
  SMOKE_DONE=1 SKIP_SETUP=1 "$0" smoke
fi

GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
GIT_DIRTY=0
if [[ -n "$(git -C "$REPO_ROOT" status --short)" ]]; then
  GIT_DIRTY=1
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="${RUN_NAME:-$MODE-$TS-${GIT_SHA:0:7}}"
RUN_DIR="$REPO_ROOT/runs/$RUN_NAME"
OUT_DIR="$RUN_DIR/output"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_DIR/source_HEAD.txt"
git -C "$REPO_ROOT" diff --binary > "$RUN_DIR/source.patch" || true
git -C "$REPO_ROOT" archive --format=tar HEAD | gzip > "$RUN_DIR/source_snapshot.tar.gz"

python3 - "$RUN_DIR/config.json" "$MODE" "$GIT_SHA" "$GIT_DIRTY" "$RUN_NAME" "$RUN_DIR" <<'PY'
import json
import sys
from datetime import datetime, timezone

path, mode, git_sha, git_dirty, run_name, run_dir = sys.argv[1:]
payload = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "mode": mode,
    "git_sha": git_sha,
    "git_dirty": bool(int(git_dirty)),
    "run_name": run_name,
    "run_dir": run_dir,
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
PY

COMMON_ARGS=(
  --size 6
  --max_loops 3
  --d_model 32
  --layers 4
  --heads 4
  --head_dim 8
  --channel_mult 2
  --l_cycles 1
  --holes_min 2
  --holes_max 4
  --eval_holes 2
  --hole_pattern unit
  --blank_loss_weight 20
  --noise_scale 0.01
  --rollout_noise_scale 0.05
  --out_dir "$OUT_DIR"
)

if [[ "$MODE" == "smoke" ]]; then
  RUN_ARGS=(
    "${COMMON_ARGS[@]}"
    --steps "${SMOKE_STEPS:-2}"
    --batch "${SMOKE_BATCH:-2}"
    --eval_n "${SMOKE_EVAL_N:-4}"
    --rollout_ks "${SMOKE_ROLLOUT_KS:-1}"
    --log_every 1
  )
else
  RUN_ARGS=(
    "${COMMON_ARGS[@]}"
    --steps "${FULL_STEPS:-300}"
    --batch "${FULL_BATCH:-32}"
    --eval_n "${FULL_EVAL_N:-256}"
    --rollout_ks "${FULL_ROLLOUT_KS:-1,4,8,16}"
    --log_every "${FULL_LOG_EVERY:-50}"
  )
fi

printf 'mode=%s\nrun_dir=%s\ngit_sha=%s\ngit_dirty=%s\n' "$MODE" "$RUN_DIR" "$GIT_SHA" "$GIT_DIRTY" | tee "$LOG_DIR/run.log"

(
  cd "$EXP_DIR"
  if [[ -n "$PYTHON_BIN" ]]; then
    "$PYTHON_BIN" study_rwkv_futureseed_loop.py "${RUN_ARGS[@]}"
  else
    UV_PROJECT_ENVIRONMENT="$REPO_ROOT/.venv" "$UV_BIN" run python study_rwkv_futureseed_loop.py "${RUN_ARGS[@]}"
  fi
) 2>&1 | tee -a "$LOG_DIR/run.log"
