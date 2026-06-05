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
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$REPO_ROOT/.cache/torch_extensions}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PATH="$REPO_ROOT/.cache/bin:$PATH"

mkdir -p "$REPO_ROOT/.cache" "$TORCH_EXTENSIONS_DIR" "$REPO_ROOT/artifacts" "$REPO_ROOT/models" "$REPO_ROOT/runs"

case "$MODE" in
  smoke|full|eqr_probe) ;;
  *)
    printf 'Usage: %s [smoke|full|eqr_probe]\n' "$0" >&2
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

PYTHON_EXTRA_PATH="${PYTHON_EXTRA_PATH:-}"
if [[ -z "$PYTHON_EXTRA_PATH" && -s "$REPO_ROOT/.cache/python-extra-path" ]]; then
  PYTHON_EXTRA_PATH="$(<"$REPO_ROOT/.cache/python-extra-path")"
fi
if [[ -n "$PYTHON_EXTRA_PATH" ]]; then
  export PYTHONPATH="$PYTHON_EXTRA_PATH${PYTHONPATH:+:$PYTHONPATH}"
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
  SMOKE_DONE=1 SKIP_SETUP=1 RUN_NAME= "$0" smoke
fi

GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
GIT_DIRTY=0
SOURCE_STATUS="$(git -C "$REPO_ROOT" status --short -- . \
  ':(exclude).cache' \
  ':(exclude).venv' \
  ':(exclude)artifacts' \
  ':(exclude)models' \
  ':(exclude)runs')"
if [[ -n "$SOURCE_STATUS" ]]; then
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

if [[ "$MODE" == "eqr_probe" ]]; then
  if [[ "${SKIP_DOWN:-0}" != "1" ]]; then
    "$REPO_ROOT/down.sh"
  fi

  EQR_ARGS=(
    --repo_root "$REPO_ROOT"
    --eqr_dir "${EQR_DIR:-$REPO_ROOT/repos/eqr}"
    --out_dir "$OUT_DIR"
    --size "${SUDOKU_SIZE:-9}"
    --steps "${EQR_STEPS:-120}"
    --batch "${EQR_BATCH:-64}"
    --eval_n "${EQR_EVAL_N:-128}"
    --holes_min "${EQR_HOLES_MIN:-4}"
    --holes_max "${EQR_HOLES_MAX:-10}"
    --hole_pattern "${HOLE_PATTERN:-random}"
    --eval_holes "${EQR_EVAL_HOLES:-8}"
    --eval_holes_list "${EQR_EVAL_HOLES_LIST:-8,12}"
    --hidden_size "${EQR_HIDDEN_SIZE:-192}"
    --heads "${EQR_HEADS:-6}"
    --layers "${EQR_LAYERS:-2}"
    --h_cycles "${EQR_H_CYCLES:-2}"
    --l_cycles "${EQR_L_CYCLES:-4}"
    --train_loops "${EQR_TRAIN_LOOPS:-4}"
    --noise_scale "${EQR_NOISE_SCALE:-0.01}"
    --noise_mode "${EQR_NOISE_MODE:-feature_diff}"
    --feature_noise_buffer_size "${EQR_FEATURE_NOISE_BUFFER_SIZE:-512}"
    --feature_noise_buffer_add "${EQR_FEATURE_NOISE_BUFFER_ADD:-64}"
    --future_seed_scale "${EQR_FUTURE_SEED_SCALE:-1.0}"
    --future_seed_gate_bias "${EQR_FUTURE_SEED_GATE_BIAS:--2.0}"
    --forward_dtype "${EQR_FORWARD_DTYPE:-bfloat16}"
    --lr "${EQR_LR:-3e-4}"
    --weight_decay "${EQR_WEIGHT_DECAY:-0.1}"
    --blank_loss_weight "${BLANK_LOSS_WEIGHT:-8.0}"
    --loop_loss "${EQR_LOOP_LOSS:-final}"
    --seed "${SEED:-52}"
    --log_every "${EQR_LOG_EVERY:-50}"
  )
  if [[ -n "${EQR_HOLE_STAGES:-}" ]]; then
    EQR_ARGS+=(--hole_stages "$EQR_HOLE_STAGES")
  fi
  if [[ "${CPU:-0}" == "1" ]]; then
    EQR_ARGS+=(--cpu)
  fi

  printf 'mode=%s\nrun_dir=%s\ngit_sha=%s\ngit_dirty=%s\n' "$MODE" "$RUN_DIR" "$GIT_SHA" "$GIT_DIRTY" | tee "$LOG_DIR/run.log"
  (
    cd "$REPO_ROOT"
    if [[ -n "$PYTHON_BIN" ]]; then
      "$PYTHON_BIN" scripts/eqr_nohydra_probe.py "${EQR_ARGS[@]}"
    else
      UV_PROJECT_ENVIRONMENT="$REPO_ROOT/.venv" "$UV_BIN" run python scripts/eqr_nohydra_probe.py "${EQR_ARGS[@]}"
    fi
  ) 2>&1 | tee -a "$LOG_DIR/run.log"

  python3 "$REPO_ROOT/scripts/record_experiment.py" --run-dir "$RUN_DIR" --mode "$MODE"
  printf 'completed run_dir=%s\n' "$RUN_DIR"
  exit 0
fi

COMMON_ARGS=(
  --size "${SUDOKU_SIZE:-6}"
  --box_rows "${BOX_ROWS:-0}"
  --box_cols "${BOX_COLS:-0}"
  --max_loops "${MAX_LOOPS:-3}"
  --d_model "${D_MODEL:-32}"
  --layers "${LAYERS:-4}"
  --heads "${HEADS:-4}"
  --head_dim "${HEAD_DIM:-8}"
  --channel_mult "${CHANNEL_MULT:-2}"
  --l_cycles "${L_CYCLES:-1}"
  --holes_min "${HOLES_MIN:-2}"
  --holes_max "${HOLES_MAX:-4}"
  --eval_holes "${EVAL_HOLES:-2}"
  --hole_pattern "${HOLE_PATTERN:-unit}"
  --blank_loss_weight "${BLANK_LOSS_WEIGHT:-20}"
  --noise_scale "${NOISE_SCALE:-0.01}"
  --rollout_noise_scale "${ROLLOUT_NOISE_SCALE:-0.05}"
  --lr "${LR:-2e-3}"
  --weight_decay "${WEIGHT_DECAY:-1e-3}"
  --feature_buffer_size "${FEATURE_BUFFER_SIZE:-8192}"
  --feature_buffer_add "${FEATURE_BUFFER_ADD:-2048}"
  --future_seed_scale "${FUTURE_SEED_SCALE:-1.0}"
  --rwkv_kernel "${RWKV_KERNEL:-auto}"
  --lambda_ "${LAMBDA:-0.95}"
  --loop_loss "${LOOP_LOSS:-final}"
  --loop_loss_start "${LOOP_LOSS_START:-1}"
  --loop_loss_power "${LOOP_LOSS_POWER:-2.0}"
  --loop_loss_min_weight "${LOOP_LOSS_MIN_WEIGHT:-0.05}"
  --unit_loss_weight "${UNIT_LOSS_WEIGHT:-0.0}"
  --unit_state_scale "${UNIT_STATE_SCALE:-0.0}"
  --unit_state_gate_bias "${UNIT_STATE_GATE_BIAS:--2.0}"
  --unit_state_mode "${UNIT_STATE_MODE:-pooled}"
  --unit_state_memory_decay "${UNIT_STATE_MEMORY_DECAY:-0.5}"
  --unit_state_token_scale "${UNIT_STATE_TOKEN_SCALE:-1.0}"
  --out_dir "$OUT_DIR"
)

if [[ -n "${HOLE_STAGES:-}" ]]; then
  COMMON_ARGS+=(--hole_stages "$HOLE_STAGES")
fi
if [[ -n "${EVAL_HOLES_LIST:-}" ]]; then
  COMMON_ARGS+=(--eval_holes_list "$EVAL_HOLES_LIST")
fi
if [[ -n "${EVAL_HOLE_PATTERNS:-}" ]]; then
  COMMON_ARGS+=(--eval_hole_patterns "$EVAL_HOLE_PATTERNS")
fi
if [[ -n "${ROLLOUT_LOOP_VALUES:-}" ]]; then
  COMMON_ARGS+=(--rollout_loop_values "$ROLLOUT_LOOP_VALUES")
fi

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

RECORD_ARGS=(--run-dir "$RUN_DIR" --mode "$MODE")
if [[ "$MODE" == "smoke" && "${UPDATE_LEADERBOARD:-0}" != "1" ]]; then
  RECORD_ARGS+=(--no-leaderboard)
fi
python3 "$REPO_ROOT/scripts/record_experiment.py" "${RECORD_ARGS[@]}"

printf 'completed run_dir=%s\n' "$RUN_DIR"
