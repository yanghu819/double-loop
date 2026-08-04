#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-$REPO_ROOT}"
EXP_DIR="$REPO_ROOT/experiments/rwkv_fs_sudoku"
RUNS_ROOT="${RUNS_ROOT:-$PERSIST_ROOT/runs}"

case "$MODE" in
  baseline)
    exec "$REPO_ROOT/scripts/run_canonical_gdn2_scale.sh"
    ;;
  gdn_legacy)
    exec "$REPO_ROOT/scripts/run_canonical_gdn_scale.sh"
    ;;
  benchmark)
    exec "$REPO_ROOT/scripts/run_sudoku_backbone_benchmark.sh" "${2:?Usage: ./run.sh benchmark <rwkv|gdn|gdn2|kda>}"
    ;;
  benchmark_suite)
    exec "$REPO_ROOT/scripts/run_sudoku_backbone_suite.sh"
    ;;
  baseline_preflight)
    exec "$REPO_ROOT/scripts/run_sudoku_baseline_preflight.sh" "${2:-}"
    ;;
  established_mlm)
    exec "$REPO_ROOT/scripts/run_established_bert_mlm.sh" "${2:?Usage: ./run.sh established_mlm <bidirectional|causal>}"
    ;;
  mqar_length_scaling)
    exec "$REPO_ROOT/scripts/run_zoology_mqar_length_scaling.sh"
    ;;
  official_bidir_mqar)
    exec "$REPO_ROOT/scripts/run_zoology_official_bidirectional_mqar.sh"
    ;;
esac

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$PERSIST_ROOT/.cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PERSIST_ROOT/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$PERSIST_ROOT/.cache/uv/python}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PERSIST_ROOT/.cache/pip}"
export HF_HOME="${HF_HOME:-$PERSIST_ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$PERSIST_ROOT/.cache/torch}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PERSIST_ROOT/.cache/torch_extensions}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PATH="$PERSIST_ROOT/.cache/bin:$PATH"

mkdir -p "$PERSIST_ROOT/.cache" "$TORCH_EXTENSIONS_DIR" "$PERSIST_ROOT/artifacts" "$PERSIST_ROOT/models" "$RUNS_ROOT"

case "$MODE" in
  smoke|full|eqr_probe|eqr_maze_probe|rwkv_maze_probe|gdn2_retrieval_probe|gdn2_zoology_mqar_probe) ;;
  *)
    printf 'Usage: %s [baseline|gdn_legacy|benchmark|benchmark_suite|baseline_preflight|established_mlm|mqar_length_scaling|official_bidir_mqar|smoke|full|eqr_probe|eqr_maze_probe|rwkv_maze_probe|gdn2_retrieval_probe|gdn2_zoology_mqar_probe]\n' "$0" >&2
    exit 2
    ;;
esac

if [[ "${SKIP_SETUP:-0}" != "1" ]]; then
  "$REPO_ROOT/setup.sh"
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" && -s "$PERSIST_ROOT/.cache/python-bin" ]]; then
  PYTHON_BIN="$(<"$PERSIST_ROOT/.cache/python-bin")"
fi
if [[ -n "$PYTHON_BIN" && ! -x "$PYTHON_BIN" ]]; then
  printf 'Configured PYTHON_BIN is not executable: %s\n' "$PYTHON_BIN" >&2
  exit 1
fi

PYTHON_EXTRA_PATH="${PYTHON_EXTRA_PATH:-}"
if [[ -z "$PYTHON_EXTRA_PATH" && -s "$PERSIST_ROOT/.cache/python-extra-path" ]]; then
  PYTHON_EXTRA_PATH="$(<"$PERSIST_ROOT/.cache/python-extra-path")"
fi
if [[ -n "$PYTHON_EXTRA_PATH" ]]; then
  export PYTHONPATH="$PYTHON_EXTRA_PATH${PYTHONPATH:+:$PYTHONPATH}"
fi

UV_BIN="${UV_BIN:-}"
if [[ -z "$PYTHON_BIN" && -z "$UV_BIN" ]]; then
  if command -v uv >/dev/null 2>&1; then
    UV_BIN="$(command -v uv)"
  else
    UV_BIN="$PERSIST_ROOT/.cache/uv-bootstrap/bin/uv"
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
if [[ "$GIT_DIRTY" == "1" && "${REQUIRE_CLEAN_SOURCE:-0}" == "1" ]]; then
  printf 'Refusing to run with a dirty source tree:\n%s\n' "$SOURCE_STATUS" >&2
  exit 5
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_NAME="${RUN_NAME:-$MODE-$TS-${GIT_SHA:0:7}}"
RUN_DIR="$RUNS_ROOT/$RUN_NAME"
OUT_DIR="$RUN_DIR/output"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_DIR/source_HEAD.txt"
git -C "$REPO_ROOT" diff --binary > "$RUN_DIR/source.patch" || true
if [[ "${SOURCE_SNAPSHOT_MODE:-full}" == "lean" ]]; then
  git -C "$REPO_ROOT" ls-files -z -- . \
    ':(exclude).cache/**' \
    ':(exclude).venv/**' \
    ':(exclude)artifacts/**' \
    ':(exclude)models/**' \
    ':(exclude)repos/**' \
    ':(exclude)runs/**' \
    | tar --null -czf "$RUN_DIR/source_snapshot.tar.gz" -C "$REPO_ROOT" --files-from -
else
  git -C "$REPO_ROOT" archive --format=tar HEAD | gzip > "$RUN_DIR/source_snapshot.tar.gz"
fi

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

if [[ "$MODE" == "eqr_maze_probe" ]]; then
  MAZE_ARGS=(
    --repo_root "$REPO_ROOT"
    --eqr_dir "${EQR_DIR:-$REPO_ROOT/repos/eqr}"
    --out_dir "$OUT_DIR"
    --grid_size "${MAZE_GRID_SIZE:-15}"
    --maze_mode "${MAZE_MODE:-perfect}"
    --wall_prob "${MAZE_WALL_PROB:-0.37}"
    --min_path_length "${MAZE_MIN_PATH_LENGTH:-32}"
    --max_path_length "${MAZE_MAX_PATH_LENGTH:-56}"
    --path_stages "${MAZE_PATH_STAGES:-}"
    --max_grid_attempts "${MAZE_MAX_GRID_ATTEMPTS:-200}"
    --max_start_attempts "${MAZE_MAX_START_ATTEMPTS:-200}"
    --steps "${MAZE_STEPS:-600}"
    --batch "${MAZE_BATCH:-128}"
    --eval_n "${MAZE_EVAL_N:-512}"
    --hidden_size "${EQR_HIDDEN_SIZE:-128}"
    --heads "${EQR_HEADS:-4}"
    --layers "${EQR_LAYERS:-1}"
    --h_cycles "${EQR_H_CYCLES:-2}"
    --l_cycles "${EQR_L_CYCLES:-4}"
    --train_loops "${EQR_TRAIN_LOOPS:-4}"
    --eval_loops "${EQR_EVAL_LOOPS:-8}"
    --expansion "${EQR_EXPANSION:-4.0}"
    --lambda_ "${EQR_LAMBDA:-0.95}"
    --noise_scale "${EQR_NOISE_SCALE:-0.0}"
    --noise_mode "${EQR_NOISE_MODE:-none}"
    --future_seed_scale "${EQR_FUTURE_SEED_SCALE:-1.0}"
    --future_seed_gate_bias "${EQR_FUTURE_SEED_GATE_BIAS:--2.0}"
    --feedback_mode "${EQR_FEEDBACK_MODE:-none}"
    --feedback_scale "${EQR_FEEDBACK_SCALE:-0.0}"
    --feedback_gate_bias "${EQR_FEEDBACK_GATE_BIAS:--2.0}"
    --state_update_mode "${EQR_STATE_UPDATE_MODE:-none}"
    --state_delta_scale "${EQR_STATE_DELTA_SCALE:-0.0}"
    --state_delta_decay "${EQR_STATE_DELTA_DECAY:-1.0}"
    --state_gate_bias "${EQR_STATE_GATE_BIAS:-2.0}"
    --decision_boundary_token_id "${EQR_DECISION_BOUNDARY_TOKEN_ID:-5}"
    --decision_boundary_scale "${EQR_DECISION_BOUNDARY_SCALE:-1.0}"
    --forward_dtype "${EQR_FORWARD_DTYPE:-bfloat16}"
    --lr "${EQR_LR:-3e-4}"
    --weight_decay "${EQR_WEIGHT_DECAY:-0.1}"
    --path_loss_weight "${MAZE_PATH_LOSS_WEIGHT:-4.0}"
    --loop_loss "${EQR_LOOP_LOSS:-all}"
    --predictive_state_weight "${EQR_PREDICTIVE_STATE_WEIGHT:-0.0}"
    --predictive_state_horizon "${EQR_PREDICTIVE_STATE_HORIZON:-1}"
    --context_improve_weight "${EQR_CONTEXT_IMPROVE_WEIGHT:-0.0}"
    --context_improve_margin "${EQR_CONTEXT_IMPROVE_MARGIN:-0.01}"
    --context_rank_weight "${EQR_CONTEXT_RANK_WEIGHT:-0.0}"
    --context_rank_margin "${EQR_CONTEXT_RANK_MARGIN:-0.25}"
    --context_rank_mode "${EQR_CONTEXT_RANK_MODE:-hard}"
    --path_mass_weight "${EQR_PATH_MASS_WEIGHT:-0.0}"
    --path_mass_start_loop "${EQR_PATH_MASS_START_LOOP:-4}"
    --path_mass_recall_margin "${EQR_PATH_MASS_RECALL_MARGIN:-0.0}"
    --path_mass_mode "${EQR_PATH_MASS_MODE:-soft}"
    --self_correction_weight "${EQR_SELF_CORRECTION_WEIGHT:-0.0}"
    --self_correction_start_loop "${EQR_SELF_CORRECTION_START_LOOP:-4}"
    --self_correction_margin "${EQR_SELF_CORRECTION_MARGIN:-0.02}"
    --hard_correction_weight "${EQR_HARD_CORRECTION_WEIGHT:-0.0}"
    --hard_correction_after_step "${EQR_HARD_CORRECTION_AFTER_STEP:-1}"
    --hard_correction_start_loop "${EQR_HARD_CORRECTION_START_LOOP:-4}"
    --hard_correction_prune_margin "${EQR_HARD_CORRECTION_PRUNE_MARGIN:-0.25}"
    --hard_correction_preserve_margin "${EQR_HARD_CORRECTION_PRESERVE_MARGIN:-0.25}"
    --path_margin_weight "${EQR_PATH_MARGIN_WEIGHT:-0.0}"
    --path_margin_start_loop "${EQR_PATH_MARGIN_START_LOOP:-4}"
    --path_margin_positive "${EQR_PATH_MARGIN_POSITIVE:-0.25}"
    --path_margin_negative "${EQR_PATH_MARGIN_NEGATIVE:-0.25}"
    --path_tversky_weight "${EQR_PATH_TVERSKY_WEIGHT:-0.0}"
    --path_tversky_after_step "${EQR_PATH_TVERSKY_AFTER_STEP:-1}"
    --path_tversky_start_loop "${EQR_PATH_TVERSKY_START_LOOP:-4}"
    --path_tversky_alpha "${EQR_PATH_TVERSKY_ALPHA:-0.7}"
    --path_tversky_beta "${EQR_PATH_TVERSKY_BETA:-0.3}"
    --grad_clip "${EQR_GRAD_CLIP:-1.0}"
    --seed "${SEED:-52}"
    --log_every "${MAZE_LOG_EVERY:-100}"
    --viz_cases "${MAZE_VIZ_CASES:-12}"
    --viz_loops "${MAZE_VIZ_LOOPS:-1,2,4,6,8,10,12,16}"
  )

  printf 'mode=%s\nrun_dir=%s\ngit_sha=%s\ngit_dirty=%s\n' "$MODE" "$RUN_DIR" "$GIT_SHA" "$GIT_DIRTY" | tee "$LOG_DIR/run.log"
  (
    cd "$REPO_ROOT"
    if [[ -n "$PYTHON_BIN" ]]; then
      "$PYTHON_BIN" scripts/eqr_maze_probe.py "${MAZE_ARGS[@]}"
    else
      UV_PROJECT_ENVIRONMENT="$PERSIST_ROOT/.venv" "$UV_BIN" run python scripts/eqr_maze_probe.py "${MAZE_ARGS[@]}"
    fi
  ) 2>&1 | tee -a "$LOG_DIR/run.log"

  python3 "$REPO_ROOT/scripts/record_experiment.py" --run-dir "$RUN_DIR" --mode "$MODE"
  printf 'completed run_dir=%s\n' "$RUN_DIR"
  exit 0
fi

if [[ "$MODE" == "rwkv_maze_probe" ]]; then
  RWKV_MAZE_ARGS=(
    --repo-root "$REPO_ROOT"
    --data-dir "${RWKV_MAZE_DATA_DIR:-$REPO_ROOT/official_eqr_compare/eqr-clean/data/maze-30x30-unique-1k}"
    --out-dir "$OUT_DIR"
    --run-name "$RUN_NAME"
    --condition "${RWKV_MAZE_CONDITION:-}"
    --steps "${RWKV_MAZE_STEPS:-800}"
    --batch "${RWKV_MAZE_BATCH:-64}"
    --eval-n "${RWKV_MAZE_EVAL_N:-512}"
    --eval-batch "${RWKV_MAZE_EVAL_BATCH:-64}"
    --d-model "${D_MODEL:-128}"
    --layers "${LAYERS:-8}"
    --heads "${HEADS:-8}"
    --head-dim "${HEAD_DIM:-16}"
    --channel-mult "${CHANNEL_MULT:-4}"
    --l-cycles "${L_CYCLES:-2}"
    --train-loops "${MAX_LOOPS:-4}"
    --eval-loops "${EVAL_LOOPS:-8}"
    --lambda "${LAMBDA:-0.95}"
    --future-seed-scale "${FUTURE_SEED_SCALE:-1.0}"
    --future-seed-decay "${FUTURE_SEED_DECAY:-0.0}"
    --future-seed-update "${FUTURE_SEED_UPDATE:-fixed}"
    --backbone "${BACKBONE:-rwkv}"
    --rwkv-kernel "${RWKV_KERNEL:-statepassing}"
    --gdn-mode "${GDN_MODE:-chunk}"
    --gdn-expand-v "${GDN_EXPAND_V:-1.0}"
    --gdn-use-short-conv "${GDN_USE_SHORT_CONV:-1}"
    --gdn-conv-size "${GDN_CONV_SIZE:-4}"
    --forward-dtype "${FORWARD_DTYPE:-bfloat16}"
    --path-weight "${RWKV_MAZE_PATH_WEIGHT:-8.0}"
    --path-binary-weight "${RWKV_MAZE_PATH_BINARY_WEIGHT:-0.0}"
    --path-budget-weight "${RWKV_MAZE_PATH_BUDGET_WEIGHT:-0.0}"
    --path-count-weight "${RWKV_MAZE_PATH_COUNT_WEIGHT:-0.0}"
    --loop-loss "${LOOP_LOSS:-all}"
    --lr "${LR:-3e-4}"
    --weight-decay "${WEIGHT_DECAY:-0.1}"
    --grad-clip "${GRAD_CLIP:-1.0}"
    --seed "${SEED:-52}"
    --log-every "${LOG_EVERY:-100}"
    --viz-cases "${RWKV_MAZE_VIZ_CASES:-64}"
  )
  if [[ "${ACTIVATION_CHECKPOINT:-0}" == "1" ]]; then
    RWKV_MAZE_ARGS+=(--activation-checkpoint)
  fi
  if [[ "${FLA_STRICT_OFFICIAL:-0}" == "1" ]]; then
    RWKV_MAZE_ARGS+=(--fla-strict-official)
  fi
  if [[ "${RWKV_MAZE_BUDGET_DECODER:-0}" == "1" ]]; then
    RWKV_MAZE_ARGS+=(--budget-decoder)
  fi

  printf 'mode=%s\nrun_dir=%s\ngit_sha=%s\ngit_dirty=%s\n' "$MODE" "$RUN_DIR" "$GIT_SHA" "$GIT_DIRTY" | tee "$LOG_DIR/run.log"
  (
    cd "$REPO_ROOT"
    if [[ -n "$PYTHON_BIN" ]]; then
      "$PYTHON_BIN" scripts/rwkv_maze_probe.py "${RWKV_MAZE_ARGS[@]}"
    else
      UV_PROJECT_ENVIRONMENT="$PERSIST_ROOT/.venv" "$UV_BIN" run python scripts/rwkv_maze_probe.py "${RWKV_MAZE_ARGS[@]}"
    fi
  ) 2>&1 | tee -a "$LOG_DIR/run.log"

  printf 'completed run_dir=%s\n' "$RUN_DIR"
  exit 0
fi

if [[ "$MODE" == "gdn2_retrieval_probe" ]]; then
  RETRIEVAL_ARGS=(
    --out-dir "$OUT_DIR"
    --run-name "$RUN_NAME"
    --condition "${RETRIEVAL_CONDITION:-}"
    --steps "${RETRIEVAL_STEPS:-300}"
    --batch "${RETRIEVAL_BATCH:-128}"
    --eval-n "${RETRIEVAL_EVAL_N:-1024}"
    --eval-batch "${RETRIEVAL_EVAL_BATCH:-128}"
    --seq-len "${RETRIEVAL_SEQ_LEN:-128}"
    --pairs "${RETRIEVAL_PAIRS:-8}"
    --key-vocab "${RETRIEVAL_KEY_VOCAB:-64}"
    --value-vocab "${RETRIEVAL_VALUE_VOCAB:-32}"
    --d-model "${D_MODEL:-128}"
    --layers "${LAYERS:-4}"
    --heads "${HEADS:-4}"
    --head-dim "${HEAD_DIM:-32}"
    --channel-mult "${CHANNEL_MULT:-4}"
    --train-loops "${MAX_LOOPS:-2}"
    --eval-loops "${EVAL_LOOPS:-4}"
    --loop-lambda "${LAMBDA:-0.95}"
    --future-seed-scale "${FUTURE_SEED_SCALE:?Set FUTURE_SEED_SCALE=0 or 1}"
    --gdn-mode "${GDN_MODE:-chunk}"
    --gdn-expand-v "${GDN_EXPAND_V:-1.0}"
    --gdn-conv-size "${GDN_CONV_SIZE:-4}"
    --forward-dtype "${FORWARD_DTYPE:-bfloat16}"
    --lr "${LR:-3e-4}"
    --weight-decay "${WEIGHT_DECAY:-0.1}"
    --grad-clip "${GRAD_CLIP:-1.0}"
    --seed "${SEED:-52}"
    --log-every "${LOG_EVERY:-100}"
    --viz-cases "${RETRIEVAL_VIZ_CASES:-8}"
    --carrier-kill-step "${RETRIEVAL_CARRIER_KILL_STEP:-200}"
    --carrier-min-past-acc "${RETRIEVAL_CARRIER_MIN_PAST_ACC:-0.5}"
  )

  printf 'mode=%s\nrun_dir=%s\ngit_sha=%s\ngit_dirty=%s\n' "$MODE" "$RUN_DIR" "$GIT_SHA" "$GIT_DIRTY" | tee "$LOG_DIR/run.log"
  (
    cd "$REPO_ROOT"
    if [[ -n "$PYTHON_BIN" ]]; then
      "$PYTHON_BIN" scripts/gdn2_future_retrieval_probe.py "${RETRIEVAL_ARGS[@]}"
    else
      UV_PROJECT_ENVIRONMENT="$PERSIST_ROOT/.venv" "$UV_BIN" run python scripts/gdn2_future_retrieval_probe.py "${RETRIEVAL_ARGS[@]}"
    fi
  ) 2>&1 | tee -a "$LOG_DIR/run.log"

  printf 'completed run_dir=%s\n' "$RUN_DIR"
  exit 0
fi

if [[ "$MODE" == "gdn2_zoology_mqar_probe" ]]; then
  ZOOLOGY_MQAR_ARGS=(
    --out-dir "$OUT_DIR"
    --run-name "$RUN_NAME"
    --condition "${MQAR_CONDITION:-}"
    --zoology-root "${ZOOLOGY_ROOT:-$PERSIST_ROOT/repos/zoology-official}"
    --zoology-sha "${ZOOLOGY_SHA:?Set ZOOLOGY_SHA to the pinned upstream commit}"
    --steps "${MQAR_STEPS:-1000}"
    --batch "${MQAR_BATCH:-128}"
    --eval-n "${MQAR_EVAL_N:-1024}"
    --eval-batch "${MQAR_EVAL_BATCH:-128}"
    --seq-len "${MQAR_SEQ_LEN:-128}"
    --pairs "${MQAR_PAIRS:-8}"
    --vocab-size "${MQAR_VOCAB_SIZE:-8192}"
    --d-model "${D_MODEL:-128}"
    --layers "${LAYERS:-4}"
    --heads "${HEADS:-4}"
    --head-dim "${HEAD_DIM:-32}"
    --channel-mult "${CHANNEL_MULT:-4}"
    --train-loops "${MAX_LOOPS:-2}"
    --eval-loops "${EVAL_LOOPS:-4}"
    --loop-lambda "${LAMBDA:-0.95}"
    --future-seed-scale "${FUTURE_SEED_SCALE:?Set FUTURE_SEED_SCALE=0 or 1}"
    --gdn-mode "${GDN_MODE:-chunk}"
    --gdn-expand-v "${GDN_EXPAND_V:-1.0}"
    --gdn-conv-size "${GDN_CONV_SIZE:-4}"
    --forward-dtype "${FORWARD_DTYPE:-bfloat16}"
    --lr "${LR:-1e-3}"
    --weight-decay "${WEIGHT_DECAY:-0.1}"
    --grad-clip "${GRAD_CLIP:-0.0}"
    --seed "${SEED:-52}"
    --log-every "${LOG_EVERY:-100}"
    --viz-cases "${MQAR_VIZ_CASES:-8}"
    --carrier-kill-step "${MQAR_CARRIER_KILL_STEP:-500}"
    --carrier-min-past-acc "${MQAR_CARRIER_MIN_PAST_ACC:-0.5}"
  )

  printf 'mode=%s\nrun_dir=%s\ngit_sha=%s\ngit_dirty=%s\n' "$MODE" "$RUN_DIR" "$GIT_SHA" "$GIT_DIRTY" | tee "$LOG_DIR/run.log"
  (
    cd "$REPO_ROOT"
    if [[ -n "$PYTHON_BIN" ]]; then
      "$PYTHON_BIN" scripts/gdn2_zoology_mqar_probe.py "${ZOOLOGY_MQAR_ARGS[@]}"
    else
      UV_PROJECT_ENVIRONMENT="$PERSIST_ROOT/.venv" "$UV_BIN" run python scripts/gdn2_zoology_mqar_probe.py "${ZOOLOGY_MQAR_ARGS[@]}"
    fi
  ) 2>&1 | tee -a "$LOG_DIR/run.log"

  printf 'completed run_dir=%s\n' "$RUN_DIR"
  exit 0
fi

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
      UV_PROJECT_ENVIRONMENT="$PERSIST_ROOT/.venv" "$UV_BIN" run python scripts/eqr_nohydra_probe.py "${EQR_ARGS[@]}"
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
  --hole_pattern "${HOLE_PATTERN:-random}"
  --official_sudoku_data_dir "${OFFICIAL_SUDOKU_DATA_DIR:-}"
  --official_sudoku_train_split "${OFFICIAL_SUDOKU_TRAIN_SPLIT:-train}"
  --official_sudoku_train_indices "${OFFICIAL_SUDOKU_TRAIN_INDICES:-}"
  --official_sudoku_eval_split "${OFFICIAL_SUDOKU_EVAL_SPLIT:-test}"
  --official_eval_seed_offset "${OFFICIAL_EVAL_SEED_OFFSET:-999}"
  --cell_order_train "${CELL_ORDER_TRAIN:-row_major}"
  --blank_loss_weight "${BLANK_LOSS_WEIGHT:-20}"
  --noise_scale "${NOISE_SCALE:-0.0}"
  --rollout_noise_scale "${ROLLOUT_NOISE_SCALE:-0.0}"
  --lr "${LR:-2e-3}"
  --weight_decay "${WEIGHT_DECAY:-1e-3}"
  --optimizer_contract "${OPTIMIZER_CONTRACT:-uniform}"
  --shared_shell_init_seed "${SHARED_SHELL_INIT_SEED:--1}"
  --feature_buffer_size "${FEATURE_BUFFER_SIZE:-8192}"
  --feature_buffer_add "${FEATURE_BUFFER_ADD:-2048}"
  --grad_accum_steps "${GRAD_ACCUM_STEPS:-1}"
  --future_seed_scale "${FUTURE_SEED_SCALE:-1.0}"
  --future_seed_decay "${FUTURE_SEED_DECAY:-0.0}"
  --future_seed_update "${FUTURE_SEED_UPDATE:-fixed}"
  --future_seed_norm_mode "${FUTURE_SEED_NORM_MODE:-unit}"
  --future_seed_gate_mode "${FUTURE_SEED_GATE_MODE:-head}"
  --future_seed_scope "${FUTURE_SEED_SCOPE:-layer}"
  --future_seed_readout_hop "${FUTURE_SEED_READOUT_HOP:-0}"
  --loop_feedback_scale "${LOOP_FEEDBACK_SCALE:-0.0}"
  --loop_feedback_detach "${LOOP_FEEDBACK_DETACH:-0}"
  --loop_feedback_corrupt_prob "${LOOP_FEEDBACK_CORRUPT_PROB:-0.0}"
  --loop_feedback_corrupt_mix "${LOOP_FEEDBACK_CORRUPT_MIX:-0.0}"
  --loop_feedback_corrupt_mode "${LOOP_FEEDBACK_CORRUPT_MODE:-random_token}"
  --loop_time_scale "${LOOP_TIME_SCALE:-0.0}"
  --scratch_mode "${SCRATCH_MODE:-none}"
  --scratch_scale "${SCRATCH_SCALE:-1.0}"
  --scratch_noise_scale "${SCRATCH_NOISE_SCALE:-0.0}"
  --scratch_gauss_weight "${SCRATCH_GAUSS_WEIGHT:-0.0}"
  --scratch_gauss_projections "${SCRATCH_GAUSS_PROJECTIONS:-0}"
  --scratch_gate_bias "${SCRATCH_GATE_BIAS:--2.0}"
  --scratch_decay_bias "${SCRATCH_DECAY_BIAS:-2.0}"
  --hidden_agg_noise_scale "${HIDDEN_AGG_NOISE_SCALE:-0.0}"
  --hidden_agg_noise_temp "${HIDDEN_AGG_NOISE_TEMP:-1.0}"
  --hidden_agg_noise_detach "${HIDDEN_AGG_NOISE_DETACH:-1}"
  --hidden_agg_noise_mode "${HIDDEN_AGG_NOISE_MODE:-gumbel}"
  --hidden_agg_noise_topk "${HIDDEN_AGG_NOISE_TOPK:-8}"
  --hidden_agg_noise_max_norm "${HIDDEN_AGG_NOISE_MAX_NORM:-0.0}"
  --exact_margin_weight "${EXACT_MARGIN_WEIGHT:-0.0}"
  --exact_margin_tau "${EXACT_MARGIN_TAU:-0.5}"
  --exact_margin_target "${EXACT_MARGIN_TARGET:-0.0}"
  --exact_margin_start_step "${EXACT_MARGIN_START_STEP:-0}"
  --forward_dtype "${FORWARD_DTYPE:-float32}"
  --backbone "${BACKBONE:-rwkv}"
  --rwkv_kernel "${RWKV_KERNEL:-auto}"
  --gdn_mode "${GDN_MODE:-chunk}"
  --gdn_expand_v "${GDN_EXPAND_V:-1.0}"
  --gdn_progressive_base_expand_v "${GDN_PROGRESSIVE_BASE_EXPAND_V:-0.0}"
  --gdn_use_short_conv "${GDN_USE_SHORT_CONV:-1}"
  --gdn_conv_size "${GDN_CONV_SIZE:-4}"
  --gdn2_gain_budget_mode "${GDN2_GAIN_BUDGET_MODE:-none}"
  --gdn2_gain_budget_sigma_cap "${GDN2_GAIN_BUDGET_SIGMA_CAP:-1.10}"
  --gdn2_gain_budget_step_cap "${GDN2_GAIN_BUDGET_STEP_CAP:-1.0}"
  --gdn2_gain_budget_sigma_cap_max "${GDN2_GAIN_BUDGET_SIGMA_CAP_MAX:-3.0}"
  --gdn2_gain_budget_infeasible_policy "${GDN2_GAIN_BUDGET_INFEASIBLE_POLICY:-raise}"
  --gdn2_fast_slow_decay_mode "${GDN2_FAST_SLOW_DECAY_MODE:-none}"
  --gdn2_fast_slow_decay_kernel_size "${GDN2_FAST_SLOW_DECAY_KERNEL_SIZE:-4}"
  --gdn2_fast_slow_decay_rho_init "${GDN2_FAST_SLOW_DECAY_RHO_INIT:-0.10}"
  --gdn2_fast_slow_decay_current_weight_init "${GDN2_FAST_SLOW_DECAY_CURRENT_WEIGHT_INIT:-0.85}"
  --gdn2_precondition_mode "${GDN2_PRECONDITION_MODE:-none}"
  --gdn2_address_mode "${GDN2_ADDRESS_MODE:-none}"
  --raven_num_slots "${RAVEN_NUM_SLOTS:-0}"
  --raven_topk "${RAVEN_TOPK:-0}"
  --lambda_ "${LAMBDA:-0.95}"
  --loop_update_mode "${LOOP_UPDATE_MODE:-fixed}"
  --loop_update_gate_init "${LOOP_UPDATE_GATE_INIT:-0.95}"
  --loop_loss "${LOOP_LOSS:-final}"
  --loop_loss_start "${LOOP_LOSS_START:-1}"
  --loop_loss_power "${LOOP_LOSS_POWER:-2.0}"
  --loop_loss_min_weight "${LOOP_LOSS_MIN_WEIGHT:-0.05}"
  --case_bank_holes "${CASE_BANK_HOLES:-}"
  --case_bank_n "${CASE_BANK_N:-0}"
  --case_bank_eval_n "${CASE_BANK_EVAL_N:-256}"
  --case_bank_loop_values "${CASE_BANK_LOOP_VALUES:-}"
  --out_dir "$OUT_DIR"
)

if [[ "${ACTIVATION_CHECKPOINT:-0}" == "1" ]]; then
  COMMON_ARGS+=(--activation_checkpoint)
fi

if [[ "${GDN_ALLOW_NEG_EIGVAL:-0}" == "1" ]]; then
  COMMON_ARGS+=(--gdn_allow_neg_eigval)
fi

if [[ "${FLA_STRICT_OFFICIAL:-0}" == "1" ]]; then
  COMMON_ARGS+=(--fla_strict_official)
fi
if [[ "${RESUME_REQUIRE_EXACT_STATE:-0}" == "1" ]]; then
  COMMON_ARGS+=(--resume_require_exact_state)
fi

if [[ -n "${RESUME_TRAIN_CHECKPOINT:-}" ]]; then
  COMMON_ARGS+=(--resume_train_checkpoint "$RESUME_TRAIN_CHECKPOINT")
fi
if [[ -n "${RESUME_TRAIN_CHECKPOINT_SHA256:-}" ]]; then
  COMMON_ARGS+=(--resume_train_checkpoint_sha256 "$RESUME_TRAIN_CHECKPOINT_SHA256")
fi
if [[ -n "${RESUME_TRAIN_SOURCE_SHA:-}" ]]; then
  COMMON_ARGS+=(--resume_train_source_sha "$RESUME_TRAIN_SOURCE_SHA")
fi
if [[ -n "${TRAIN_CHECKPOINT_DIR:-}" ]]; then
  COMMON_ARGS+=(--train_checkpoint_dir "$TRAIN_CHECKPOINT_DIR")
fi
if [[ -n "${SAVE_TRAIN_CHECKPOINT_EVERY:-}" ]]; then
  COMMON_ARGS+=(--save_train_checkpoint_every "$SAVE_TRAIN_CHECKPOINT_EVERY")
fi

if [[ -n "${HOLE_STAGES:-}" ]]; then
  COMMON_ARGS+=(--hole_stages "$HOLE_STAGES")
fi
if [[ -n "${EVAL_HOLES_LIST:-}" ]]; then
  COMMON_ARGS+=(--eval_holes_list "$EVAL_HOLES_LIST")
fi
if [[ -n "${OFFICIAL_EVAL_BLANK_RANGES:-}" ]]; then
  COMMON_ARGS+=(--official_eval_blank_ranges "$OFFICIAL_EVAL_BLANK_RANGES")
fi
if [[ -n "${EVAL_CHECKPOINT_STEPS:-}" ]]; then
  COMMON_ARGS+=(--eval_checkpoint_steps "$EVAL_CHECKPOINT_STEPS")
fi
if [[ -n "${EVAL_CHECKPOINT_STAGE_OFFSETS:-}" ]]; then
  COMMON_ARGS+=(--eval_checkpoint_stage_offsets "$EVAL_CHECKPOINT_STAGE_OFFSETS")
fi
if [[ -n "${EVAL_CHECKPOINT_HOLES_LIST:-}" ]]; then
  COMMON_ARGS+=(--eval_checkpoint_holes_list "$EVAL_CHECKPOINT_HOLES_LIST")
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
  if [[ ${FULL_ROLLOUT_KS+x} ]]; then
    FULL_ROLLOUT_KS_ARG="$FULL_ROLLOUT_KS"
  else
    FULL_ROLLOUT_KS_ARG="1,4,8,16"
  fi
  RUN_ARGS=(
    "${COMMON_ARGS[@]}"
    --steps "${FULL_STEPS:-300}"
    --batch "${FULL_BATCH:-32}"
    --eval_n "${FULL_EVAL_N:-256}"
    --rollout_ks "$FULL_ROLLOUT_KS_ARG"
    --log_every "${FULL_LOG_EVERY:-50}"
  )
fi

printf 'mode=%s\nrun_dir=%s\ngit_sha=%s\ngit_dirty=%s\n' "$MODE" "$RUN_DIR" "$GIT_SHA" "$GIT_DIRTY" | tee "$LOG_DIR/run.log"

run_training_process() {
  cd "$EXP_DIR"
  if [[ -n "$PYTHON_BIN" ]]; then
    exec "$PYTHON_BIN" study_rwkv_futureseed_loop.py "${RUN_ARGS[@]}"
  fi
  exec env UV_PROJECT_ENVIRONMENT="$PERSIST_ROOT/.venv" \
    "$UV_BIN" run python study_rwkv_futureseed_loop.py "${RUN_ARGS[@]}"
}

if [[ -n "${EXACT_TRAIN_PID_FILE:-}" ]]; then
  mkdir -p "$(dirname "$EXACT_TRAIN_PID_FILE")"
  run_training_process > >(tee -a "$LOG_DIR/run.log") 2>&1 &
  TRAIN_PID=$!
  PID_TMP="${EXACT_TRAIN_PID_FILE}.tmp.$$"
  printf '%s\n' "$TRAIN_PID" > "$PID_TMP"
  mv "$PID_TMP" "$EXACT_TRAIN_PID_FILE"
  cleanup_exact_train_pid() {
    rm -f "$EXACT_TRAIN_PID_FILE" "$PID_TMP"
  }
  terminate_exact_train() {
    trap - INT TERM
    if kill -0 "$TRAIN_PID" 2>/dev/null; then
      kill -TERM "$TRAIN_PID"
      wait "$TRAIN_PID" 2>/dev/null || true
    fi
    cleanup_exact_train_pid
    exit 143
  }
  trap cleanup_exact_train_pid EXIT
  trap terminate_exact_train INT TERM
  set +e
  wait "$TRAIN_PID"
  TRAIN_STATUS=$?
  set -e
  cleanup_exact_train_pid
  trap - EXIT INT TERM
  if [[ "$TRAIN_STATUS" -ne 0 ]]; then
    exit "$TRAIN_STATUS"
  fi
else
  run_training_process 2>&1 | tee -a "$LOG_DIR/run.log"
fi

RECORD_ARGS=(--run-dir "$RUN_DIR" --mode "$MODE")
if [[ "${UPDATE_LEADERBOARD:-}" == "0" ]] || \
  [[ "$MODE" == "smoke" && "${UPDATE_LEADERBOARD:-0}" != "1" ]]; then
  RECORD_ARGS+=(--no-leaderboard)
fi
python3 "$REPO_ROOT/scripts/record_experiment.py" "${RECORD_ARGS[@]}"

printf 'completed run_dir=%s\n' "$RUN_DIR"
