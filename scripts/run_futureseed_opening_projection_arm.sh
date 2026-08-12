#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PERSIST_ROOT="${PERSIST_ROOT:-/huyang2/double-loop}"
EXPECTED_UUID="${EXPECTED_UUID:?P-LOOP-002 requires EXPECTED_UUID for the admitted GPU}"
EXPECTED_FLA_SHA="9c8e42e762fce087c27b673af4922795d9edb85e"
CONFIG_PATH="$REPO_ROOT/configs/sudoku/futureseed_opening_projection.env"
TRAINER="$REPO_ROOT/experiments/rwkv_fs_sudoku/study_rwkv_futureseed_loop.py"
ARM="${1:?usage: $0 <canonical|opening_projection> <probe|formal>}"
RUN_KIND="${2:?usage: $0 <canonical|opening_projection> <probe|formal>}"

case "$ARM" in
  canonical|opening_projection) ;;
  *) printf 'Unknown P-LOOP-002 arm: %s\n' "$ARM" >&2; exit 2 ;;
esac
case "$RUN_KIND" in
  probe|formal) ;;
  *) printf 'Unknown P-LOOP-002 run kind: %s\n' "$RUN_KIND" >&2; exit 2 ;;
esac
if [[ "$RUN_KIND" == "probe" && "$ARM" != "opening_projection" ]]; then
  printf 'P-LOOP-002 step3001 probe is candidate-only.\n' >&2
  exit 2
fi
if [[ "${CUDA_VISIBLE_DEVICES:-0}" != "0" ]]; then
  printf 'P-LOOP-002 requires CUDA_VISIBLE_DEVICES=0.\n' >&2
  exit 3
fi
export CUDA_VISIBLE_DEVICES=0

VISIBLE_GPU="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)"
if [[ "$VISIBLE_GPU" != "0, $EXPECTED_UUID" ]]; then
  printf 'Unexpected visible GPU contract: %s\n' "$VISIBLE_GPU" >&2
  exit 4
fi
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]]; then
  printf 'P-LOOP-002 refuses to overlap an existing GPU compute process.\n' >&2
  exit 5
fi
if git -C "$REPO_ROOT" symbolic-ref -q HEAD >/dev/null; then
  printf 'P-LOOP-002 requires a detached source worktree.\n' >&2
  exit 6
fi
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
  printf 'P-LOOP-002 requires a clean source worktree.\n' >&2
  exit 7
fi
if [[ ! -f "$CONFIG_PATH" || ! -f "$TRAINER" ]]; then
  printf 'P-LOOP-002 source files are incomplete.\n' >&2
  exit 8
fi
if ! grep -q 'choices=FUTURE_SEED_GRADIENT_MODES' "$TRAINER" || \
  ! grep -q 'resume_allow_future_seed_gradient_upgrade' "$TRAINER"; then
  printf 'P-LOOP-002 trainer gradient contract is not present in this SHA.\n' >&2
  exit 8
fi

export PERSIST_ROOT
export RUNS_ROOT="${RUNS_ROOT:-$PERSIST_ROOT/runs}"
export PYTHON_BIN="${PYTHON_BIN:-$PERSIST_ROOT/official_eqr_compare/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  printf 'CUDA Python is not executable: %s\n' "$PYTHON_BIN" >&2
  exit 9
fi
BASE_CACHE="${BASE_CACHE:-$PERSIST_ROOT/.cache/p-loop-002}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$BASE_CACHE/xdg}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$BASE_CACHE/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$BASE_CACHE/torchinductor}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$BASE_CACHE/torch_extensions}"
export TMPDIR="${TMPDIR:-$BASE_CACHE/tmp}"
export TORCH_HOME="${TORCH_HOME:-$BASE_CACHE/torch}"
export HF_HOME="${HF_HOME:-$BASE_CACHE/huggingface}"
export PATH="$PERSIST_ROOT/.cache/bin:$PATH"
export PYTHONPATH="$REPO_ROOT/experiments/rwkv_fs_sudoku:$PERSIST_ROOT/.cache/fla-active:$PERSIST_ROOT/.cache/python-extra-pylib${PYTHONPATH:+:$PYTHONPATH}"
export FLA_EXPECTED_SOURCE_SHA="$EXPECTED_FLA_SHA"
export FLA_SOURCE_SHA_MARKER="$PERSIST_ROOT/.cache/fla-source-sha"
unset FLA_SOURCE_ROOT

set -a
# shellcheck disable=SC1091
source "$CONFIG_PATH"
set +a

ACTUAL_PARENT_SHA="$(sha256sum "$RESUME_TRAIN_CHECKPOINT" | awk '{print $1}')"
if [[ "$ACTUAL_PARENT_SHA" != "$RESUME_TRAIN_CHECKPOINT_SHA256" ]]; then
  printf 'Parent checkpoint SHA mismatch: %s != %s\n' \
    "$ACTUAL_PARENT_SHA" "$RESUME_TRAIN_CHECKPOINT_SHA256" >&2
  exit 10
fi
if [[ "$(<"$FLA_SOURCE_SHA_MARKER")" != "$EXPECTED_FLA_SHA" ]]; then
  printf 'Pinned FLA marker mismatch.\n' >&2
  exit 11
fi

GIT_SHA="$(git -C "$REPO_ROOT" rev-parse HEAD)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
export FUTURE_SEED_GRADIENT_MODE="$ARM"
export RESUME_ALLOW_FUTURE_SEED_GRADIENT_UPGRADE=0
if [[ "$ARM" == "opening_projection" ]]; then
  export RESUME_ALLOW_FUTURE_SEED_GRADIENT_UPGRADE=1
fi
if [[ "$RUN_KIND" == "probe" ]]; then
  FULL_STEPS=3001
  FULL_EVAL_N=8
  OFFICIAL_EVAL_BLANK_RANGES=51-55
  EVAL_HOLES_LIST=53
  EVAL_CHECKPOINT_STEPS=3001
  EVAL_CHECKPOINT_HOLES_LIST=53
  SAVE_TRAIN_CHECKPOINT_EVERY=1
  CASE_BANK_N=0
  FULL_LOG_EVERY=1
  DEFAULT_NAME="p-loop-002-opening-projection-probe-s3001-${TIMESTAMP}-${GIT_SHA:0:7}"
else
  FULL_STEPS=3100
  FULL_EVAL_N=512
  OFFICIAL_EVAL_BLANK_RANGES=51-55,56-60,61-64
  EVAL_HOLES_LIST=50,53,58,64
  EVAL_CHECKPOINT_STEPS=3100
  EVAL_CHECKPOINT_HOLES_LIST=50,53,58,64
  SAVE_TRAIN_CHECKPOINT_EVERY=100
  CASE_BANK_N=8
  FULL_LOG_EVERY=25
  DEFAULT_NAME="p-loop-002-${ARM}-s3100-${TIMESTAMP}-${GIT_SHA:0:7}"
fi
export RUN_NAME="${RUN_NAME:-$DEFAULT_NAME}"
RUN_DIR="$RUNS_ROOT/$RUN_NAME"
OUT_DIR="$RUN_DIR/output"
LOG_DIR="$RUN_DIR/logs"
export TRAIN_CHECKPOINT_DIR="${TRAIN_CHECKPOINT_DIR:-$PERSIST_ROOT/models/$RUN_NAME/checkpoints}"
if [[ -e "$RUN_DIR" || -e "$TRAIN_CHECKPOINT_DIR" ]]; then
  printf 'Refusing to reuse P-LOOP-002 output path for %s.\n' "$RUN_NAME" >&2
  exit 12
fi

mkdir -p "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
  "$TORCH_EXTENSIONS_DIR" "$TMPDIR" "$TORCH_HOME" "$HF_HOME" \
  "$OUT_DIR" "$LOG_DIR" "$TRAIN_CHECKPOINT_DIR"
git -C "$REPO_ROOT" rev-parse HEAD > "$RUN_DIR/source_HEAD.txt"
git -C "$REPO_ROOT" diff --binary > "$RUN_DIR/source.patch"
git -C "$REPO_ROOT" ls-files -z -- . \
  ':(exclude).cache/**' ':(exclude).venv/**' ':(exclude)artifacts/**' \
  ':(exclude)models/**' ':(exclude)repos/**' ':(exclude)runs/**' \
  | tar --null -czf "$RUN_DIR/source_snapshot.tar.gz" -C "$REPO_ROOT" --files-from -

"$PYTHON_BIN" - "$RUN_DIR/config.json" "$RUN_NAME" "$RUN_DIR" "$GIT_SHA" \
  "$EXPECTED_UUID" "$ARM" "$RUN_KIND" "$RESUME_TRAIN_CHECKPOINT" \
  "$RESUME_TRAIN_CHECKPOINT_SHA256" <<'PY'
import json
import pathlib
import sys
from datetime import datetime, timezone

(output, run_name, run_dir, git_sha, gpu_uuid, arm, run_kind,
 parent, parent_sha) = sys.argv[1:]
payload = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "mode": "full",
    "git_sha": git_sha,
    "git_dirty": False,
    "run_name": run_name,
    "run_dir": run_dir,
    "plan_id": "P-LOOP-002",
    "gpu_uuid": gpu_uuid,
    "arm": arm,
    "run_kind": run_kind,
    "parent_checkpoint": parent,
    "parent_checkpoint_sha256": parent_sha,
    "inference_delta": 0,
}
pathlib.Path(output).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

TRAIN_ARGS=(
  --size "$SUDOKU_SIZE" --box_rows "$BOX_ROWS" --box_cols "$BOX_COLS"
  --max_loops "$MAX_LOOPS" --d_model "$D_MODEL" --layers "$LAYERS"
  --heads "$HEADS" --head_dim "$HEAD_DIM" --channel_mult "$CHANNEL_MULT"
  --l_cycles "$L_CYCLES" --eval_holes "$EVAL_HOLES"
  --hole_pattern "$HOLE_PATTERN"
  --official_sudoku_data_dir "$OFFICIAL_SUDOKU_DATA_DIR"
  --official_sudoku_train_split "$OFFICIAL_SUDOKU_TRAIN_SPLIT"
  --official_sudoku_train_indices "$OFFICIAL_SUDOKU_TRAIN_INDICES"
  --official_sudoku_eval_split "$OFFICIAL_SUDOKU_EVAL_SPLIT"
  --official_eval_seed_offset "$OFFICIAL_EVAL_SEED_OFFSET"
  --cell_order_train "$CELL_ORDER_TRAIN" --blank_loss_weight "$BLANK_LOSS_WEIGHT"
  --noise_scale "$NOISE_SCALE" --rollout_noise_scale "$ROLLOUT_NOISE_SCALE"
  --lr "$LR" --weight_decay "$WEIGHT_DECAY"
  --optimizer_contract "$OPTIMIZER_CONTRACT"
  --shared_shell_init_seed "$SHARED_SHELL_INIT_SEED"
  --feature_buffer_size "$FEATURE_BUFFER_SIZE"
  --feature_buffer_add "$FEATURE_BUFFER_ADD"
  --grad_accum_steps "$GRAD_ACCUM_STEPS"
  --future_seed_scale "$FUTURE_SEED_SCALE"
  --future_seed_decay "$FUTURE_SEED_DECAY"
  --future_seed_update "$FUTURE_SEED_UPDATE"
  --future_seed_norm_mode "$FUTURE_SEED_NORM_MODE"
  --future_seed_gate_mode "$FUTURE_SEED_GATE_MODE"
  --future_seed_scope "$FUTURE_SEED_SCOPE"
  --future_seed_readout_hop "$FUTURE_SEED_READOUT_HOP"
  --future_seed_content_mode "$FUTURE_SEED_CONTENT_MODE"
  --future_seed_gradient_mode "$FUTURE_SEED_GRADIENT_MODE"
  --loop_feedback_scale "$LOOP_FEEDBACK_SCALE"
  --loop_feedback_detach "$LOOP_FEEDBACK_DETACH"
  --loop_feedback_corrupt_prob "$LOOP_FEEDBACK_CORRUPT_PROB"
  --loop_feedback_corrupt_mix "$LOOP_FEEDBACK_CORRUPT_MIX"
  --loop_feedback_corrupt_mode "$LOOP_FEEDBACK_CORRUPT_MODE"
  --loop_time_scale "$LOOP_TIME_SCALE"
  --scratch_mode "$SCRATCH_MODE" --scratch_scale "$SCRATCH_SCALE"
  --scratch_noise_scale "$SCRATCH_NOISE_SCALE"
  --scratch_gauss_weight "$SCRATCH_GAUSS_WEIGHT"
  --scratch_gauss_projections "$SCRATCH_GAUSS_PROJECTIONS"
  --scratch_gate_bias "$SCRATCH_GATE_BIAS" --scratch_decay_bias "$SCRATCH_DECAY_BIAS"
  --hidden_agg_noise_scale "$HIDDEN_AGG_NOISE_SCALE"
  --hidden_agg_noise_temp "$HIDDEN_AGG_NOISE_TEMP"
  --hidden_agg_noise_detach "$HIDDEN_AGG_NOISE_DETACH"
  --hidden_agg_noise_mode "$HIDDEN_AGG_NOISE_MODE"
  --hidden_agg_noise_topk "$HIDDEN_AGG_NOISE_TOPK"
  --hidden_agg_noise_max_norm "$HIDDEN_AGG_NOISE_MAX_NORM"
  --exact_margin_weight "$EXACT_MARGIN_WEIGHT"
  --exact_margin_tau "$EXACT_MARGIN_TAU"
  --exact_margin_target "$EXACT_MARGIN_TARGET"
  --exact_margin_start_step "$EXACT_MARGIN_START_STEP"
  --forward_dtype "$FORWARD_DTYPE" --backbone "$BACKBONE"
  --gdn_mode "$GDN_MODE" --gdn_expand_v "$GDN_EXPAND_V"
  --gdn_progressive_base_expand_v "$GDN_PROGRESSIVE_BASE_EXPAND_V"
  --gdn_use_short_conv "$GDN_USE_SHORT_CONV" --gdn_conv_size "$GDN_CONV_SIZE"
  --gdn2_gain_budget_mode "$GDN2_GAIN_BUDGET_MODE"
  --gdn2_fast_slow_decay_mode "$GDN2_FAST_SLOW_DECAY_MODE"
  --gdn2_precondition_mode "$GDN2_PRECONDITION_MODE"
  --gdn2_address_mode "$GDN2_ADDRESS_MODE" --gdn2_update_mode "$GDN2_UPDATE_MODE"
  --gdn2_state_expert_mode "$GDN2_STATE_EXPERT_MODE"
  --gdn2_cross_layer_init "$GDN2_CROSS_LAYER_INIT"
  --raven_num_slots "$RAVEN_NUM_SLOTS" --raven_topk "$RAVEN_TOPK"
  --lambda_ "$LAMBDA" --loop_update_mode "$LOOP_UPDATE_MODE"
  --loop_update_gate_init "$LOOP_UPDATE_GATE_INIT" --loop_loss "$LOOP_LOSS"
  --loop_loss_start "$LOOP_LOSS_START" --loop_loss_power "$LOOP_LOSS_POWER"
  --loop_loss_min_weight "$LOOP_LOSS_MIN_WEIGHT"
  --case_bank_holes "$CASE_BANK_HOLES" --case_bank_n "$CASE_BANK_N"
  --case_bank_eval_n "$CASE_BANK_EVAL_N"
  --case_bank_loop_values "$CASE_BANK_LOOP_VALUES"
  --resume_train_checkpoint "$RESUME_TRAIN_CHECKPOINT"
  --resume_train_checkpoint_sha256 "$RESUME_TRAIN_CHECKPOINT_SHA256"
  --resume_train_source_sha "$RESUME_TRAIN_SOURCE_SHA" --resume_require_exact_state
  --train_checkpoint_dir "$TRAIN_CHECKPOINT_DIR"
  --save_train_checkpoint_every "$SAVE_TRAIN_CHECKPOINT_EVERY"
  --hole_stages "$HOLE_STAGES" --eval_holes_list "$EVAL_HOLES_LIST"
  --official_eval_blank_ranges "$OFFICIAL_EVAL_BLANK_RANGES"
  --eval_checkpoint_steps "$EVAL_CHECKPOINT_STEPS"
  --eval_checkpoint_holes_list "$EVAL_CHECKPOINT_HOLES_LIST"
  --steps "$FULL_STEPS" --batch "$FULL_BATCH" --eval_n "$FULL_EVAL_N"
  --rollout_ks "$FULL_ROLLOUT_KS" --log_every "$FULL_LOG_EVERY"
  --seed "$SEED" --out_dir "$OUT_DIR"
)
if [[ "$FLA_STRICT_OFFICIAL" == "1" ]]; then TRAIN_ARGS+=(--fla_strict_official); fi
if [[ "$GDN_ALLOW_NEG_EIGVAL" == "1" ]]; then TRAIN_ARGS+=(--gdn_allow_neg_eigval); fi
if [[ "$ACTIVATION_CHECKPOINT" == "1" ]]; then TRAIN_ARGS+=(--activation_checkpoint); fi
if [[ "$RESUME_ALLOW_FUTURE_SEED_GRADIENT_UPGRADE" == "1" ]]; then
  TRAIN_ARGS+=(--resume_allow_future_seed_gradient_upgrade)
fi

printf 'plan=P-LOOP-002 arm=%s kind=%s run_dir=%s git_sha=%s gpu_uuid=%s\n' \
  "$ARM" "$RUN_KIND" "$RUN_DIR" "$GIT_SHA" "$EXPECTED_UUID" | tee "$LOG_DIR/run.log"
(
  cd "$REPO_ROOT/experiments/rwkv_fs_sudoku"
  "$PYTHON_BIN" "$TRAINER" "${TRAIN_ARGS[@]}"
) 2>&1 | tee -a "$LOG_DIR/run.log"

RESULT="$OUT_DIR/futureseed_loop_seed52.json"
CHECKPOINT="$TRAIN_CHECKPOINT_DIR/train_state_step$(printf '%06d' "$FULL_STEPS").pt"
if [[ ! -s "$RESULT" || ! -s "$CHECKPOINT" ]]; then
  printf 'P-LOOP-002 missing expected result or checkpoint: %s %s\n' \
    "$RESULT" "$CHECKPOINT" >&2
  exit 13
fi
if grep -E -i -n 'nan|out of memory|traceback|fallback' "$LOG_DIR/run.log"; then
  printf 'P-LOOP-002 integrity pattern found in run log.\n' >&2
  exit 14
fi
"$PYTHON_BIN" - "$RESULT" "$CHECKPOINT" "$ARM" "$RUN_KIND" "$GIT_SHA" <<'PY'
import hashlib
import json
import pathlib
import sys

result_path, checkpoint_path, arm, run_kind, git_sha = sys.argv[1:]
payload = json.loads(pathlib.Path(result_path).read_text(encoding="utf-8"))
args = payload["args"]
train = payload["metrics"]["train"]
if args.get("future_seed_gradient_mode") != arm:
    raise SystemExit("result gradient mode mismatch")
resume = train.get("resume_train_checkpoint", {})
contract = resume.get("semantic_contract", {})
if (
    not contract.get("matched")
    or contract.get("saved_at_step") != 3000
    or resume.get("saved_at_step") != 3000
):
    raise SystemExit("exact-resume contract is missing or mismatched")
accepted = bool(contract.get("accepted_future_seed_gradient_upgrade"))
if accepted != (arm == "opening_projection"):
    raise SystemExit("gradient-upgrade acceptance mismatch")
if payload["metrics"].get("task", {}).get("future_seed_gradient_mode") != arm:
    raise SystemExit("task gradient mode mismatch")
if run_kind == "formal":
    ranges = payload["metrics"].get("official_eval_by_blank_range", {})
    if list(ranges) != ["b51_55", "b56_60", "b61_64"]:
        raise SystemExit(f"official range set mismatch: {list(ranges)}")
    case_bank = payload["metrics"].get("case_bank", {}).get("holes", {})
    if set(case_bank) != {"official_b51_55", "official_b56_60", "official_b61_64"}:
        raise SystemExit("formal case bank is incomplete")
    for row in case_bank.values():
        if not pathlib.Path(row["all_cases_json"]).is_file():
            raise SystemExit("formal all-cases export is missing")
for path in (result_path, checkpoint_path):
    digest = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    print(f"{digest}  {path}")
print(f"verified_source_sha={git_sha}")
PY
