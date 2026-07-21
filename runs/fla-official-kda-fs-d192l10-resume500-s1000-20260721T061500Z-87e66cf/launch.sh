#!/usr/bin/env bash
set -euo pipefail

PERSIST_ROOT=/huyang2/double-loop
WORKTREE="${WORKTREE:?WORKTREE is required}"
RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:?RESUME_CHECKPOINT is required}"
SOURCE_SHA=87e66cf91c086283e6981235c93152579fb93dc4
RUN_ROOT="$PERSIST_ROOT/runs/$RUN_NAME"
OUT_DIR="$RUN_ROOT/output"
MODEL_DIR="$PERSIST_ROOT/models/$RUN_NAME/checkpoints"
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

mkdir -p "$RUN_ROOT/logs" "$OUT_DIR" "$MODEL_DIR" "$TMPDIR"
if [[ "$(git -C "$WORKTREE" rev-parse HEAD)" != "$SOURCE_SHA" ]]; then
  echo "wrong detached source SHA" >&2
  exit 10
fi
if [[ ! -f "$RESUME_CHECKPOINT" ]]; then
  echo "missing exact resume checkpoint" >&2
  exit 11
fi

started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf '%s\n' "$SOURCE_SHA" > "$RUN_ROOT/source_HEAD.txt"
printf '{"timestamp_utc":"%s","mode":"full","git_sha":"%s","git_dirty":false,"run_name":"%s","run_dir":"%s","tracking_note":"exact KDA step500 optimizer/RNG continuation under the P-LA-004 matched recipe"}\n' \
  "$started_at" "$SOURCE_SHA" "$RUN_NAME" "$RUN_ROOT" > "$RUN_ROOT/config.json"
{
  printf 'plan_id=P-LA-005\n'
  printf 'source_sha=%s\n' "$SOURCE_SHA"
  printf 'started_at_utc=%s\n' "$started_at"
  printf 'cuda_visible_devices=%s\n' "$CUDA_VISIBLE_DEVICES"
  printf 'worktree=%s\n' "$WORKTREE"
  printf 'resume_checkpoint=%s\n' "$RESUME_CHECKPOINT"
  printf 'save_every=100\n'
} > "$RUN_ROOT/launch.env"
nvidia-smi --query-gpu=name,index,memory.used,memory.total,utilization.gpu --format=csv,noheader > "$RUN_ROOT/gpu_before.csv"
tar -czf "$RUN_ROOT/source_snapshot.tar.gz" -C "$WORKTREE" \
  experiments/rwkv_fs_sudoku/study_rwkv_futureseed_loop.py \
  experiments/rwkv_fs_sudoku/check_fla_delta_backbones.py \
  scripts/run_fla_official_futureseed_compare.sh \
  run.sh

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
"$PYTHON_BIN" study_rwkv_futureseed_loop.py \
  --size 9 --box_rows 0 --box_cols 0 \
  --max_loops 5 --d_model 192 --layers 10 --heads 6 --head_dim 32 --channel_mult 4 --l_cycles 2 \
  --holes_min 46 --holes_max 55 --eval_holes 53 --hole_pattern random \
  --official_sudoku_data_dir "$PERSIST_ROOT/data/sudoku-extreme-full" \
  --official_sudoku_train_split train --official_sudoku_eval_split test --official_eval_seed_offset 999 \
  --blank_loss_weight 8 --noise_scale 0 --rollout_noise_scale 0 \
  --lr 0.0015 --weight_decay 0.001 --feature_buffer_size 8192 --feature_buffer_add 2048 \
  --grad_accum_steps 4 --future_seed_scale 1 --future_seed_decay 0 \
  --future_seed_update fixed --future_seed_norm_mode unit \
  --loop_feedback_scale 0 --loop_feedback_detach 0 --loop_feedback_corrupt_prob 0 \
  --loop_feedback_corrupt_mix 0 --loop_feedback_corrupt_mode random_token --loop_time_scale 0 \
  --scratch_mode none --scratch_scale 1 --scratch_noise_scale 0 --scratch_gauss_weight 0 \
  --scratch_gauss_projections 0 --scratch_gate_bias -2 --scratch_decay_bias 2 \
  --hidden_agg_noise_scale 0 --hidden_agg_noise_temp 1 --hidden_agg_noise_detach 1 \
  --hidden_agg_noise_mode gumbel --hidden_agg_noise_topk 8 --hidden_agg_noise_max_norm 0 \
  --exact_margin_weight 0 --exact_margin_tau 0.5 --exact_margin_target 0 --exact_margin_start_step 0 \
  --forward_dtype bfloat16 --backbone kda --rwkv_kernel auto --gdn_mode chunk \
  --gdn_expand_v 2 --gdn_use_short_conv 1 --gdn_conv_size 4 --lambda_ 0.95 \
  --loop_update_mode fixed --loop_update_gate_init 0.95 --loop_loss all \
  --loop_loss_start 1 --loop_loss_power 2 --loop_loss_min_weight 0.05 \
  --case_bank_holes "" --case_bank_n 4 --case_bank_eval_n 256 --case_bank_loop_values 1,2,3,4,5 \
  --out_dir "$OUT_DIR" --fla_strict_official \
  --resume_train_checkpoint "$RESUME_CHECKPOINT" --train_checkpoint_dir "$MODEL_DIR" \
  --save_train_checkpoint_every 100 --hole_stages 46-50:100,51-55:900 \
  --eval_holes_list 53 --official_eval_blank_ranges 46-50,51-55,56-64 \
  --eval_checkpoint_steps 1000 --eval_checkpoint_holes_list 53 \
  --steps 1000 --batch 32 --eval_n 512 --rollout_ks "" --log_every 100 \
  2>&1 | tee "$RUN_ROOT/logs/run.log"
