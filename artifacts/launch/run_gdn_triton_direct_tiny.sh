#!/usr/bin/env bash
set -euo pipefail

cd /huyang2/double-loop/.worktrees/gdn-fla-9336e94-20260625T1708

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/huyang2/double-loop/.worktrees/gdn-fla-9336e94-20260625T1708/experiments/rwkv_fs_sudoku

/opt/conda/bin/python experiments/rwkv_fs_sudoku/study_rwkv_futureseed_loop.py \
  --backbone gdn \
  --gdn_mode triton_recurrent \
  --gdn_use_short_conv 0 \
  --size 9 \
  --hole_stages 8-12:2 \
  --eval_holes 8 \
  --eval_holes_list 8 \
  --d_model 32 \
  --layers 2 \
  --heads 2 \
  --head_dim 16 \
  --channel_mult 2 \
  --l_cycles 1 \
  --max_loops 2 \
  --future_seed_scale 1 \
  --forward_dtype bfloat16 \
  --steps 2 \
  --batch 4 \
  --eval_n 8 \
  --rollout_ks "" \
  --log_every 1 \
  --out_dir /huyang2/double-loop/artifacts/gdn_triton_direct_tiny
