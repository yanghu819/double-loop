#!/usr/bin/env bash
set -euo pipefail
ROOT=/huyang2/double-loop
WT=$ROOT/.worktrees/gdn-real-bwd-clean-645011a-20260625T2130
cd "$WT"
PYTHONPATH=$WT/experiments/rwkv_fs_sudoku /opt/conda/bin/python experiments/rwkv_fs_sudoku/check_gdn_triton_kernel.py --matrix --out $ROOT/artifacts/gdn_real_bwd_clean_matrix_645011a.json
env SMOKE_DONE=1 SKIP_SETUP=1 SOURCE_SNAPSHOT_MODE=lean PYTHON_BIN=/opt/conda/bin/python CUDA_VISIBLE_DEVICES=0 \
  SUDOKU_SIZE=9 HOLE_PATTERN=random HOLE_STAGES=8-16:50,16-24:50 EVAL_HOLES=16 EVAL_HOLES_LIST=8,16,24 \
  BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 GDN_EXPAND_V=1.0 \
  D_MODEL=64 LAYERS=2 HEADS=4 HEAD_DIM=16 CHANNEL_MULT=2 L_CYCLES=1 MAX_LOOPS=2 \
  FULL_STEPS=100 FULL_BATCH=32 FULL_EVAL_N=256 FULL_ROLLOUT_KS= FULL_LOG_EVERY=25 \
  FORWARD_DTYPE=bfloat16 LR=0.002 BLANK_LOSS_WEIGHT=20 FUTURE_SEED_SCALE=1.0 \
  RUN_NAME=gdn-realbwd-fs-s100-20260625T2130-645011a ./run.sh full
