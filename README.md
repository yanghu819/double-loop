# Double Loop

Clean research prototype for:

```text
RWKV recurrent backbone
+ FutureSeed cross-layer recurrent state initialization
+ EqR-style depth loop
+ feature-difference stochastic rollout
```

The current probe task is 6x6 Sudoku. Sudoku is used as a structured local
testbed for latent reasoning, not as a solver target.

## Active Code

- `experiments/rwkv_fs_sudoku/study_rwkv_futureseed_loop.py`

The maintained script has no attention/GRU fallback, no Sudoku rule helper, no
search/backtracking, no repeat-view Sudoku representation, and no learned
posterior guidance branch.

## Active Result

- `experiments/rwkv_fs_sudoku/runs/final_feature_diff_main_6x6`

Best kept result:

| K | oracle exact | confidence | consistency | residual | majority | token disagreement |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 0.5039 | 0.3477 | 0.3477 | 0.3438 | 0.3477 | 0.1171 |

## Run

```bash
cd experiments/rwkv_fs_sudoku
uv run python study_rwkv_futureseed_loop.py \
  --size 6 --steps 300 --batch 32 --eval_n 256 --max_loops 3 \
  --d_model 32 --layers 4 --heads 4 --head_dim 8 --channel_mult 2 \
  --l_cycles 1 \
  --holes_min 2 --holes_max 4 --eval_holes 2 \
  --hole_pattern unit \
  --blank_loss_weight 20 \
  --noise_scale 0.01 \
  --rollout_ks 1,4,8,16 \
  --rollout_noise_scale 0.05 --log_every 50 \
  --out_dir runs/final_feature_diff_main_6x6
```

## Checkpoint

- `checkpoint/clean-feature-diff-main-2026-06-01`
