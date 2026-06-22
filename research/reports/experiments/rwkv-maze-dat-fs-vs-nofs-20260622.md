# rwkv-maze-dat-fs-vs-nofs-20260622

## 1. Metainfo

- Plan ID: `P-MAZE-006`
- Status: in-progress
- Machine: AIStation GPU1 only
- Remote work dir: `/huyang2/double-loop/.worktrees/rwkv-maze-dat-20260622`
- Date: 2026-06-22 Asia/Shanghai
- Local implementation timestamp: 2026-06-22T11:36:02Z

## 2. Hypothesis

The current Maze bottleneck is a broad-mask attractor. Existing objectives make
PATH recall cheap to preserve and false positives cheap to keep, so later loops
copy a high-recall superset instead of correcting it.

A generic denoising-attractor training pressure should give recurrent loops a
learned reason to move from corrupted output states back toward the clean target.
If FutureSeed supplies useful global/future context, the FutureSeed arm should
preserve true path cells better while pruning false positives.

## 3. Configuration

- Dataset: official `maze-30x30-unique-1k`
- Runner: `scripts/rwkv_maze_probe.py`
- Model: causal RWKV state-passing Maze backbone
- Architecture: `D128/L8/H8/head_dim16`, train loops `4`, eval loops `16`
- Pair:
  - no-FutureSeed DAT: `--future-seed-scale 0`
  - FutureSeed DAT: `--future-seed-scale 1`
- Shared DAT mechanism:
  - `--feedback-mode pred`
  - `--dat-weight 0.5`
  - `--dat-loops 2`
  - random token corruption, add-PATH corruption, delete-PATH corruption,
    model high-confidence error corruption
- Forbidden: search, repair, selector, oracle rollout, best-of-K, maze topology
  rules, CPU smoke, GPU2.

## 4. Environment

To be filled from remote launch.

## 5. Commands

No-FutureSeed:

```bash
CUDA_VISIBLE_DEVICES=0 /opt/conda/bin/python scripts/rwkv_maze_probe.py \
  --repo-root /huyang2/double-loop/.worktrees/rwkv-maze-dat-20260622 \
  --data-dir /huyang2/double-loop/official_eqr_compare/eqr-clean/data/maze-30x30-unique-1k \
  --out-dir /huyang2/double-loop/runs/rwkv-maze-dat-fs-vs-nofs-20260622/nofs \
  --run-name rwkv-maze-dat-nofs-s1000-20260622 \
  --condition no_futureseed_dat \
  --steps 1000 --batch 64 --eval-n 512 --eval-batch 64 \
  --d-model 128 --layers 8 --heads 8 --head-dim 16 --channel-mult 4 \
  --train-loops 4 --eval-loops 16 --future-seed-scale 0 \
  --feedback-mode pred --dat-weight 0.5 --dat-loops 2 \
  --path-weight 8 --log-every 100 --viz-cases 128
```

FutureSeed:

```bash
CUDA_VISIBLE_DEVICES=0 /opt/conda/bin/python scripts/rwkv_maze_probe.py \
  --repo-root /huyang2/double-loop/.worktrees/rwkv-maze-dat-20260622 \
  --data-dir /huyang2/double-loop/official_eqr_compare/eqr-clean/data/maze-30x30-unique-1k \
  --out-dir /huyang2/double-loop/runs/rwkv-maze-dat-fs-vs-nofs-20260622/futureseed \
  --run-name rwkv-maze-dat-fs-s1000-20260622 \
  --condition futureseed_dat \
  --steps 1000 --batch 64 --eval-n 512 --eval-batch 64 \
  --d-model 128 --layers 8 --heads 8 --head-dim 16 --channel-mult 4 \
  --train-loops 4 --eval-loops 16 --future-seed-scale 1 \
  --feedback-mode pred --dat-weight 0.5 --dat-loops 2 \
  --path-weight 8 --log-every 100 --viz-cases 128
```

## 6. Artifacts

Pending.

## 7. Results

Pending.

Required readouts:

- loop1/4/8/16 path F1, precision, recall, pred_path_frac, FP/FN
- loop16 minus loop1 FP and FN
- no-FutureSeed vs FutureSeed delta
- DAT diagnostics: denoise CE, stability CE/MSE, improvement loss
- hard-case visualizations with FP/FN coloring

## 8. Conclusions

Pending.

Kill criteria:

- Any run leaves GPU1 or uses CPU smoke.
- Step300 precision remains `<0.35` and loop16 FP is not lower than loop1 FP.
- FutureSeed gain comes only from larger pred_path_frac or recall with FP growth.
- DAT blocks opening entirely through step500.

## 9. Submission Record

N/A.
