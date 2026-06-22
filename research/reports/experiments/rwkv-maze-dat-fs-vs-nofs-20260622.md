# rwkv-maze-dat-fs-vs-nofs-20260622

## 1. Metainfo

- Plan ID: `P-MAZE-006`
- Status: discarded
- Machine: AIStation GPU1 only
- Remote work dir: `/huyang2/double-loop/.worktrees/rwkv-maze-dat-20260622`
- Source SHA: `a19825e`
- Runtime Python: `/opt/conda/bin/python`
- GPU: NVIDIA A800-SXM4-80GB
- Date: 2026-06-22 Asia/Shanghai

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
- Model: causal RWKV7 CUDA state-passing Maze backbone
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
- Actual runtime change: planned `batch=64` OOMed on A800 80GB. Both arms were
  relaunched at matched `batch=32`, `eval_batch=32`. The model, data, loops,
  DAT mechanism, eval size, and kill rules were unchanged.
- Environment fix: AIStation `/opt/conda/bin/ninja --version` returns code
  `245`, so PyTorch rejected CUDA extension builds. A project-local wrapper
  `scripts/ninja_compat_wrapper.sh` normalizes only the version-query exit code
  and forwards real build invocations unchanged.
- Forbidden: search, repair, selector, oracle rollout, best-of-K, maze topology
  rules, CPU smoke, GPU2.

## 4. Environment

- Remote Python: `/opt/conda/bin/python`
- Torch: `2.7.0+cu126`
- CUDA available: yes
- Cache paths:
  - `TORCH_EXTENSIONS_DIR=/huyang2/double-loop/.cache/torch_extensions`
  - wrapper bin: `/huyang2/double-loop/.cache/bin/ninja`
- Worktree status before launch: detached at `a19825e`, clean after removing
  generated `core.*` files.

## 5. Commands

Launcher:

```bash
/huyang2/double-loop/artifacts/launch/run_rwkv_maze_dat_20260622.sh nofs
/huyang2/double-loop/artifacts/launch/run_rwkv_maze_dat_20260622.sh fs
```

Effective shared command shape:

```bash
CUDA_VISIBLE_DEVICES=0 \
TORCH_EXTENSIONS_DIR=/huyang2/double-loop/.cache/torch_extensions \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TORCH_CUDA_ARCH_LIST=8.0 \
/opt/conda/bin/python scripts/rwkv_maze_probe.py \
  --repo-root /huyang2/double-loop/.worktrees/rwkv-maze-dat-20260622 \
  --data-dir /huyang2/double-loop/official_eqr_compare/eqr-clean/data/maze-30x30-unique-1k \
  --d-model 128 --layers 8 --heads 8 --head-dim 16 --channel-mult 4 \
  --train-loops 4 --eval-loops 16 \
  --steps 1000 --batch 32 --eval-n 512 --eval-batch 32 \
  --feedback-mode pred --dat-weight 0.5 --dat-loops 2 \
  --path-weight 8 --log-every 100 --viz-cases 128
```

The two arms only changed `--future-seed-scale 0` vs `1` and output directory.

## 6. Artifacts

- Local dashboard: `runs/rwkv-maze-dat-fs-vs-nofs-20260622/index.html`
- Local comparison JSON: `runs/rwkv-maze-dat-fs-vs-nofs-20260622/comparison.json`
- Local logs:
  - `runs/rwkv-maze-dat-fs-vs-nofs-20260622/logs/nofs.log`
  - `runs/rwkv-maze-dat-fs-vs-nofs-20260622/logs/futureseed.log`
- Abort records:
  - `runs/rwkv-maze-dat-fs-vs-nofs-20260622/nofs/abort.json`
  - `runs/rwkv-maze-dat-fs-vs-nofs-20260622/futureseed/abort.json`
- Remote archive:
  `/huyang2/double-loop/artifacts/launch/rwkv-maze-dat-fs-vs-nofs-20260622.tgz`

Visualization gap: per-case input/target/loop1/loop4/loop8/loop16 grids were not
produced because both runs were intentionally killed at step300 before the
runner reached final `write_visuals()`. This is a runner issue to fix before the
next abortable Maze probe.

## 7. Results

Both arms hit the kill criterion at step300: precision is still in the
broad-mask regime, and loop16 does not reduce false positives relative to loop1.

| arm | step | loop1 F1 | loop16 F1 | gain | pred frac | FP/case | FN/case | DAT loss |
|---|---:|---:|---:|---:|---|---|---|---:|
| no-FS | 100 | 0.4677 | 0.4677 | +0.0000 | 0.4311 -> 0.4311 | 269.3 -> 269.3 | 0.1 -> 0.1 | 0.2727 |
| no-FS | 200 | 0.4676 | 0.4675 | -0.0001 | 0.4317 -> 0.4319 | 269.7 -> 269.9 | 0.0 -> 0.0 | 0.2235 |
| no-FS | 300 | 0.4683 | 0.4680 | -0.0003 | 0.4300 -> 0.4309 | 268.3 -> 269.0 | 0.2 -> 0.1 | 0.2179 |
| FS | 100 | 0.4673 | 0.4673 | +0.0000 | 0.4322 -> 0.4322 | 270.1 -> 270.1 | 0.0 -> 0.0 | 0.2693 |
| FS | 200 | 0.4680 | 0.4679 | -0.0001 | 0.4308 -> 0.4310 | 269.0 -> 269.2 | 0.1 -> 0.1 | 0.2196 |
| FS | 300 | 0.4691 | 0.4688 | -0.0003 | 0.4268 -> 0.4274 | 265.9 -> 266.4 | 0.7 -> 0.6 | 0.2108 |

Matched step300 comparison:

- FS minus no-FS loop16 F1: `+0.0008`
- FS minus no-FS loop16 FP: `-2.6` FP/case
- FS minus no-FS loop gain: `+0.0000`
- Both loop gains are negative: `-0.0003`.
- Both later loops slightly increase predicted PATH fraction and FP at step300.

## 8. Conclusions

Discard `P-MAZE-006` as positive evidence. DAT feedback is trainable and the
loss decreases, but it converges to the same broad-mask fixed point. FutureSeed
slightly shifts the operating point, but it does not create loop-time false
positive pruning.

Mechanism lesson: previous-prediction feedback plus denoising-to-label is still
too close to self-reconstruction. It teaches the model to make corrupted
outputs look like the same high-recall mask, not to compare candidate path
hypotheses and remove false branches.

Paper impact: this strengthens the boundary claim. FutureSeed has a strong
cheap-bidirectional Sudoku result, but official Maze remains negative for
loop-correction. Maze should stay as failure analysis unless the next mechanism
changes the state dynamics in a way that makes loops move a decision boundary
without search, repair, selector, or maze-specific rules.

Next useful engineering fix: abortable probes must write hard-case visuals before
termination. Without that, kill-rule runs lose the most important qualitative
evidence.

## 9. Submission Record

N/A.
