# rwkv-maze-boundary-fs-vs-nofs-20260621

## 1. Metainfo

- Plan ID: P-MAZE-004
- Status: discarded as final benchmark; retained as weak mechanism evidence
- Machine: GPU1 A100
- Start time UTC: 2026-06-21T13:40:00Z
- Completion time UTC: 2026-06-21T15:59:37Z
- Source commit: `3df9e0275d378b7ad5b92c8da8f518df3a1db94a`
- Remote work dir: `/huyang2/double-loop/.worktrees/rwkv-maze-boundary-detach-3df9e02`

## 2. Hypothesis

Official Maze path-token weighting makes broad PATH masks too cheap. A generic
PATH-vs-non-PATH decision-boundary objective plus per-sample PATH-mass
calibration should reduce false positives without using maze rules. If
FutureSeed is genuinely cheap bidirectional context for causal RWKV, it should
benefit more once the objective no longer rewards broad coverage.

## 3. Configuration

- Data: official `maze-30x30-unique-1k`
- Model: causal RWKV7 state-passing token classifier
- D_MODEL: 128
- Layers: 8
- Heads/head dim: 8 / 16
- Train/eval loops: 4 / 8
- Batch: 32
- Steps: 800
- Objective:
  - token CE with PATH token weight `2`
  - PATH margin BCE weight `1`
  - per-sample PATH mass budget weight `1`
- Conditions: `FUTURE_SEED_SCALE=0` and `FUTURE_SEED_SCALE=1`
- Forbidden: selector, search, repair, maze rules, postprocessing, seed sweep

## 4. Environment

- AIStation row: GPU1 only
- Python: /opt/conda/bin/python
- CUDA binding: `CUDA_VISIBLE_DEVICES=0`
- RWKV extension cache: `/huyang2/double-loop/.cache/torch_extensions`

## 5. Commands

Smoke command shape:

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 SKIP_DOWN=1 \
PYTHON_BIN=/opt/conda/bin/python RWKV_KERNEL=statepassing \
RWKV_MAZE_STEPS=1 RWKV_MAZE_BATCH=4 RWKV_MAZE_EVAL_N=8 \
D_MODEL=64 LAYERS=2 HEADS=4 HEAD_DIM=16 CHANNEL_MULT=2 \
L_CYCLES=1 MAX_LOOPS=1 EVAL_LOOPS=2 \
RUN_NAME=rwkv-maze-boundary-smoke-3df9e02-bg ./run.sh rwkv_maze_probe
```

Main command shape, run once with `FUTURE_SEED_SCALE=0` and once with
`FUTURE_SEED_SCALE=1`:

```bash
PATH=/huyang2/double-loop/.cache/bin:/opt/conda/bin:$PATH \
TORCH_CUDA_ARCH_LIST=8.0 CUDA_VISIBLE_DEVICES=0 \
SMOKE_DONE=1 SKIP_SETUP=1 SKIP_DOWN=1 \
PYTHON_BIN=/opt/conda/bin/python RWKV_KERNEL=statepassing \
FORWARD_DTYPE=bfloat16 \
XDG_CACHE_HOME=/huyang2/double-loop/.cache \
TORCH_EXTENSIONS_DIR=/huyang2/double-loop/.cache/torch_extensions \
RWKV_MAZE_DATA_DIR=/huyang2/double-loop/official_eqr_compare/eqr-clean/data/maze-30x30-unique-1k \
RWKV_MAZE_STEPS=800 RWKV_MAZE_BATCH=32 RWKV_MAZE_EVAL_N=512 \
D_MODEL=128 LAYERS=8 HEADS=8 HEAD_DIM=16 CHANNEL_MULT=4 \
L_CYCLES=2 MAX_LOOPS=4 EVAL_LOOPS=8 \
RWKV_MAZE_PATH_WEIGHT=2 RWKV_MAZE_PATH_BINARY_WEIGHT=1 \
RWKV_MAZE_PATH_BUDGET_WEIGHT=1 LOOP_LOSS=all \
FUTURE_SEED_SCALE=<0-or-1> RUN_NAME=<run> ./run.sh rwkv_maze_probe
```

## 6. Artifacts

- Local archive: `runs/rwkv-maze-boundary-fs-vs-nofs-20260621/`
- Dashboard: `runs/rwkv-maze-boundary-fs-vs-nofs-20260621/index.html`
- Aggregate plot: `runs/rwkv-maze-boundary-fs-vs-nofs-20260621/comparison_summary.png`
- Hard-case plot: `runs/rwkv-maze-boundary-fs-vs-nofs-20260621/hard_case_boundary_grid.png`
- Comparison JSON: `runs/rwkv-maze-boundary-fs-vs-nofs-20260621/comparison.json`
- Score JSON: `runs/rwkv-maze-boundary-fs-vs-nofs-20260621/score.json`
- no-FutureSeed run:
  - Remote: `/huyang2/double-loop/.worktrees/rwkv-maze-boundary-detach-3df9e02/runs/rwkv-maze-boundary-nofs-s800-20260621T151104Z-3df9e02`
  - Local log: `runs/rwkv-maze-boundary-fs-vs-nofs-20260621/nofs/logs/run.log`
  - Local result: `runs/rwkv-maze-boundary-fs-vs-nofs-20260621/nofs/output/rwkv_maze_probe.json`
- FutureSeed run:
  - Remote completed rerun: `/huyang2/double-loop/.worktrees/rwkv-maze-boundary-detach-3df9e02/runs/rwkv-maze-boundary-fs-s800-20260621T153154Z-3df9e02`
  - Local log: `runs/rwkv-maze-boundary-fs-vs-nofs-20260621/futureseed/logs/run.log`
  - Local result: `runs/rwkv-maze-boundary-fs-vs-nofs-20260621/futureseed/output/rwkv_maze_probe.json`
- Note: the first FutureSeed attempt was interrupted when the GPU1 development
  environment halted before writing final JSON. It was not used as a completed
  result; the same configuration was rerun from scratch.

## 7. Results

Final eval, 512 official Maze cases:

| Condition | loop8 path F1 | precision | recall | pred PATH frac | FP/case | FN/case | loop gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| no FutureSeed | 0.4089 | 0.3256 | 0.5828 | 0.2360 | 143.2 | 49.7 | +0.0004 |
| FutureSeed | 0.4278 | 0.3270 | 0.6280 | 0.2527 | 152.9 | 44.3 | +0.0008 |

Delta FutureSeed - no FutureSeed:

- path F1: `+0.0189`
- precision: `+0.0014`
- recall: `+0.0452`
- predicted PATH fraction: `+0.0167`
- FP/case: `+9.7`
- FN/case: `-5.3`

Training diagnostics at step800:

| Condition | loss | CE | boundary BCE | budget loss | margin pos | margin neg | soft PATH frac |
|---|---:|---:|---:|---:|---:|---:|---:|
| no FutureSeed | 0.7931 | 0.3147 | 0.3921 | 0.0863 | 0.4831 | -6.6917 | 0.2200 |
| FutureSeed | 0.7628 | 0.3028 | 0.3799 | 0.0801 | 0.5253 | -6.9870 | 0.2139 |

Comparison to the previous broad-mask objective:

- Previous causal RWKV path-weight objective reached higher loop8 F1:
  no-FutureSeed `0.4690`, FutureSeed `0.4666`.
- That score came from broad coverage: pred PATH frac about `0.40`, recall
  about `0.95`, and FP around `250` per case.
- The new boundary objective reduced predicted PATH mass and FP count, but it
  did so by adding many false negatives.

## 8. Conclusions

Decision: discard this objective as a final benchmark, keep it as weak mechanism
evidence.

What this experiment answered:

- The broad-mask loophole is real: once false positives are made costly, the
  model no longer predicts nearly half the board as PATH.
- The current boundary objective is not enough: it lowers false positives but
  overprunes true path cells, so final F1 drops below the broader path-weighted
  objective.
- FutureSeed helps slightly under this pressure. The gain is mostly recall
  preservation, not precision; this is consistent with cheap future context
  helping the model keep true path cells alive under pruning.
- Loop correction remains unsolved. loop1 and loop8 are nearly the same
  operating point in both conditions.

Paper-level implication:

This is not evidence that FutureSeed+loop beats EqR on Maze. It is evidence
that Maze evaluation must separate broad coverage from actual path recovery, and
that FutureSeed has a small robustness signal when a causal RWKV is pushed away
from the broad-mask shortcut. The next high-ROI paper experiment should not
sweep boundary weights. It should either:

1. use a proxy/objective where broad coverage cannot be a cheap answer by
   construction, or
2. introduce a simple generic recurrent decision/state mechanism that explicitly
   lets later loops reduce false positives while preserving true path cells.

## 9. Submission Record

Not applicable.
