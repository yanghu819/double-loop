# rwkv-maze-boundary-fs-vs-nofs-20260621

## 1. Metainfo

- Plan ID: P-MAZE-004
- Status: in-progress
- Machine: GPU1 A100
- Start time UTC: 2026-06-21T13:40:00Z
- Source commit: pending
- Remote work dir: pending

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

Planned command shape:

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 SKIP_DOWN=1 \
PYTHON_BIN=/opt/conda/bin/python RWKV_KERNEL=statepassing \
RWKV_MAZE_STEPS=800 RWKV_MAZE_BATCH=32 RWKV_MAZE_EVAL_N=512 \
D_MODEL=128 LAYERS=8 HEADS=8 HEAD_DIM=16 CHANNEL_MULT=4 \
L_CYCLES=2 MAX_LOOPS=4 EVAL_LOOPS=8 \
RWKV_MAZE_PATH_WEIGHT=2 RWKV_MAZE_PATH_BINARY_WEIGHT=1 \
RWKV_MAZE_PATH_BUDGET_WEIGHT=1 LOOP_LOSS=all \
FUTURE_SEED_SCALE=<0-or-1> RUN_NAME=<run> ./run.sh rwkv_maze_probe
```

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

Not applicable.
