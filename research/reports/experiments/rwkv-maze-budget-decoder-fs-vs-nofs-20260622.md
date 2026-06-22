# rwkv-maze-budget-decoder-fs-vs-nofs-20260622

## 1. Metainfo

- Plan ID: P-MAZE-005
- Status: in-progress
- Machine: GPU1 A100
- Start time UTC: 2026-06-22T03:33:52Z
- Source commit: pending implementation commit
- Remote work dir: pending detached SHA worktree

## 2. Hypothesis

The previous Maze failures may not mean the causal RWKV cannot identify the
true path. They may mean its raw PATH-vs-non-PATH argmax boundary is badly
calibrated: path-weight training produces broad masks, while boundary pressure
overprunes. A generic learned path-budget head can test whether the useful signal
is already present as a ranking.

Prediction:

- If ranking is the bottleneck's missing piece, the learned-budget decoder should
  reduce predicted PATH mass and FP while preserving recall better than the
  boundary objective.
- If budgeted decoding still trades FP for FN, the problem is not just
  calibration; the representation/ranking itself does not separate true path
  cells from false positives.
- If FutureSeed helps under this decoder, it should show as better budgeted F1
  or better recall at a similar predicted budget.

## 3. Configuration

- Data: official `maze-30x30-unique-1k`
- Model: causal RWKV7 state-passing token classifier
- Decoder: learned global PATH fraction head; budgeted PATH mask is top cells by
  model PATH margin under the model-predicted budget
- D_MODEL: 128
- Layers: 8
- Heads/head dim: 8 / 16
- Train/eval loops: 4 / 8
- Batch: 32
- Steps: 800
- Objective:
  - token CE with PATH token weight `8`
  - learned path-count BCE weight `1`
  - no boundary BCE or budget L1 pressure
- Conditions: `FUTURE_SEED_SCALE=0` and `FUTURE_SEED_SCALE=1`
- Forbidden: selector, search, repair, maze rules, oracle true count, seed sweep

## 4. Environment

- AIStation row: GPU1 only
- Python: `/opt/conda/bin/python`
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
RWKV_MAZE_PATH_COUNT_WEIGHT=1 RWKV_MAZE_BUDGET_DECODER=1 \
RUN_NAME=<smoke> ./run.sh rwkv_maze_probe
```

Main command shape, run once with `FUTURE_SEED_SCALE=0` and once with
`FUTURE_SEED_SCALE=1`:

```bash
PATH=/huyang2/double-loop/.cache/bin:/opt/conda/bin:$PATH \
TORCH_CUDA_ARCH_LIST=8.0 CUDA_VISIBLE_DEVICES=0 \
SMOKE_DONE=1 SKIP_SETUP=1 SKIP_DOWN=1 \
PYTHON_BIN=/opt/conda/bin/python RWKV_KERNEL=statepassing \
FORWARD_DTYPE=bfloat16 SOURCE_SNAPSHOT_MODE=lean \
XDG_CACHE_HOME=/huyang2/double-loop/.cache \
TORCH_EXTENSIONS_DIR=/huyang2/double-loop/.cache/torch_extensions \
RWKV_MAZE_DATA_DIR=/huyang2/double-loop/official_eqr_compare/eqr-clean/data/maze-30x30-unique-1k \
RWKV_MAZE_STEPS=800 RWKV_MAZE_BATCH=32 RWKV_MAZE_EVAL_N=512 \
D_MODEL=128 LAYERS=8 HEADS=8 HEAD_DIM=16 CHANNEL_MULT=4 \
L_CYCLES=2 MAX_LOOPS=4 EVAL_LOOPS=8 \
RWKV_MAZE_PATH_WEIGHT=8 RWKV_MAZE_PATH_BINARY_WEIGHT=0 \
RWKV_MAZE_PATH_BUDGET_WEIGHT=0 RWKV_MAZE_PATH_COUNT_WEIGHT=1 \
RWKV_MAZE_BUDGET_DECODER=1 LOOP_LOSS=all \
FUTURE_SEED_SCALE=<0-or-1> RUN_NAME=<run> ./run.sh rwkv_maze_probe
```

Kill criteria:

- If step400 has both raw loop8 and budget loop8 path F1 near zero, stop as
  optimization failure.
- If budget decoder reduces FP only by large FN growth, finish the pair but
  classify as calibration-not-enough.
- If GPU utilization is low while memory is high, stop exact PID and inspect.

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

Not applicable.
