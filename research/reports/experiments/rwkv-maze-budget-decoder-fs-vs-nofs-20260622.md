# rwkv-maze-budget-decoder-fs-vs-nofs-20260622

## 1. Metainfo

- Plan ID: P-MAZE-005
- Status: in-progress
- Machine: GPU1 A100
- Start time UTC: 2026-06-22T03:33:52Z
- Source commit: `97404e11ac9b55f83659a4f559a444a4d2856999`
- Remote work dir: `/huyang2/double-loop/.worktrees/rwkv-maze-budget-detach-97404e1`

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

- Remote smoke: `/huyang2/double-loop/.worktrees/rwkv-maze-budget-detach-97404e1/runs/rwkv-maze-budget-smoke-rebuild-20260622T044054-97404e1`
- Remote noFS run: `/huyang2/double-loop/.worktrees/rwkv-maze-budget-detach-97404e1/runs/rwkv-maze-budget-nofs-s800-20260622T0442Z-97404e1`
- Remote FutureSeed run: `/huyang2/double-loop/.worktrees/rwkv-maze-budget-detach-97404e1/runs/rwkv-maze-budget-fs-s800-20260622T0501Z-97404e1`
- Local archive/dashboard:
  `runs/rwkv-maze-budget-decoder-fs-vs-nofs-20260622/index.html`
- Statepassing diagnostic logs:
  `/huyang2/double-loop/artifacts/launch/rwkv_maze_stage_diag_statepassing_rebuild_20260622T043915Z.log`

Engineering note:

- Initial statepassing smoke hung before the first training step with GPU util near
  zero. A stage diagnostic isolated the hang to `StatePassingRWKV7.apply`.
- The same model completed forward/backward with `rwkv_kernel=torch`, so the new
  budget decoder code was not the cause.
- A synthetic statepassing call also hung at `T=96`; deleting only the project
  cache directory
  `/huyang2/double-loop/.cache/torch_extensions/rwkv7_statepassing_clampw_n16`
  forced a rebuild. After rebuild, the real Maze forward/backward completed.
- Lesson: after GPU/container restart, stale project-local torch extension cache
  can make statepassing appear to hang. Rebuild the specific head-dim extension
  before changing modeling code.

## 7. Results

| condition | raw loop8 F1 | raw gain | raw P/R | raw pred frac | raw FP/FN | budget loop8 F1 | budget gain | budget P/R | budget pred frac | budget FP/FN | count abs err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| noFS | 0.4652 | -0.0006 | 0.3114 / 0.9333 | 0.3963 | 245.8 / 8.0 | 0.3387 | +0.0026 | 0.3361 / 0.3432 | 0.1345 | 80.4 / 78.2 | 0.0098 |
| FutureSeed | 0.4684 | +0.0000 | 0.3108 / 0.9640 | 0.4103 | 254.7 / 4.3 | 0.3289 | -0.0001 | 0.3318 / 0.3277 | 0.1301 | 78.2 / 80.0 | 0.0095 |

Step-level signal:

- By step100 both conditions had already learned a high-recall raw broad mask:
  raw loop8 path F1 about `0.4673`, precision about `0.31`, recall near `1.0`,
  and predicted PATH fraction about `0.432`.
- The path-count head also learned quickly: final predicted path fraction was
  `0.1382` noFS and `0.1326` FutureSeed against true `0.1337`.
- The budget decoder used that count signal to reduce false positives from about
  `246-255` raw FP/case to about `78-80`, but it also introduced about `78-80`
  false negatives per case.
- Loop did not produce correction in either condition. Raw loop1->loop8 and
  budget loop1->loop8 gains were all within about `0.003`.

## 8. Conclusions

- Discard this specific budget-decoder direction as a scoring path. It answers a
  useful question but does not solve Maze: total PATH mass calibration is not the
  missing piece.
- The bottleneck is true-vs-false PATH ranking and late-loop correction. The
  model can estimate how many cells should be PATH, but the top-ranked set does
  not contain enough true-path cells.
- FutureSeed is neutral-to-negative on this Maze probe: budget loop8 F1 is
  `0.3289` versus noFS `0.3387`. This reinforces the boundary from official EqR
  and causal RWKV Maze: Maze broad-mask/path-ranking failure is not the same as
  the Sudoku cheap-bidirectional failure.
- Next high-ROI work should not sweep count/budget loss weights. It should either
  change the generic training signal toward ranking/self-correction, or select a
  causal proxy where future-context access is the actual bottleneck.

## 9. Submission Record

Not applicable.
