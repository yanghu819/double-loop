# Official Sudoku Full Backbone Long Scale

## 1. Metainfo

- Plan ID: `P-SUDOKU-002`
- Status: in progress
- Local branch: `codex/gpu1-experiment-tracking`
- Scheduled time: `2026-06-25 14:58:07 +0800`
- Machine: AIStation `GPU1` only
- Remote work dir: `/huyang2/double-loop`

## 2. Hypothesis

The 600-step official-data gate showed that the full standalone
FutureSeed-RWKV-loop backbone opens while the matched no-FS backbone stays on a
high-loss plateau. The high-value question is whether this opening compounds
with more training, or whether it saturates at a tiny nonzero exact rate.

Prediction:

- If FutureSeed is a real scalable optimization mechanism for this backbone,
  the 1500-step FS arm should improve clearly beyond the 600-step exact
  `0.0137`, ideally `>=0.05`, with positive loop5-loop1 gain.
- If step count is not the bottleneck, FS will keep CE near `1.0` and exact
  near the previous tiny value; then the next axis should be state dynamics or
  model/data scale, not more same-config epochs.
- A matched no-FS arm is worth running only if the FS arm opens enough to need
  a fair delta.

## 3. Configuration

- Data: official EqR Sudoku Extreme arrays,
  `sudoku-extreme-1k-aug-1000`
- Dataloader: standalone FutureSeed runner official `.npy` path
- Architecture: `D_MODEL=192`, `LAYERS=10`, `HEADS=12`, `HEAD_DIM=16`,
  `CHANNEL_MULT=4`
- Loop: `L_CYCLES=2`, `MAX_LOOPS=5`, final-loop loss
- Train: `FULL_STEPS=1500`, `FULL_BATCH=128`, official train split
- Eval: official test split fixed subset, `FULL_EVAL_N=2048`
- First arm: `FUTURE_SEED_SCALE=1`
- Conditional matched arm: `FUTURE_SEED_SCALE=0` only if FS opens
- Rollout metrics: explicitly disabled with `FULL_ROLLOUT_KS=`
- Forbidden: right-to-left scan, solver, repair, selector, oracle rollout,
  Sudoku-specific postprocessing, seed sweep

## 4. Environment

Expected:

- CUDA device: GPU1 only via `CUDA_VISIBLE_DEVICES=0`
- Worktree: detached at the pushed setup SHA
- Python: existing remote environment under `/huyang2/double-loop`
- RWKV kernel: CUDA statepassing
- Caches/artifacts/runs/models under `/huyang2/double-loop`

## 5. Commands

No CPU smoke. Syntax/static validation only before push:

```bash
bash -n run.sh
python -m py_compile experiments/rwkv_fs_sudoku/study_rwkv_futureseed_loop.py
```

Remote launch shape:

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 \
SOURCE_SNAPSHOT_MODE=lean UPDATE_LEADERBOARD=0 \
XDG_CACHE_HOME=/huyang2/double-loop/.cache \
PIP_CACHE_DIR=/huyang2/double-loop/.cache/pip \
HF_HOME=/huyang2/double-loop/.cache/huggingface \
TORCH_HOME=/huyang2/double-loop/.cache/torch \
TORCH_EXTENSIONS_DIR=/huyang2/double-loop/.cache/torch_extensions \
PATH=/huyang2/double-loop/.cache/bin:$PATH \
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
SUDOKU_SIZE=9 D_MODEL=192 LAYERS=10 HEADS=12 HEAD_DIM=16 CHANNEL_MULT=4 \
L_CYCLES=2 MAX_LOOPS=5 FULL_STEPS=1500 FULL_BATCH=128 FULL_EVAL_N=2048 \
FULL_ROLLOUT_KS= ROLLOUT_LOOP_VALUES= FULL_LOG_EVERY=100 \
BLANK_LOSS_WEIGHT=8 RWKV_KERNEL=statepassing FORWARD_DTYPE=bfloat16 \
FUTURE_SEED_SCALE=1 RUN_NAME=official-sudoku-fullbackbone-long-fs-<timestamp>-<sha> \
./run.sh full
```

Kill criteria:

- If step1000 exact remains `<=0.02` and CE remains `>1.0`, stop after
  archiving and do not launch no-FS.
- If step100 takes more than 20 minutes, stop by exact PID and relaunch smaller
  `FULL_BATCH=80`.
- If source is dirty or `FULL_ROLLOUT_KS=` does not disable rollout metrics,
  stop before using the result.

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

Not applicable.
