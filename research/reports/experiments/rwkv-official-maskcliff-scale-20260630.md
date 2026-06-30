# RWKV Official Sudoku Mask-Cliff Scale Gate

## 1. Metainfo

- Plan ID: `P-DIAG-002`
- Status: in-progress
- Local branch: `codex/gpu1-experiment-tracking`
- Scheduled time: `2026-06-30 20:31:29 +0800`
- Machine: AIStation `GPU1` only
- Remote work dir: `/huyang2/double-loop`
- Intended launch: detached worktree at the pushed SHA for this record

## 2. Hypothesis

We need to separate two explanations for the current official Sudoku plateau:

- GDN replacement is the regression: the original full RWKV+native FutureSeed
  backbone should beat the GDN plateau on the same official arrays.
- High blank-count official boards are the cliff: lower-blank buckets should
  open more strongly than the hardest bucket even under the RWKV backbone.

This is one diagnostic run, not a sweep. If it does not answer this split, the
next step should be a different mechanism question rather than another nearby
size/seed/loss table.

## 3. Configuration

- Data: official EqR Sudoku arrays, `sudoku-extreme-1k-aug-1000`
- Backbone: full standalone RWKV + native terminal-state FutureSeed
- Architecture: `D_MODEL=256`, `LAYERS=12`, `HEADS=16`, `HEAD_DIM=16`,
  `CHANNEL_MULT=4`
- Loop: `L_CYCLES=2`, `MAX_LOOPS=8`, final-loop loss
- Train: `FULL_STEPS=2000`, `FULL_BATCH=96`, `BLANK_LOSS_WEIGHT=8`
- Eval: official test split, `FULL_EVAL_N=512`
- Official blank-range eval: `46-50,51-55,56-64`
- Case-bank visualization: per blank-range groups, loops `1,2,3,5,8`
- CUDA path: `RWKV_KERNEL=statepassing`, `FORWARD_DTYPE=bfloat16`
- Forbidden: no CPU smoke, no GPU2, no selector/search/repair/oracle rollout,
  no Sudoku-specific rule or postprocessing, no no-FS/GDN table in this run.

## 4. Environment

- CUDA device: `CUDA_VISIBLE_DEVICES=0`
- Worktree: to be created under `/huyang2/double-loop/.worktrees`
- Caches/artifacts/runs/models under `/huyang2/double-loop`
- Python: reuse the existing remote environment from prior official Sudoku runs

## 5. Commands

Local static validation only:

```bash
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
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
SUDOKU_SIZE=9 BACKBONE=rwkv RWKV_KERNEL=statepassing \
D_MODEL=256 LAYERS=12 HEADS=16 HEAD_DIM=16 CHANNEL_MULT=4 \
L_CYCLES=2 MAX_LOOPS=8 FULL_STEPS=2000 FULL_BATCH=96 FULL_EVAL_N=512 \
HOLES_MIN=46 HOLES_MAX=64 EVAL_HOLES=56 EVAL_HOLES_LIST=56 \
OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64 \
EVAL_CHECKPOINT_STEPS=500,1000,1500,2000 \
SAVE_TRAIN_CHECKPOINT_EVERY=500 \
TRAIN_CHECKPOINT_DIR=/huyang2/double-loop/models/<run_name> \
FULL_ROLLOUT_KS= ROLLOUT_LOOP_VALUES= FULL_LOG_EVERY=100 \
BLANK_LOSS_WEIGHT=8 FORWARD_DTYPE=bfloat16 FUTURE_SEED_SCALE=1 \
CASE_BANK_N=2 CASE_BANK_EVAL_N=256 CASE_BANK_LOOP_VALUES=1,2,3,5,8 \
RUN_NAME=<run_name> ./run.sh full
```

Kill criteria:

- If step100 takes more than 15 minutes, stop by exact PID and relaunch the same
  diagnostic at `FULL_BATCH=64`, not a new idea.
- If CUDA OOMs, relaunch once at `D_MODEL=192/LAYERS=12` only if the OOM happens
  before meaningful training starts.
- If step500 CE is clearly worse than the historical D192 RWKV/GDN openings and
  blank accuracy remains near the no-FS plateau, stop and classify this as not a
  useful RWKV scale continuation.

## 6. Artifacts

Pending launch.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

Not a submission run.
