# GDN FutureSeed Official Sudoku Mask-Cliff Diagnostic

## 1. Metainfo

- Plan ID: `P-DIAG-003`
- Status: in-progress
- Local branch: `codex/gpu1-experiment-tracking`
- Scheduled time: `2026-06-30 21:57:06 +0800`
- Machine: AIStation `GPU1` only
- Remote work dir: `/huyang2/double-loop`
- Intended source SHA: current pushed branch SHA for this record

## 2. Hypothesis

P-DIAG-002 showed that full RWKV+native FutureSeed almost solves the lower
official blank bucket but hits a hard exact cliff above about 50 blanks. This
run asks whether GDN+FutureSeed has the same cliff shape.

Prediction:

- If GDN+FutureSeed also solves `46-50` blanks but fails `51+`, the bottleneck
  is the current FutureSeed+loop global-consistency mechanism, not just GDN.
- If GDN+FutureSeed fails even `46-50`, then GDN state/backbone is an additional
  bottleneck relative to RWKV.

This is one diagnostic run, not an ablation table. It should not change loss,
seed, width, or dynamics.

## 3. Configuration

- Data: official EqR Sudoku arrays, `sudoku-extreme-1k-aug-1000`
- Backbone: GDN Triton recurrent real backward + native terminal-state FutureSeed
- Architecture: `D_MODEL=192`, `LAYERS=10`, `HEADS=12`, `HEAD_DIM=16`,
  `CHANNEL_MULT=4`
- Loop: `L_CYCLES=2`, `MAX_LOOPS=5`, final-loop loss
- Train: `FULL_STEPS=1500`, `FULL_BATCH=128`, `BLANK_LOSS_WEIGHT=8`
- Eval: official test split, `FULL_EVAL_N=512`
- Official blank-range eval: `46-50,51-55,56-64`
- Case-bank visualization: per blank-range groups, loops `1,2,3,5`
- CUDA path: `GDN_MODE=triton_recurrent`, `GDN_USE_SHORT_CONV=0`,
  `FORWARD_DTYPE=bfloat16`
- Forbidden: no CPU smoke, no GPU2, no selector/search/repair/oracle rollout,
  no Sudoku-specific rule or postprocessing, no no-FS arm in this run.

## 4. Environment

Pending launch.

## 5. Commands

Local static validation only:

```bash
python -m py_compile experiments/rwkv_fs_sudoku/study_rwkv_futureseed_loop.py
bash -n run.sh
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
SUDOKU_SIZE=9 BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 \
D_MODEL=192 LAYERS=10 HEADS=12 HEAD_DIM=16 CHANNEL_MULT=4 \
L_CYCLES=2 MAX_LOOPS=5 FULL_STEPS=1500 FULL_BATCH=128 FULL_EVAL_N=512 \
HOLES_MIN=46 HOLES_MAX=64 EVAL_HOLES=56 EVAL_HOLES_LIST=56 \
OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64 \
EVAL_CHECKPOINT_STEPS=500,1000,1500 \
FULL_ROLLOUT_KS= ROLLOUT_LOOP_VALUES= FULL_LOG_EVERY=100 \
BLANK_LOSS_WEIGHT=8 FORWARD_DTYPE=bfloat16 FUTURE_SEED_SCALE=1 \
CASE_BANK_N=2 CASE_BANK_EVAL_N=256 CASE_BANK_LOOP_VALUES=1,2,3,5 \
RUN_NAME=<run_name> ./run.sh full
```

Kill criteria:

- If step100 takes more than 15 minutes, stop by exact PID and relaunch once at
  `FULL_BATCH=96`.
- If step500 blank accuracy remains near the no-FS plateau (`~0.27`) and CE is
  still high, stop and classify GDN as not open under this budget.
- If run completes, judge only by blank-range shape, not mixed exact alone.

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

Not a submission run.
