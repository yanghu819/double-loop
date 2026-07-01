# GDN FutureSeed Deep-State Transition Scaling

## 1. Metainfo

- Plan ID: `P-DIAG-006`
- Status: in-progress
- Local branch: `codex/gpu1-experiment-tracking`
- Scheduled time: `2026-07-01 15:20:00 +0800`
- Machine: AIStation `GPU1` only
- Remote work dir: `/huyang2/double-loop`
- Source SHA: pending commit for this plan
- Parent evidence:
  `gdn-transition-resume-expv4-lr15-s4500-20260701T0622Z-33b477e`

## 2. Hypothesis

P-DIAG-005 changed the diagnosis: the `51-55` blank cliff is no longer a hard
zero once recurrent value/state capacity is expanded and the hard-stage sees
more compute.

Observed before this run:

- D192/L10/expand_v4 step4500 holes53 loop5 exact: `0.0615`
- D192/L10/expand_v4 final mixed loop1->loop5 exact: `0.0088 -> 0.0537`
- D192/L10/expand_v4 official `51-55` loop5 exact/blank: `0.0459 / 0.6555`
- `46-50` remains solved at `1.0000`, so the transition is localized around
  global consistency under heavier masking.

Mechanism question: is the remaining failure mostly lack of effective model
and recurrent-state capacity, or did the D192/L10 state formulation already
hit the useful scaling wall?

Prediction:

- If bitter-lesson capacity scaling is the right axis, a bigger D224/L12
  expand_v4 model should beat the D192/L10 expand_v4 step3000 transition
  point and ideally reach official `51-55` loop5 exact `>= 0.05`.
- If it is just more expensive without better exact/CE slope, do not continue
  width/depth tables. The next move should be a cleaner generic recurrent
  state formulation.

This is a single high-information scaling gate, not a width/depth sweep.

## 3. Configuration

- Data: official EqR Sudoku arrays, `sudoku-extreme-1k-aug-1000`
- Backbone: GDN Triton recurrent real backward + native terminal-state
  FutureSeed
- Architecture: `D_MODEL=224`, `LAYERS=12`, `HEADS=14`, `HEAD_DIM=16`,
  `CHANNEL_MULT=4`
- Memory/state: `GDN_EXPAND_V=4.0`
- Loop: `L_CYCLES=2`, `MAX_LOOPS=5`, final-loop loss
- Curriculum: `HOLE_STAGES=46-50:500,51-55:2500`
- Train: `FULL_STEPS=3000`, microbatch `FULL_BATCH=48`,
  `GRAD_ACCUM_STEPS=3`, effective batch `144`
- Eval: official test split, `FULL_EVAL_N=1024`
- Checkpoint eval: steps `1000,2000,3000`, holes53
- Official blank-range eval: `46-50,51-55,56-64`
- Case-bank visualization: loops `1,2,3,5`
- CUDA path: `GDN_MODE=triton_recurrent`, `GDN_USE_SHORT_CONV=0`,
  `FORWARD_DTYPE=bfloat16`, `ACTIVATION_CHECKPOINT=1`
- Optimization: `LR=0.0015`, `WEIGHT_DECAY=0.001`, `BLANK_LOSS_WEIGHT=8`
- Forbidden: no CPU smoke, no GPU2, no selector/search/repair/oracle rollout,
  no Sudoku-specific rule, no seed/loss/width/depth table.

## 4. Environment

Pending launch.

## 5. Commands

Remote launch shape:

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 \
SOURCE_SNAPSHOT_MODE=lean UPDATE_LEADERBOARD=0 \
XDG_CACHE_HOME=/huyang2/double-loop/.cache \
PIP_CACHE_DIR=/huyang2/double-loop/.cache/pip \
HF_HOME=/huyang2/double-loop/.cache/huggingface \
TORCH_HOME=/huyang2/double-loop/.cache/torch \
TORCH_EXTENSIONS_DIR=/huyang2/double-loop/.cache/torch_extensions \
PYTHON_BIN=/opt/conda/bin/python \
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
SUDOKU_SIZE=9 BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 \
GDN_EXPAND_V=4.0 D_MODEL=224 LAYERS=12 HEADS=14 HEAD_DIM=16 CHANNEL_MULT=4 \
L_CYCLES=2 MAX_LOOPS=5 HOLE_STAGES=46-50:500,51-55:2500 \
FULL_STEPS=3000 FULL_BATCH=48 GRAD_ACCUM_STEPS=3 FULL_EVAL_N=1024 \
HOLES_MIN=46 HOLES_MAX=55 EVAL_HOLES=53 EVAL_HOLES_LIST=50,53,56 \
OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64 \
EVAL_CHECKPOINT_STEPS=1000,2000,3000 EVAL_CHECKPOINT_HOLES_LIST=53 \
SAVE_TRAIN_CHECKPOINT_EVERY=1000 \
FULL_ROLLOUT_KS= ROLLOUT_LOOP_VALUES= FULL_LOG_EVERY=100 \
BLANK_LOSS_WEIGHT=8 LR=0.0015 WEIGHT_DECAY=0.001 \
FORWARD_DTYPE=bfloat16 FUTURE_SEED_SCALE=1 ACTIVATION_CHECKPOINT=1 \
CASE_BANK_N=2 CASE_BANK_EVAL_N=256 CASE_BANK_LOOP_VALUES=1,2,3,5 \
RUN_NAME=<run_name> ./run.sh full
```

Kill criteria:

- If CUDA OOMs, stop and archive; do not lower one knob repeatedly.
- If step100 takes more than `25` minutes, stop exact PID and archive as
  throughput-low-ROI.
- If step1000 holes53 loop5 exact is below D192/L10 expand_v4 step1000
  (`0.0205`) and CE is not clearly better, stop unless the run is already near
  a scheduled checkpoint.
- If step2000 remains below D192/L10 step2000 (`0.0215`) with worse blank_acc,
  stop and classify blind deep scaling as low ROI.

Success criteria:

- Strong: official `51-55` loop5 exact `>= 0.05`, or holes53 checkpoint loop5
  exact `>= 0.06` by step3000.
- Weak: clear better slope than D192/L10 at matched step, with case-bank hard
  wrong cells continuing to fall through loop5.
- Negative: more expensive but no better exact/CE/visual slope; move to generic
  recurrent state formulation.

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

Not a submission run.
