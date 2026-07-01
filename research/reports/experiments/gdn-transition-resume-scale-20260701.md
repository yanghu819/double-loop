# GDN FutureSeed 51-55 Resume Scaling

## 1. Metainfo

- Plan ID: `P-DIAG-005`
- Status: in-progress
- Local branch: `codex/gpu1-experiment-tracking`
- Scheduled time: `2026-07-01 14:10:33 +0800`
- Machine: AIStation `GPU1` only
- Remote work dir: `/huyang2/double-loop`
- Source SHA: current pushed branch SHA for this record
- Parent run: `gdn-transition-memory-expv4-s3000-20260701T0301Z-29c8aea`
- Parent checkpoint:
  `/huyang2/double-loop/.worktrees/gdn-transition-memory-29c8aea-20260701T0301Z/runs/gdn-transition-memory-expv4-s3000-20260701T0301Z-29c8aea/checkpoints/train_state_step003000.pt`

## 2. Hypothesis

P-DIAG-004 showed that larger recurrent value/state memory changes the 51-55
blank cliff:

- Previous normal-state GDN had `51-55` exact `0`.
- `GDN_EXPAND_V=4.0` opened `51-55` to exact `0.0098` and blank accuracy
  `0.6019`.
- Holes53 checkpoint exact had weak slope: step1000 `0.0205`, step2000
  `0.0215`, step3000 `0.0303`.

Mechanism question: is the 51-55 transition now mostly clean compute/data-bound,
or does the same recurrent state formulation hit another plateau?

Prediction:

- If clean scaling is enough, continuing the same large-state model should push
  holes53 checkpoint exact and 51-55 exact clearly upward, ideally past `0.05`.
- If checkpoint4200/4500 remain flat despite more hard examples, then same-state
  longer training is not the answer. The next step must change the generic
  recurrent state formulation.

This is one resume scaling gate, not a long-run table.

## 3. Configuration

- Data: official EqR Sudoku arrays, `sudoku-extreme-1k-aug-1000`
- Backbone: GDN Triton recurrent real backward + native terminal-state FutureSeed
- Resume checkpoint: parent step3000 checkpoint listed above
- Fixed architecture: `D_MODEL=192`, `LAYERS=10`, `HEADS=12`, `HEAD_DIM=16`,
  `CHANNEL_MULT=4`
- Memory/state: `GDN_EXPAND_V=4.0`
- Loop: `L_CYCLES=2`, `MAX_LOOPS=5`, final-loop loss
- Curriculum total after resume: `HOLE_STAGES=46-50:500,51-55:4000`
- Additional training after resume: 1500 hard-stage steps, total step 4500
- Train: microbatch `FULL_BATCH=64`, `GRAD_ACCUM_STEPS=2`, effective batch `128`
- Eval: official test split, `FULL_EVAL_N=1024`
- Checkpoint eval: steps `3600,4200,4500`, holes53
- Official blank-range eval: `46-50,51-55,56-64`
- Case-bank visualization: loops `1,2,3,5`
- CUDA path: `GDN_MODE=triton_recurrent`, `GDN_USE_SHORT_CONV=0`,
  `FORWARD_DTYPE=bfloat16`
- Forbidden: no CPU smoke, no GPU2, no selector/search/repair/oracle rollout,
  no Sudoku-specific rule, no seed/loss/expand-v/width table.

## 4. Environment

Pending launch.

## 5. Commands

Local static validation only:

```bash
bash -n run.sh
git diff --check
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
PYTHON_BIN=/opt/conda/bin/python \
RESUME_TRAIN_CHECKPOINT=<parent-step3000.pt> \
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
SUDOKU_SIZE=9 BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 \
GDN_EXPAND_V=4.0 D_MODEL=192 LAYERS=10 HEADS=12 HEAD_DIM=16 CHANNEL_MULT=4 \
L_CYCLES=2 MAX_LOOPS=5 HOLE_STAGES=46-50:500,51-55:4000 \
FULL_STEPS=4500 FULL_BATCH=64 GRAD_ACCUM_STEPS=2 FULL_EVAL_N=1024 \
HOLES_MIN=46 HOLES_MAX=55 EVAL_HOLES=53 EVAL_HOLES_LIST=50,53,56 \
OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64 \
EVAL_CHECKPOINT_STEPS=3600,4200,4500 EVAL_CHECKPOINT_HOLES_LIST=53 \
FULL_ROLLOUT_KS= ROLLOUT_LOOP_VALUES= FULL_LOG_EVERY=100 \
BLANK_LOSS_WEIGHT=8 FORWARD_DTYPE=bfloat16 FUTURE_SEED_SCALE=1 \
CASE_BANK_N=2 CASE_BANK_EVAL_N=256 CASE_BANK_LOOP_VALUES=1,2,3,5 \
RUN_NAME=<run_name> ./run.sh full
```

Kill criteria:

- If resume checkpoint cannot be loaded, stop and archive as infrastructure
  failure; do not restart from scratch.
- If GPU memory exceeds the 80GB device limit or CUDA OOMs, stop exact PID and
  archive.
- If checkpoint4200 holes53 exact is flat around `0.03` and blank accuracy has
  no meaningful slope, let final eval finish only if near; otherwise stop.
- If final `51-55` exact remains below `0.02`, mark same-state long scaling
  as low ROI and do not run 6000/9000 steps.

Success criteria:

- Strong: `51-55` loop5 exact `>=0.05`.
- Weak: holes53 checkpoint exact continues upward from `0.0303` and case-bank
  hard wrong cells keep dropping without damaging `46-50`.
- Negative: exact/CE/visual cases plateau; move to generic recurrent state
  formulation rather than more same-state training.

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

Not a submission run.
