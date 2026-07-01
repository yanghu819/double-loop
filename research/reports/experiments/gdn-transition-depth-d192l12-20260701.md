# GDN FutureSeed Conservative Depth Scaling

## 1. Metainfo

- Plan ID: `P-DIAG-007`
- Status: failed
- Local branch: `codex/gpu1-experiment-tracking`
- Scheduled time: `2026-07-01 15:45:00 +0800`
- Machine: AIStation `GPU1` only
- Remote work dir: `/huyang2/double-loop`
- Source SHA: `be24dc165e8ff61469ce87fefdbf261ff4e8352d`
- Parent evidence:
  `gdn-transition-resume-expv4-lr15-s4500-20260701T0622Z-33b477e`
  and failed D224/L12 NaN boundary

## 2. Hypothesis

D224/L12 + expand_v4 failed at step100 with NaN under the proven D192 LR. That
does not answer whether capacity scaling works; it only says the aggressive
width+depth jump is not optimizer-stable.

Mechanism question: does a smaller capacity jump, specifically D192/L10 to
D192/L12 while keeping recurrent value/state expansion, improve the 51-55
transition without destabilizing training?

Prediction:

- If depth is a useful capacity axis, D192/L12 should beat D192/L10 expand_v4
  at matched steps and ideally push holes53 loop5 exact toward `>=0.06`.
- If it NaNs or underperforms D192/L10, then blind depth/state scaling is not
  the immediate upper-bound path. The next work should stabilize the generic
  recurrent state/optimizer rather than add more layers.

This is one conservative scaling gate, not a depth table.

## 3. Configuration

- Data: official EqR Sudoku arrays, `sudoku-extreme-1k-aug-1000`
- Backbone: GDN Triton recurrent real backward + native terminal-state
  FutureSeed
- Architecture: `D_MODEL=192`, `LAYERS=12`, `HEADS=12`, `HEAD_DIM=16`,
  `CHANNEL_MULT=4`
- Memory/state: `GDN_EXPAND_V=4.0`
- Loop: `L_CYCLES=2`, `MAX_LOOPS=5`, final-loop loss
- Curriculum: `HOLE_STAGES=46-50:500,51-55:2500`
- Train: `FULL_STEPS=3000`, microbatch `FULL_BATCH=64`,
  `GRAD_ACCUM_STEPS=2`, effective batch `128`
- Eval: official test split, `FULL_EVAL_N=1024`
- Checkpoint eval: steps `1000,2000,3000`, holes53
- Official blank-range eval: `46-50,51-55,56-64`
- Case-bank visualization: loops `1,2,3,5`
- CUDA path: `GDN_MODE=triton_recurrent`, `GDN_USE_SHORT_CONV=0`,
  `FORWARD_DTYPE=bfloat16`, `ACTIVATION_CHECKPOINT=1`
- Optimization: `LR=0.0015`, `WEIGHT_DECAY=0.001`, `BLANK_LOSS_WEIGHT=8`
- Forbidden: no CPU smoke, no GPU2, no selector/search/repair/oracle rollout,
  no Sudoku-specific rule, no LR/seed/width/depth table.

## 4. Environment

- AIStation row: `GPU1`
- Remote worktree:
  `/huyang2/double-loop/.worktrees/gdn-transition-depth-d192l12-be24dc1-20260701T0748Z`
- Remote run:
  `/huyang2/double-loop/.worktrees/gdn-transition-depth-d192l12-be24dc1-20260701T0748Z/runs/gdn-transition-depth-d192l12-expv4-s3000-20260701T0748Z-be24dc1`
- Python: `/opt/conda/bin/python`
- Torch: `2.7.0+cu126`
- Source SHA in executed worktree:
  `be24dc165e8ff61469ce87fefdbf261ff4e8352d`
- Git dirty at launch: `0`
- GPU row: `GPU1`, `CUDA_VISIBLE_DEVICES=0`

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
GDN_EXPAND_V=4.0 D_MODEL=192 LAYERS=12 HEADS=12 HEAD_DIM=16 CHANNEL_MULT=4 \
L_CYCLES=2 MAX_LOOPS=5 HOLE_STAGES=46-50:500,51-55:2500 \
FULL_STEPS=3000 FULL_BATCH=64 GRAD_ACCUM_STEPS=2 FULL_EVAL_N=1024 \
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

- If step100 is NaN, stop exact PID and archive. Do not lower LR repeatedly.
- If step100 takes more than `20` minutes, stop as throughput-low-ROI.
- If step1000 holes53 loop5 exact is below D192/L10 expand_v4 step1000
  (`0.0205`) and blank_acc/CE are not better, stop unless a checkpoint is
  already near.
- If step2000 remains below D192/L10 step2000 (`0.0215`) with worse blank_acc,
  stop and classify depth scaling as low ROI.

Success criteria:

- Strong: official `51-55` loop5 exact `>= 0.05`, or holes53 checkpoint loop5
  exact `>= 0.06` by step3000.
- Weak: better matched-step slope than D192/L10 and case-bank hard boards keep
  changing through loop5.
- Negative: instability or no matched-step improvement; move to a cleaner
  generic recurrent state formulation.

## 6. Artifacts

- Local run archive:
  `runs/gdn-transition-depth-d192l12-expv4-s3000-20260701T0748Z-be24dc1`
- Abort metadata:
  `runs/gdn-transition-depth-d192l12-expv4-s3000-20260701T0748Z-be24dc1/abort.json`
- Config:
  `runs/gdn-transition-depth-d192l12-expv4-s3000-20260701T0748Z-be24dc1/config.json`
- Log:
  `runs/gdn-transition-depth-d192l12-expv4-s3000-20260701T0748Z-be24dc1/logs/run.log`
- Source provenance:
  `runs/gdn-transition-depth-d192l12-expv4-s3000-20260701T0748Z-be24dc1/source_HEAD.txt`
  `runs/gdn-transition-depth-d192l12-expv4-s3000-20260701T0748Z-be24dc1/source.patch`
  `runs/gdn-transition-depth-d192l12-expv4-s3000-20260701T0748Z-be24dc1/source_snapshot.tar.gz`

## 7. Results

The conservative D192/L12 depth run was stopped at the first logged training
point:

```text
[future_seed_loop stage=1:46-50] step=0100 ce=nan total=nan loop1=nan loop_last=nan
```

Abort:

- Timestamp: `2026-07-01T07:52:28Z`
- Reason: NaN loss at step100 under D192/L12, `GDN_EXPAND_V=4.0`,
  activation-checkpoint depth scaling
- Killed exact PIDs: `1101`, `1103`, `1135`, `1136`, `1137`
- GPU after kill: `0 MiB`, utilization `0%`

No quality metric should be read from this run.

## 8. Conclusions

Decision: failed, discard this activation-checkpoint depth run.

The important inference is that P-DIAG-006 and P-DIAG-007 share
`ACTIVATION_CHECKPOINT=1`, and both fail at step100 with NaN before the hard
51-55 stage. The D192/L10 expand_v4 runs without activation checkpoint were
numerically stable.

This makes the next question sharper:

- Is the NaN caused by activation checkpointing around the custom GDN recurrent
  path?
- Or does any 12-layer expand_v4 setup fail under the current optimizer?

Next decision:

- Do not lower LR as a table.
- Run one no-activation-checkpoint D192/L12 gate with smaller microbatch to keep
  memory within GPU1. If it trains, activation checkpoint is the culprit. If it
  still NaNs, depth/optimizer stability is the culprit.

## 9. Submission Record

Not a submission run.
