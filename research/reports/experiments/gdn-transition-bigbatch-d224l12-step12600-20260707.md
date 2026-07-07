# gdn-transition-bigbatch-d224l12-step12600-20260707

## 1. Metainfo

- Plan ID: `P-DIAG-026`
- Status: approved / waiting for verified GPU1 A800 resource
- Planned run name: `gdn-transition-bigbatch-d224l12-step12600-20260707T0345Z-e44147f`
- Branch: `codex/gpu1-experiment-tracking`
- Source SHA: `e44147fde8fada0a8373ffce3f3b9ded4d93ffc1`
- Machine: AIStation `GPU1` only
- Planned start: 2026-07-07 03:45 UTC

## 2. Hypothesis

P-DIAG-021 showed the strongest clean native FutureSeed GDN result so far:
loop5 full exact `0.2363`, loop1 to loop5 exact gain `+0.2129`, official
`51-55` exact `0.3555`, and `56-64` exact `0.1582`. The weak point is not
whether FutureSeed and loop compute work; they do. The weak point is that hard
holes60/64 checkpoint exact oscillates, which makes the hard-range upper bound
hard to trust.

P-DIAG-025 ruled out a small learned overwrite-rate gate: it was active, but did
not stabilize or improve the hard range. This run tests a simpler scaling
explanation: maybe each optimizer update sees too few hard boards, so board-level
consistency gradients are noisy. If true, a larger effective batch should make
holes60/64 smoother and preserve or improve hard exact without changing model
semantics.

## 3. Configuration

- Base checkpoint:
  `/huyang2/double-loop/.worktrees/gdn-transition-cyclicbridge-loop8-d224l12-step12000-20260703T0757Z/runs/gdn-transition-cyclicbridge-loop5-d224l12-step12000-restart1-20260704T0527Z-9c621a5/checkpoints/train_state_step012000.pt`
- Data:
  `/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000`
- Sudoku: official 9x9, random holes, train split `train`, eval split `test`
- Backbone: native FutureSeed GDN
- Model: `D_MODEL=224`, `LAYERS=12`, `HEADS=14`, `HEAD_DIM=16`
- State: `GDN_EXPAND_V=4.0`, `GDN_MODE=triton_recurrent`, `GDN_USE_SHORT_CONV=0`
- Loops: `MAX_LOOPS=5`, `LOOP_LOSS=all`, fixed loop update
- Batch: microbatch `FULL_BATCH=32`, `GRAD_ACCUM_STEPS=8`, effective batch `256`
- Train: resume from global step12000 to `FULL_STEPS=12600`
- Eval:
  - checkpoint eval steps `12300,12600`
  - checkpoint holes `53,60,64`
  - final official blank ranges `46-50,51-55,56-64`
  - case bank loops `1,3,5`

## 4. Environment

Pending. AIStation API restarted GPU1, but the row is currently queued as generic
`GPU:1`. Launch is blocked until GPU1 is verified as `NVIDIA-A800-SXM4-80GB`
with SSH and CUDA available. GPU2 is not allowed.

## 5. Commands

Planned remote launch:

```bash
export GITHUB_TOKEN="${GITHUB_TOKEN}"
export REPO_OWNER=yanghu819
export REPO_NAME=double-loop
cd /huyang2/double-loop
git fetch origin codex/gpu1-experiment-tracking
git worktree add --detach /huyang2/double-loop/.worktrees/pdiag026-bigbatch-e44147f-20260707 e44147fde8fada0a8373ffce3f3b9ded4d93ffc1
cd /huyang2/double-loop/.worktrees/pdiag026-bigbatch-e44147f-20260707

SMOKE_DONE=1 SKIP_SETUP=1 SOURCE_SNAPSHOT_MODE=lean CUDA_VISIBLE_DEVICES=0 \
SUDOKU_SIZE=9 HOLE_PATTERN=random \
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
OFFICIAL_SUDOKU_TRAIN_SPLIT=train OFFICIAL_SUDOKU_EVAL_SPLIT=test \
RESUME_TRAIN_CHECKPOINT=/huyang2/double-loop/.worktrees/gdn-transition-cyclicbridge-loop8-d224l12-step12000-20260703T0757Z/runs/gdn-transition-cyclicbridge-loop5-d224l12-step12000-restart1-20260704T0527Z-9c621a5/checkpoints/train_state_step012000.pt \
D_MODEL=224 LAYERS=12 HEADS=14 HEAD_DIM=16 CHANNEL_MULT=4 \
BACKBONE=gdn GDN_MODE=triton_recurrent GDN_EXPAND_V=4.0 GDN_USE_SHORT_CONV=0 \
FUTURE_SEED_SCALE=1 MAX_LOOPS=5 LOOP_LOSS=all LOOP_UPDATE_MODE=fixed \
FULL_BATCH=32 GRAD_ACCUM_STEPS=8 FULL_STEPS=12600 FULL_EVAL_N=512 FULL_LOG_EVERY=100 \
HOLE_STAGES=46-50:100,51-55:5900,56-64:300,51-64:500,51-55:300,56-64:500,51-64:600,51-55:300,56-64:500,51-64:600,56-64:300,51-64:300 \
EVAL_HOLES=53 EVAL_HOLES_LIST=53,60,64 \
EVAL_CHECKPOINT_STEPS=12300,12600 EVAL_CHECKPOINT_HOLES_LIST=53,60,64 \
OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64 \
CASE_BANK_HOLES=official_b46_50:46-50,official_b51_55:51-55,official_b56_64:56-64 \
CASE_BANK_N=4 CASE_BANK_EVAL_N=256 CASE_BANK_LOOP_VALUES=1,3,5 \
FULL_ROLLOUT_KS= RUN_NAME=gdn-transition-bigbatch-d224l12-step12600-20260707T0345Z-e44147f \
./run.sh full
```

Kill criteria:

- Do not launch unless GPU1 is verified as A800.
- If step100 throughput makes step12600 infeasible within the lease, stop by exact
  PID and write `abort.json`.
- If checkpoint holes60/64 are clearly below P-DIAG-021/P-DIAG-025 neighboring
  curves, stop and classify batch256 as low ROI.

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

No submission. This is a mechanism/scaling diagnostic only.
