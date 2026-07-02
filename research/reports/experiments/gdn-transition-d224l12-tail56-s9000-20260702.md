# GDN FutureSeed D224/L12 Tail 56-64 Continuation To 9000

## 1. Metainfo

- Plan ID: `P-DIAG-013`
- Status: in-progress
- Local branch: `codex/gpu1-experiment-tracking`
- Planned time: `2026-07-02 12:45 +0800`
- Machine: AIStation `GPU1` only
- Remote work dir: `/huyang2/double-loop`
- Source SHA: pending launch commit
- Parent run:
  `gdn-transition-d224l12-nockpt-continue-s6000-20260702T0209Z-c00167e`
- Parent checkpoint:
  `/huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-continue-c00167e-20260702T0209Z/runs/gdn-transition-d224l12-nockpt-continue-s6000-20260702T0209Z-c00167e/checkpoints/train_state_step006000.pt`

## 2. Hypothesis

P-DIAG-012 proved that the D224/L12 native-FutureSeed GDN recipe is no longer
stuck at the 51-55 blank cliff once we give it enough hard-stage tokens:
official `51-55` loop5 exact reached `0.2832`, and holes53 checkpoint exact
rose monotonically through step6000.

The remaining question is whether the same clean scaling recipe can move the
next cliff, `56-64`, or whether that bucket needs a larger model/state or a
different recurrent state update.

Prediction:

- Positive: continuing from step6000 with a 56-64 tail should raise holes60 and
  official `56-64` exact, while not destroying the newly opened `51-55`.
- Negative: if holes60 stays near zero or `51-55` collapses, this recipe has
  likely saturated its hard-distribution transfer. Then the next move is bigger
  model/state or a simple generic state-dynamics change, not more same-shape
  tail training.

This is a single continuation run. It is not a seed/LR/loss/gate sweep and it
does not add Sudoku-specific rules.

## 3. Configuration

- Resume checkpoint: P-DIAG-012 step6000 train state
- Data: official EqR Sudoku arrays, `sudoku-extreme-1k-aug-1000`
- Backbone: GDN Triton recurrent real backward + native terminal-state
  FutureSeed
- Architecture: `D_MODEL=224`, `LAYERS=12`, `HEADS=14`, `HEAD_DIM=16`,
  `CHANNEL_MULT=4`
- Memory/state: `GDN_EXPAND_V=4.0`
- Loop: `L_CYCLES=2`, `MAX_LOOPS=5`
- Loop supervision: `LOOP_LOSS=all`
- Curriculum: `HOLE_STAGES=46-50:100,51-55:5900,56-64:3000`
- Train: total `FULL_STEPS=9000`, microbatch `FULL_BATCH=32`,
  `GRAD_ACCUM_STEPS=4`, effective batch `128`
- Eval: official test split, `FULL_EVAL_N=512`
- Checkpoint eval: steps `7000,8000,9000`, holes `53,60`
- Official blank-range eval: `46-50,51-55,56-64`
- CUDA path: `GDN_MODE=triton_recurrent`, `GDN_USE_SHORT_CONV=0`,
  `FORWARD_DTYPE=bfloat16`, no activation checkpoint
- Optimization: `LR=0.0015`, `WEIGHT_DECAY=0.001`, `BLANK_LOSS_WEIGHT=8`
- Forbidden: no CPU smoke, no GPU2, no selector/search/repair/oracle rollout,
  no Sudoku-specific rule, no seed/LR/loss/gate table.

## 4. Environment

Pending launch.

## 5. Commands

Remote launch shape:

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 \
SOURCE_SNAPSHOT_MODE=lean UPDATE_LEADERBOARD=0 \
PYTHON_BIN=/opt/conda/bin/python \
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
RESUME_TRAIN_CHECKPOINT=/huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-continue-c00167e-20260702T0209Z/runs/gdn-transition-d224l12-nockpt-continue-s6000-20260702T0209Z-c00167e/checkpoints/train_state_step006000.pt \
SUDOKU_SIZE=9 BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 \
GDN_EXPAND_V=4.0 D_MODEL=224 LAYERS=12 HEADS=14 HEAD_DIM=16 CHANNEL_MULT=4 \
L_CYCLES=2 MAX_LOOPS=5 LOOP_LOSS=all \
HOLE_STAGES=46-50:100,51-55:5900,56-64:3000 \
FULL_STEPS=9000 FULL_BATCH=32 GRAD_ACCUM_STEPS=4 FULL_EVAL_N=512 \
HOLES_MIN=46 HOLES_MAX=64 EVAL_HOLES=60 EVAL_HOLES_LIST=53,60 \
OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64 \
EVAL_CHECKPOINT_STEPS=7000,8000,9000 EVAL_CHECKPOINT_HOLES_LIST=53,60 \
SAVE_TRAIN_CHECKPOINT_EVERY=500 \
FULL_ROLLOUT_KS= ROLLOUT_LOOP_VALUES= FULL_LOG_EVERY=100 \
BLANK_LOSS_WEIGHT=8 LR=0.0015 WEIGHT_DECAY=0.001 \
FORWARD_DTYPE=bfloat16 FUTURE_SEED_SCALE=1 \
CASE_BANK_N=3 CASE_BANK_EVAL_N=192 CASE_BANK_LOOP_VALUES=1,2,3,5 \
RUN_NAME=<run_name> ./run.sh full
```

Kill criteria:

- Stop exact PID on OOM or NaN.
- If checkpoint7000/8000 shows holes60 exact still near zero and holes53 or
  official `51-55` is clearly collapsing, stop instead of burning full tail.
- Stop if throughput makes step9000 infeasible under the current GPU1 lease.

Success criteria:

- Positive: step9000 holes60 loop5 exact clearly above the P-DIAG-012
  `56-64` regime, or final official `56-64` loop5 exact `>= 0.15`.
- Guardrail: final official `51-55` loop5 exact should stay `>= 0.20`.
- Strong: official `56-64` loop5 exact `>= 0.20` while `51-55` remains strong.

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

Not a submission run.
