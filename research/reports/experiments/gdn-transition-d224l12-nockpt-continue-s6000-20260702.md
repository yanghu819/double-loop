# GDN FutureSeed D224/L12 No-Checkpoint Continue To 6000

## 1. Metainfo

- Plan ID: `P-DIAG-012`
- Status: in-progress
- Local branch: `codex/gpu1-experiment-tracking`
- Planned time: `2026-07-02 10:05 +0800`
- Machine: AIStation `GPU1` only
- Remote work dir: `/huyang2/double-loop`
- Source SHA: pending launch commit
- Parent run: `gdn-transition-d224l12-nockpt-long-s3000-20260701T1125Z-4346594`
- Parent checkpoint:
  `/huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-long-4346594-20260701T1125Z/runs/gdn-transition-d224l12-nockpt-long-s3000-20260701T1125Z-4346594/checkpoints/train_state_step003000.pt`

## 2. Hypothesis

P-DIAG-011 did not beat the D192/L10 4500 transition baseline, but it was not
flat. Holes53 loop5 exact improved `0.0176 -> 0.0195 -> 0.0352`, blank accuracy
improved `0.5284 -> 0.5449 -> 0.5750`, and final loop refinement was real
(`loop1 -> loop5` mixed exact `0.0215 -> 0.0410`).

The single question here is:

Does the larger D224/L12 backbone simply need a longer hard-stage run to cross
the D192/L10 transition gate, or is it a worse compute path despite being
larger?

Prediction:

- If D224/L12 is a genuine longer-scale winner, continuing to 6000 should beat
  the D192/L10 4500 reference: holes53 loop5 exact `>= 0.0615` or official
  `51-55` exact `> 0.0459`.
- If it mostly improves blank accuracy but not exact, same-family bigger-network
  scaling is not the efficient route to the upper bound.

This is one continuation run requested for long-run scaling. It is not a
seed/LR/loss/gate table.

## 3. Configuration

- Resume checkpoint: P-DIAG-011 step3000 train state
- Data: official EqR Sudoku arrays, `sudoku-extreme-1k-aug-1000`
- Backbone: GDN Triton recurrent real backward + native terminal-state
  FutureSeed
- Architecture: `D_MODEL=224`, `LAYERS=12`, `HEADS=14`, `HEAD_DIM=16`,
  `CHANNEL_MULT=4`
- Memory/state: `GDN_EXPAND_V=4.0`
- Loop: `L_CYCLES=2`, `MAX_LOOPS=5`
- Loop supervision: `LOOP_LOSS=all`
- Curriculum: `HOLE_STAGES=46-50:100,51-55:5900`
- Train: total `FULL_STEPS=6000`, microbatch `FULL_BATCH=32`,
  `GRAD_ACCUM_STEPS=4`, effective batch `128`
- Eval: official test split, `FULL_EVAL_N=512`
- Checkpoint eval: steps `4000,5000,6000`, holes53
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
RESUME_TRAIN_CHECKPOINT=/huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-long-4346594-20260701T1125Z/runs/gdn-transition-d224l12-nockpt-long-s3000-20260701T1125Z-4346594/checkpoints/train_state_step003000.pt \
SUDOKU_SIZE=9 BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 \
GDN_EXPAND_V=4.0 D_MODEL=224 LAYERS=12 HEADS=14 HEAD_DIM=16 CHANNEL_MULT=4 \
L_CYCLES=2 MAX_LOOPS=5 LOOP_LOSS=all \
HOLE_STAGES=46-50:100,51-55:5900 \
FULL_STEPS=6000 FULL_BATCH=32 GRAD_ACCUM_STEPS=4 FULL_EVAL_N=512 \
HOLES_MIN=46 HOLES_MAX=55 EVAL_HOLES=53 EVAL_HOLES_LIST=50,53,56 \
OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64 \
EVAL_CHECKPOINT_STEPS=4000,5000,6000 EVAL_CHECKPOINT_HOLES_LIST=53 \
SAVE_TRAIN_CHECKPOINT_EVERY=500 \
FULL_ROLLOUT_KS= ROLLOUT_LOOP_VALUES= FULL_LOG_EVERY=100 \
BLANK_LOSS_WEIGHT=8 LR=0.0015 WEIGHT_DECAY=0.001 \
FORWARD_DTYPE=bfloat16 FUTURE_SEED_SCALE=1 \
CASE_BANK_N=3 CASE_BANK_EVAL_N=192 CASE_BANK_LOOP_VALUES=1,2,3,5 \
RUN_NAME=<run_name> ./run.sh full
```

Kill criteria:

- Stop exact PID on OOM or NaN.
- At checkpoint4000, stop if holes53 loop5 exact regresses below step3000 and
  blank accuracy does not improve meaningfully.
- Stop if throughput makes step6000 impossible within the current GPU1 lease.

Success criteria:

- Positive: step6000 holes53 loop5 exact `>= 0.0615` or official `51-55`
  exact `> 0.0459`.
- Strong: official `51-55` exact `>= 0.08`.
- Negative: exact remains below D192/L10 reference despite more compute. That
  would argue for changing recurrent state formulation, not more same-family
  brute force.

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

Not a submission run.
