# GDN FutureSeed 51-55 Transition Memory Scaling

## 1. Metainfo

- Plan ID: `P-DIAG-004`
- Status: in-progress
- Local branch: `codex/gpu1-experiment-tracking`
- Scheduled time: `2026-07-01 11:01:25 +0800`
- Machine: AIStation `GPU1` only
- Remote work dir: `/huyang2/double-loop`
- Intended source SHA: current pushed branch SHA for this record

## 2. Hypothesis

P-DIAG-002 and P-DIAG-003 showed the same shape for RWKV+FutureSeed and
GDN+FutureSeed: the model almost solves `46-50` blank Sudoku boards, then exact
accuracy collapses to zero at `51+` blanks.

Mechanism question: is this cliff mainly because the recurrent memory/state is
too small to carry enough global alternatives through the loop?

Prediction:

- If recurrent memory/state capacity is the bottleneck, a large value/state
  expansion should open `51-55` blanks while preserving the solved `46-50`
  bucket.
- If `46-50` stays solved but `51-55` remains exact `0`, then simple memory
  scaling is not enough; the next step needs a different generic state update
  or training dynamic, not an expand-v table.

This is one high-information probe, not an ablation table.

## 3. Configuration

- Data: official EqR Sudoku arrays, `sudoku-extreme-1k-aug-1000`
- Backbone: GDN Triton recurrent real backward + native terminal-state FutureSeed
- Fixed token architecture: `D_MODEL=192`, `LAYERS=10`, `HEADS=12`,
  `HEAD_DIM=16`, `CHANNEL_MULT=4`
- Memory/state scaling: `GDN_EXPAND_V=4.0`
- Loop: `L_CYCLES=2`, `MAX_LOOPS=5`, final-loop loss
- Curriculum: `HOLE_STAGES=46-50:500,51-55:2500`
- Train: microbatch `FULL_BATCH=64`, `GRAD_ACCUM_STEPS=2`, effective batch `128`
- Eval: official test split, `FULL_EVAL_N=1024`
- Official blank-range eval: `46-50,51-55,56-64`
- Checkpoint eval: steps `1000,2000,3000`
- Case-bank visualization: per blank-range groups, loops `1,2,3,5`
- CUDA path: `GDN_MODE=triton_recurrent`, `GDN_USE_SHORT_CONV=0`,
  `FORWARD_DTYPE=bfloat16`
- Forbidden: no CPU smoke, no GPU2, no selector/search/repair/oracle rollout,
  no Sudoku-specific rule, no expand-v/seed/loss/width table.

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
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
SUDOKU_SIZE=9 BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 \
GDN_EXPAND_V=4.0 D_MODEL=192 LAYERS=10 HEADS=12 HEAD_DIM=16 CHANNEL_MULT=4 \
L_CYCLES=2 MAX_LOOPS=5 HOLE_STAGES=46-50:500,51-55:2500 \
FULL_STEPS=3000 FULL_BATCH=64 GRAD_ACCUM_STEPS=2 FULL_EVAL_N=1024 \
HOLES_MIN=46 HOLES_MAX=55 EVAL_HOLES=53 EVAL_HOLES_LIST=50,53,56 \
OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64 \
EVAL_CHECKPOINT_STEPS=1000,2000,3000 EVAL_CHECKPOINT_HOLES_LIST=53 \
FULL_ROLLOUT_KS= ROLLOUT_LOOP_VALUES= FULL_LOG_EVERY=100 \
BLANK_LOSS_WEIGHT=8 FORWARD_DTYPE=bfloat16 FUTURE_SEED_SCALE=1 \
CASE_BANK_N=2 CASE_BANK_EVAL_N=256 CASE_BANK_LOOP_VALUES=1,2,3,5 \
RUN_NAME=<run_name> ./run.sh full
```

Kill criteria:

- OOM or CUDA kernel failure: stop exact PID and archive as infrastructure
  boundary.
- Step100 wall time above 15 minutes: stop and relaunch one smaller probe with
  microbatch `32`, `GRAD_ACCUM_STEPS=4`.
- If checkpoint1000 on holes53 has loop5 exact `0` and blank accuracy below
  `0.50`, stop as optimization failure.
- If checkpoint2000 has `51-55` exact still `0` and no blank/wrong-cell slope,
  finish only if final eval is near; otherwise archive as negative memory gate.

Success criteria:

- Strong: `51-55` loop5 exact `>=0.05` while `46-50` exact remains high.
- Weak: `51-55` blank accuracy and hard-case wrong-cell trajectories improve
  clearly without hurting `46-50`.
- Negative: `51-55` exact remains `0`; do not run expand-v values as a table.

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Submission Record

Not a submission run.
