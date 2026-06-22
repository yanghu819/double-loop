# official-eqr-h96-fs-gate-20260622

## 1. Metainfo

- Plan ID: `P-EQR-007`
- Status: in-progress
- Machine: AIStation GPU1 only
- Remote worktree: `/huyang2/double-loop/.worktrees/official-eqr-h96-gate`
- Remote official EqR base: `/huyang2/double-loop/official_eqr_compare`
- Launcher source SHA: to be filled after commit
- Date: 2026-06-22 Asia/Shanghai

## 2. Hypothesis

FutureSeed should be tested as a cheap bidirectional/context mechanism under a
real capacity or compute bottleneck, not as a small add-on to a full EqR mixer.
If it has paper-level value inside the official EqR codebase, then a compressed
EqR model should benefit more from FutureSeed than the full D128 model did.

This run compresses the official Maze EqR train config from `hidden_size=128` to
`hidden_size=96` while keeping `num_heads=8`. That reduces attention, MLP, state,
embedding, and readout width in a generic way; it does not add task-specific
rules, repair, search, selector, or oracle decoding.

## 3. Configuration

- Official EqR upstream SHA: `aba94e9cde0f273ce644db5261cd6915ba6561f0`
- Dataset: official `maze-30x30-unique-1k`
- Objective: path-token-weighted official EqR token objective,
  `PATH_TOKEN_WEIGHT=8`
- Pair:
  - compressed base: `arch.hidden_size=96 arch.num_heads=8`
  - compressed FutureSeed: same plus `arch.future_seed_scale=1`,
    `arch.future_seed_gate_bias=-2`
- Budget: `EPOCHS=256`, `TRAIN_EPOCHS_PER_ITER=256`,
  `GLOBAL_BATCH_SIZE=128`, visual eval loops `1,4,8,16`, max cases `256`
- Existing boundary references:
  - full official EqR D128 pathw8 loop16 F1 about `0.46824`
  - full official EqR D128+FutureSeed pathw8 loop16 F1 about `0.46835`
  - causalized EqR D128+FutureSeed delta only `+0.00009`

## 4. Environment

To be filled from remote launch.

## 5. Commands

Train compressed base:

```bash
CUDA_VISIBLE_DEVICES=0 EPOCHS=256 TRAIN_EPOCHS_PER_ITER=256 GLOBAL_BATCH_SIZE=128 \
PATH_TOKEN_WEIGHT=8 EQR_EXTRA_OVERRIDES='arch.hidden_size=96 arch.num_heads=8' \
RUN_NAME=official-eqr-h96-base-pathw8-e256-20260622TBD \
  /huyang2/double-loop/.worktrees/official-eqr-h96-gate/scripts/official_eqr_compare/run_official_eqr_compare.sh train-base
```

Train compressed FutureSeed:

```bash
CUDA_VISIBLE_DEVICES=0 EPOCHS=256 TRAIN_EPOCHS_PER_ITER=256 GLOBAL_BATCH_SIZE=128 \
PATH_TOKEN_WEIGHT=8 EQR_EXTRA_OVERRIDES='arch.hidden_size=96 arch.num_heads=8' \
FUTURE_SEED_SCALE=1 FUTURE_SEED_GATE_BIAS=-2 \
RUN_NAME=official-eqr-h96-fs-pathw8-e256-20260622TBD \
  /huyang2/double-loop/.worktrees/official-eqr-h96-gate/scripts/official_eqr_compare/run_official_eqr_compare.sh train-futureseed
```

Visual/eval commands will be filled after checkpoint paths are known.

## 6. Artifacts

Pending.

## 7. Results

Pending.

Required readouts:

- loop1/4/8/16 path F1, precision, recall, pred_path_frac, FP/FN
- loop1 to loop16 gain
- params and train wall time
- whether FutureSeed reaches D128 baseline with materially fewer params
- hard-case visuals

## 8. Conclusions

Pending.

Kill criteria:

- Any run leaves GPU1 or uses CPU smoke.
- Official EqR data/config path differs between arms.
- FutureSeed only changes token accuracy while path F1/FP/FN stay unchanged.
- FutureSeed gain comes from larger pred_path_frac / broader mask.
- Both compressed arms reproduce the same broad-mask operating point with
  `|delta path F1| < 0.01`; then stop Maze positive-evidence experiments and
  keep Maze only as failure analysis.

## 9. Submission Record

N/A.
