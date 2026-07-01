# GDN FutureSeed 51-55 Transition Memory Scaling

## 1. Metainfo

- Plan ID: `P-DIAG-004`
- Status: completed
- Local branch: `codex/gpu1-experiment-tracking`
- Scheduled time: `2026-07-01 11:01:25 +0800`
- Machine: AIStation `GPU1` only
- Remote work dir: `/huyang2/double-loop`
- Source SHA: `29c8aea514c3647d77f867e7c473e6aee358ff47`
- Run name: `gdn-transition-memory-expv4-s3000-20260701T0301Z-29c8aea`

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

- Remote worktree:
  `/huyang2/double-loop/.worktrees/gdn-transition-memory-29c8aea-20260701T0301Z`
- Run dir:
  `/huyang2/double-loop/.worktrees/gdn-transition-memory-29c8aea-20260701T0301Z/runs/gdn-transition-memory-expv4-s3000-20260701T0301Z-29c8aea`
- Device: GPU1 only, `CUDA_VISIBLE_DEVICES=0`
- GPU: A800-SXM4-80GB
- Python: `/opt/conda/bin/python`
- Torch: `2.7.0+cu126`
- Git dirty at launch: false
- Peak CUDA memory: `63453.6 MB` allocated, `64420.0 MB` reserved

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

- Local run copy:
  `runs/gdn-transition-memory-expv4-s3000-20260701T0301Z-29c8aea`
- Git-archived visualization/report copy:
  `research/reports/visualizations/gdn-transition-memory-scale-20260701`
- Main JSON:
  `research/reports/visualizations/gdn-transition-memory-scale-20260701/output/futureseed_loop_seed52.json`
- Case-bank HTML:
  `research/reports/visualizations/gdn-transition-memory-scale-20260701/output/case_bank/official_b51_55/index.html`
  and
  `research/reports/visualizations/gdn-transition-memory-scale-20260701/output/case_bank/official_b56_64/index.html`
- Dashboard:
  `research/reports/visualizations/gdn-transition-memory-scale-20260701/visualizations/index.html`
- Logs:
  `research/reports/visualizations/gdn-transition-memory-scale-20260701/logs/run.log`
- Checkpoints were generated remotely for checkpoint eval, but were not pulled
  into the Git archive.

## 7. Results

Training:

| step | stage | train CE |
|---:|---|---:|
| 100 | 46-50 | 1.3350 |
| 300 | 46-50 | 0.0669 |
| 500 | 46-50 | 0.0216 |
| 1000 | 51-55 | 0.9747 |
| 2000 | 51-55 | 0.9189 |
| 2400 | 51-55 | 0.8414 |
| 3000 | 51-55 | 0.8863 |

Checkpoint eval on holes53:

| checkpoint | loop5 exact | loop5 blank_acc |
|---:|---:|---:|
| 1000 | 0.0205 | 0.5112 |
| 2000 | 0.0215 | 0.5390 |
| 3000 | 0.0303 | 0.5580 |

Final mixed official eval, `eval_n=1024`:

| loop | exact | blank_acc | clue_ok |
|---:|---:|---:|---:|
| 1 | 0.0020 | 0.4425 | 0.8652 |
| 2 | 0.0273 | 0.5054 | 0.9961 |
| 3 | 0.0283 | 0.5447 | 0.9990 |
| 4 | 0.0381 | 0.5564 | 1.0000 |
| 5 | 0.0381 | 0.5571 | 1.0000 |

Official blank-range eval:

| blank range | loop1 exact | loop1 blank_acc | loop5 exact | loop5 blank_acc |
|---|---:|---:|---:|---:|
| 46-50 | 0.0820 | 0.9392 | 1.0000 | 1.0000 |
| 51-55 | 0.0000 | 0.4654 | 0.0098 | 0.6019 |
| 56-64 | 0.0000 | 0.4112 | 0.0078 | 0.5157 |

Case-bank trajectories:

| group | case | holes | loop1 wrong/conflict | loop2 | loop3 | loop5 |
|---|---|---:|---:|---:|---:|---:|
| 51-55 | solved b0123 | 53 | 20 / 26 | 14 / 22 | 2 / 4 | 0 / 0 |
| 51-55 | almost b0052 | 53 | 25 / 23 | 13 / 20 | 5 / 13 | 3 / 9 |
| 51-55 | hard b0177 | 55 | 23 / 25 | 13 / 17 | 8 / 10 | 5 / 3 |
| 51-55 | hard b0047 | 55 | 17 / 19 | 10 / 16 | 8 / 12 | 6 / 3 |
| 56-64 | solved b0091 | 57 | 25 / 27 | 18 / 23 | 11 / 20 | 0 / 0 |
| 56-64 | almost b0138 | 56 | 28 / 26 | 13 / 22 | 10 / 19 | 4 / 10 |
| 56-64 | hard b0210 | 56 | 24 / 25 | 20 / 23 | 10 / 19 | 5 / 9 |
| 56-64 | hard b0056 | 57 | 21 / 26 | 21 / 24 | 14 / 21 | 9 / 13 |

Comparison to the previous short GDN mask-cliff diagnostic:

- Previous P-DIAG-003 at D192/L10/1000 steps, normal state:
  `46-50 exact=0.9863`, `51-55 exact=0`, `56-64 exact=0`,
  `51-55 blank_acc=0.5457`.
- This run with `GDN_EXPAND_V=4.0` and focused 51-55 hard stage:
  `46-50 exact=1.0000`, `51-55 exact=0.0098`, `56-64 exact=0.0078`,
  `51-55 blank_acc=0.6019`.
- Mixed official loop gain is real:
  loop1->loop5 exact `0.0020 -> 0.0381`, blank_acc
  `0.4425 -> 0.5571`.

## 8. Conclusions

Decision: mark as weak positive, not a solved result.

The strong success criterion was not met: `51-55` loop5 exact is `0.0098`,
well below the `>=0.05` gate. Still, the result is not the same cliff as the
previous diagnostic. Larger recurrent value/state plus a focused transition
curriculum opens nonzero exact on `51-55`, improves `51-55` blank accuracy from
about `0.5457` to `0.6019`, preserves the solved `46-50` bucket, and the
case-bank shows loop5 genuinely reducing wrong cells in hard boards.

Mechanism insight:

- The `51+` cliff is not simply "FutureSeed cannot handle many blanks".
  Recurrent state capacity changes the boundary.
- The remaining bottleneck is exact/global consistency: train CE is still
  high around `0.88` at step3000, and many hard cases stop with a few wrong
  but stable cells.
- Do not run an `expand_v` table. The single `expand_v=4` probe already
  answers the useful question: memory/state scaling matters, but simple state
  expansion alone is not enough.

Next high-ROI move:

- If we keep scaling, change only one meaningful axis: a more efficient bigger
  state run with stronger throughput, or a simple generic recurrent state
  formulation that preserves alternatives across loops.
- Avoid loss-weight, seed, margin, noise, or threshold sweeps. They have already
  failed to convert blank accuracy into full-board exact.

## 9. Submission Record

Not a submission run.
