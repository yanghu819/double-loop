# GDN FutureSeed D224/L12 No-Checkpoint Continue To 6000

## 1. Metainfo

- Plan ID: `P-DIAG-012`
- Status: done
- Local branch: `codex/gpu1-experiment-tracking`
- Planned time: `2026-07-02 10:05 +0800`
- Launched: `2026-07-02 10:09 +0800`
- Completed: `2026-07-02 12:20 +0800`
- Machine: AIStation `GPU1` only
- Remote work dir: `/huyang2/double-loop`
- Source SHA: `c00167e1f6d8623d632d1598203323feda3fc5c0`
- Run name:
  `gdn-transition-d224l12-nockpt-continue-s6000-20260702T0209Z-c00167e`
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

- AIStation row: `GPU1`
- Host: `430g41c4b65qj-0`
- GPU: `NVIDIA A100-SXM4-80GB`
- Remote worktree:
  `/huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-continue-c00167e-20260702T0209Z`
- Run dir:
  `/huyang2/double-loop/.worktrees/gdn-transition-d224l12-nockpt-continue-c00167e-20260702T0209Z/runs/gdn-transition-d224l12-nockpt-continue-s6000-20260702T0209Z-c00167e`
- Python: `/opt/conda/bin/python`
- Torch/CUDA from run log: `torch=2.7.0+cu126`, device `cuda`
- Peak CUDA allocated/reserved is recorded in `output/futureseed_loop_seed52.json`.
- Checkpoints stayed on remote only and were excluded from the Git archive.

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

- Local archive:
  `runs/gdn-transition-d224l12-nockpt-continue-s6000-20260702T0209Z-c00167e`
- Remote archive:
  `/huyang2/double-loop/artifacts/gdn-transition-d224l12-nockpt-continue-s6000-20260702T0209Z-c00167e-metadata-light.tgz`
- Config:
  `runs/gdn-transition-d224l12-nockpt-continue-s6000-20260702T0209Z-c00167e/config.json`
- Score:
  `runs/gdn-transition-d224l12-nockpt-continue-s6000-20260702T0209Z-c00167e/score.json`
- Main output:
  `runs/gdn-transition-d224l12-nockpt-continue-s6000-20260702T0209Z-c00167e/output/futureseed_loop_seed52.json`
- Checkpoint eval JSON:
  `runs/gdn-transition-d224l12-nockpt-continue-s6000-20260702T0209Z-c00167e/output/checkpoint_eval_step004000.json`
  through `checkpoint_eval_step006000.json`
- Visual index:
  `runs/gdn-transition-d224l12-nockpt-continue-s6000-20260702T0209Z-c00167e/visualizations/index.html`
- Case banks:
  `output/case_bank/official_b46_50`,
  `output/case_bank/official_b51_55`,
  `output/case_bank/official_b56_64`

## 7. Results

Primary score:

| metric | value |
|---|---:|
| `metrics.eval_clean.loop5.label_exact` | `0.20703125` |
| final loop5 blank accuracy | `0.62158668` |
| final loop1 exact / blank | `0.0234375 / 0.53704417` |
| final loop5 exact / blank | `0.20703125 / 0.62158668` |

Final loop curve on the mixed official eval:

| loop | exact | blank_acc |
|---:|---:|---:|
| 1 | `0.0234` | `0.5370` |
| 2 | `0.0352` | `0.5881` |
| 3 | `0.1289` | `0.6136` |
| 4 | `0.2031` | `0.6214` |
| 5 | `0.2070` | `0.6216` |

Holes53 checkpoint eval:

| total step | loop5 exact | loop5 blank_acc |
|---:|---:|---:|
| 100 | `0.0000` | `0.2443` |
| 300 | `0.0020` | `0.4704` |
| 1000 | `0.0176` | `0.5284` |
| 2000 | `0.0195` | `0.5449` |
| 3000 | `0.0352` | `0.5750` |
| 4000 | `0.0684` | `0.5949` |
| 5000 | `0.1250` | `0.6111` |
| 6000 | `0.1680` | `0.6120` |

Official blank-range eval:

| range | loop1 exact / blank | loop5 exact / blank |
|---|---:|---:|
| `46-50` | `0.9375 / 0.9983` | `1.0000 / 1.0000` |
| `51-55` | `0.0000 / 0.5814` | `0.2832 / 0.7139` |
| `56-64` | `0.0000 / 0.4883` | `0.0801 / 0.5405` |

Case-bank summaries:

| group | final exact | final blank_acc | selected cases |
|---|---:|---:|---|
| `official_b46_50` | `1.0000` | `1.0000` | 3 solved-by-loop |
| `official_b51_55` | `0.3333` | `0.7392` | 3 solved, 3 almost, 3 hard failures |
| `official_b56_64` | `0.0990` | `0.5612` | 3 solved, 2 almost, 3 hard failures |

## 8. Conclusions

Decision: keep. This run decisively answers the P-DIAG-012 question: D224/L12
was not worse because capacity scaling was useless; it needed much longer
hard-stage training. The holes53 exact curve is not flat. It moves
`0.0352 -> 0.0684 -> 0.1250 -> 0.1680` from step3000 to step6000, and final
official `51-55` exact reaches `0.2832`, far above the previous D192/L10
transition reference (`0.0459`).

The important mechanism signal is loop depth, not just local token accuracy:
on the final mixed eval, loop1 exact is only `0.0234`, while loop5 exact is
`0.2070`. On the official `51-55` bucket, loop1 exact is still `0`, but loop5
exact is `0.2832`. That means the loop is converting partial blank accuracy
into globally valid boards after the model has enough capacity and hard-stage
training.

Remaining bottleneck: the `56-64` bucket is opened but still weak
(`0.0801` exact). The next scaling decision should not be another tiny
loss/gate/noise tweak. The high-ROI directions are either:

- continue clean capacity/data/compute scaling around this stable D224/L12
  recipe, or
- move to a similarly simple generic recurrent-state update only if it can
  preserve this long-training slope.

This is strong evidence for the bitter-lesson path: no Sudoku repair, no
selector, no oracle rollout, no maze/Sudoku rule. More model/state plus more
hard data and every-loop supervision finally moved the transition cliff.

## 9. Submission Record

Not a submission run.
