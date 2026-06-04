# 9x9 FutureSeed RWKV Scale-Up: D192 L10 H40

Purpose: test whether a larger FutureSeed RWKV recurrent backbone moves the 9x9 random-hole viability cliff outward, rather than only saturating the easier h24-h32 distribution.

Source SHA on GPU1: `a5b3b4ae7cd6bbf98f1b065887cac04697373529`

Run: `scaleup-d192l10-h40-20260604T0844Z-a5b3b4a`

Recorded UTC: `2026-06-04T08:57:25.192348+00:00`

## Setup

- GPU row: GPU1 only, `CUDA_VISIBLE_DEVICES=0`
- Kernel path: `RWKV_KERNEL=cuda`
- Board: 9x9 Sudoku, 3x3 boxes, random holes
- Model: `D_MODEL=192`, `LAYERS=10`, `HEADS=12`, `HEAD_DIM=16`, `CHANNEL_MULT=4`
- Loop: `L_CYCLES=2`, `MAX_LOOPS=5`, final-loop loss
- FutureSeed: `FUTURE_SEED_SCALE=1`
- Noise: `NOISE_SCALE=0`, `ROLLOUT_NOISE_SCALE=0`; no active feature-noise injection in this run
- Curriculum: `8-20:300,20-32:500,32-40:300`
- Eval: `FULL_EVAL_N=512`, holes `24,32,36,40,44`

## Training Trace

| step | stage | ce | loop1 loss | loop last loss |
| ---: | --- | ---: | ---: | ---: |
| 100 | 8-20 | 1.1645 | 1.1670 | 1.1645 |
| 200 | 8-20 | 0.4785 | 0.5112 | 0.4785 |
| 300 | 8-20 | 0.0683 | 0.1078 | 0.0683 |
| 400 | 20-32 | 0.0629 | 0.1858 | 0.0629 |
| 500 | 20-32 | 0.0311 | 0.1205 | 0.0311 |
| 600 | 20-32 | 0.0206 | 0.1099 | 0.0206 |
| 700 | 20-32 | 0.0209 | 0.1302 | 0.0209 |
| 800 | 20-32 | 0.0201 | 0.1276 | 0.0201 |
| 900 | 32-40 | 0.1074 | 0.3919 | 0.1074 |
| 1000 | 32-40 | 0.0932 | 0.3253 | 0.0932 |
| 1100 | 32-40 | 0.0585 | 0.3665 | 0.0585 |

Training took `1255.2s`. Step 100 completed well under the 15-minute kill threshold, and GPU utilization stayed high during the run.

## Hole Transfer

Exact solve rate by loop:

| holes | loop1 | loop2 | loop3 | loop4 | loop5 | loop5-loop1 | loop5 blank acc |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 24 | 0.8145 | 0.9863 | 0.9883 | 0.9883 | 0.9883 | +0.1738 | 0.9980 |
| 32 | 0.3320 | 0.9043 | 0.9199 | 0.9199 | 0.9199 | +0.5879 | 0.9930 |
| 36 | 0.0645 | 0.7285 | 0.7910 | 0.8008 | 0.8086 | +0.7441 | 0.9796 |
| 40 | 0.0020 | 0.4258 | 0.5156 | 0.5273 | 0.5332 | +0.5312 | 0.9531 |
| 44 | 0.0000 | 0.1035 | 0.1680 | 0.1875 | 0.1914 | +0.1914 | 0.8893 |

Primary leaderboard score is h32 loop5 exact: `0.919922`.

The harder-transfer readout is h40 loop5 exact: `0.533203`. h44 remains below the viable regime at `0.191406`, but it is not random failure: loop depth still recovers nontrivial exact solves from zero loop1 exact.

## Insight

Scale-up worked in the sense that h40 crossed a meaningful exact-solve threshold. The cliff moved outward from h32 toward h40.

The important mechanism readout is not only loop1 to loop2. On h40, loop2 reaches `0.4258`, while loop5 reaches `0.5332`; on h44, loop2 reaches `0.1035`, while loop5 reaches `0.1914`. That means later recurrent refinement is still buying global consistency on the hardest transfer cases. FutureSeed gives a strong global initialization, but it does not replace loop depth.

The failure mode at h44 is also clear: blank accuracy is still `0.8893`, while exact is only `0.1914`. This is a board-level consistency cliff, not just local digit classification. The next high-ROI direction is not another width-only repeat; it should test whether longer loop horizon, loop-supervision schedule, or harder curriculum changes h44 consistency.

## Decision

Continue the FutureSeed RWKV backbone direction. This run is strong enough to tag, but it does not justify a blind size sweep.

Next experiment should be one of:

1. Longer-loop hard transfer: keep D192/L10, train/eval loops beyond 5 on h40-h44, to test whether h44 is loop-horizon-limited.
2. Curriculum pressure: add a short h40-h44 final stage, to test whether the current cliff is caused by train distribution ending at h40.

Avoid:

- repeating D192/L10 with only small seed changes;
- reviving feature noise before we have a clear h44 consistency hypothesis;
- selector work, because this run used K1 only and no oracle-selector gap was tested.

## Provenance Note

`config.json` and `metadata.json` report `git_dirty=true`. The dirty bit was caused by a pre-launch `core.2644` temporary file from earlier diagnostics, not by source changes. That file was removed after launch; `source_HEAD.txt`, `source.patch`, and the source snapshot are archived in this run directory.

## Artifacts

- `config.json`
- `score.json`
- `metadata.json`
- `logs/run.log`
- `logs/launcher.log`
- `launch.env`
- `output/futureseed_loop_seed52.json`
- `output/futureseed_loop_seed52.md`
- `output/futureseed_loop_case_seed52.html`
- `source_HEAD.txt`
- `source.patch`
- `source_snapshot.tar.gz`
