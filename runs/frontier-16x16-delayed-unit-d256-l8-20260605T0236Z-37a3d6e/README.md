# 16x16 Delayed Loop + Unit Consistency Probe

Purpose: test whether the negative 16x16 boundary can be rescued by loss shaping alone: delayed intermediate-loop CE plus a differentiable row/column/box digit-mass consistency loss.

Source SHA on GPU1: `37a3d6ebfd64d4ce18d35345a4c0d5458369f6ea`

Run: `frontier-16x16-delayed-unit-d256-l8-20260605T0236Z-37a3d6e`

Recorded UTC: `2026-06-05T03:11:29.595970+00:00`

## Setup

- GPU row: GPU1 only, `CUDA_VISIBLE_DEVICES=0`
- Board: 16x16 Sudoku, 4x4 boxes, random holes
- Model: `D_MODEL=256`, `LAYERS=12`, `HEADS=16`, `HEAD_DIM=16`, `CHANNEL_MULT=4`
- Loop: `MAX_LOOPS=8`, `LOOP_LOSS=delayed`, `LOOP_LOSS_START=5`, `LOOP_LOSS_POWER=2.0`
- Loop weights: `[0, 0, 0, 0, 0.0333, 0.1333, 0.3000, 0.5333]`
- Unit loss: `UNIT_LOSS_WEIGHT=0.25`
- FutureSeed: `FUTURE_SEED_SCALE=1`
- Active rollout noise: none, `NOISE_SCALE=0`, `ROLLOUT_NOISE_SCALE=0`
- Kernel: `RWKV_KERNEL=cuda`
- Curriculum: `8-24:300,24-32:500,32-40:400`
- Eval holes: `24,32,40,48`, `FULL_EVAL_N=256`

## Training Trace

| step | stage | ce | total | loop1 loss | loop last loss | unit loss |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 100 | 8-24 | 1.3620 | 1.3743 | 1.3611 | 1.3620 | 0.0494 |
| 200 | 8-24 | 1.2306 | 1.2427 | 1.2288 | 1.2306 | 0.0484 |
| 300 | 8-24 | 1.1119 | 1.1228 | 1.1108 | 1.1119 | 0.0439 |
| 400 | 24-32 | 1.2642 | 1.2811 | 1.2629 | 1.2642 | 0.0677 |
| 500 | 24-32 | 1.1679 | 1.1846 | 1.1850 | 1.1679 | 0.0669 |
| 600 | 24-32 | 1.1365 | 1.1527 | 1.1420 | 1.1365 | 0.0646 |
| 700 | 24-32 | 1.1106 | 1.1269 | 1.1229 | 1.1106 | 0.0652 |
| 800 | 24-32 | 1.0258 | 1.0409 | 1.0387 | 1.0258 | 0.0604 |
| 900 | 32-40 | 1.0946 | 1.1128 | 1.1091 | 1.0946 | 0.0726 |
| 1000 | 32-40 | 1.0991 | 1.1180 | 1.1152 | 1.0991 | 0.0759 |
| 1100 | 32-40 | 1.0155 | 1.0330 | 1.0345 | 1.0155 | 0.0697 |
| 1200 | 32-40 | 1.0642 | 1.0823 | 1.0826 | 1.0642 | 0.0723 |

Training took `2305.5s`.

## Hole Transfer

Exact solve rate by loop8:

| holes | exact | valid | solved | blank acc |
| ---: | ---: | ---: | ---: | ---: |
| 24 | 0.0000 | 0.0000 | 0.0000 | 0.4106 |
| 32 | 0.0000 | 0.0000 | 0.0000 | 0.3831 |
| 40 | 0.0000 | 0.0000 | 0.0000 | 0.3903 |
| 48 | 0.0000 | 0.0000 | 0.0000 | 0.3559 |

K8 oracle exact was also `0.0000`, with zero trajectory disagreement. This is not a selector problem.

## Insight

Loss-level shaping did not rescue 16x16. Delayed loop CE avoided the worst form of naive equal all-loop supervision, but the added unit digit-mass loss did not restore local accuracy or create full-board validity.

Compared with prior 16x16 runs:

- final-loop foothold: final CE `0.5792`, h32 blank accuracy `0.8287`, h32 exact `0.0039`;
- naive all-loop: final CE `1.0498`, h32 blank accuracy `0.3872`, h32 exact `0.0`;
- delayed + unit loss: final CE `1.0642`, h32 blank accuracy `0.3831`, h32 exact `0.0`.

The unit loss is too weak as a structural mechanism because nearly uniform per-cell predictions already satisfy row/column/box digit mass. It regularizes sharpened predictions but does not create a communication path that assembles a Sudoku-valid board.

## Decision

Stop loss-only rescue attempts for 16x16. The next high-ROI experiment should add explicit row/column/box unit-state passing or unit tokens, then compare against the old final-loop foothold under the same 16x16 low-hole curriculum. Do not run more `UNIT_LOSS_WEIGHT` grids.

## Provenance Note

The run was clean at launch: `git_dirty=false`. The source snapshot tarball is retained on GPU1/local disk and represented in GitHub by `source_snapshot.tar.gz.sha256` and `source_snapshot.ls.txt` because it exceeds GitHub's `100MB` file limit.

## Artifacts

- `config.json`
- `score.json`
- `metadata.json`
- `logs/run.log`
- `logs/launcher.outer.log`
- `run.pid`
- `output/futureseed_loop_seed52.json`
- `output/futureseed_loop_seed52.md`
- `output/futureseed_loop_case_seed52.html`
- `source_HEAD.txt`
- `source.patch`
- `source_snapshot.tar.gz` on GPU1/local disk
- `source_snapshot.tar.gz.sha256`
- `source_snapshot.ls.txt`
