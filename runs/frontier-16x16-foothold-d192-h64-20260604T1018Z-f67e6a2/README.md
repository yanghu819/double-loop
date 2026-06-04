# 16x16 FutureSeed RWKV Foothold Probe

Purpose: after 12x12 succeeded, test whether 16x16 has any low-hole foothold under the same FutureSeed RWKV recurrent-loop paradigm.

Source SHA on GPU1: `f67e6a2e909dc39ce5ba48f622d93b1ec43f5ab3`

Run: `frontier-16x16-foothold-d192-h64-20260604T1018Z-f67e6a2`

Recorded UTC: `2026-06-04T10:41:38.512425+00:00`

## Setup

- GPU row: GPU1 only, `CUDA_VISIBLE_DEVICES=0`
- Board: 16x16 Sudoku, 4x4 boxes, random holes
- Model: `D_MODEL=192`, `LAYERS=10`, `HEADS=12`, `HEAD_DIM=16`, `CHANNEL_MULT=4`
- Loop: `L_CYCLES=2`, `MAX_LOOPS=5`, final-loop loss
- FutureSeed: `FUTURE_SEED_SCALE=1`
- Active noise injection: none, `NOISE_SCALE=0`, `ROLLOUT_NOISE_SCALE=0`
- Kernel: `RWKV_KERNEL=cuda`
- Curriculum: `16-32:300,32-48:400,48-64:300`
- Eval holes: `32,48,64,80`, `FULL_EVAL_N=256`

## Training Trace

| step | stage | ce | loop1 loss | loop last loss |
| ---: | --- | ---: | ---: | ---: |
| 100 | 16-32 | 1.4472 | 1.4453 | 1.4472 |
| 200 | 16-32 | 1.1872 | 1.2001 | 1.1872 |
| 300 | 16-32 | 0.9933 | 1.0013 | 0.9933 |
| 400 | 32-48 | 1.1440 | 1.1541 | 1.1440 |
| 500 | 32-48 | 1.1415 | 1.1574 | 1.1415 |
| 600 | 32-48 | 1.0380 | 1.0596 | 1.0380 |
| 700 | 32-48 | 0.9125 | 0.9613 | 0.9125 |
| 800 | 48-64 | 0.9115 | 1.0472 | 0.9115 |
| 900 | 48-64 | 0.7611 | 0.9899 | 0.7611 |
| 1000 | 48-64 | 0.5792 | 0.9110 | 0.5792 |

Training took `1353.9s`.

## Hole Transfer

Exact solve rate by loop:

| holes | loop1 | loop3 | loop5 | loop5 blank acc |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 0.0000 | 0.0039 | 0.0039 | 0.8287 |
| 48 | 0.0000 | 0.0000 | 0.0000 | 0.7393 |
| 64 | 0.0000 | 0.0000 | 0.0000 | 0.6552 |
| 80 | 0.0000 | 0.0000 | 0.0000 | 0.5704 |

## Insight

This is the negative boundary. The run does optimize locally: CE drops to `0.5792`, and loop5 blank accuracy reaches `0.8287` at h32. But full-board exact does not emerge: h32 exact is only `0.0039`, and h48/h64/h80 are exactly zero.

The failure is not speed, CUDA, or OOM. GPU1 ran at high utilization with `66.9GB` used. The failure is global consistency on a 16x16 constraint graph. Loop depth slightly improves local blank accuracy but is neutral on full-board exact.

`git_dirty=true` in metadata came from prior remote `leaderboard.csv` updates and aborted-run metadata, not source-code edits. The source SHA is archived as `f67e6a2`.

## Decision

Current FutureSeed RWKV loop paradigm supports 12x12 but not 16x16 full-board solving under this budget and architecture. Pushing 16x16 next needs a new mechanism or training strategy, not a blind repeat.

High-ROI next directions:

1. Add a consistency objective or differentiable unit-conflict penalty for larger boards.
2. Train with stronger intermediate loop supervision rather than final-loop-only loss.
3. Use hierarchical/unit-token structure so 16x16 does not require pure sequence-level propagation across 256 cells.

Avoid:

- more 16x16 D192 seed repeats;
- high-hole 16x16 schedules before a h32/h48 foothold exists;
- larger snapshots inside Git.

## Provenance Note

The source snapshot tarball is `139.91MB`, above GitHub's hard `100MB` file limit. It is retained on GPU1/local disk, while GitHub tracks `source_snapshot.tar.gz.sha256` and `source_snapshot.ls.txt`.

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
- `source_snapshot.tar.gz` on GPU1/local disk
- `source_snapshot.tar.gz.sha256`
- `source_snapshot.ls.txt`
