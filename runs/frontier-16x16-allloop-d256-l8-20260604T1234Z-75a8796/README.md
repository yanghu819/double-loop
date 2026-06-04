# 16x16 All-Loop Supervision Rescue Probe

Purpose: test the user's hypothesis that supervising every recurrent loop, combined with larger hidden size and deeper loop budget, can rescue 16x16 Sudoku.

Source SHA on GPU1: `75a8796b822a80f66ebe9e3d1d5dc341d6505891`

Run: `frontier-16x16-allloop-d256-l8-20260604T1234Z-75a8796`

Recorded UTC: `2026-06-04T13:13:37.037272+00:00`

## Setup

- GPU row: GPU1 only, `CUDA_VISIBLE_DEVICES=0`
- Board: 16x16 Sudoku, 4x4 boxes, random holes
- Model: `D_MODEL=256`, `LAYERS=12`, `HEADS=16`, `HEAD_DIM=16`, `CHANNEL_MULT=4`
- Loop: `MAX_LOOPS=8`, `LOOP_LOSS=all`
- FutureSeed: `FUTURE_SEED_SCALE=1`
- Active noise injection: none, `NOISE_SCALE=0`, `ROLLOUT_NOISE_SCALE=0`
- Kernel: `RWKV_KERNEL=cuda`
- Curriculum: `8-24:300,24-32:500,32-40:400`
- Eval holes: `24,32,40,48`, `FULL_EVAL_N=256`

## Training Trace

| step | stage | ce | total | loop1 loss | loop last loss |
| ---: | --- | ---: | ---: | ---: | ---: |
| 100 | 8-24 | 1.3641 | 1.3643 | 1.3651 | 1.3641 |
| 200 | 8-24 | 1.2472 | 1.2467 | 1.2453 | 1.2472 |
| 300 | 8-24 | 1.0901 | 1.0910 | 1.0963 | 1.0901 |
| 400 | 24-32 | 1.2407 | 1.2407 | 1.2430 | 1.2407 |
| 500 | 24-32 | 1.1559 | 1.1564 | 1.1616 | 1.1559 |
| 600 | 24-32 | 1.1198 | 1.1185 | 1.1144 | 1.1198 |
| 700 | 24-32 | 1.1087 | 1.1094 | 1.1146 | 1.1087 |
| 800 | 24-32 | 1.0134 | 1.0152 | 1.0255 | 1.0134 |
| 900 | 32-40 | 1.0743 | 1.0747 | 1.0797 | 1.0743 |
| 1000 | 32-40 | 1.0823 | 1.0833 | 1.0900 | 1.0823 |
| 1100 | 32-40 | 1.0064 | 1.0079 | 1.0176 | 1.0064 |
| 1200 | 32-40 | 1.0498 | 1.0510 | 1.0595 | 1.0498 |

Training took `2287.7s`.

## Hole Transfer

Exact solve rate by loop:

| holes | loop1 | loop5 | loop8 | loop8 blank acc |
| ---: | ---: | ---: | ---: | ---: |
| 24 | 0.0000 | 0.0000 | 0.0000 | 0.4102 |
| 32 | 0.0000 | 0.0000 | 0.0000 | 0.3872 |
| 40 | 0.0000 | 0.0000 | 0.0000 | 0.3955 |
| 48 | 0.0000 | 0.0000 | 0.0000 | 0.3560 |

## Insight

Naive equal supervision on every loop did not rescue 16x16. It made optimization worse.

Compared with the previous 16x16 final-loop foothold run:

- final-loop D192/L10 loop5 reached final CE `0.5792` and h32 blank accuracy `0.8287`;
- all-loop D256/L12 loop8 reached final CE only `1.0498` and h32 blank accuracy `0.3872`;
- both had exact solve near zero, but all-loop lost most local accuracy too.

The loop losses also did not separate. At step1200, loop1 loss is `1.0595` and loop8 loss is `1.0498`, so deeper recurrence is not acting like a learned attractor. Equal per-loop CE appears to force early loops to solve before a useful latent state exists, making the optimization problem harder instead of stabilizing it.

## Decision

Do not continue naive `LOOP_LOSS=all` for 16x16. The right version of "supervise every loop" probably needs a shaped schedule, not equal CE everywhere:

1. discounted loop loss with low early-loop weight;
2. consistency/distillation loss between loop states instead of hard labels at every loop;
3. conflict/unit-level losses that reward global constraint repair;
4. curriculum where intermediate-loop supervision turns on only after the final loop has a foothold.

## Provenance Note

The run was clean at launch: `git_dirty=false`. The source snapshot tarball is retained on GPU1/local disk and represented in GitHub by `source_snapshot.tar.gz.sha256` and `source_snapshot.ls.txt` because it exceeds GitHub's `100MB` file limit.

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
