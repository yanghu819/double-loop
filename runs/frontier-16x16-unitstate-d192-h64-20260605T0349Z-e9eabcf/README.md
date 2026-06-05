# 16x16 Unit-State Passing Probe

Purpose: test whether the 16x16 failure is a missing structure-communication problem rather than a hidden-size, loop-count, or loss-weight problem.

Source SHA on GPU1: `e9eabcff07c207335ec8930c479ee841996e3b90`

Run: `frontier-16x16-unitstate-d192-h64-20260605T0349Z-e9eabcf`

Recorded UTC: `2026-06-05T04:01:47.735307+00:00`

## Setup

- GPU row: GPU1 only, `CUDA_VISIBLE_DEVICES=0`
- Board: 16x16 Sudoku, 4x4 boxes, random holes
- Model: `D_MODEL=192`, `LAYERS=10`, `HEADS=12`, `HEAD_DIM=16`, `CHANNEL_MULT=4`
- Loop: `MAX_LOOPS=5`, final-loop CE
- Unit state: `UNIT_STATE_SCALE=1.0`, `UNIT_STATE_GATE_BIAS=-1.0`
- Unit loss: disabled, `UNIT_LOSS_WEIGHT=0`
- FutureSeed: `FUTURE_SEED_SCALE=1`
- Active rollout noise: none, `NOISE_SCALE=0`, `ROLLOUT_NOISE_SCALE=0`
- Kernel: `RWKV_KERNEL=cuda`
- Curriculum: `16-32:300,32-48:400,48-64:300`
- Eval holes: `32,48,64,80`, `FULL_EVAL_N=256`

## Training Trace

| step | stage | ce | loop1 loss | loop last loss | unit loss |
| ---: | --- | ---: | ---: | ---: | ---: |
| 100 | 16-32 | 0.2129 | 0.2063 | 0.2129 | 0.0105 |
| 200 | 16-32 | 0.0467 | 0.0726 | 0.0467 | 0.0026 |
| 300 | 16-32 | 0.0273 | 0.0481 | 0.0273 | 0.0017 |
| 400 | 32-48 | 0.0956 | 0.1383 | 0.0956 | 0.0087 |
| 500 | 32-48 | 0.0844 | 0.1322 | 0.0844 | 0.0082 |
| 600 | 32-48 | 0.0736 | 0.1314 | 0.0736 | 0.0068 |
| 700 | 32-48 | 0.0764 | 0.1281 | 0.0764 | 0.0076 |
| 800 | 48-64 | 0.1497 | 0.2687 | 0.1497 | 0.0157 |
| 900 | 48-64 | 0.1344 | 0.2667 | 0.1344 | 0.0133 |
| 1000 | 48-64 | 0.1102 | 0.2491 | 0.1102 | 0.0110 |

Training took `1407.9s`.

## Loop Depth

| loop | exact | valid | solved | blank acc |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.2031 | 0.2031 | 0.2031 | 0.9520 |
| 2 | 0.6836 | 0.6836 | 0.6836 | 0.9866 |
| 3 | 0.6758 | 0.6758 | 0.6758 | 0.9865 |
| 4 | 0.6875 | 0.6875 | 0.6875 | 0.9871 |
| 5 | 0.6914 | 0.6914 | 0.6914 | 0.9872 |

## Hole Transfer

Exact solve rate by loop5:

| holes | exact | valid | solved | blank acc |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 0.6914 | 0.6914 | 0.6914 | 0.9872 |
| 48 | 0.2148 | 0.2148 | 0.2148 | 0.9631 |
| 64 | 0.0117 | 0.0117 | 0.0117 | 0.9156 |
| 80 | 0.0000 | 0.0000 | 0.0000 | 0.8386 |

## Insight

This is the first strong 16x16 result. Under the same D192/L10/loop5 low-hole curriculum as the old foothold run, explicit row/column/box unit-state passing changes the regime:

- old final-loop foothold: final CE `0.5792`, h32 exact `0.0039`, h48+ exact `0.0`;
- delayed unit-loss rescue: final CE `1.0642`, h32 exact `0.0`;
- unit-state passing: final CE `0.1102`, h32 exact `0.6914`, h48 exact `0.2148`, h64 exact `0.0117`.

The mechanism is structural, not a selector trick. K1 already reaches h32 exact `0.6914`, and loop depth still matters: loop1 exact is `0.2031`, while loop5 reaches `0.6914`.

## Decision

Promote explicit unit-state passing to the mainline for larger Sudoku. The next high-ROI scaling experiment is not another loss ablation; it is a harder frontier probe:

1. 16x16 higher-hole curriculum, targeting h64/h80.
2. 25x25 feasibility with unit-state enabled and conservative low-hole curriculum.
3. Cleaner architecture pass: make unit-state a named module in docs and evaluate whether true unit tokens beat pooled unit-state.

## Launch Notes

Two earlier worktree launches failed before training because the worktree PATH picked a broken `/opt/conda/bin/ninja`. They were aborted and relaunched with `/huyang2/double-loop/.cache/bin` first in PATH. No CPU fallback was used.

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
