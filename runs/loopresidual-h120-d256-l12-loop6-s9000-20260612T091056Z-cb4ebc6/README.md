# Loop-Residual h120 Checkpoint Abort

- Run: `loopresidual-h120-d256-l12-loop6-s9000-20260612T091056Z-cb4ebc6`
- Machine: AIStation GPU1 only
- Source SHA: `cb4ebc6ca768987931a2fec4295ff8164bc24902`
- Status: stopped after the first hard-stage checkpoint at step `6800`

## Hypothesis

The run tested whether writing FutureSeed memory back through the RWKV loop state is enough to keep late loops revising hard h120 puzzles. This is a mechanism test, not a table-filling ablation: if loop-residual state update opens h120, continue; if h120 stays closed and late loops stop adding exact solves, switch to a cleaner mutable scratch state.

## Configuration

- Board: 12x12, 3x4 boxes
- Model: `D_MODEL=256`, `LAYERS=12`, `HEADS=8`, `HEAD_DIM=32`, `CHANNEL_MULT=4`
- Loop: `MAX_LOOPS=6`, `LOOP_LOSS=all`
- FutureSeed: `FUTURE_SEED_UPDATE=loop_residual`, `FUTURE_SEED_DECAY=0.75`
- Curriculum: `16-36:200,36-60:300,60-72:500,72-84:1200,84-96:1800,96-108:1800,108-120:3200`
- Batch/eval: `FULL_BATCH=72`, `FULL_EVAL_N=512`, `FORWARD_DTYPE=bfloat16`, `RWKV_KERNEL=statepassing`

## Checkpoint Result

Step `6800`, after about `8619.4s`; train CE was `0.4452`.

| holes | loop1 | loop2 | loop3 | loop4 | loop5 | loop6 | loop6-loop3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| h96 exact | 0.0000 | 0.4844 | 0.6465 | 0.6445 | 0.6445 | 0.6484 | +0.0020 |
| h108 exact | 0.0000 | 0.0371 | 0.2031 | 0.2188 | 0.2285 | 0.2266 | +0.0234 |
| h120 exact | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| h132 exact | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |

Blank accuracy at loop6 was h96 `0.9819`, h108 `0.9309`, h120 `0.6178`, h132 `0.1700`.

## Decision

Stop. The run made h96 strong and h108 partially useful, but it failed the actual h120 question: h120 exact remained zero and late-loop exact gain was zero. Continuing this exact route to later checkpoints has lower expected information gain than testing a separated mutable scratch state.

The next experiment is `scratchpad-gated-h120-d256-l12-loop6-s9000-20260612T1048Z-b03829b`: fixed FutureSeed context plus gated mutable scratch, no noise and no Gaussian regularizer.

## Artifacts

- `config.json`
- `source_HEAD.txt`
- `source.patch`
- `logs/run.log`
- `output/checkpoint_eval_step006800.json`
- `output/abort.json`
