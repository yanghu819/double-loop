# Gated Scratchpad h120 Interrupted Run

- Run: `scratchpad-gated-h120-d256-l12-loop6-s9000-20260612T1048Z-b03829b`
- Machine: AIStation GPU1 only
- Source SHA: `b03829be2b4ec387f4bf772e5144b6c5dab7af74`
- Status: interrupted by AIStation environment halt before checkpoint

## Hypothesis

This was the first clean test of a separated mutable scratch state: keep FutureSeed context fixed, add a gated writable scratchpad across loops, and do not add feature noise or Gaussian regularization. The question was whether late loop state has a writable workspace instead of trying to revise the backbone/FutureSeed context directly.

## Configuration

- Board: 12x12, 3x4 boxes
- Model: `D_MODEL=256`, `LAYERS=12`, `HEADS=8`, `HEAD_DIM=32`, `CHANNEL_MULT=4`
- Loop: `MAX_LOOPS=6`, `LOOP_LOSS=all`
- FutureSeed: `FUTURE_SEED_UPDATE=fixed`, `FUTURE_SEED_DECAY=0.0`
- Scratch: `SCRATCH_MODE=gated`, no scratch noise, no Gaussian regularizer
- Curriculum: `16-36:200,36-60:300,60-72:500,72-84:1200,84-96:1800,96-108:1800,108-120:3200`
- Batch/eval: `FULL_BATCH=72`, `FULL_EVAL_N=512`, `FORWARD_DTYPE=bfloat16`, `RWKV_KERNEL=statepassing`

## Partial Signal

The run reached step `3600` before the AIStation environment halted. No eval checkpoint was produced, so this is not a score-bearing result.

Training did show the scratch state was being used rather than ignored. `scratch_delta` rose from `0.374` at step100 to `3.289` at step3600. CE also followed the expected curriculum shape: step100 `1.5808`, step1000 `0.0863`, then the h84/h96 stage jump and recovery with step3600 CE `0.1353`.

## Decision

Do not treat this as evidence for or against h120 exact. Relaunch a shortened first-checkpoint run with the same prefix curriculum and `108-120:1000`, so the experiment directly answers the step6800 h120 question within the available lease.

Follow-up run: `scratchpad-gated-h120-step6800-d256-l12-loop6-20260613T0238Z-b03829b`.

## Artifacts

- `config.json`
- `source_HEAD.txt`
- `source.patch`
- `logs/run.log`
