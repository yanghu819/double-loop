# D320 h120 Capacity First Checkpoint

- Run: `capprobe-d320-h120-step6800-d320-l12-loop6-b48-20260613T1522Z-b03829b`
- Machine: AIStation GPU1 only
- Source SHA: `b03829be2b4ec387f4bf772e5144b6c5dab7af74`
- Status: completed the first h120 hard-stage checkpoint at step `6800`
- Primary recorded score: `0.0000` from `metrics.eval_clean.loop6.label_exact`

## Hypothesis

The scratchpad branch showed that adding a writable latent state, noise, and
Gaussian residual pressure did not turn local fill accuracy into board-level
consistency. This run returned to a cleaner bitter-lesson question:

Can a wider clean FutureSeed+loop backbone make h120 viable without task repair,
selector search, or Sudoku-specific structure?

The test intentionally changed only meaningful scale: `D_MODEL=320` with the
same loop6/all-loop h120 first-checkpoint curriculum. Batch had to drop to `48`
to fit the 80GB GPU.

## Configuration

- Board: 12x12, 3x4 boxes
- Model: `D_MODEL=320`, `LAYERS=12`, `HEADS=8`, `HEAD_DIM=40`, `CHANNEL_MULT=4`
- Loop: `MAX_LOOPS=6`, `LOOP_LOSS=all`
- FutureSeed: `FUTURE_SEED_UPDATE=fixed`, `FUTURE_SEED_DECAY=0.0`
- Scratchpad: `SCRATCH_MODE=none`
- Curriculum: `16-36:200,36-60:300,60-72:500,72-84:1200,84-96:1800,96-108:1800,108-120:1000`
- Batch/eval: `FULL_BATCH=48`, `FULL_EVAL_N=512`, `FORWARD_DTYPE=bfloat16`, `RWKV_KERNEL=statepassing`

## Checkpoint Result

Step `6800`, after about `8345.7s`; train CE was `0.7799`.

| holes | loop1 | loop2 | loop3 | loop4 | loop5 | loop6 | loop6-loop3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| h96 exact | 0.0000 | 0.2246 | 0.4141 | 0.4141 | 0.4316 | 0.4336 | +0.0195 |
| h108 exact | 0.0000 | 0.0078 | 0.0371 | 0.0508 | 0.0488 | 0.0508 | +0.0137 |
| h120 exact | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| h132 exact | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |

Blank accuracy at loop6 was h96 `0.9643`, h108 `0.8773`, h120 `0.5000`,
and h132 `0.1673`.

The h120 loop curve is the important failure signature:

| loop | exact | blank_acc | valid |
| --- | ---: | ---: | ---: |
| 1 | 0.0000 | 0.3826 | 0.0000 |
| 2 | 0.0000 | 0.4725 | 0.0000 |
| 3 | 0.0000 | 0.4957 | 0.0000 |
| 4 | 0.0000 | 0.4993 | 0.0000 |
| 5 | 0.0000 | 0.5004 | 0.0000 |
| 6 | 0.0000 | 0.5000 | 0.0000 |

Loops still help locally, but only until about loop3. After that, exact, valid,
and blank accuracy all saturate.

## Decision

Do not repeat D320 at the same effective budget. The wider model fit and trained,
but it was worse than the D256 clean h120 curve at the comparable first h120
checkpoint:

- D256 clean curve at step6800: h96/h108/h120 loop6 exact about `0.9941 / 0.8203 / 0.0137`
- D320 capacity probe at step6800: h96/h108/h120 loop6 exact `0.4336 / 0.0508 / 0.0000`

The result is not "scale never helps." It says this specific scaling move is
inefficient: increasing width while cutting batch and keeping the same step
budget does not preserve the h96/h108 foundation, so it cannot test a clean h120
upper bound.

The next high-ROI scaling move should be one of:

- return to D256 and spend compute where the curve was actually positive: longer
  h120 hard-stage or better data/optimization at h120;
- add activation checkpointing only if it enables a real effective-budget
  increase, such as wider model without batch collapse or much longer training;
- test one very simple FutureSeed state-update change only if it directly targets
  late-loop correction, not latent diversity or Sudoku repair.

Do not run more same-budget D320 repeats, scratch-noise sweeps, selector work, or
Sudoku repair rules.

## Artifacts

- `config.json`
- `metadata.json`
- `score.json`
- `source_HEAD.txt`
- `source.patch`
- `source_snapshot.tar.gz.sha256`
- `source_snapshot.ls.txt`
- `logs/run.log`
- `output/checkpoint_eval_step006800.json`
- `output/futureseed_loop_seed52.json`
- `output/futureseed_loop_seed52.md`
- `output/futureseed_loop_case_seed52.html`

The full `source_snapshot.tar.gz` exists on GPU1 under the run directory but was
not committed because it is about 149 MB; the committed SHA, patch, snapshot
hash, and file list identify the source state.
