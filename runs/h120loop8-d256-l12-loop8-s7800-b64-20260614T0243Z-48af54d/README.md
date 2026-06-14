# h120 Loop8 Scale Test

- Run: `h120loop8-d256-l12-loop8-s7800-b64-20260614T0243Z-48af54d`
- Machine: AIStation GPU1 only
- Source SHA: `48af54d007786b85fcb93adabb3244123714991c`
- Status: completed through step `7800`
- Primary recorded score: `0.0156` from `metrics.eval_clean.loop8.label_exact`

## Hypothesis

The clean D256 h120 +4k run showed that hard-stage compute still helps, but the
loop curve saturated after loop5. This run asked a narrow scaling question:

If h120 is limited by recurrent depth, then increasing `MAX_LOOPS` from 6 to 8
should create extra late-loop correction at h120, even before another long
hard-stage extension.

This is one scale-frontier test, not a table-filling loop-count sweep.

## Configuration

- Board: 12x12, random holes, 3x4 boxes
- Model: `D_MODEL=256`, `LAYERS=12`, `HEADS=8`, `HEAD_DIM=32`, `CHANNEL_MULT=4`
- Loop: `MAX_LOOPS=8`, `LOOP_LOSS=all`
- FutureSeed: `FUTURE_SEED_UPDATE=fixed`, `FUTURE_SEED_DECAY=0.0`
- Scratchpad: `SCRATCH_MODE=none`
- Curriculum: `16-36:200,36-60:300,60-72:500,72-84:1200,84-96:1800,96-108:1800,108-120:2000`
- Checkpoints: h120-stage offsets `1000,2000`, holes `96,108,120,132`
- Batch/eval: `FULL_BATCH=64`, `FULL_EVAL_N=512`, `FORWARD_DTYPE=bfloat16`, `RWKV_KERNEL=statepassing`

## Results

Checkpoint curve:

| readout | h96 loop8 exact | h108 loop8 exact | h120 loop8 exact | h120 blank_acc | h132 loop8 exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| +1k, step6800 | 0.9961 | 0.8320 | 0.0059 | 0.5462 | 0.0000 |
| +2k, step7800 | 0.9961 | 0.8926 | 0.0234 | 0.5986 | 0.0000 |

Final h120 loop curve:

| loop | exact | blank_acc | valid |
| --- | ---: | ---: | ---: |
| 1 | 0.0000 | 0.3298 | 0.0000 |
| 2 | 0.0000 | 0.4907 | 0.0000 |
| 3 | 0.0078 | 0.5810 | 0.0078 |
| 4 | 0.0137 | 0.5974 | 0.0137 |
| 5 | 0.0156 | 0.6012 | 0.0156 |
| 6 | 0.0156 | 0.6018 | 0.0156 |
| 7 | 0.0156 | 0.6021 | 0.0156 |
| 8 | 0.0156 | 0.6021 | 0.0156 |

At step7800, h120 loop6 and loop8 exact are both `0.0234`. In the final eval,
h120 loop6 and loop8 exact are both `0.0156`. Extra loops add almost no blank
accuracy and no extra exact solves after loop6.

## Decision

This is a negative scale result for deeper loop count. The run keeps the h96 and
h108 foundation alive, so the failure is not that the model never learned the
easier curriculum. The failure is that loop7 and loop8 do not keep correcting
h120 boards once the fixed FutureSeed loop has reached its local trajectory.

Compared with the clean h120 +4k run, this shorter loop8 run is also weaker at
similar hard-stage exposure: step7800 h120 exact is `0.0234`, while the loop6
long curve reached `0.0332` at the comparable +2k checkpoint and `0.0723` final.
That makes more h120 hard-stage compute a better clean scaling axis than simply
adding loop count.

Next work should not be a loop-count sweep. Use either genuinely larger
effective compute, or a simple FutureSeed state update that lets later loops
revise state instead of rereading a fixed state. Keep selector work, Sudoku
repair, scratch noise/Gaussian sweeps, and same-budget width repeats paused.

No tag was created because the primary score is below `0.50`.

## Artifacts

- `config.json`
- `metadata.json`
- `score.json`
- `source_HEAD.txt`
- `source.patch`
- `source_snapshot.tar.gz.sha256`
- `source_snapshot.ls.txt`
- `logs/run.log`
- `logs/driver.log`
- `output/checkpoint_eval_step006800.json`
- `output/checkpoint_eval_step007800.json`
- `output/futureseed_loop_seed52.json`
- `output/futureseed_loop_seed52.md`
- `output/futureseed_loop_case_seed52.html`

The full `source_snapshot.tar.gz` exists on GPU1 under the run directory but was
not committed because it is about 149 MB; the committed SHA, patch, snapshot
hash, and file list identify the source state.
