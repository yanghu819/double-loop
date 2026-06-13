# h120 +4k Hard-Stage Curve Extension

- Run: `h120long4k-d256-l12-loop6-s9800-20260613T1943Z-f75ca4f`
- Machine: AIStation GPU1 only
- Source SHA: `f75ca4f911f505957709b6572b0ff3fc180edf62`
- Status: completed the +4k h120 hard-stage curve through step `9800`
- Primary recorded score: `0.0723` from `metrics.eval_clean.loop6.label_exact`

## Hypothesis

The D320 capacity probe showed that wider same-budget training loses the h96/h108
foundation and does not open h120. The previous D256 clean h120 curve was
positive but shallow through +3k h120 hard-stage steps. This run asked the next
clean scaling question:

Does pure h120 hard-stage compute keep moving exact solve beyond +3k, or has the
D256 fixed-FutureSeed loop effectively saturated?

This is one curve-extension run, not an ablation table.

## Configuration

- Board: 12x12, random holes, 3x4 boxes
- Model: `D_MODEL=256`, `LAYERS=12`, `HEADS=8`, `HEAD_DIM=32`, `CHANNEL_MULT=4`
- Loop: `MAX_LOOPS=6`, `LOOP_LOSS=all`
- FutureSeed: `FUTURE_SEED_UPDATE=fixed`, `FUTURE_SEED_DECAY=0.0`
- Scratchpad: `SCRATCH_MODE=none`
- Curriculum: `16-36:200,36-60:300,60-72:500,72-84:1200,84-96:1800,96-108:1800,108-120:4000`
- Checkpoints: h120-stage offsets `1000,2000,3000,4000`, holes `96,108,120,132`
- Batch/eval: `FULL_BATCH=72`, `FULL_EVAL_N=512`, `FORWARD_DTYPE=bfloat16`, `RWKV_KERNEL=statepassing`

## Checkpoint Curve

| readout | h96 loop6 exact | h108 loop6 exact | h120 loop6 exact | h120 blank_acc | h132 loop6 exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| +1k, step6800 | 0.9941 | 0.8203 | 0.0137 | 0.5658 | 0.0000 |
| +2k, step7800 | 0.9941 | 0.8926 | 0.0332 | 0.6062 | 0.0000 |
| +3k, step8800 | 0.9922 | 0.9004 | 0.0410 | 0.6382 | 0.0000 |
| +4k, step9800 | 0.9961 | 0.9062 | 0.0586 | 0.6655 | 0.0000 |

The final full eval after about `11824.9s` reached h120 loop6 exact `0.0723`,
blank accuracy `0.6704`, and valid Sudoku `0.0801`; train CE was `0.2030`.

h120 final loop curve:

| loop | exact | blank_acc | valid |
| --- | ---: | ---: | ---: |
| 1 | 0.0000 | 0.3800 | 0.0000 |
| 2 | 0.0000 | 0.5534 | 0.0000 |
| 3 | 0.0391 | 0.6515 | 0.0410 |
| 4 | 0.0664 | 0.6680 | 0.0703 |
| 5 | 0.0723 | 0.6699 | 0.0801 |
| 6 | 0.0723 | 0.6704 | 0.0801 |

## Decision

Pure h120 hard-stage compute is still positive. The curve extends beyond the old
+3k point: checkpoint h120 exact moves from `0.0410` to `0.0586`, and the final
score-bearing eval reaches `0.0723`.

But the slope is still shallow. The model is not close to a solved h120 regime:
blank accuracy is only about `0.67`, h132 stays `0.0`, and loop gain saturates
after loop5. This means D256 long compute has better ROI than same-budget D320,
but extending h120 stage forever is not the main next move.

Next best experiment should combine the useful compute path with one simple
late-loop state revision mechanism. Keep it clean: no selector work, no Sudoku
repair, no scratch noise/Gaussian sweeps, and no same-budget D320 repeats.

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
- `output/checkpoint_eval_step006800.json`
- `output/checkpoint_eval_step007800.json`
- `output/checkpoint_eval_step008800.json`
- `output/checkpoint_eval_step009800.json`
- `output/futureseed_loop_seed52.json`
- `output/futureseed_loop_seed52.md`
- `output/futureseed_loop_case_seed52.html`

The full `source_snapshot.tar.gz` exists on GPU1 under the run directory but was
not committed because it is about 149 MB; the committed SHA, patch, snapshot
hash, and file list identify the source state.
