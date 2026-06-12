# h120 Hard-Stage Curve

- Run: `h120curve-12x12-d256-l12-loop6-s9000-ckpt1k2k3k-20260612T0455Z-2e95bd8`
- Time: 2026-06-12 UTC
- Source SHA: `2e95bd8513402634d973a53e2ae87a625ae69a5c`
- Machine: AIStation GPU1, A800 80GB
- Purpose: answer whether longer `108-120` hard-stage training opens h120, using one run with fixed in-run checkpoint evals at 1k/2k/3k hard-stage steps.

## Config

- Board: 12x12, random holes, 3x4 boxes
- Model: D256, L12, heads8, head_dim32, loop6, `LOOP_LOSS=all`
- Kernel/dtype: RWKV statepassing CUDA, bfloat16
- Batch/eval: train batch72, eval512
- Curriculum: `16-36:200,36-60:300,60-72:500,72-84:1200,84-96:1800,96-108:1800,108-120:3200`
- Checkpoint evals: final-stage offsets `1000,2000,3000`, holes `96,108,120,132`

## Results

| readout | h96 loop6 exact | h108 loop6 exact | h120 loop6 exact | h120 blank_acc | h132 loop6 exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| h120 stage +1k, step6800 | 0.9941 | 0.8203 | 0.0137 | 0.5658 | 0.0000 |
| h120 stage +2k, step7800 | 0.9941 | 0.8926 | 0.0332 | 0.6062 | 0.0000 |
| h120 stage +3k, step8800 | 0.9922 | 0.9004 | 0.0410 | 0.6382 | 0.0000 |
| final, step9000 | 0.9902 | 0.9355 | 0.0449 | 0.6196 | 0.0000 |

h120 loop curve at final eval:

| loop | exact | blank_acc |
| --- | ---: | ---: |
| 1 | 0.0000 | 0.3600 |
| 2 | 0.0000 | 0.5211 |
| 3 | 0.0195 | 0.6045 |
| 4 | 0.0371 | 0.6181 |
| 5 | 0.0410 | 0.6189 |
| 6 | 0.0449 | 0.6196 |

## Insight

Longer h120 hard-stage training is real but sublinear. h120 exact improves from `0.0137` at +1k to `0.0449` final, and h108 strengthens to `0.9355`. This is better than the prior 1600-step h120 run (`0.0273`), but the curve does not look like a near-threshold system that simply needs one more short extension.

The main bottleneck is still global consistency at h120. Blank accuracy rises into the low `0.62-0.64` range, but exact remains below `0.05`; loop depth adds exact through loop5/6 but the gain is small after loop4. This supports the next move being either meaningfully more capacity/compute or a simple FutureSeed state-update mechanism that lets late loops revise state, not selector search or Sudoku repair.

## Artifacts

- Metrics: `output/futureseed_loop_seed52.json`
- Checkpoints: `output/checkpoint_eval_step006800.json`, `output/checkpoint_eval_step007800.json`, `output/checkpoint_eval_step008800.json`
- Report: `output/futureseed_loop_seed52.md`
- Case HTML: `output/futureseed_loop_case_seed52.html`
- Hardest-case visual: `output/h120_hardest_case_visual.html`
- Hardest-case PNG: `output/h120_hardest_case_visual.png`
- Hardest-case parsed summary: `output/h120_hardest_case_summary.json`
- Logs: `logs/run.log`, `logs/driver.log`

## Hardest Case Visual Read

The saved seed52 h120 case exposes the failure mode directly. Loop1 gets only
`22/120` hidden cells right. Loop3 lifts that to `51/120`, but loop4 drops to
`47/120`, loop5 stays at `47/120`, and loop6 reaches only `48/120`. The final
board still has `72` wrong hidden cells and `34` duplicate row/column/box
conflict units.

This supports the current interpretation: loop depth is useful early, but the
fixed FutureSeed state does not keep producing meaningful revisions late in the
trajectory. The next modeling change should be a simple state update or gated
revision mechanism, not Sudoku-specific repair or selector logic.
