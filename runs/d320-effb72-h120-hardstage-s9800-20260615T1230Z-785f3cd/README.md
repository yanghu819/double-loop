# D320 Effective-Batch H120 Hard-Stage Segment

Status: interrupted by GPU1 halt before the first planned checkpoint eval.

- Run: `d320-effb72-h120-hardstage-s9800-20260615T1230Z-785f3cd`
- Source SHA: `785f3cdca74aab1f6e9d6a4cc252a17e3aa63731`
- Resume source: `d320-effb72-h120-resume4000-s6800-20260615T0916Z-785f3cd/checkpoints/latest.pt`
- Model: clean fixed FutureSeed + loop, D320/L12/loop6/all-loop, bf16 statepassing.
- Batch: microbatch 24, `GRAD_ACCUM_STEPS=3`, effective batch 72.
- Curriculum target: extend `108-120` hard stage to 4000 total steps, total step target 9800.
- Exclusions: no scratch, no feature noise, no selector, no Sudoku repair.

## Observed Before Halt

| step | h120-stage CE | note |
| --- | ---: | --- |
| 6900 | 0.5913 | resumed from step6800 |
| 7000 | 0.7010 | step7000 train checkpoint saved |
| 7100 | 0.6122 | continued after checkpoint |
| 7200 | 0.4983 | below step6800 CE |
| 7300 | 0.3414 | strong positive h120-stage signal |
| 7400 | 0.4571 | noisy but still below early stage |
| 7500 | 0.4718 | last logged step before GPU1 halt |

The only persisted training state from this segment is `step7000/latest.pt`; later logged steps were lost when GPU1 halted.

## Decision

Do not score this as a model result. It is a platform-interrupted segment with useful trend evidence. The follow-up run `d320-effb72-h120-resume7000-s9800-20260615T1358Z-785f3cd` resumed from the saved step7000 checkpoint and completed the planned 9800-step experiment.
