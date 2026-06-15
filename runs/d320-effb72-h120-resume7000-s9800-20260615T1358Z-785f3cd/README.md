# D320 Effective-Batch H120 Hard-Stage Completion

Status: completed.

- Run: `d320-effb72-h120-resume7000-s9800-20260615T1358Z-785f3cd`
- Source SHA: `785f3cdca74aab1f6e9d6a4cc252a17e3aa63731`
- Resume source: `d320-effb72-h120-hardstage-s9800-20260615T1230Z-785f3cd/checkpoints/latest.pt`
- Model: clean fixed FutureSeed + loop, D320/L12/loop6/all-loop, bf16 statepassing.
- Batch: microbatch 24, `GRAD_ACCUM_STEPS=3`, effective batch 72.
- Curriculum target: extended `108-120` hard stage to 4000 total steps, total step target 9800.
- Exclusions: no scratch, no feature noise, no selector, no Sudoku repair.

## Key Results

Checkpoint h120 loop6 curve:

| step | h120 exact | h120 blank_acc |
| --- | ---: | ---: |
| 7800 | 0.0078 | 0.5721 |
| 8800 | 0.0254 | 0.6197 |
| 9800 | 0.0566 | 0.6438 |

Final full eval:

| holes | loop6 exact | loop6 blank_acc |
| --- | ---: | ---: |
| 96 | 0.9785 | 0.9989 |
| 108 | 0.9004 | 0.9884 |
| 120 | 0.0547 | 0.6484 |
| 132 | 0.0000 | 0.1659 |

Final score: `0.0546875` on `metrics.eval_clean.loop6.label_exact`.

## Insight

D320 effective-batch scaling is real but not magic. The old D320/B48 run collapsed because effective training was weak. With microbatch 24 and accumulation 3, D320 keeps a strong h96/h108 foundation and opens h120 from zero as hard-stage compute increases.

The slope is positive: h120 exact moves `0.0078 -> 0.0254 -> 0.0566` across the 7800/8800/9800 checkpoints. This supports the bitter-lesson path: better general training infrastructure, larger effective compute, and longer hard-stage exposure move the frontier without selector, repair, or Sudoku-specific rules.

But D320 does not yet beat the best D256 long-compute line at comparable h120 hard-stage exposure. The result is approximately tied with the D256 +4k checkpoint (`0.0586`) and below the later D256 resumable continuation (`0.1777`). Width needs enough time; it is not a shortcut around the h120 global-consistency cliff.

## Decision

Continue D320 only from this checkpoint if the next run buys a genuinely later point, such as 11800 or 12800. Do not run D320/B48, selector, repair, scratch/noise, deeper-loop sweeps, or another short first-checkpoint repeat.
