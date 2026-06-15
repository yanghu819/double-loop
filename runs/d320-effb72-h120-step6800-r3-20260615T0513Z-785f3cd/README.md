# D320 Effective-Batch h120 First Segment

Run: `d320-effb72-h120-step6800-r3-20260615T0513Z-785f3cd`

Source SHA: `785f3cdca74aab1f6e9d6a4cc252a17e3aa63731`

Status: platform interrupted after the step4000 train checkpoint; resumed by `d320-effb72-h120-resume4000-s6800-20260615T0916Z-785f3cd`.

## Purpose

This was the first real D320 effective-batch scale run after validating gradient accumulation. It tested whether D320 can train with effective batch 72 instead of the weak D320/B48 setting that collapsed h96/h108 foundation.

This is not a separate scientific endpoint. It is the first segment of the same D320 scale test.

## Configuration

- Board: 12x12, random holes, 3x4 boxes.
- Model: D320/L12, heads10, head_dim32, loop6, `LOOP_LOSS=all`.
- Batch: microbatch 24, `GRAD_ACCUM_STEPS=3`, effective batch 72.
- Runtime: GPU1 only, bf16, RWKV statepassing CUDA.
- Mechanism: clean fixed FutureSeed, no scratch, no noise, no selector, no repair.
- Curriculum: `16-36:200,36-60:300,60-72:500,72-84:1200,84-96:1800,96-108:1800,108-120:1000`.

## Partial Curve

| step | stage | CE | loop1 CE | note |
| ---: | --- | ---: | ---: | --- |
| 100 | 16-36 | 1.6071 | 1.6039 | warmup |
| 500 | 36-60 | 0.7394 | 0.8196 | stage2 learning |
| 1000 | 60-72 | 0.1235 | 0.3223 | first checkpoint |
| 2000 | 72-84 | 0.1699 | 0.4011 | second checkpoint |
| 3000 | 84-96 | 0.4083 | 0.7139 | hard stage still high |
| 4000 | 84-96 | 0.2313 | 0.6353 | checkpoint before platform halt |

## Decision

Resume, do not treat this as a failed model run. The useful signal is that D320 effective-batch preserved early and mid-stage optimization well enough to justify finishing the h108/h120 stages.
