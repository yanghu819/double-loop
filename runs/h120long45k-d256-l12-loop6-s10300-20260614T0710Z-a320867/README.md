# h120 Long 4.5k D256 Loop6 Scale Run

Run: `h120long45k-d256-l12-loop6-s10300-20260614T0710Z-a320867`

Source SHA: `a320867364a05679e2595b5fcd4992ffde09189a`

Started: `2026-06-14T07:03:03Z`

Stopped: `2026-06-14T10:30:20Z`

## Hypothesis

After loop8 failed to add new h120 solves and D320/B72 was too slow for one lease, the next clean scale question was whether a D256/L12/loop6 model keeps improving if the 108-120 hard stage is extended to 4500 steps. This tests the current FutureSeed+loop upper-bound path without selector, repair, scratch noise, or Sudoku-specific tricks.

## Configuration

- Board: 12x12, random holes, implicit 3x4 boxes.
- Model: D256/L12, heads8, head_dim32, loop6, `LOOP_LOSS=all`.
- FutureSeed: fixed context, no mutable scratch, no loop-residual update.
- Kernel/runtime: CUDA GPU1 only, bf16, RWKV statepassing.
- Curriculum: `16-36:200,36-60:300,60-72:500,72-84:1200,84-96:1800,96-108:1800,108-120:4500`.
- Batch/eval: batch72, eval512, checkpoint eval at steps 6800/7800/8800/9800/10300.

## Results

The run was stopped after step9800 because the lease risk was high and the step9800 checkpoint had already landed. Final step10300 was not attempted.

| checkpoint | h96 loop6 exact | h108 loop6 exact | h120 loop3 exact | h120 loop6 exact | h120 loop6 blank_acc | h132 loop6 exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 6800 | 0.9941 | 0.8203 | 0.0039 | 0.0137 | 0.5658 | 0.0000 |
| 7800 | 0.9941 | 0.8926 | 0.0156 | 0.0332 | 0.6062 | 0.0000 |
| 8800 | 0.9922 | 0.9004 | 0.0254 | 0.0410 | 0.6382 | 0.0000 |
| 9800 | 0.9961 | 0.9062 | 0.0312 | 0.0586 | 0.6655 | 0.0000 |

## Insight

This run almost exactly reproduces the previous clean +4k h120 checkpoint curve through step9800: h120 loop6 exact climbs `0.0137 -> 0.0332 -> 0.0410 -> 0.0586`. That is useful because it says the scale signal is real and reproducible, not a one-off lucky run. But it also says the slope is shallow: h108 is already around `0.90`, while h120 is still below `0.06` and h132 is fully closed.

The loop still matters, but it is not enough by itself. At step9800, h120 loop3 exact is `0.0312` and loop6 exact is `0.0586`; the extra recurrent work adds solved boards, but not enough to make h120 a supported regime. The main failure is still complete-board consistency at the frontier, not local filling throughput.

## Decision

Clean D256 hard-stage compute remains the best validated scaling path, but another identical longer run has low information gain. The next high-ROI move should either make training resumable enough to truly scale hard-stage compute beyond one lease, or introduce a very simple FutureSeed/loop state update aimed at late correction. Do not spend more GPU time on deeper loop-count sweeps, selector work, Sudoku repair, scratch noise/Gaussian sweeps, or non-resumable D320/B72 repeats.
