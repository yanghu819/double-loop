# h120 Loop6 All-Loop Scale Test

Run: `h120loopall-retry-12x12-d256-l12-loop6-s7400-20260611T1640Z-8ccc4a9`

Source SHA: `8ccc4a9afb73b927ca8d3976837abf372ddd2e04`

Remote: GPU1 AIStation, A100 80GB after GPU1 restart.

## Question

Can the clean FutureSeed+loop path push beyond h108 if we spend the next budget on generic loop scaling instead of Sudoku-specific structure?

The tested change is deliberately simple: use loop6 and supervise every loop with `LOOP_LOSS=all`. This asks whether the recurrent trajectory becomes more stable under hard holes, while staying inside the same flattened FutureSeed backbone.

## Config

- Board: 12x12 Sudoku, random holes, 3x4 boxes.
- Model: D256, 12 layers, 8 heads, head_dim 32, channel_mult 4.
- Loop: max_loops 6, `LOOP_LOSS=all`.
- Curriculum: `16-36:200,36-60:300,60-72:500,72-84:1200,84-96:1800,96-108:1800,108-120:1600`.
- Batch/eval: batch 72, eval 512, K1 rollout only.
- Runtime: bfloat16, RWKV7 statepassing CUDA kernel.

## Results

| eval holes | loop6 exact | loop6 blank_acc | decision |
| --- | ---: | ---: | --- |
| h96 | 0.9922 | 0.9999 | solved under this budget |
| h108 | 0.9160 | 0.9924 | strongly supported |
| h120 | 0.0273 | 0.5866 | barely opened, still frontier |
| h132 | 0.0000 | 0.1530 | closed |

h120 loop curve:

| loop | exact | blank_acc |
| ---: | ---: | ---: |
| 1 | 0.0000 | 0.3441 |
| 2 | 0.0000 | 0.4916 |
| 3 | 0.0117 | 0.5719 |
| 4 | 0.0273 | 0.5843 |
| 5 | 0.0273 | 0.5860 |
| 6 | 0.0273 | 0.5866 |

Training finished with CE `0.3605`, total loss `0.5267`, loop1 CE `1.1588`, and train time `8556.3s`.

## Read

This is a real frontier movement but not a clean h120 solve. The old clean h108 run had h120 exact `0.0000` and h120 blank_acc `0.3439`; this run moves h120 to exact `0.0273` and blank_acc `0.5866`. More importantly, it makes h108 robust: h108 exact rises from `0.1797` to `0.9160`.

The limiting signal is also clear. Extra loops help until loop4 on h120, then exact saturates. K1 has zero selector gap, so this is not a rollout-selection problem. The model is still not locally reliable enough at h120, and the last stage remains hard even after a strong h108 foundation.

## Decision

Treat loop6 plus all-loop supervision as useful for scaling the clean path through h108 and as the first nonzero h120 opening. Do not spend next budget on selector work or Sudoku repair rules. The next high-ROI direction is either more h120-stage compute/capacity or one simple FutureSeed state update that improves late-loop revision without adding task-specific structure.

No tag was created because primary score is below `0.50`.
