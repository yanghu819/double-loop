# h84long-12x12-d256-l12-loop5-s2400-20260611T043811Z-f3e48c2

Recorded at: 2026-06-11T05:35:06Z
Remote run: `/huyang2/double-loop/.worktrees/h84long-12x12-20260611T043525Z-f3e48c2/runs/h84long-12x12-d256-l12-loop5-s2400-20260611T043811Z-f3e48c2`
Source SHA: `f3e48c2befc5d1125ca960ff6a525e1cf70a4fcc`
GPU row: GPU1 only

## Question

The short h84 frontier-shift probe reached h84 loop5 exact `0.0234`. This run asks whether the same clean FutureSeed+loop model can move the 12x12 frontier by simply giving h84 a longer hard stage, or whether h84 already needs a new mechanism.

## Setup

- Board: 12x12 Sudoku, 3x4 boxes, random holes.
- Model: FutureSeed loop backbone, D256, 12 layers, 16 heads, head dim 16.
- Loop budget: max loop 5.
- Curriculum: `16-36:200,36-60:300,60-72:400,72-84:1500`.
- Eval: primary h84, eval_n 384; transfer h60/h72/h84/h96; case-bank h84 eval_n 1024.
- Precision/kernel: bf16 forward path, RWKV kernel auto.
- Noise: no stochastic rollout noise; feature-diff noise scale stayed at the default zero path.

## Results

- Train CE: h84 stage step1000 `0.3859`, step1200 `0.2662`, step1500 `0.2395`, step1800 `0.1898`, step2000 `0.1492`, step2400 `0.1934`.
- Loop5 exact by holes: h60 `0.9297`, h72 `0.7161`, h84 `0.1172`, h96 `0.0000`.
- h84 loop progression: loop1 `0.0000`, loop2 `0.0000`, loop3 `0.0625`, loop4 `0.1068`, loop5 `0.1172`.
- h84 case-bank: exact `0.1211`, blank_acc `0.8884`, selected 8 solved-by-loop, 8 almost-solved, and 8 hard-failure cases.
- Rollout selector gap: K1/K4/K8 oracle exact all `0.1172`; selector exact also `0.1172`; trajectory disagreement `0.0`.

## Decision

This is a real frontier move. Extending the h84 hard stage improved h84 exact from `0.0234` to `0.1172` under the same model and loop budget. h72 also rose from `0.5469` in the short h84 probe to `0.7161`, so the longer hard-stage pressure did not merely overfit h84 at the expense of easier transfer.

The result does not say the paradigm is solved. h96 remains closed at exact `0.0000`, and h84 blank accuracy is `0.8895` while exact is only `0.1172`, so whole-board consistency is still the bottleneck. Loop depth is useful and not cosmetic: h84 exact is zero at loop1/2, opens at loop3, and improves again at loop4/5.

Next high-ROI step: scale the clean path, not Sudoku-specific tricks. The next run should either raise model capacity with the same h84 curriculum, or raise loop budget with a simple FutureSeed state update change. Do not spend time on rollout selector work here; the oracle gap is zero.
