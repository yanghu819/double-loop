# hardstage-long-h72-12x12-d256-l12-loop5-s1500-20260610T0538Z-0942939

Recorded at: 2026-06-10T06:18:17.952975+00:00
Remote run: `/huyang2/double-loop/.worktrees/hardstage-h72-20260610T053842Z-0942939/runs/hardstage-long-h72-12x12-d256-l12-loop5-s1500-20260610T0538Z-0942939`
Source SHA: `09429392f2a88ef09ac7b6fc2d0952e34397593f`
GPU row: GPU1 only

## Question

After diagnostics showed h72 failures were stable wrong attractors, simple loop-credit and FutureSeed damping both failed. This run asks a more bitter-lesson question: does more hard-stage compute on the same clean FutureSeed+loop model move the 12x12 h72 cliff, or is the current mechanism already saturated?

## Setup

- Board: 12x12 Sudoku, 3x4 boxes, random holes.
- Model: clean FutureSeed loop backbone, D256, 12 layers, 16 heads, head dim 16.
- Loop budget: train/eval max loop 5.
- Precision/kernel: bf16 forward path, RWKV kernel auto.
- Curriculum: `16-36:300,36-60:500,60-72:700`.
- Batch/eval: full batch 80, eval_n 384, case-bank h72 eval_n 1024.
- No new architecture change relative to the h72 diagnostics baseline; the intended intervention is extending the 60-72 hard stage from 300 to 700 steps.

## Results

- Train CE: step1100 0.1525, step1200 0.1294, step1300 0.1572, step1400 0.0823, step1500 0.0802.
- Loop5 exact by holes: h48 0.9635, h60 0.7786, h72 0.1745, h84 0.0000.
- h72 loop progression: loop1 0.0000, loop2 0.0052, loop3 0.1354, loop4 0.1719, loop5 0.1745.
- h72 blank accuracy: loop5 0.9251.
- h72 case-bank: exact 0.1865, blank_acc 0.9293, selected 8 solved-by-loop, 8 almost-solved, 8 hard-failure cases.

## Decision

Positive. The h72 cliff moved strongly without adding Sudoku-specific repair logic or more complicated FutureSeed damping. Against the 2026-06-10 diagnostics baseline, h72 loop5 exact improved from 0.0677 to 0.1745 and h60 from 0.6484 to 0.7786. This supports continuing the clean scaling direction first.

The next high-information run is already launched: same model and mechanism, but with a longer 60-72 hard stage (`16-36:300,36-60:500,60-72:1500`). If h72 keeps rising, continue compute/curriculum scaling; if it flattens, the next step should change general modeling capacity or loop dynamics, not add Sudoku-specific conflict heuristics.
