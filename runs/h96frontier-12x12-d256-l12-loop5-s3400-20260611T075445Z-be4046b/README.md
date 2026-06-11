# h96frontier-12x12-d256-l12-loop5-s3400-20260611T075445Z-be4046b

Recorded at: 2026-06-11T09:05:00Z
Remote run: `/huyang2/double-loop/.worktrees/h96frontier-12x12-20260611T075445Z-be4046b/runs/h96frontier-12x12-d256-l12-loop5-s3400-20260611T075445Z-be4046b`
Source SHA: `be4046be273f07267edbc98a60d3164d13436c32`
GPU row: GPU1 only

## Question

The clean D256 h84 long-stage run opened h84 at exact `0.1172`, while h96 remained `0.0`. This run asks whether h96 is also curriculum-bound: if we push the hard stage from 72-84 to 84-96 while keeping the same clean FutureSeed+loop model, does h96 open at all?

## Setup

- Board: 12x12 Sudoku, 3x4 boxes, random holes.
- Model: clean FutureSeed+loop backbone, D256, 12 layers, 16 heads, head dim 16.
- Loop budget: max loop 5.
- Curriculum: `16-36:200,36-60:300,60-72:400,72-84:1000,84-96:1500`.
- Eval: h72/h84/h96/h108, eval_n 384.
- Case bank: h96, eval_n 1024.
- Precision/kernel: bf16 forward path, RWKV kernel auto.

## Results

- Train CE: h96 stage step2000 `0.4962`, step2300 `0.4194`, step2800 `0.3443`, step3200 `0.3788`, step3400 `0.3931`.
- Loop5 exact by holes: h72 `0.8750`, h84 `0.4740`, h96 `0.0104`, h108 `0.0000`.
- h96 loop progression: loop1 `0.0000`, loop3 `0.0000`, loop4 `0.0078`, loop5 `0.0104`.
- h96 case-bank: exact `0.0068`, blank_acc `0.7652`, selected 7 solved-by-loop, 8 almost-solved, and 8 hard-failure cases.
- Rollout selector gap: K1 oracle exact `0.0104`; selector exact `0.0104`; trajectory disagreement `0.0`.

## Decision

h96 is no longer fully closed. The same clean FutureSeed+loop model opens h96 from `0.0` to `0.0104` by moving curriculum pressure to 84-96. This is small but meaningful because h96 loop1/3 remain zero and loop4/5 create the nonzero solves.

This is not a solved regime. h96 blank accuracy is only `0.7769`, h96 exact is barely nonzero, and h108 remains `0.0`. The h96 hard stage also never reaches the low CE range seen in the h84 long run. That means more hard-stage compute could plausibly move h96, but the curve is already much slower than h84.

Next high-ROI step is not selector work; the K1 oracle gap is zero. The useful fork is either a longer h96 hard stage under the same D256 model, or a simple loop/FutureSeed-state mechanism intended to lower h96 CE faster. Short width probes are not enough evidence.
