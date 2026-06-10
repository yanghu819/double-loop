# hardstage-longer-h72-12x12-d256-l12-loop5-s2300-20260610T0611Z-0942939

Recorded at: 2026-06-10T07:02:25.413243+00:00
Remote run: `/huyang2/double-loop/.worktrees/hardstage2300-h72-20260610T061045Z-0942939/runs/hardstage-longer-h72-12x12-d256-l12-loop5-s2300-20260610T0611Z-0942939`
Source SHA: `09429392f2a88ef09ac7b6fc2d0952e34397593f`
GPU row: GPU1 only

## Question

The s1500 hard-stage run showed that more 60-72 exposure moved the h72 cliff. This run asks whether the same clean FutureSeed+loop model still has useful headroom, or whether longer hard-stage training has already saturated.

## Setup

- Board: 12x12 Sudoku, 3x4 boxes, random holes.
- Model: FutureSeed loop backbone, D256, 12 layers, 16 heads, head dim 16.
- Loop budget: max loop 5.
- Curriculum: `16-36:300,36-60:500,60-72:1500`.
- Batch/eval: full batch 80, eval_n 384, case-bank h72 eval_n 1024.
- Precision/kernel: bf16 forward path, RWKV kernel auto.

## Results

- Train CE: step1500 0.0802, step1900 0.0753, step2000 0.0980, step2100 0.1034, step2200 0.0986, step2300 0.0758.
- Loop5 exact by holes: h48 0.9818, h60 0.8750, h72 0.4010, h84 0.0078.
- h72 loop progression: loop1 0.0000, loop2 0.0156, loop3 0.3359, loop4 0.3906, loop5 0.4010.
- h84 loop progression: loop1 0.0000, loop2 0.0000, loop3 0.0026, loop4 0.0052, loop5 0.0078.
- h72 case-bank: exact 0.3955, blank_acc 0.9534, selected 8 solved-by-loop, 8 almost-solved, 8 hard-failure cases.

## Decision

Strong positive for clean scaling on h72. Against s1500, h72 exact improved from 0.1745 to 0.4010 and h60 improved from 0.7786 to 0.8750. Loop remains useful: h72 loop3 to loop5 adds +0.0651 exact.

The next frontier is h84. This run barely opens h84 at 0.0078, so a short h84 curriculum probe was launched to test whether targeted 72-84 exposure can move that boundary under the same model. If h84 stays near zero, the next lever should be capacity or loop dynamics rather than more h72-only training.
