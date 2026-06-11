# capprobe-d320-12x12-h84-s900-20260611T054106Z-d29bb4e

Recorded at: 2026-06-11T06:02:00Z
Remote run: `/huyang2/double-loop/.worktrees/capprobe-d320-h84-20260611T054106Z-d29bb4e/runs/capprobe-d320-12x12-h84-s900-20260611T054106Z-d29bb4e`
Source SHA: `d29bb4e922d6ee4430e1a9e8724218b86480b711`
GPU row: GPU1 only

## Question

After h84 opened under the D256 long-stage run, the next scaling question is whether simply widening the clean FutureSeed+loop backbone is a practical path toward h96. GPU1 had only a short remaining lease, so this was a capacity feasibility probe, not a full score run.

## Setup

- Board: 12x12 Sudoku, 3x4 boxes, random holes.
- Model: FutureSeed loop backbone, D320, 12 layers, 20 heads, head dim 16.
- Loop budget: max loop 5.
- Curriculum: `16-36:100,36-60:150,60-72:200,72-84:450`.
- Eval: h72/h84/h96, eval_n 192, K1 rollout only.
- Batch: 48.
- Precision/kernel: bf16 forward path, RWKV kernel auto.

## Results

- GPU feasibility: no OOM; observed memory around 49 GB on the A800 80GB.
- Train CE: step100 `1.6489`, step300 `1.8029`, step500 `1.7790`, step700 `1.1760`, step900 `0.8789`.
- Loop5 exact by holes: h72 `0.0000`, h84 `0.0000`, h96 `0.0000`.
- Loop5 blank accuracy: h72 `0.6675`, h84 `0.5523`, h96 `0.4264`.
- h84 loop progression: loop1/3/5 exact all `0.0000`.

## Decision

This is a negative short-budget signal, not a final rejection of D320. The model fits in memory, but at 900 steps it is still far behind the D256 long-stage run and has not reached the regime where loop depth can repair boards.

Do not launch a full D320 run just because it fits. If capacity scaling is tested again, it should receive the same 2400-step h84 curriculum as the D256 baseline, or use a schedule that reaches low CE much earlier. Under tight lease time, the better next use of GPU is not a wider short probe; it is either a full-budget capacity run after restart or a loop/FutureSeed-state mechanism run with a clear h84/h96 readout.
