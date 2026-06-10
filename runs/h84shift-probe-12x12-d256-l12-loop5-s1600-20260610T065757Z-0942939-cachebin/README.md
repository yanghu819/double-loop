# h84shift-probe-12x12-d256-l12-loop5-s1600-20260610T065757Z-0942939-cachebin

Recorded at: 2026-06-10T07:35:05Z
Remote run: `/huyang2/double-loop/.worktrees/h84shift-12x12-20260610T065757Z-0942939/runs/h84shift-probe-12x12-d256-l12-loop5-s1600-20260610T065757Z-0942939-cachebin`
Source SHA: `09429392f2a88ef09ac7b6fc2d0952e34397593f`
GPU row: GPU1 only

## Question

The longer h72 run barely opened h84 at exact 0.0078. This short probe asks whether directly shifting the curriculum frontier to 72-84 holes can move h84 under the same model, or whether h84 already needs capacity or loop-dynamics changes.

## Setup

- Board: 12x12 Sudoku, 3x4 boxes, random holes.
- Model: FutureSeed loop backbone, D256, 12 layers, 16 heads, head dim 16.
- Loop budget: max loop 5.
- Curriculum: `16-36:200,36-60:300,60-72:400,72-84:700`.
- Eval: primary h84, eval_n 256, case-bank h84 eval_n 512.
- Precision/kernel: bf16 forward path, RWKV kernel auto.
- Note: `git_dirty=1` in the remote metadata came from failed startup run directories inside the temporary worktree. `source.patch` is empty, and no source code changed.

## Results

- Train CE: step900 0.1666 before h84 stage, then h84 stage step1000 0.3859, step1200 0.2662, step1500 0.2395, step1600 0.2173.
- Loop5 exact by holes: h60 0.9297, h72 0.5469, h84 0.0234, h96 0.0000.
- h84 loop progression: loop1 0.0000, loop2 0.0000, loop3 0.0156, loop4 0.0195, loop5 0.0234.
- h84 case-bank: exact 0.0352, blank_acc 0.8582, selected 6 solved-by-loop, 6 almost-solved, 6 hard-failure cases.

## Decision

Positive but weak. Targeted h84 curriculum improves h84 over the h72-only s2300 run, from 0.0078 to 0.0234, while h72 also rises to 0.5469. That says the frontier can move with cleaner curriculum pressure, but h84 remains a hard boundary under this model.

Next high-ROI step is not another short h84 repeat. The useful fork is either a longer h84 stage after restarting GPU time, or a capacity/loop-dynamics scale-up that keeps the h84 curriculum.
