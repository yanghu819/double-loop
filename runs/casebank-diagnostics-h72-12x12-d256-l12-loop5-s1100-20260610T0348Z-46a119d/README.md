# h72 Case-Bank Diagnostics

Run: `casebank-diagnostics-h72-12x12-d256-l12-loop5-s1100-20260610T0348Z-46a119d`

Git SHA: `46a119da651a1c0e109647490cc754dfa70159bd`

GPU: AIStation `GPU1`, A100 80GB, `CUDA_VISIBLE_DEVICES=0`

## Question

At the 12x12 h72 cliff, are the last failures low-confidence uncertainty, or stable wrong states that the loop cannot escape?

## Setup

This run keeps the existing FutureSeed/RWKV loop model and training recipe, then expands only the diagnostic case bank:

- 12x12 Sudoku, random holes
- D256, L12, 16 heads, head dim 16, 2 low-level cycles, 5 loops
- Curriculum: `16-36:300,36-60:500,60-72:300`
- Batch 80, bf16 forward, CUDA RWKV7 path
- Eval holes: `48,60,72,84`
- h72 case bank: `eval_n=1024`, 8 cases each for solved-by-loop, almost-solved, hard-failure

## Results

Loop5 exact by holes:

| holes | exact | blank acc |
| ---: | ---: | ---: |
| 48 | 0.9531 | 0.9975 |
| 60 | 0.6484 | 0.9833 |
| 72 | 0.0677 | 0.8980 |
| 84 | 0.0000 | 0.7026 |

Main h60 loop progression:

| loop | exact | blank acc |
| ---: | ---: | ---: |
| 1 | 0.0000 | 0.7654 |
| 2 | 0.2266 | 0.9602 |
| 3 | 0.5911 | 0.9809 |
| 4 | 0.6354 | 0.9829 |
| 5 | 0.6484 | 0.9833 |

h72 case bank:

- Final exact: `0.0742`
- Final blank accuracy: `0.9023`
- Selected cases: 8 solved-by-loop, 8 almost-solved, 8 hard-failure
- Final wrong cells across almost/hard cases: 48
- Final wrong cells changed in the last shown loop: 1 / 48
- Stable-since histogram for final wrong cells: loop1=19, loop2=12, loop3=16, loop5=1
- Almost-solved final wrongs: mean margin `0.192`, mean entropy `0.906`
- Hard-failure final wrongs: mean margin `0.482`, mean entropy `0.693`

## Interpretation

The useful part of loop is real: on h60, exact rises from `0.0000` at loop1 to `0.6484` at loop5. The first loop produces locally plausible but globally incomplete states; repeated refinement turns many of those into valid boards.

The h72 cliff is different. The remaining wrong cells are not mostly late-loop oscillations. Almost all final wrong cells are already stable by loop1/2/3, and the hard failures are more confident than the almost-solved failures. That means the loop often enters a wrong attractor early and then keeps polishing it.

Rollout does not fix this run: K8 oracle equals the selector score, so this is not a selector problem. It is a loop dynamics / credit assignment problem.

## Decision

Do not add Sudoku-specific repair rules. The next high-ROI work should be a simple, generic change to the recurrent refinement dynamics:

- make loop updates less eager to lock early predictions, or
- add a learned residual/decay gate that lets later loops revise stable states, or
- supervise intermediate loops so they learn to keep uncertainty where the global state is unresolved.

The key test for the next run is whether h72 improves without hurting h60 and without adding hand-coded Sudoku structure.

