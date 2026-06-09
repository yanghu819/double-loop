# Clean Better Lesson Scale Probe

## Question

After removing Sudoku-specific row/column/box priors, can pure RWKV7 + FutureSeed + outer loop scale into 16x16 Sudoku by spending more model and recurrent compute?

This run intentionally avoided:

- row/column/box state pooling
- persistent unit memory
- unit-consistency loss
- structured hole patterns
- feature noise

## Runs

| run | result | decision |
| --- | --- | --- |
| `clean-scale-16x16-d256-l12-loop8-h64-b48-20260608T035314Z-f227f8f` | OOM before step100, process reached `79.28GiB` on 80GB | batch48 infeasible |
| `clean-scale-16x16-d256-l12-loop8-h64-b32-20260608T035725Z-f227f8f` | OOM before step100, process reached `79.30GiB` on 80GB | batch32 infeasible |
| `clean-scale-16x16-d256-l12-loop8-h64-b16-20260608T040045Z-f227f8f` | fits at about `42.9GB`, but h64 curriculum had step300 CE `1.4504` on 24-40 holes | stop hard curriculum |
| `clean-scale-16x16-d256-l12-loop8-h32-b16-20260608T041120Z-f227f8f` | fits at about `42.9GB`; 8-16 holes step300 CE `1.0208`; 16-24 holes step500 CE `1.2165` | stop as low ROI |
| `bf16-b32-shaped16x16-d256-l12-loop8-h24-s600-20260609T0638Z-a9a0e9d` | `FORWARD_DTYPE=bfloat16` makes D256/L12/loop8 batch32 fit at about `67.8GB`; h24 blank_acc improves to `0.3711`, but exact remains `0.0000` | keep bf16 as scale default; exact still closed |

## Insight

The clean mainline does not fail because it lacks a Sudoku rule patch. It first hits two more general scaling bottlenecks.

1. Activation memory was the immediate fp32 80GB limit. D256/L12/loop8 on 16x16 could not run at batch32 in fp32, but bf16 forward now makes batch32 fit at about `67.8GB`. This is a real generic scaling-engineering win.
2. Optimization/credit assignment is now the next limit. BF16 batch32 improves local blank accuracy, but exact remains zero and loop-last is only slightly better than loop1. Extra recurrent compute is still not being converted into strong refinement.

The prior unit-memory results are therefore reinterpreted as a diagnostic, not a method claim: they showed that the bottleneck is communication substrate. But the clean Better Lesson direction should not solve that by hardcoding Sudoku structure.

## Decision

Do not add Sudoku priors back.

Next work should be generic and scalable:

1. Use `FORWARD_DTYPE=bfloat16` by default on GPU1 clean scale runs.
2. Add activation checkpointing so the bf16 path can spend the memory headroom on larger width/depth or longer loops.
3. Better loop credit assignment that makes later loops learn a refinement operator without task rules.
4. A clean scaling curve on the same proxy after the above generic fixes: 9x9, 12x12, 16x16 with random holes only.

No tag was created because all runs were aborted by decision rules before producing a score.
