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

## Insight

The clean mainline does not fail because it lacks a Sudoku rule patch. It first hits two more general scaling bottlenecks.

1. Activation memory is the immediate 80GB limit. D256/L12/loop8 on 16x16 cannot run at batch32 or batch48 in the current implementation. This is a generic scaling-engineering issue, not a task-prior issue.
2. Optimization/credit assignment is the next limit. Even with batch16, pure FutureSeed+loop learns slowly on 16x16. The loop-last loss is not materially better than loop1 early in training, which means extra recurrent compute is not yet being converted into useful refinement.

The prior unit-memory results are therefore reinterpreted as a diagnostic, not a method claim: they showed that the bottleneck is communication substrate. But the clean Better Lesson direction should not solve that by hardcoding Sudoku structure.

## Decision

Do not add Sudoku priors back.

Next work should be generic and scalable:

1. Activation checkpointing or gradient accumulation so the same architecture can use an effective batch size above 16.
2. Better loop credit assignment that makes later loops learn a refinement operator without task rules.
3. A clean scaling curve on the same proxy after the above generic fixes: 9x9, 12x12, 16x16 with random holes only.

No tag was created because all runs were aborted by decision rules before producing a score.
