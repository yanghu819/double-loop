# Scale Frontier Summary: 9x9 to 16x16

Purpose: identify the largest Sudoku board currently supported by the FutureSeed RWKV recurrent-loop paradigm on GPU1.

## Result

The original flattened FutureSeed RWKV loop supports 12x12 but not 16x16. The new explicit unit-state mechanism opens a 16x16 foothold under the tested budget.

| board | run | key result | decision |
| --- | --- | --- | --- |
| 9x9 | `scaleup-d192l10-h40-20260604T0844Z-a5b3b4a` | h40 loop5 exact `0.5332`, h44 `0.1914` | supported |
| 12x12 | `frontier-12x12-d256-h72-20260604T0920Z-f67e6a2` | h60 loop5 exact `0.5052`, h72 `0.0443`, h84 `0.0` | largest supported board |
| 16x16 high-hole | `frontier-16x16-d192-h112-20260604T1006Z-f67e6a2` | high-loss at h64-h96; aborted | over-hard curriculum |
| 16x16 foothold | `frontier-16x16-foothold-d192-h64-20260604T1018Z-f67e6a2` | final CE `0.5792`, but h32 exact `0.0039`, h48+ exact `0.0` | not supported |
| 16x16 all-loop | `frontier-16x16-allloop-d256-l8-20260604T1234Z-75a8796` | `LOOP_LOSS=all`, D256/L12/loop8; final CE `1.0498`, h24-h48 exact `0.0` | naive per-loop CE is negative |
| 16x16 delayed+unit loss | `frontier-16x16-delayed-unit-d256-l8-20260605T0236Z-37a3d6e` | delayed loop CE + unit digit-mass loss; final CE `1.0642`, h24-h48 exact `0.0`, h32 blank_acc `0.3831` | loss-only rescue is negative |
| 16x16 unit-state | `frontier-16x16-unitstate-d192-h64-20260605T0349Z-e9eabcf` | explicit row/col/box state passing; h32 exact `0.6914`, h48 `0.2148`, h64 `0.0117` | 16x16 opened |
| 16x16 scaled unit-state | `frontier-16x16-unitstate-d256-l12-loop7-h80-b32-20260605T0658Z-f170ff7` | D256/L12/loop7, h48 exact `0.4180`, h64 `0.0430`, h80 `0.0`; batch48+ OOM | scale helps but h80 closed |
| 16x16 unit-memory | `frontier-16x16-unitmem-d192-l10-loop7-h80-b48-20260605T0803Z-79004cf` | persistent row/col/box unit memory, h48 exact `0.6992`, h64 `0.2305`, h80 `0.0` | structure beats direct scale |
| 16x16 unit-memory h80 curriculum | `frontier-16x16-unitmem-h80curr-d192-l10-loop7-b48-20260605T0850Z-0be64f4` | longer 64-80 stage; h64 exact `0.4297`, h72 `0.2031`, h80 `0.0625`, h88 `0.0039` | h80 opened |
| 25x25 unit-memory feasibility | `frontier-25x25-unitmem-feas-d160-l8-loop5-b32-20260605T0937Z-710bb46` | D160/L8/loop5; h50 exact `0.5156`, h75 `0.0742`, h100 `0.0` | 25x25 low-hole foothold |

## Insight

The scale boundary is not raw CUDA throughput. 16x16 ran with high utilization and acceptable memory. The boundary is global consistency: the model can reduce local CE on 16x16, but it does not assemble complete valid boards.

Loop depth becomes more important at 12x12: loop1 exact is zero across evaluated holes, while loop5 reaches h60 exact `0.5052`. At 16x16, loop depth no longer rescues exact solve, which means the current recurrence no longer propagates enough structure over 256 cells.

The delayed+unit-loss probe rules out a loss-only patch. The unit digit-mass objective is satisfied by near-uniform predictions and did not create a full-board communication path; K8 oracle exact remained `0.0`, so this is not a selector failure.

The unit-state run changes the conclusion: the 16x16 cliff was not mainly hidden size or loop count. It was missing row/column/box communication. With explicit unit-state pooling and broadcast, the same D192/L10/loop5 frontier jumps from h32 exact `0.0039` to `0.6914`.

The D256/L12/loop7 harder-hole run shows that scale is useful but insufficient. It improves h48 exact to `0.4180` and h64 exact to `0.0430`, but h80 remains `0.0`. Loop depth matters more on hard holes, ending the h64-80 stage with loop-last CE `0.1867` versus loop1 CE `0.5612`, but eval exact saturates around loop5. Because batch64 and batch48 both OOM on the 80GB GPU, the current no-checkpointing unit-state design is close to its direct scaling limit.

The persistent unit-memory run is the strongest scaling insight so far. A smaller D192/L10/loop7 model with recurrent row/col/box memory beats the D256/L12 pooled unit-state run: h48 exact `0.6992` versus `0.4180`, h64 `0.2305` versus `0.0430`, and final CE `0.1226` versus `0.1867`. The bottleneck is therefore unit-level recurrent state, not just capacity.

The h80-focused memory curriculum opens the next frontier: h80 exact rises from `0.0` to `0.0625`, h72 reaches `0.2031`, and h64 reaches `0.4297`. h88 is barely nonzero at `0.0039`; h96 remains closed. That means h80 was partly a curriculum boundary, while h88/h96 are still global consistency frontiers.

The 25x25 conservative feasibility run establishes the first larger-board foothold. With D160/L8/loop5 memory mode, h50 exact reaches `0.5156` and h75 reaches `0.0742`; h100 remains `0.0`. The model can optimize 625-cell Sudoku under low/mid holes, so 25x25 is no longer a pure infeasibility case.

## Decision

Treat 12x12 as the largest supported scale for the original flattened paradigm. Treat 16x16 as supported by the new unit-memory mainline through h80. Treat 25x25 as supported at low holes and feasible at h75. h100 on 25x25 and h88/h96 on 16x16 are still frontier territory.

Next high-ROI work should change the mechanism:

1. Target the 25x25 h75-h100 transition with memory mode, e.g. `50-75:300,75-100:700`, eval h75/h90/h100/h110.
2. For 16x16 h88/h96, prefer true unit tokens inside the RWKV sequence or a targeted hard curriculum over more width.
3. Add activation checkpointing only if it enables a structurally different experiment, not just a blind D384/L16 scale-up.
4. Keep selector work paused until K-oracle improves; current K1 is already the main result.
