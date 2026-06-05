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

## Insight

The scale boundary is not raw CUDA throughput. 16x16 ran with high utilization and acceptable memory. The boundary is global consistency: the model can reduce local CE on 16x16, but it does not assemble complete valid boards.

Loop depth becomes more important at 12x12: loop1 exact is zero across evaluated holes, while loop5 reaches h60 exact `0.5052`. At 16x16, loop depth no longer rescues exact solve, which means the current recurrence no longer propagates enough structure over 256 cells.

The delayed+unit-loss probe rules out a loss-only patch. The unit digit-mass objective is satisfied by near-uniform predictions and did not create a full-board communication path; K8 oracle exact remained `0.0`, so this is not a selector failure.

The unit-state run changes the conclusion: the 16x16 cliff was not mainly hidden size or loop count. It was missing row/column/box communication. With explicit unit-state pooling and broadcast, the same D192/L10/loop5 frontier jumps from h32 exact `0.0039` to `0.6914`.

The D256/L12/loop7 harder-hole run shows that scale is useful but insufficient. It improves h48 exact to `0.4180` and h64 exact to `0.0430`, but h80 remains `0.0`. Loop depth matters more on hard holes, ending the h64-80 stage with loop-last CE `0.1867` versus loop1 CE `0.5612`, but eval exact saturates around loop5. Because batch64 and batch48 both OOM on the 80GB GPU, the current no-checkpointing unit-state design is close to its direct scaling limit.

## Decision

Treat 12x12 as the largest supported scale for the original flattened paradigm. Treat 16x16 as supported by the new unit-state mainline, with h64/h80 still frontier territory. Direct width/depth scaling gives measurable but sublinear returns and does not open h80 under the current mechanism.

Next high-ROI work should change the mechanism:

1. Add persistent or true row/col/box unit tokens, because pooled unit-state plus direct scale improves h64 but leaves h80 exact at `0.0`.
2. If staying with pooled unit-state, run a longer h48-80 curriculum before adding more width; h64 improved, so curriculum length may still matter.
3. Add activation checkpointing only if it enables a structurally different experiment, not just a blind D384/L16 scale-up.
4. Delay 25x25 until h80 has a nonzero exact signal; otherwise 25x25 mostly amplifies the current global consistency bottleneck.
5. Keep selector work paused until K-oracle improves; current K1 is already the main result.
