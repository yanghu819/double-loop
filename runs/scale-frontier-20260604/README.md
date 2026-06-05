# Scale Frontier Summary: 9x9 to 16x16

Purpose: identify the largest Sudoku board currently supported by the FutureSeed RWKV recurrent-loop paradigm on GPU1.

## Result

The current paradigm supports 12x12 but not 16x16 full-board solving under the tested budget.

| board | run | key result | decision |
| --- | --- | --- | --- |
| 9x9 | `scaleup-d192l10-h40-20260604T0844Z-a5b3b4a` | h40 loop5 exact `0.5332`, h44 `0.1914` | supported |
| 12x12 | `frontier-12x12-d256-h72-20260604T0920Z-f67e6a2` | h60 loop5 exact `0.5052`, h72 `0.0443`, h84 `0.0` | largest supported board |
| 16x16 high-hole | `frontier-16x16-d192-h112-20260604T1006Z-f67e6a2` | high-loss at h64-h96; aborted | over-hard curriculum |
| 16x16 foothold | `frontier-16x16-foothold-d192-h64-20260604T1018Z-f67e6a2` | final CE `0.5792`, but h32 exact `0.0039`, h48+ exact `0.0` | not supported |
| 16x16 all-loop | `frontier-16x16-allloop-d256-l8-20260604T1234Z-75a8796` | `LOOP_LOSS=all`, D256/L12/loop8; final CE `1.0498`, h24-h48 exact `0.0` | naive per-loop CE is negative |
| 16x16 delayed+unit loss | `frontier-16x16-delayed-unit-d256-l8-20260605T0236Z-37a3d6e` | delayed loop CE + unit digit-mass loss; final CE `1.0642`, h24-h48 exact `0.0`, h32 blank_acc `0.3831` | loss-only rescue is negative |

## Insight

The scale boundary is not raw CUDA throughput. 16x16 ran with high utilization and acceptable memory. The boundary is global consistency: the model can reduce local CE on 16x16, but it does not assemble complete valid boards.

Loop depth becomes more important at 12x12: loop1 exact is zero across evaluated holes, while loop5 reaches h60 exact `0.5052`. At 16x16, loop depth no longer rescues exact solve, which means the current recurrence no longer propagates enough structure over 256 cells.

The delayed+unit-loss probe rules out a loss-only patch. The unit digit-mass objective is satisfied by near-uniform predictions and did not create a full-board communication path; K8 oracle exact remained `0.0`, so this is not a selector failure.

## Decision

Treat 12x12 as the largest supported Sudoku scale for the current paradigm. Do not run more 16x16 seed repeats or high-hole repeats.

Next high-ROI work should change the mechanism:

1. Add hierarchical row/col/box tokens or unit-level state passing.
2. Add shaped intermediate loop supervision only as a supporting objective; the naive equal `LOOP_LOSS=all` and delayed unit-loss rescue were both negative.
3. Keep selector work paused until K-oracle improves; the latest K8 oracle gap is zero.
4. Fix experiment tracking so source snapshots are stored outside Git once they exceed GitHub's hard file limit.
