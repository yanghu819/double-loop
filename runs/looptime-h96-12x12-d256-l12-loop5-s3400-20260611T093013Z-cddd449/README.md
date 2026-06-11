# Loop-Time H96 Mechanism Run

Run: `looptime-h96-12x12-d256-l12-loop5-s3400-20260611T093013Z-cddd449`

Source SHA: `cddd449cc1cc8b8270559aa8cf7c59b301c27bfc`

Timestamp: `2026-06-11T09:30:13Z`

## Question

The clean h96 frontier run barely opened h96, and exact only appeared at late loops. The mechanism question was whether a shared recurrent update needs a small loop-phase signal so early loops can construct a board and late loops can correct it.

This is not a Sudoku-specific trick. The only new signal is a learned projection of two sinusoidal loop-time features, enabled by `LOOP_TIME_SCALE=1`.

## Setup

- Board: `12x12`, random holes.
- Model: FutureSeed RWKV loop, `D_MODEL=256`, `LAYERS=12`, `HEADS=16`, `HEAD_DIM=16`, `L_CYCLES=2`, `MAX_LOOPS=5`.
- Curriculum: `16-36:200,36-60:300,60-72:400,72-84:1000,84-96:1500`.
- Batch/eval: `FULL_BATCH=80`, `FULL_EVAL_N=384`, h96 case-bank `eval_n=1024`.
- Precision: `FORWARD_DTYPE=bfloat16`, RWKV kernel `auto`.
- Mechanism: `LOOP_TIME_SCALE=1`, no loop feedback, fixed FutureSeed update.

## Result

Loop-time did not beat the clean h96 baseline.

| Readout | clean h96 baseline | loop-time |
| --- | ---: | ---: |
| h72 loop5 exact | `0.8750` | `0.8542` |
| h84 loop5 exact | `0.4740` | `0.4115` |
| h96 loop5 exact | `0.0104` | `0.0052` |
| h108 loop5 exact | `0.0000` | `0.0000` |
| h96 blank_acc loop1 | `0.4211` | `0.5102` |
| h96 blank_acc loop5 | `0.7769` | `0.7549` |
| h96 case-bank exact | `0.0068` | `0.0059` |

Training CE was mixed. Loop-time was slower early, briefly better around step900/1900, and finished slightly better than baseline CE but with worse exact:

- step500: `0.7079` vs baseline `0.5030`.
- step900: `0.1421` vs baseline `0.1666`.
- step1900: `0.1398` vs baseline `0.1545`.
- step3400: `0.3667` vs baseline `0.3931`.

## Decision

Stop loop-time conditioning in this form. It makes early-loop predictions much stronger, but it does not improve the final global-consistency frontier. The important failure mode is not that the loop lacks a clock; it is that the late loop still falls into high-confidence coupled mistakes.

The next high-ROI direction should not be a loop-time scale sweep. Either spend compute on the clean h96 curriculum, or change the state update more directly so late loops can revise stable wrong attractors without adding task-specific Sudoku rules.
