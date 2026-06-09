# 12x12 h72 FutureSeed Loop Case Bank

Run: `casebank-h72-12x12-d256-l12-loop5-s1100-20260609T1013Z-8a2eb57`

Source SHA: `8a2eb57e0a09cbbe581746c247f20c365d61dfb3`

Recorded UTC: `2026-06-09T10:46:32.292147+00:00`

## Purpose

This run reproduces the 12x12 clean frontier and adds concrete h72 case-bank visualizations. The goal is mechanism analysis, not score chasing.

## Setup

- GPU row: GPU1 only.
- Board: 12x12 Sudoku, 3x4 boxes, random holes.
- Model: D256, 12 layers, 16 heads, head dim 16.
- Loop: max loops 5, final-loop CE.
- FutureSeed: scale 1.0, fixed update, no feedback.
- Curriculum: `16-36:300,36-60:500,60-72:300`.
- Eval holes: `48,60,72,84`, eval N 384.
- Case bank: h72, eval N 512, 4 cases each for solved-by-loop, almost-solved, hard-failure.

## Result

This is an exact reproduction of the prior 12x12 frontier metrics, plus case-bank export.

| holes | loop5 exact | loop5 blank acc | valid |
| ---: | ---: | ---: | ---: |
| 48 | 0.8880 | 0.9944 | 0.8958 |
| 60 | 0.5052 | 0.9743 | 0.5104 |
| 72 | 0.0443 | 0.8881 | 0.0521 |
| 84 | 0.0000 | 0.6993 | 0.0000 |

Case-bank h72 sample:

| kind | count | read |
| --- | ---: | --- |
| solved-by-loop | 4 | loop depth turns bad loop1 boards into exact solves |
| almost-solved | 4 | loop reaches 1 wrong hidden cell but stalls |
| hard-failure | 4 | loop improves local accuracy but leaves about 5 coupled wrong cells |

## Mechanism Insight

The solved-by-loop cases are the cleanest evidence for loop value. Typical trajectory:

- loop1: 29-32 wrong hidden cells and 31-34 duplicate-conflict units.
- loop2: large local repair, usually down to 5-11 wrong cells.
- loop3: near-global coordination, often 0-3 wrong cells.
- loop5: final consistency cleanup for solved cases.

The almost-solved cases show the current limit. They reach 1 wrong hidden cell by loop3, but loop5 makes no further change. The hard-failure cases also improve strongly, but stall near 5 wrong hidden cells with remaining duplicate conflicts. So the loop is real, but the last few coupled errors are sticky.

## Artifacts

- h72 case bank index: `output/case_bank/h72/index.html`
- case metadata: `output/case_bank/h72/cases.json`
- run JSON: `output/futureseed_loop_seed52.json`
- run report: `output/futureseed_loop_seed52.md`
- logs: `logs/run.log`

