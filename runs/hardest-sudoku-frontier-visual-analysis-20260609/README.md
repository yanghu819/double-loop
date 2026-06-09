# Hardest Doable Sudoku Frontier Visual Analysis

Generated UTC: `2026-06-09T10:58:28.087897+00:00`

## Selection

Selected task: `12x12 random-holes h72` from `frontier-12x12-d256-h72-20260604T0920Z-f67e6a2`.

This is the hardest currently archived clean task that is still doable. It uses random holes only, no row/column/box unit-memory mechanism, and reaches loop5 exact `4.43%` at h72. The next harder 12x12 point, h84, is closed at exact `0.00%`; clean 16x16 is also still closed. That makes h72 the best current mechanism microscope: it has real successes, real failures, and a visible cliff.

## Mechanism Readout

- Loop1 at h72: exact `0.00%`, blank accuracy `58.77%`.
- Loop2 at h72: exact `0.26%`, blank accuracy `83.05%`.
- Loop3 at h72: exact `2.34%`, blank accuracy `88.12%`.
- Loop5 at h72: exact `4.43%`, blank accuracy `88.81%`.
- Loop gain: exact `4.43%`, blank accuracy `30.03%`.
- Independent-cell exact estimate at h72 is `1.94e-04`; real exact is `4.43%`. The solved boards are therefore not just independent lucky cells. The loop sometimes lands in a coherent whole-board basin.
- FutureSeed h72 diagnostic: gate mean `0.540`, state norm `8.639`. The current archive records only aggregate seed strength, not per-cell or per-loop causal contribution.

## Main Insight

FutureSeed alone is not enough at this frontier: loop1 exact is zero. The loop is the part that turns a noncausal global seed into board-level refinement.

The loop has two regimes. Loop1 to loop2 mostly repairs local cell reliability. Loop2 to loop5 adds much less blank accuracy, but it still moves exact upward, which means later loops are doing the harder board-level coordination work. The current failure mode is not simply "more single-cell accuracy"; h72 already has high blank accuracy and still has low exact because a few coupled mistakes break the whole board.

## Concrete Cases

- 12x12 archived case: `runs/frontier-12x12-d256-h72-20260604T0920Z-f67e6a2/output/futureseed_loop_case_seed52.html`. This is the concrete case exported by the original run. It is useful for reading loop refinement from the same trained model, but the archive did not save a full h72 case bank.
- New h72 case bank: `runs/casebank-h72-12x12-d256-l12-loop5-s1100-20260609T1013Z-8a2eb57/output/case_bank/h72/index.html`. This reproduced the frontier and exported 4 solved-by-loop, 4 almost-solved, and 4 hard-failure h72 boards.
- 16x16 clean failure case: `runs/bf16-b32-shaped16x16-d256-l12-loop8-h24-s600-20260609T0638Z-a9a0e9d/output/futureseed_loop_case_seed52.html`. This shows why 16x16 clean is not yet the right case-level microscope: exact is still closed.

The case bank confirms the loop story concretely. Solved-by-loop boards start with about 29-32 wrong hidden cells at loop1, drop to 5-11 at loop2, reach 0-3 by loop3, and finish exact by loop5. Almost-solved and hard-failure boards show the current limit: the last 1-5 coupled wrong cells can become sticky.

## Files

- `index.html`
- `frontier_selection.json`

