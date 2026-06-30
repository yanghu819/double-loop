# EqR vs FutureSeed Same-Case Sudoku Loop Visualization

Date: 2026-06-30

Git SHA: `ee21f3b97a9dd641ef7ed553d0fa761d2ebf2086`

Run: `eqr-vs-fs-sudoku-hardcase-20260630T1836Z-ee21f3b`

Visualization: `research/reports/visualizations/eqr-vs-futureseed-sudoku-hardcase-20260630/index.html`

## Question

On the exact same hard 9x9 Sudoku case, how does the official EqR recurrent rollout change with loop count, and how does that compare to our GDN + native FutureSeed loop trajectory?

This is a diagnostic visualization, not an ablation table. The target question is whether later loops keep resolving global inconsistency, or whether they only improve local blank accuracy and then stall.

## Setup

- Case source: GDN + native FutureSeed latent-noise case bank, `official_hard_failure_01_b0022`.
- Puzzle: 58 hidden cells.
- Official EqR checkpoint: `/huyang2/double-loop/official_eqr_sudoku_repro_20260623/downloaded_checkpoints/sudoku-extreme/eqr.pth`.
- EqR runtime: official EqR repo and official venv from the previous Sudoku reproduction.
- GPU: GPU1 only.
- No repair, search, selector, oracle rollout, or hand Sudoku rule post-processing.

## Result

| model | loop1 | loop2 | loop3 | loop5 | later |
|---|---:|---:|---:|---:|---|
| Official EqR | 13 wrong / 18 conflict units | 0 wrong / valid | 0 wrong / valid | 0 wrong / valid | loop8/16/32/64 remain exact |
| GDN + native FutureSeed | 29 wrong / 27 conflict units | 16 wrong / 24 conflict units | 6 wrong / 11 conflict units | 6 wrong / 11 conflict units | not exported in this case bank |

## Insight

The gap is not just local token learning. Our FutureSeed backbone does useful work early: it cuts wrong hidden cells from 29 to 6 by loop3 on this hard case. The failure is that the later loop stops moving the global board state once it reaches a near-solution with a few mutually coupled conflicts.

Official EqR behaves differently on the same input: loop1 is imperfect, but loop2 reaches a valid exact board and stays there. That supports the current diagnosis: EqR has a training/objective/state dynamic that turns recurrent computation into global convergence, while our current FutureSeed loop mostly front-loads correction and then freezes.

## Decision

Do not add Sudoku repair or selector. The next useful experiment should target the clean mechanism gap:

- keep FutureSeed as the cheap bidirectional/context source;
- make the loop state update or training target preserve a global consistency correction path after the early easy fixes;
- compare to official EqR with same-case and aggregate exact metrics.

The most direct next probe is to align loop supervision/training pressure with EqR's behavior: later loops should be trained to reduce full-board conflict and preserve already-correct blanks, not just improve per-blank CE.

## Artifacts

- `research/reports/visualizations/eqr-vs-futureseed-sudoku-hardcase-20260630/index.html`
- `research/reports/visualizations/eqr-vs-futureseed-sudoku-hardcase-20260630/comparison.json`
- `research/reports/visualizations/eqr-vs-futureseed-sudoku-hardcase-20260630/eqr_custom_case.json`

