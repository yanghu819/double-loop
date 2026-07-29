# Historical Experiment Reports

These reports are preserved evidence. They are not active recommendations.
Many documents intentionally describe rejected hypotheses or superseded code.

Current launch instructions live in the repository root `README.md`; current
maintenance status lives in `docs/DEPRECATIONS.md`.

The retained architecture decision and latest strict four-backbone rerun are
in `futureseed-sudoku-four-backbone-clean-rerun-20260724.md`. The original
canonicalization run remains in
`futureseed-sudoku-four-backbone-baseline-20260723.md`.

The next clean scaling test, separating nominal steps from previously unseen
independent-board coverage, is tracked in
`gdn-unseen-data-coverage-s31500-20260723.md`.

The latest FutureSeed2 mechanism boundary is documented in
`futureseed2-block-memory-20260729.md`: exact same-layer terminal-state carry
preserves first-loop opening but destroys later-loop refinement, so simple
block memory is rejected without a hyperparameter sweep.
