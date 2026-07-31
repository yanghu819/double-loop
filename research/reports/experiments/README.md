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

The latest FutureSeed2 mechanism boundaries are documented in
`futureseed2-block-memory-20260729.md` and
`futureseed2-multihop-readout-20260729.md`. Exact same-layer terminal-state
carry destroys later-loop refinement; producer-compatible two-hop readout is
active but slower and weakens hard closure. Both simple time/depth radius
extensions are rejected without hyperparameter sweeps.

The latest generic memory-address tests are documented in
`gdn2-address-payload-random-order-20260730.md` and
`gdn2-address-matched-total-compute-20260731.md`. Canonical-position Q/K
substantially improves randomly ordered GDN2+FutureSeed training and produces
real later-loop correction. Its mean official hard blank advantage remains
`+0.2683` at matched step9300, but official 51-64 exact remains zero. The
separate `kda-occurrence-address-probe-20260730.md` shows that standard
occurrence rotary does not make overwritten KDA values queryable and is
rejected.
