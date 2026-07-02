# gdn-transition-stategeom-d224l12-hd32-s6000-20260702T1108Z-c50b48b

D224/L12 native-FutureSeed GDN state-geometry scaling probe.

The change was intentionally simple: keep `D_MODEL=224` and `LAYERS=12`, but
replace `HEADS=14, HEAD_DIM=16` with `HEADS=7, HEAD_DIM=32`. This doubles the
per-layer GDN recurrent state cells from `14336` to `28672` without adding
Sudoku rules, repair, search, selector, or a custom loss.

Result: trainable but low ROI. Step1000 holes53 loop5 exact/blank was
`0.017578125/0.52627349`, effectively tied with the cheaper H14/D16 D224
baseline at step1000 (`0.0176/0.5284`). H7/D32 used about `71.5GB` and took
about `3212s` to step1000, so it was stopped before step3000/6000.

Decision: do not run a head_dim/head_count table. The next scaling axis should
improve state efficiency or state update dynamics, not brute-force per-head
state geometry.
