# FutureSeed causal control at step 6000

This is the paired visual diagnostic for `P-CAUSAL-001`. Both arms use strict
official FLA GDN2, the same official Sudoku data, D192/L10/H6/D32 architecture,
loop5 every-loop CE, seed52, and exactly 6000 optimizer steps. The scientific
difference is native FutureSeed state initialization.

The independent fixed-hole gate gives FutureSeed/no-FutureSeed hard loop5 exact
means of `0.0651/0.0000`, blank accuracy `0.6451/0.2762`, and train CE
`0.7974/1.6035`. On the separate paired 384-board visualization pool, loop5
exact is `0.0703/0.0000` and blank accuracy is `0.6335/0.2753`.

The key mechanism result is recurrent correction. Across the paired
51-55/56-60/61-64 ranges, no-FutureSeed removes only about `0.12/0.13/0.12`
wrong cells from loop1 to loop5. FutureSeed removes `7.24/3.39/8.23` and creates
complete solutions. Open `output/index.html` for same-puzzle loop1/3/5 boards.

This supports a strong finite-compute FutureSeed advantage. It does not decide
the preregistered 12k frontier claim, so the no-FutureSeed trajectory continues
unchanged to step9000 and step12000.
