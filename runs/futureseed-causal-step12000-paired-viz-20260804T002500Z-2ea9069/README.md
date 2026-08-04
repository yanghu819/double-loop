# FutureSeed causal control at step 12000

This is the preregistered endpoint for `P-CAUSAL-001`. Both arms use strict
official FLA GDN2, the same official Sudoku data and curriculum,
D192/L10/H6/D32, loop5 with every-loop CE, seed52, effective batch128, and
exactly 12,000 optimizer steps. Native FutureSeed state initialization is the
only scientific difference.

On the independent fixed 512-board-per-hole gate, FutureSeed/no-FutureSeed
hard loop5 exact means are `0.29297/0.00000`, blank accuracy is
`0.77412/0.28971`, and train CE is `0.59694/1.58121`. Hard loop1-to-loop5
blank gain is `+0.19911/+0.00083`. No-FutureSeed cannot solve even the fixed
holes50 batch, while FutureSeed solves all holes50 boards by loop5.

On the separate paired 384-board visualization pool, loop5 exact is
`0.27865/0.00000` and blank accuracy is `0.77248/0.28330`. Across paired
51-55/56-60/61-64 ranges, no-FutureSeed removes only
`0.16/0.02/0.21` mean wrong cells from loop1 to loop5. FutureSeed removes
`10.34/6.93/19.32` and creates exact boards. A selected 64-blank board changes
`39 -> 14 -> 2 -> 0 -> 0` wrong cells with FutureSeed while its matched causal
control stays at `50 -> 51 -> 51 -> 51 -> 51`.

The registered persistent-frontier criterion passes by a wide margin. The
result supports a task-local causal claim: FutureSeed opens an information and
optimization path that makes later recurrent loops useful. It does not yet
support a universal language or reasoning claim; the next useful evidence is
the same strict causal comparison on a non-Sudoku task, not longer no-FutureSeed
training or a seed table.

Open `output/index.html` for same-puzzle input, target, and loop1/3/5 boards.
Wrong cells are highlighted in red. The complete per-arm case banks are under
`output/arms/`.
