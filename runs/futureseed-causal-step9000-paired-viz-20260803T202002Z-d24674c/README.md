# FutureSeed causal control at step 9000

This is the paired visual diagnostic for P-CAUSAL-001. Both arms use strict
official FLA GDN2, the same official Sudoku data, D192/L10/H6/D32 architecture,
loop5 every-loop CE, seed52, and exactly 9000 optimizer steps. The only
scientific difference is native FutureSeed state initialization.

On the independent fixed 512-board-per-hole gate, FutureSeed/no-FutureSeed
hard loop5 exact means are 0.2142/0.0000, blank accuracy is 0.7422/0.2837, and
train CE is 0.6524/1.5967. Hard loop1-to-loop5 blank gain is
+0.18439/-0.00033.

On the separate paired 384-board visualization pool, loop5 exact is
0.1953/0.0000 and blank accuracy is 0.7414/0.2822. Across the paired
51-55/56-60/61-64 ranges, no-FutureSeed removes only 0.55/0.20/0.18 mean wrong
cells from loop1 to loop5. FutureSeed removes 8.56/6.84/17.47 and creates exact
boards. One selected 64-blank case changes 36 -> 17 -> 6 -> 1 -> 0 wrong cells
with FutureSeed while its matched no-FutureSeed prediction stays near 50 wrong.

The causal control has now consumed 50% more optimizer steps than the frozen
FutureSeed step6000 gate and still does not reach that earlier FutureSeed
readout. This establishes a greater-than-1.5x optimizer-step compression lower
bound for the step6000 hard level. It does not replace the preregistered
step12000 frontier comparison, so training continues unchanged.

Open output/index.html for same-puzzle input, target, and loop1/3/5 boards.
