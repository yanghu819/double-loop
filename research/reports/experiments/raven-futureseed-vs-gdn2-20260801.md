# P-RAVEN-001: Raven + FutureSeed versus GDN2 + FutureSeed

## Mechanism hypothesis

FutureSeed passes a terminal recurrent state from an earlier depth into a later
depth, giving a causal recurrent mixer a cheap summary of future tokens. GDN2
updates a dense matrix state at every token. Raven instead learns which memory
slots a token may modify and leaves every unselected slot exactly unchanged.
If FutureSeed is limited by repeated state overwrite, sparse slot routing should
preserve its useful future summary better than GDN2.

## Prediction

At the same number of recurrent-state scalars, Raven should reduce training CE
faster or improve the official 51-55 blank exact/blank-accuracy metrics. Its
loop5 result must not merely repeat or regress loop1. A score gain bought by a
larger state does not count.

## Fairness contract

- Official FLA source: 31d15f7554bd5df05d3da6f75e09146279d2b1a8.
- Exact official classes and GSA/GDN2 CUDA kernels; backend dispatch disabled.
- GPU1 only, one seed (52), no CPU model smoke.
- Same official Sudoku arrays, source SHA, D192/L10/H6/D32 shell, optimizer,
  effective batch 128, FutureSeed1, loop5, and all-loop CE.
- GDN2 recurrent state: 32 * 32 = 1024 values per head.
- Raven recurrent state: 16 * (32 + 32) = 1024 values per head.
- Raven top-2 routing gives the paper's 12.5% write occupancy.
- GDN2 keeps its established official short convolution; this is a conservative
  comparison against the stronger baseline, not hidden parameter matching.

## Budget and kill criteria

Run one 500-step arm per architecture. Stop on source/class/kernel mismatch,
fallback, NaN, state-budget mismatch, or absent CE descent by step300. Do not
sweep slots, top-k, seed, loss, or learning rate after a negative result.

## Paper claim enabled by success

At matched recurrent memory, sparse learned allocation is a better carrier for
FutureSeed than dense delta-rule memory on a global-consistency task. This would
connect FutureSeed's cheap future context with Raven's reduced memory
interference without a task-specific rule.

## Result

Pending.
