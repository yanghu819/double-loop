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

Discarded as a Sudoku/FutureSeed carrier replacement after the single matched
GPU1 gate. Both arms used source SHA
`f2306f8cf34098e45187828755546172435bcc17`; the post-processing-only
comparator fix is commit `4181234b1dfc4ec024905d8f9581fc552af9c723`.

The first launch (`20260801T082102Z`) was externally halted when the AIStation
GPU1 lease expired before step 100. It produced no checkpoint and is not
evidence. The exact same preregistered experiment was relaunched after GPU1 was
renewed and completed both arms.

### CUDA and fairness checks

- The contract used the exact official `fla.layers.raven.Raven` and
  `fla.layers.gdn2.GatedDeltaNet2` classes from FLA SHA `31d15f7`.
- Raven ran the official sparse router and `chunk_gsa`; GDN2 ran its official
  chunk/Triton path. Silent fallback and CPU model execution were disabled.
- Raven's `(key_state, value_state)` cache was losslessly packed for
  FutureSeed and exactly reconstructed before the official layer call.
- Both arms used 1024 recurrent-state values per head, identical Sudoku data,
  outer shell, initialization seed, optimizer, effective batch, curriculum,
  FutureSeed, loop count, and all-loop CE.

### Matched results

| Metric | GDN2 + FutureSeed | Raven + FutureSeed | Raven delta |
|---|---:|---:|---:|
| parameters | 5.462M | 4.650M | -14.9% |
| train CE, step500 | 1.0186 | 1.1388 | +0.1202 |
| seconds / optimizer step | 6.041 | 7.703 | +27.5% |
| peak allocated VRAM | 8025 MiB | 7262 MiB | -9.5% |
| 46-50 blank loop5 exact | 0.7969 | 0.0000 | -0.7969 |
| 46-50 blank loop5 blank accuracy | 0.9918 | 0.7771 | -0.2147 |
| 51-55 blank loop5 exact | 0.0000 | 0.0000 | 0.0000 |
| 51-55 blank loop5 blank accuracy | 0.5257 | 0.4475 | -0.0782 |
| 56-64 blank loop5 exact | 0.0000 | 0.0000 | 0.0000 |
| 56-64 blank loop5 blank accuracy | 0.4680 | 0.4052 | -0.0628 |

On the shared primary visualization, GDN2 changes `23 -> 21 -> 22 -> 22 ->
22` wrong cells over loops 1-5; Raven changes `34 -> 31 -> 31 -> 30 -> 30`.
Raven therefore performs a small amount of recurrent correction, but starts
from a much weaker state and never solves a board in the primary mixed batch.
Its mixed exact stays `0 -> 0`, while GDN2 improves `0.0098 -> 0.0195`.

### Decision and lesson

Raven saves about 15% parameters and 9.5% allocated memory, but that saving
does not compensate for 27.5% slower training and materially worse quality.
Sparse slot preservation does not improve this short, dense global-constraint
task at matched recurrent-state capacity. The likely boundary is task fit:
Raven is designed to retain selected content over long contexts, whereas every
Sudoku cell participates in dense repeated constraints.

Do not sweep Raven slots, top-k, seed, learning rate, or loss on this result.
Retain Raven as a possible long-context retrieval carrier, not as the current
Sudoku/FutureSeed scaling backbone. The result rejects this matched Raven
replacement; it does not claim Raven is universally weaker or that its
paper-default larger slot budget cannot help retrieval tasks.
