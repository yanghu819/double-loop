# FutureSeed cheap bidirectional RWKV9 probe

Hypothesis: FutureSeed provides a cheap future/noncausal boundary condition for a causal RWKV recurrence.

Protocol: paired GPU1 run at source SHA `955e266`, same CUDA RWKV7 state-passing backbone and curriculum, seed `52`, 800 train steps, 9x9 random holes, batch 192, `MAX_LOOPS=5`. The only intended mechanism difference was `FUTURE_SEED_SCALE=0` vs `FUTURE_SEED_SCALE=1`.

Provenance note: the run metadata reports `git_dirty=true` because the experiment tracker wrote leaderboard/visualization artifacts during recording. The model source was launched from detached SHA `955e266ca018744d7e92d39216e30c69c4840230`; no model-code patch was part of the intended comparison.

Result: strongly positive on this 9x9 proxy.

- no-FS loop5 h12 exact: 0.0156; early/late blank acc: 0.4697 / 0.8382; gap -0.3685
- FS loop5 h12 exact: 0.9492; early/late blank acc: 0.9947 / 0.9955; gap -0.0008
- loop5 exact gain: +0.9336
- absolute early/late gap reduction: +0.3677
- step800 train CE: no-FS 0.4779, FS 0.0097
- train time: no-FS 467.0s, FS 459.1s

Mechanism read: no-FS learns late cells much more easily than early cells, consistent with causal recurrence lacking future clues. FutureSeed nearly removes that position bias and makes loop1 already strong, then loop3/5 refine further.

Decision: keep FutureSeed as the main module. The next fair test is to port the same diagnostic to the official EqR code path or to a causal Maze backbone, not to sweep Sudoku seeds.

Boundary: this supports the FutureSeed-as-cheap-bidirectional-context story on a Sudoku proxy. It is not an official EqR Maze result.
