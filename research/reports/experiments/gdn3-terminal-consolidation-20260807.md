# P-GDN3-008: Terminal-State Consolidation Sweep

## 1. Metainfo

- Status: approved; implementation and launch gates pending
- Date: 2026-08-07
- Branch: `codex/gdn3-terminal-consolidation-20260807`
- Benchmark: official/full-diversity hard 9x9 Sudoku 51-64 blanks
- Compute: AIStation task-mode GPU1 only
- GPU UUID: `GPU-53e9f3b4-2966-65d3-6614-09c540921519`
- Seed: 52 only
- Parent: D256/L12/H8/K32/V32 position-QK GDN3 plus native terminal FutureSeed
- Parent checkpoint:
  `/huyang2/double-loop/models/gdn3-position-qk-d256l12-s12000-20260806T131527Z-9f2ee8d/checkpoints/train_state_step003000.pt`
- Parent checkpoint SHA256:
  `6339c3cb2b5fc5230a581d6633716483e35ff8e4522f06a9d7aaf26512f023da`
- Parent source SHA: `9f2ee8d1738032bc5f09b55db0b81d507780b376`
- Frozen control:
  `p-fs3-001-terminal-s3100-20260806T200859Z-3e167b6`

## 2. Evidence Boundary

P-FS3-001/002/003 and P-GDN3-005/006/007 all activate their registered
mechanisms without improving hard exact. They close analytic FutureSeed
residuals, learned producer compression/routing, scalar erase/write coupling,
an independent recurrent expert, and a controller that reads the incoming
layer state before the official scan.

The source audit exposes one untested boundary. P007 computes
`Q_unit @ S_in` once from the state received from the previous layer, then
precomputes all K/V/erase/write residuals before the token scan. It does not
allow the current layer to revise its own terminal state after that state has
seen the complete sequence. P008 tests that missing transition operation. It
is not another transfer-content map, output readout, parallel state, scalar
prior, or Sudoku-specific bias.

## 3. Mechanism Hypothesis

A single causal GDN2 pass must simultaneously accumulate evidence and commit
the memory that native FutureSeed transfers to the next layer. Early writes
cannot use evidence that arrives later in that same pass. If hard closure is
limited by this commit order, a second same-order state transition initialized
from the first terminal state should consolidate globally informed memory and
improve late-loop board correction.

This predicts a nonzero, board- and token-varying correction address; a
nonzero change in terminal state after the correction sweep; and stronger
loop3-to5 hard-board correction. If the transition is active but exact and
same-board correction do not improve, post-scan consolidation is not the local
bottleneck and the mechanism is closed without changing its source, scale,
scan count, or duration.

## 4. Candidate

The parent position-QK official-GDN2 pass is unchanged and still supplies the
model output. On producer layers0-10, its token outputs before output
normalization feed one per-layer, head-shared, zero-initialized V32-to-K32
linear map. This produces `K_corr` for one second pinned-official
`chunk_gdn2` transition that:

- starts from the first pass terminal state;
- reuses the first pass V payload and erase/write gates;
- uses zero correction decay;
- runs in the same traversal order;
- returns only the consolidated terminal state for native FutureSeed.

At migration `K_corr=0`, so every correction update is exactly zero and the
second terminal state equals the first terminal state. The model output is
never replaced or augmented by correction output. After learning, each token
edits a state that already contains the complete first-pass sequence, and the
correction state evolves token by token within the official recurrence.

Layer11 has no downstream receiver and therefore has no refiner or unused
parameter. The mechanism adds exactly `11 * 32 * 32 = 11,264` parameters, no
new persistent state, no reverse scan, no second independent core, no search,
and no Sudoku logic. It adds one official correction scan in each of 11
producer paths. This fixed operation is the candidate itself, not a scan-count
or model-depth sweep.

## 5. Frozen Configuration

- D256/L12/H8/K32/V32, channel multiplier4
- position-QK addressing, random traversal
- native terminal FutureSeed, loop5, equal CE at every loop
- 12 pinned official-FLA GDN2/Triton parent layers
- 11 head-shared V32-to-K32 correction projections
- one same-order correction transition on producer layers0-10
- official full-diversity curriculum and exact step3000 parent
- BF16, effective batch128, optimizer/RNG/data order exact resume
- LR0.0015, weight decay0.001, seed52
- candidate-only step3000-to3100 continuation

The frozen control is not rerun. There is no correction source, output scale,
scan count, layer subset, seed, optimizer, loss, batch, width, depth, or
duration follow-up.

## 6. Launch Gates

Before formal execution, one strict GPU1 CUDA contract must prove:

1. exactly CUDA index0 and the registered UUID;
2. pinned FLA source SHA `9c8e42e762fce087c27b673af4922795d9edb85e`;
3. 12 parent and 11 correction official `chunk_gdn2` paths, with 23
   `ChunkGDN2FunctionBackward` graphs in a full native-FutureSeed pass;
4. exact 11,264-parameter migration and no unexpected parameter/state;
5. bit-exact full output and all 12 terminal states at zero correction K;
6. per-layer exact output/state identity under synthetic nonzero incoming
   state, including correction paths0-10;
7. finite nonzero gradients in all 11 correction projections from downstream
   native FutureSeed;
8. finite nonzero correction address, board/token variation, terminal-state
   residual, and within-sweep state progression after a synthetic opening;
9. head-permutation equivariance and changed terminal state when first-pass
   token outputs or terminal state are shuffled across boards;
10. exact-resume optimizer/RNG/data-order migration and complete step3001
    checkpoint plus metrics JSON;
11. no fallback, NaN, OOM, source/data drift, or hidden CPU model path.

Any contract or production-probe miss closes this implementation. It does not
authorize an output residual, projection-source change, fewer/more sweeps, or
layer subset.

Contract R1 on pushed SHA
`6ea0a12c86edf07854416cadda30c6990a336c61` exited before exercising the
candidate transition. The diagnostics dictionary referenced
`terminal_consolidation_diag` in the position-QK path, while the call producing
that value had been inserted by an ambiguous patch anchor into the mutually
exclusive fast-slow path. The resulting `NameError` occurred on the first
control forward; no identity, official-kernel, gradient, activation, science,
or cost assertion ran or failed. PGID28464 exited naturally, allocations
cleared, and the log plus non-science abort are retained under
`/huyang2/double-loop/artifacts/launch/p-gdn3-008/`. R2 moves only that existing
call to its registered position immediately after the first position-QK
official chunk. The mechanism, parameters, parent, configuration, prediction,
and every gate remain unchanged.

## 7. Science And Cost Gates

At step3100, activation requires all of:

- exactly 11 producer correction paths active;
- correction K relative RMS and board/token variation each finite and
  `>=1e-4`;
- terminal residual relative RMS and board variation finite and `>=1e-4`;
- finite nonzero within-sweep state progression and correction-projection RMS;
- no fallback.

Quality passes by exactly one route:

1. hard51-64 macro loop5 exact improves by at least `+0.02`, with every hard
   range losing no more than `0.01` blank accuracy; or
2. mixed loop5 exact improves by at least `+0.03`, official61-64 does not
   regress, and same-board loop3-to5 wrong-cell correction is stronger.

The second official state transition is the intended compute cost. Before any
result, warmed elapsed overhead is capped below `120%` and peak allocated
memory overhead below `80%` versus the frozen control. OOM, timing instability,
or either ceiling miss kills the run regardless of quality.

Any activation, quality, or cost miss discards this mechanism. Do not rescue
projection source/scale, correction decay/gates, scan count, layer subset,
seed, LR, loss, batch, model width/depth, or duration.

## 8. Required Readout

Report and archive:

- mixed and official51-55/56-60/61-64 loop1-5 exact/blank/wrong cells;
- train CE and same-board loop3-to5 correction counts;
- correction address, board/token variation, terminal residual, state
  progression, and projection-weight activation;
- independently warmed throughput, peak allocation/reservation, and timing
  stability;
- config, source, parent, checkpoint, metrics, log, and comparison hashes;
- same-board loop1-5 visualization using frozen-control case IDs.

## 9. Decision

Pending implementation, pushed-source contract, and exact step3001 probe.
