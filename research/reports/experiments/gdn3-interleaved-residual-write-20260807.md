# P-GDN3-010: Interleaved Residual Write

## 1. Metainfo

- Status: discarded at strict CUDA contract
- Date: 2026-08-07
- Branch: `codex/gdn3-interleaved-write-20260807`
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
- Frozen control: `p-fs3-001-terminal-s3100-20260806T200859Z-3e167b6`

## 2. Evidence Boundary

P-GDN3-005 changes gate coherence, P006 adds a separate state expert, P007
conditions precomputed update tensors on inherited state, and P008 adds a
post-scan transition. All activate without sufficient exact closure. P009 then
tests a stable transform between two external official calls; forward identity
passes, but the required two-call terminal-state backward chain is absent.

The remaining high-information boundary is update rank *inside one
differentiable official call*. Standard GDN2 performs one erase/write event per
logical token. No completed experiment gives the same state a second learned
write subspace per token while preserving the pinned official recurrence and a
single autograd function. P010 tests that capacity rather than another state
bank, readout residual, gate prior, cache bridge, or Sudoku-specific operation.

## 3. Mechanism Hypothesis

Hard closure may require storing two competing or complementary payloads at a
token before the ordinary erase/write update commits the state. A single
rank-one write forces both payloads through one address/value pair. A learned
auxiliary write immediately followed by the unchanged parent update should add
general state-update rank while letting the parent's erase and decay regulate
the new information locally.

The falsifiable prediction is that the auxiliary V path receives gradient at
exact identity, its independent K residual receives gradient after a synthetic
V opening, all 12 layers produce board- and token-varying auxiliary writes, and
state magnitude remains bounded. Active auxiliary writes without a registered
exact gain close the mechanism. Failure of identity, single-call official
provenance, gradients, or stability closes it before science.

## 4. Candidate

For each original token, construct two adjacent microsteps inside one call to
the unchanged pinned `chunk_gdn2`:

1. the auxiliary microstep reuses the parent Q and K plus a learned independent
   K residual, uses learned auxiliary V, and fixes `g=0`, `b=0`, `w=1`;
2. the original parent Q/K/V/decay/erase/write microstep follows unchanged;
3. only outputs from the original microsteps are retained.

At migration, per-layer `D256->K256` residual and `D256->V256` auxiliary
projections are zero. Auxiliary V is therefore exactly zero, making the first
microstep the identity transition
`S_aux = S + k_aux * 0^T = S`; the subsequent parent update sees the exact
parent state. The V projection has a first-stage gradient through the nonzero
parent K. After synthetic V opening, the K residual obtains a second-stage
gradient. The 162 physical microsteps remain in one official autograd function,
so P009's external initial-state bridge is not used.

This adds exactly `12 * 2 * 256 * 256 = 1,572,864` parameters and no recurrent
state. It doubles physical transition count but does not add another core,
reverse scan, search, rule, selector, repair, or task-specific feature. There
is one fixed auxiliary write; rank, ordering, gates, and microstep count are not
tunable follow-ups.

## 5. Frozen Configuration

- D256/L12/H8/K32/V32, channel multiplier4
- position-QK addressing and the registered random traversal
- native terminal FutureSeed, loop5, equal CE at every loop
- 12 pinned official-FLA GDN2/Triton layers
- one zero-init auxiliary K residual and V projection per layer
- auxiliary microstep immediately before each original update
- official full-diversity curriculum and exact step3000 parent
- BF16, effective batch128, optimizer/RNG/data order exact resume
- LR0.0015, weight decay0.001, seed52
- candidate-only step3000-to3100 continuation

The frozen control is not rerun. There is no rank, microstep order/count,
auxiliary gate/decay, projection sharing, scale, seed, optimizer, loss, batch,
width/depth, or duration follow-up.

## 6. Launch Gates

Before formal execution, one strict GPU1 CUDA contract must prove:

1. exactly CUDA index0 and the registered UUID;
2. pinned FLA source SHA `9c8e42e762fce087c27b673af4922795d9edb85e`;
3. exactly 12 official GDN2 layers and 12
   `ChunkGDN2FunctionBackward` graphs, with one 162-step call per layer and no
   fallback;
4. exact 1,572,864-parameter migration and no new recurrent state;
5. bit-exact full output and all 12 terminal states at zero auxiliary V,
   including synthetic nonzero incoming states, against the 81-step parent;
6. finite nonzero first-stage gradients in all 12 auxiliary V projections and,
   after synthetic V opening, finite nonzero gradients in all 12 K residuals;
7. exact auxiliary identity at zero V and changed state/output when V opens or
   K residual/source is shuffled;
8. head permutation equivariance, finite tensor geometry, and no accidental
   dependency on Sudoku rules or traversal boundary;
9. exact-resume optimizer/RNG/data-order migration and complete step3001
   checkpoint plus metrics JSON;
10. no NaN, OOM, source/data drift, hidden CPU model path, or concurrent GPU
    process.

Any contract or production-probe miss closes this implementation. It does not
authorize tolerance relaxation, a second autograd call, custom backward,
different microstep order/gates/count, or projection/rank rescue.

## 7. Science, Stability, And Cost Gates

At step3100, activation and stability require all of:

- exactly 12 auxiliary write paths active;
- auxiliary V and K-residual RMS `>=1e-4`, with finite board/token/head
  variation;
- auxiliary state-write relative RMS `>=1e-4` and finite;
- terminal-state RMS finite at every loop and no more than `4x` the frozen
  control at the matching loop;
- no fallback.

Quality passes by exactly one route:

1. hard51-64 macro loop5 exact improves by at least `+0.02`, with every hard
   range losing no more than `0.01` blank accuracy; or
2. mixed loop5 exact improves by at least `+0.03`, official61-64 does not
   regress, and same-board loop3-to5 wrong-cell correction is stronger.

The physical transition count doubles inside the same official core. Before
any result, warmed elapsed overhead is capped below `120%` and peak allocated
memory overhead below `80%` versus the frozen control. OOM, unstable timing, or
either ceiling miss kills the run regardless of quality.

Any activation, stability, quality, or cost miss discards this mechanism. Do
not rescue rank, auxiliary order/count/gates/decay, projection sharing or
scale, seed, LR, loss, batch, model width/depth, or duration.

## 8. Required Readout

Report and archive mixed and official51-55/56-60/61-64 loop1-5
exact/blank/wrong cells; train CE and same-board correction; auxiliary K/V and
state-write activation; terminal-state geometry; independently warmed
throughput, peak allocation/reservation and timing; config/source/parent/
checkpoint/metrics/log hashes; and same-board loop1-5 visualization.

## 9. Decision

Discard. Complete source SHA
`1c3c3e7f2ae244f611cc890250721d6115348c08` was pushed and read back, and the
clean detached worktree passed source, GPU UUID and exclusivity preflight. The
strict R1 CUDA contract then rejected the candidate before gradients or a
step3001 probe: with both auxiliary projections exactly zero, the 162-step
single-call path changed the parent model output by max absolute
`0.04559326171875` relative to the 81-step official call. This violates the
registered bit-exact parent migration, even though the added write payload is
zero.

Contract log:
`/huyang2/double-loop/artifacts/launch/p-gdn3-010/contract-1c3c3e7-r1.log`
(`SHA256 840b156267676d0ce6f019bfe51f7fc7d81fab4c8c0ae1ea31ccbe9fe713988b`).
Abort record:
`/huyang2/double-loop/artifacts/launch/p-gdn3-010/contract-1c3c3e7-r1.abort.json`
(`SHA256 fae2959fd0de398ec11c8e5bcb7f62eb34e1635ac43eb2bd76f4c2b2cfe42985`).
The process exited naturally with status1 and GPU1 returned to 0 MiB with no
compute application. No exact-resume probe, continuation, score or visualization
exists. Do not rescue auxiliary gate/decay, order/count, rank, projection,
scale, tolerance, seed, LR, loss, batch, width/depth, or duration.
