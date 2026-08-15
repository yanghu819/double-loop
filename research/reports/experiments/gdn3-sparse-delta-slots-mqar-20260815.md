# P-GDN3-050: Sparse Committed-Delta Pair Slots

## 1. Metainfo

- Status: preregistered; implementation in progress
- Task: directional MQAR L1024, four future and four past queries
- Parent: D128/L2/H4/K32/V32 pinned-official GDN2 plus native FutureSeed
- Endpoint: one candidate-only 10-epoch/batch32/seed123 run from the frozen
  P-REPRO-001 serialized initialization

## 2. Mechanism Hypothesis

P-REPRO-001 proves that the protocol is exactly reproducible and that
`1546/2024 = 76.38%` of native errors retrieve a valid value under the wrong
key. P047 can recover the value set, while P049 shows that another dense
canonical-address matrix does not restore ownership. The falsifiable P050
hypothesis is therefore that committed edits need explicit pair isolation
inside live recurrent memory.

Each layer keeps its native GDN2 state, scan and FutureSeed. In parallel, the
exact committed edit already computed by the official GDN2 forward is written
to one of 16 learned address slots through pinned-official `chunk_gsa`. Its
factorized state has a key factor `[K,16]` and value factor `[16,V]`, so slot
`m` owns both sides of one update and cross-slot key/value products never form.
The local sparse read enters through a zero-initialized per-head gate. At the
layer boundary, the factor product is normalized and enters the receiver's
native KxV FutureSeed through a separate zero-initialized gate.

This is not P025's dense correction matrix, P038's two complete dense states,
P039's Raven address controller, P047/P049's semantic/dense companion, or a
Raven replacement. It uses Raven/GSA only as a sparse pair-state organization
for exact GDN2 committed edits; the native GDN2 remains the main data plane.

## 3. Fixed Topology And Budget

- exactly one native official GDN2 scan plus one official GSA scan per layer;
- exactly 16 slots, hard top-1 forward routing with a soft straight-through
  gradient; no top-k, temperature, slot-count or router sweep;
- exactly 4,108 new parameters: two H4x16xK32 anchor tensors, eight local read
  gates and four receiving FutureSeed gates;
- exactly 4,096 additional factor-state values per layer;
- no Sudoku logic, selector, search, second seed or training-setting change;
- elapsed/post-warm/warmed-step each `<2.0x` frozen native replay B and peak
  allocation `<1.60x`.

## 4. Contract And Activation Gate

The strict single-GPU contract must prove the exact GPU UUID, pinned FLA and
Zoology SHAs, two `ChunkGDN2FunctionBackward` and two
`ChunkGSAFunctionBackward` nodes, exact parent parameters, bit-exact zero-gate
full output and nonzero incoming-main-state behavior, finite nonzero first-stage
gate gradients and opened-gate anchor gradients, head permutation equivariance,
token-order/state dependency, exact parameter/state/scan counts, bounded
`s in [0,1]`, nonpositive decay, finite state and no fallback.

The endpoint requires both layers' committed edits, local reads, factor states
and slot routing to be finite and nonzero; all 16 slots must be used, normalized
usage entropy must be at least `.50`, maximum slot usage at most `.30`, both
local gates and the one receiving sparse-FutureSeed gate at least `1e-4`, and
mean off-diagonal slot-key cosine below `.90`.

## 5. Quality Gate

Compare only with frozen P-REPRO-001 replay B, whose trained parameters and all
4,000 predictions exactly reproduce replay A. P050 passes only if all hold:

- balanced accuracy `>=.85` and at least `+.10` over `.494`;
- future and past accuracy each `>=.82`;
- joint exact `>=.60`;
- fewer total errors;
- wrong-key valid-value swap fraction falls by at least `.10`.

Any integrity, activation, quality or cost miss closes sparse delta slots. No
slot count, routing, gate, anchor, normalization, seed, LR, loss, batch, width,
depth or duration rescue is authorized. A pass admits one Sudoku transfer; a
miss redirects to a genuinely different live pair-memory operator.

## 6. Results

Pending the one registered contract and endpoint.

## 7. Decision

Pending.
