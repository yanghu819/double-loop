# P-GDN3-067 Erase-then-Delta FutureSeed

## 1. Metainfo

- Plan: `P-GDN3-067`
- State: approved; implementation in progress
- First decision field: fixed directional MQAR L1024/K4
- Resource: one AIStation A800 80GB, CUDA index 0 only
- Frozen reference: `P-GDN3-059` Momentum DeltaNet + native `[S,M]`
  FutureSeed

## 2. Evidence And Hypothesis

P059 leaves 223 errors, including 151 valid-value/wrong-owner swaps. Direct
erase/write-key decoupling in P031 activated strongly but destroyed retrieval:
the independent erase direction replaced the working same-owner correction and
became almost orthogonal to the write key. P029's unconstrained two-write
GatedDeltaProduct also failed. P041 already tests the published Q-Delta
`k + lambda q` mixed prediction address and collapses, so renaming that family
is not a new experiment. P066 shows that a stable global least-squares state
also loses directional ownership.

The untested intervention is a structured two-stage edit: erase stale memory
at an independently learned address, then retain the complete standard delta
correction at the current owner key. The independent path can remove an old
owner without replacing the write/read coordinate. If this extra cleanup is
the missing operation, it should reduce P059's adjacent-owner tail while
preserving both retrieval directions. If the mechanism activates but does not
beat P059, address-level erase-before-write is closed in this regime.

Primary equation source: Erase-then-Delta Attention, arXiv `2606.26560`.
The user-provided `yanghu819/GDN_decouple_k` and the paper are mathematical
inspiration only; no code is copied. Production execution uses the pinned MIT
FLA `chunk_gated_delta_product` operator at SHA `9c8e42e...d85e`.

## 3. Fixed Recurrence

For each token/head, let `D=Diag(exp(g))`, normalized erase/write addresses be
`e,k`, scalar gates be `gamma,beta in (0,1)`, and payload be `v`. Apply

```text
S_decay = D S_(t-1)
S_erase = (I - gamma e e^T) S_decay
S_t     = (I - beta k k^T) S_erase + beta k v^T
o_t     = q^T S_t / sqrt(K)
```

This is exactly two official product microsteps per token:

1. `(key=e, value=0, gate=gamma)`;
2. `(key=k, value=v, gate=beta)`.

Decay is applied once before the pair and readout occurs only after the second
microstep. The first payload is structurally zero, not a learned value. The
candidate has one K32xV32 state per head/layer, no side state, cache, selector,
second pass, Sudoku logic or fallback. Native FutureSeed transports the whole
terminal state to the adjacent receiver after per-head RMS normalization and
one learned gate.

## 4. Why Prior Failures Do Not Cover It

- P031 replaced the standard correction's erase address. P067 preserves the
  complete same-key delta correction and prepends a pure erase.
- P029 learned two arbitrary keys, values and writes. P067 fixes the first
  payload to zero and gives the pair explicit erase-then-correct semantics.
- P041 mixes `q` into the live prediction address. P067 never uses query as a
  write or erase controller.
- P005-P008 and later wrappers operate before/after the scan or in side state.
  P067 changes every live token transition inside the official product scan.
- P066 globally solves a normal equation. P067 preserves local sequential
  owner edits and a fixed-size KxV state.

This is a foundational recurrence and therefore trains from scratch on the
validated binding-error regime. A 100-step Sudoku graft is not a valid first
verdict.

## 5. Fixed Protocol

- Model: D128/L2/H4/K32/V32, output gate enabled
- State: 4,096 values/layer
- Operator: pinned official `chunk_gated_delta_product`, exactly two ordered
  microsteps/token, scalar positive erase/write gates, one forget gate
- FutureSeed: one adjacent full-state route, scale 1
- Data: fixed mixed-direction MQAR L1024/K4; 10,000 train, 1,000 validation
- Train: 10 epochs, batch32, seed123, inherited AdamW/LR/WD/schedule
- Initialization: from scratch under the same deterministic initializer/data;
  no control rerun and no parent checkpoint graft
- Sweep: none

## 6. Strict CUDA Contract

The exact pushed clean worktree must prove on the sole A800:

1. target GPU UUID, pinned FLA SHA/files and no compute overlap/fallback;
2. exactly two official `ChunkGatedDeltaProductFunctionBackward` paths;
3. exact two-microstep order, first payload identically zero, second payload
   learned, one decay application and one K32xV32 terminal state;
4. official output/state parity to an explicit FP32 EDA recurrence with a
   nonzero incoming state, relative RMS <=0.03;
5. finite nonzero gradients for Q, both key addresses, V, gamma, beta, decay,
   output path, incoming state and the receiver FutureSeed gate;
6. incoming-state dependence, transform-order dependence and head permutation
   equivariance <=0.005;
7. exact parameter/state accounting and production-length finite forward and
   backward.

## 7. Registered Decision Gates

Activation/stability, all required:

- two active layers, exactly one active native FutureSeed route;
- erase/write key relative RMS >=0.10 and mean absolute cosine <=0.98;
- both gate means in `[0.01,0.99]` with token variation >=1e-4;
- erase effect relative RMS >=1e-4, first payload max absolute value exactly 0;
- terminal state RMS in `[1e-4,1e4]` with board variation >1e-6.

Quality versus frozen P059, all required:

- balanced accuracy >=`.94925` (P059 +`.005`);
- future and past accuracy each regress by at most `.003`;
- joint exact >=`.829` (P059 +`.005`);
- total errors <=200;
- wrong-key valid-value swaps <=120 and swap share <=`.60`.

Cost versus P059, all required:

- elapsed, post-warm wall and independent warmed step <=2.25x;
- peak training CUDA allocation <=1.50x.

Any integrity, activation, stability, quality or cost miss closes this exact
EDA configuration. No erase/write gate, key tie/angle, initialization, step
order, scalar/vector gate, product count, FutureSeed gate, seed, LR, loss,
batch, width, depth or duration rescue is authorized.

## 8. Artifacts And Decision

Pending exact pushed source, clean detached worktree, strict contract and one
fixed endpoint.

## 9. Submission Record

Not applicable.
