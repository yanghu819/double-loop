# P-GDN3-040: Linear + Exact Product Direct-Sum State

## 1. Metainfo

- Status: preregistered; implementation pending strict CUDA contract
- Date: 2026-08-15
- Benchmark: directional MQAR L1024 wrong-key regime
- Fixed setting: D128/L2/H4/V32, native FutureSeed, 10 epochs, batch32,
  seed123
- Arms: contemporaneous native K32 GDN2 control, then one K96 candidate
- Resource: one task-mode GPU, no concurrent model process

## 2. Evidence Boundary

P031/P036 show that decoupling erase and write keys destroys coherent row
ownership. P032 shows that replacing K32 with a wider native K64 address does
not by itself improve binding. P038 shows that routing whole tokens into two
full states lowers conditional swaps but hurts general retrieval. P024's
degree-two dual hash also failed, but it replaced the native linear address
with a compact analytic product and therefore did not test a complementary
product state.

P040 preserves the complete learned K32 read/erase/write map byte-for-byte and
adds an exact product-address block in the same recurrent state. It tests one
remaining state-organization question: whether linear addresses solve ordinary
retrieval while an uncompressed second-order block disambiguates bindings that
otherwise collide. This is not a cache, router, parallel expert, second scan,
decoupled key, or Sudoku-specific mechanism.

## 3. Mechanism

For every head and token, the existing projected K32 query/key is normalized
as `q0,k0`. The first 16 raw coordinates are split into two independently
normalized K8 factors `(q1,q2)` and `(k1,k2)`. The candidate constructs

`phi(q) = [q0, vec(q1 outer q2)]`,

`phi(k) = [k0, vec(k1 outer k2)]`.

Both blocks have unit norm. The first 32 coordinates are exactly the native
address; the extra 64 coordinates are the full, uncompressed product basis.
One pinned-official `chunk_gdn2` scan operates on the coherent K96xV32 state.
Its explicit read scale remains `1/sqrt(32)`, so the native block contributes
exactly the original read and the product block is an additive complement.

Native decay and erase coordinates are unchanged. Product decay is the mean
of its two factor log-decays and product erase is their geometric mean. Both
remain bounded. V, write gate, output gate/projection, short convolutions and
native FutureSeed are unchanged. FutureSeed transports the entire K96xV32
terminal state. The candidate adds zero parameters, 8,192 state values/layer,
zero scans and no selector or auxiliary objective.

## 4. Falsifiable Prediction

If L1024 errors reflect pairwise address collisions that a linear feature map
cannot separate, the product block should stay active and board/token-specific,
reduce wrong-key valid-value swaps by at least 0.10, reduce total errors, and
raise both future and past retrieval. It must reach balanced accuracy at least
0.85, exceed the locked 0.7475 baseline and its contemporaneous control by at
least 0.10, and reach joint exact at least 0.60.

If the product state activates but misses any quality gate, exact second-order
address complement is closed. Do not rescue factor dimension, direct-sum
weight, scale, gate lift, seed, LR, loss, batch, width, depth or duration. If
it broadly collapses retrieval while conditional swap fraction falls, count
that as failure, not binding improvement.

## 5. Contract And Matched Initialization

Before formal training, exact pushed source in a clean detached worktree must
prove on the selected single GPU:

1. exact CUDA index0/UUID, pinned FLA and clean Zoology provenance;
2. equal control/candidate parameters and byte-identical mapped parent tensors;
3. exactly two official GDN2 modules, six Triton short convolutions and two
   `ChunkGDN2FunctionBackward` paths;
4. exact K32 native Q/K, decay and erase sub-block identity;
5. unit native/product blocks and the identity
   `phi(q).phi(k)=q0.k0+(q1.k1)(q2.k2)` within BF16 tolerance;
6. head-permutation equivariance and nonzero dependence on each product factor;
7. exact K96xV32 terminal-state shape, product-only incoming-state dependence,
   finite full logits and no fallback;
8. finite nonzero gradients through factor rows, native-only rows, V and write
   projections in both layers;
9. exact train/test hashes and fixed matched initialization.

## 6. Activation And Quality Gates

Both layers must report one official scan, K32+K64 topology, 12,288 state
values/layer and zero new parameters. Product Q/K RMS, token variation and
board variation must be nonzero. Product/native state and read RMS ratios must
be finite in `[1e-4,1e4]`, terminal state RMS in `[1e-4,1e4]`, decay nonpositive,
erase in `[0,1]`, and native FutureSeed active.

All quality checks are binding:

- balanced accuracy at least 0.85;
- balanced gain at least +0.10 versus both 0.7475 and runtime control;
- future and past accuracy each at least 0.85 and runtime control +0.10;
- joint exact at least 0.60 and runtime control +0.10;
- fewer total errors;
- wrong-key valid-value swap fraction among errors lower by at least 0.10.

## 7. Cost And Kill Gates

State values and key-axis kernel work are exactly 3x the K32 control. Elapsed,
post-warm wall and independently warmed-step ratios must each remain below
3.25x; peak allocation must remain below 2.50x. These limits are fixed before
the result. Any integrity, activation, quality or cost miss closes P040 and
writes a structured abort. No nearby rescue is authorized. Only a full pass
permits one matched hard-Sudoku transfer.

## 8. Required Readout

Archive future/past/balanced/joint metrics, CE curves, total errors, wrong-key
swaps, direct-sum algebra errors, product address/state/read geometry, native
FutureSeed diagnostics, parameter/state/scan counts, elapsed/wall/warmed/
allocation ratios and source/config/checkpoint/score/log hashes.

## 9. Decision

R1 source `09c1aed9` stopped in the strict contract after the full CUDA
forward, K96 state-dependency path and official backward graph had executed.
The final gradient audit incorrectly accessed official GDN2 `f_proj` as one
linear layer, while the pinned implementation exposes a two-linear
`Sequential`. It raised `AttributeError` before writing `contract.json`.
`abort.json` records `phase=contract` and `scientific_failure=false`; GPU memory
and processes cleared. R2 changes only the checker locator from
`f_proj.weight` to `f_proj[1].weight`. Mechanism, data, prediction, quality,
cost and kill gates remain byte-for-byte unchanged. No quality result exists.

Pending strict R2 CUDA contract and the single fixed matched endpoint.
