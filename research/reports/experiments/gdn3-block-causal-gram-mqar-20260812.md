# P-GDN3-026: Block-Causal Prefix-Gram Query Conditioner

## 1. Metainfo

- Status: pre-registered; not yet scored
- First decision field: directional MQAR L1024 wrong-key regime
- Fixed parent: frozen P-GDN3-020 static bounded Log-SPD
- Fixed model/data: D128/L2/H4/K32, native FutureSeed, 4 KV pairs,
  10k/1k examples, 10 epochs, batch32, seed123
- Increment over P020: 8 parameters, no payload state and no extra official scan

## 2. Evidence And Hypothesis

P020 improved the current-runtime balanced accuracy from `0.1735` to
`0.48225`, but joint exact remained `0.044` and `94.16%` of remaining errors
were correct-value/wrong-key swaps. Static full-matrix geometry helps, while
diagonal conditioners, extra state banks, correction memory, product hashes,
Raven routing and V-axis Bi-Axis variants did not close binding.

Hypothesis: collision geometry changes across a long sequence. A strictly
causal full-Gram conditioner derived from completed blocks can improve the
next block's reads without changing the canonical key/write/state frame.

## 3. Mechanism

Use fixed block size 64. Normalize raw static-Log-SPD keys, compute each
completed block Gram, and form an exclusive prefix Gram for every later block.
Let `A=(K G/tr(G)-I)/(1+||K G/tr(G)-I||_F)` and
`C=I-(tanh(a)/4)A`, with one scalar `a` per layer/head. Transform query only
through `q+q(C-I)`; keys, writes and recurrent state remain canonical. The
first block is exactly identity. Because `||A||_2<1`, factor eigenvalues stay
in `(0.75,1.25)` and condition below `1.67`.

All prefix Grams are computed before one unchanged pinned-official
`chunk_gdn2` call. Current/future blocks cannot influence an earlier block.

## 4. Novel Boundary

This is not a cache, selector, second state, Raven controller, V-lifetime
wrapper, Bi-Axis moving frame, static P020 parameter expansion, or multiple
scan. It tests whether sample- and prefix-dependent full key geometry improves
reads from the one main canonical GDN2 state.

## 5. Strict Contract

The exact pushed SHA on one A100 must prove pinned FLA/Zoology provenance,
Triton ShortConv, exactly two official GDN2 backward paths and calls, exact
P020 parent tensors, `+8` parameters, zero-init bit-exact output/nonzero-state/
FutureSeed/parent-gradient identity, finite nonzero gradients in all eight
gates, first-block identity, exclusive block causality, block-token and head
permutation equivariance, mechanism dependency, factor bounds and no fallback.

## 6. Zero-Parameter Admission

Before training, load frozen P020 checkpoint/hash and score the full-strength
conditioner on the fixed 128-example prefix without changing logits. Training
opens only if median effective-rank gain is `>=0.01`, at least `60%` of records
improve rank, median anisotropy ratio is `<=0.90`, at least `60%` improve
anisotropy, median cross-key binding-margin gain is positive, and at least
`55%` of all/future/past margins improve. Any miss closes before training.

## 7. Registered Decision Gate

If admission passes, run exactly one 10-epoch candidate from scratch. It must:

- activate all eight gates and keep factors inside fixed bounds;
- reach balanced accuracy `>=0.65` and at least `+0.10` over P020;
- reach joint exact `>=0.15` and each direction `>=0.62`;
- reduce wrong-key-valid-value swap fraction by at least `0.10`;
- keep fit, post-warm wall and warmed-step overhead below `35%`, and peak
  allocation overhead below `20%`.

Any miss closes online Gram conditioning. No block size, strength cap, seed,
LR, loss, batch, width/depth, epoch or query/key placement rescue.

## 8. Artifacts

`scripts/run_zoology_gdn2_block_gram.sh` runs contract, diagnostic admission
and at most one candidate. It archives source/config/log/score/checkpoint hashes
under `/huyang2/double-loop/runs`.

## 9. Decision

Pending the exact pushed-SHA A100 contract and diagnostic.
