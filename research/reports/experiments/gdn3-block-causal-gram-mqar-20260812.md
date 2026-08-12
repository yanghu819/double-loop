# P-GDN3-026: Block-Causal Prefix-Gram Query Conditioner

## 1. Metainfo

- Status: discarded at the frozen diagnostic admission; no candidate training
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

The terminal formal run is
`p-gdn3-026-block-gram-l1024-20260812T124430Z-af732ef` on task-mode
`NVIDIA A100-SXM4-80GB`, CUDA index 0, UUID
`GPU-0da20a4f-5e67-e47d-7aab-8c6efa2864ad`. Exact source
`af732ef26fce4d87d60e88e9523a100cd3b52ada` was pushed and read back before
the clean detached run. Key SHA256 values are:

- score: `bf4be80fd8901904473acbba6fdc5792dff1e0ddfbde93ea0f7095b18edfceb8`
- contract: `1797ce2cb18636826dd08851b6e86dbd31b3ca79eeb3b843bac59deccaee6e1e`
- formal log: `f5dcf286f9b8a484cf18de9587ed33d49fd1b76c9e76bf727e40c9026bcd0729`
- source snapshot: `d30e2fcc80e748851f282ba1bc055441d4f845eb52c8b8439c63265377d46146`
- artifact manifest: `8cbdef1cb3ad84ad75b30774b61a46e3e68c8a8f2c124951010015319d81c89d`

Two earlier executions consumed no diagnostic or training budget. Source
`54e6710` exited before model construction because `ZOOLOGY_ROOT` was not
exported; source `925498b` then passed the full CUDA contract but exited before
the diagnostic because a hand-transcribed frozen swap fraction lacked the
precision of the SHA-locked `1950/2071` evidence. Their non-science aborts are
retained rather than counted as experiment arms.

## 9. Decision

Discard. The terminal strict CUDA contract passes all registered mechanical
claims: one visible A100; pinned FLA/Zoology provenance; Triton ShortConv;
exactly two official calls/backwards; exact parent output, incoming-state and
gradient identity; exactly eight new parameters; all eight finite nonzero
gate gradients; block causality and token/head equivariance. An opened factor
is nontrivial but bounded, with eigenvalues `0.9283..1.0369`, condition
`1.1131`, and mean output delta `0.12288`.

The frozen 128-example, 2-layer, 15-active-block diagnostic rejects the
quality premise before training:

- effective-rank fraction median gain is `-0.00001999`, versus required
  `+0.01`; only `39.95%` improve versus `60%`;
- anisotropy median ratio is `1.000088`, versus required `<=0.90`; only
  `40.09%` improve versus `60%`;
- binding-margin median gain is exactly `0`; only `36.57%` improve versus
  `55%`;
- future/past improved fractions are `31.05%/42.09%`, both below `55%`.

The intervention was evaluated at full fixed strength, remained stable
(factor eigenvalues `0.7616..1.0138`, condition `1.3232`) and did not alter
logits, so this is a direct falsification rather than failed optimization.
No 10-epoch candidate, checkpoint, cost score or Sudoku transfer was run.

Close block size, strength, factor map and query/key placement rescues. Static
P020 remains a useful signal, but online prefix-Gram geometry is not the
missing binding mechanism. Combined with the failed caches, banks, correction
state, hashes, Raven controllers and Bi-Axis variants, the next research move
should target loop dynamics/training signal with a contemporaneous runtime
control, not another address/state wrapper.
