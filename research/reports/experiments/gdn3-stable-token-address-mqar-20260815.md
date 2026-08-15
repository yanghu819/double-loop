# P-GDN3-046: Cross-Layer Shared Stable-Token Address

## 1. Metainfo

- Status: complete; discarded at the fixed L1024 quality and warmed-step gates
- Task: directional MQAR, sequence length 1024, four future and four past queries
- Carrier: D128/L2/H4/K32/V32 pinned-official GDN2 plus native FutureSeed
- Fixed endpoint: 10 epochs, batch32, seed123, contemporaneous control then candidate
- External idea boundary: `yanghu819/GDN_decouple_k` at
  `c7667fd11d95d3d147f59bb3d4492989909b69ae`

## 2. Mechanism Hypothesis

The external proposal splits one key into separate write and erase keys. Native
GDN2 already has an independent read query, while its one write/erase key owns
the same state rows coherently. P-GDN3-031 and P-GDN3-036 show that adding an
erase key destroys retrieval, and P042-P045 close Q/K alignment, gauges,
frames, and whitening losses. Functional key splitting is not the answer.

The useful part of “decoupling” may instead be source separation: address
identity should not drift with the contextual hidden state that produces value
payload and gates. P-ABIND-003 is the retained positive Sudoku result: one
stable anchor residual shared by Q/K raises mean hard blank accuracy
`0.2037->0.2844`. Its registered next proof, generic associative retrieval,
was never run.

Directional MQAR supplies the sharp prediction. The same semantic key appears
at write and query positions. A token-only anchor gives those occurrences an
exact shared address residual even when their contextual hidden states and
positions differ. If occurrence drift is causal, this should reduce wrong-key
valid-value swaps and improve both directions. If the mechanism activates but
accuracy falls, stable shared addresses do not transfer and the family closes.

## 3. Exact Intervention

For input token embedding `e_t`, one cross-layer shared zero-initialized map is
used:

```text
a_t = W_shared * (e_t / rms(e_t))
q_l,t = q_native_l,t + a_t
k_l,t = k_native_l,t + a_t
```

The same `a_t` is used at both GDN2 layers and contains no positional
embedding. It is added after the native Q/K projections and Triton short
convolutions, before the unchanged in-kernel L2 normalization. Native
V/decay/erase/write/output paths, coherent key ownership, K32xV32 state,
official `chunk_gdn2`, and terminal FutureSeed are unchanged.

`W_shared: D128 -> H4*K32` adds exactly 16,384 parameters, zero recurrent
values and zero scans. Zero initialization makes the full model and every
nonzero-incoming-state layer path exactly native. There is no task rule,
selector, cache, search, repair, target-dependent address, or second key.

## 4. Independent Data Contract

- frozen directional-MQAR L1024 train/test cache and generator;
- train/test examples `10000/1000`, four key/value pairs, seed123;
- exact fixed train/test hashes;
- one saved matched initialization and one warmup-batch hash shared by arms;
- order is native FutureSeed control, then stable-token-address candidate.

## 5. Integrity Contract

Before formal training require the sole visible GPU UUID, clean detached
pushed/read-back source, pinned FLA/Zoology hashes, two official
`GatedDeltaNet2`/`ChunkGDN2FunctionBackward` layers and Triton convolutions.
The candidate must have 677,968 parameters versus 661,584 native, exactly one
shared 16,384-parameter projection, and 4,096 recurrent values per layer.

At zero initialization require exact full logits, nonzero-incoming output and
terminal state, parent tensors, parent-gradient topology, and official
backward-node count. A third byte-identical native model calibrates official
BF16/Triton replay noise on the same batch. Candidate parent-gradient max
absolute and aggregate relative RMS differences may not exceed the larger of
the fixed `0.125/0.01` bounds and `1.5x` native-vs-native replay noise. The
shared projection gradient must be finite/nonzero. With a nonzero contract
weight, repeated occurrences of one token must have address max error
`<=1e-6`, different tokens must vary, both layers must change Q/K, both must
receive the same residual tensor, and terminal states must remain finite and
board-varying. Any fallback or topology/identity miss is terminal.

## 6. Fixed Falsifiers And Gate

Endpoint activation requires one shared projection, two active layers,
projection/residual/token variation at least `1e-4`, board variation above
`1e-6`, repeated-token identity, finite variable terminal states, and native
FutureSeed on the receiving layer.

All quality checks are conjunctive:

- balanced accuracy `>=0.55` and `>=control+0.10`;
- future and past accuracy each `>=0.50` and `>=control+0.08`;
- joint exact `>=0.06` and `>=control+0.04`;
- fewer total query errors;
- wrong-key valid-value swap fraction reduced by at least `0.05`.

Any miss closes stable token-address residuals. Do not tune projection scale,
normalization, token-plus-position versus token-only, per-layer versus shared
maps, rank, seed, data, LR, loss, width, depth, or duration.

## 7. Cost And Scaling

Fixed ceilings are `<1.20x` for elapsed, post-warm wall and independently
warmed step, and `<1.12x` peak allocated CUDA memory. No result-dependent
relaxation is allowed.

## 8. Results And Decision

The R4 strict CUDA contract passed. Zero-initialized full output and both
nonzero-incoming output/state paths were bit exact, both layers retained
official `ChunkGDN2FunctionBackward`, and the candidate had exactly 677,968
parameters versus 661,584 native with unchanged 4,096-value recurrent states.
The apparent parent-gradient discrepancy was official BF16/Triton replay
noise: byte-identical native replay differed by max/relative-RMS
`0.312055/1.042848`, while candidate-parent differences were
`0.311873/1.053548`, inside the preregistered `1.5x` calibrated envelope.

| Metric | Control | Candidate | Delta |
|---|---:|---:|---:|
| Balanced accuracy | 0.36625 | 0.42450 | +0.05825 |
| Future accuracy | 0.35150 | 0.43350 | +0.08200 |
| Past accuracy | 0.38100 | 0.41550 | +0.03450 |
| Joint exact | 0.00200 | 0.01200 | +0.01000 |
| Total query errors | 2,535 | 2,302 | -233 |
| Wrong-key valid-value swaps | 1,574 | 1,717 | +143 |
| Swap fraction among errors | 0.620907 | 0.745873 | +0.124966 |

The mechanism was fully active. The shared projection weight RMS was
`0.017893`, address residual RMS/token std/board std were
`1.054060/0.103689/0.000393`, repeated-token address error was exactly zero,
and both layers changed Q/K. Native FutureSeed remained active. Stable token
identity therefore improves directional retrieval, especially future reads,
but it does not close bindings: joint exact remains low and both the absolute
wrong-key count and its error fraction worsen. This is a useful value-set
retrieval signal, not a successful memory architecture.

Elapsed, post-warm wall, independently warmed step, and peak-allocation ratios
were `0.69544/0.71670/1.37958/1.04818x`. Arm-order compilation makes the first
two ratios non-causal; the independent warmed-step benchmark is the valid
steady-state cost and misses the fixed `1.20x` ceiling. There was no NaN, OOM,
fallback, source/data drift, or infrastructure failure. Exit status 2 is the
registered science miss.

P-GDN3-046 is discarded with no Sudoku transfer. Close projection scale,
normalization, token-plus-position, per-layer versus shared maps, rank, seed,
data, LR, loss, width, depth and duration rescue. Together with P031/P036 and
P042-P045, this closes direct decoupled erase keys and nearby Q/K/address-map
refinements. A successor must alter scalable live-state organization or the
committed recurrent transition while preserving coherent ownership.

## 9. Provenance

- Pushed/read-back source: `b5a791db55cee166bfa1bdae79cde1d4e1b23dde`
- Formal run:
  `/huyang2/double-loop/runs/p-gdn3-046-stable-token-address-l1024-20260815T-r4-b5a791d`
- Contract score SHA256:
  `22e3c992f536f22b2bfd3389210e142146f2ed3f8e6a6addb518deaadde376b1`
- Endpoint score SHA256:
  `7009abf32f925df1f099af41c616327103fcf0d5ead7c89c8e1dd3616731bcbf`
- Control checkpoint SHA256:
  `59ca8e2b8aa9fc9527772526956395094888858c9ecced74569216a8ba1cd6ce`
- Candidate checkpoint SHA256:
  `2f20458d7355cd345988ec4799b0b69961f600a67406ddcb4e56b87764690c17`
- Formal log SHA256:
  `a95ddc4335caea342929c97405c546634c6ca5b1586b9e6af37168ae1a60eb80`
- Source snapshot SHA256:
  `2c17b0c0c606a76a61b5ce675a494363df2bfd6f2457c47dd8125a1c83b75e81`
