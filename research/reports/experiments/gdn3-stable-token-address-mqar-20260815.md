# P-GDN3-046: Cross-Layer Shared Stable-Token Address

## 1. Metainfo

- Status: proposed; implementation complete, pending strict CUDA contract
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

## 8. Next Decision

If every gate passes, authorize one hard-Sudoku scale transfer of the same
shared-namespace principle. If any gate fails, close the stable-address source
transfer and the complete decoupled-key refinement line. The next work must
target a new scalable live-state organization, not another address map.

## 9. Submission Record

Not applicable. This is a local architecture science gate.
