# P-GDN3-047: Binding-Certificate Companion State

## 1. Metainfo

- Status: complete; discarded at the fixed quality/activation gate
- Task: directional MQAR, sequence length 1024, four future and four past queries
- Carrier: D128/L2/H4/K32/V32 pinned-official GDN2 plus native FutureSeed
- Fixed endpoint: 10 epochs, batch32, seed123, contemporaneous control then candidate
- Decision target: reduce correct-value/wrong-key binding errors before any Sudoku transfer

## 2. Mechanism Hypothesis

P-GDN3-046 shows a real but incomplete effect. A cross-layer stable token address
improves balanced accuracy `.36625->.42450` and reduces total errors by 233, yet
wrong-key valid-value swaps rise `1574->1717`. A paired transition audit shows
1,059 wrong queries become correct while 826 correct queries become wrong. Stable
identity therefore helps recover the value set but does not tell the downstream
network which semantic key actually owned the retrieved value.

The falsifiable hypothesis is that GDN2 needs a binding certificate alongside
its normal value state. The main state should remain the only predictor memory;
an auxiliary state should store the stable token identity committed under the
same address and gates. A query can then retrieve both the value and evidence of
the key that produced it. The existing residual stream and MLP contain the query
identity, so they can learn a generic consistency test without a task selector.

This boundary is not covered by prior failures. P025 stores committed value edits
in a compressed correction bank. P013 duplicates the full value payload and has
a serially dead learned-address path. P006 learns an independent recurrent expert.
P038 routes values among full states, and P-CAUSAL-014 only widens V. None stores
canonical key identity in a state aligned exactly to the native main address.

## 3. Exact Intervention

For each layer and token, compute the unchanged native tensors `q,k,v,g,b,w`.
The first pinned-official scan is the exact parent:

```text
(o_main, S_main) = GDN2(q, k, v, g, b, w, S_in)
```

From the input token embedding `e_t`, form a position-free stable payload:

```text
z_t = reshape(e_t / rms(e_t), H4, V32)
(o_cert, C) = GDN2(q, k, z_t, g, b, w, 0)
o = o_main + tanh(a_layer,head) * (o_cert / rms_V(o_cert))
```

The original output norm/projection then consumes `o`. Each layer has four
zero-initialized scalar logits, so the full model adds exactly 8 parameters.
The certificate adds one transient H4xK32xV32 state, 4,096 values per layer, and
one official scan per layer. It is rebuilt within each layer and is not carried
by FutureSeed or across macro loops. Main K32xV32 state, native FutureSeed,
projections, gates, optimizer and data stay unchanged.

There is no learned address, cache admission, search, repair, rule, oracle,
selector, target dependence or second predicted-value bank. Same token IDs have
exactly identical certificate payloads at every occurrence.

## 4. Independent Data Contract

- frozen directional-MQAR L1024 train/test generator and hashes;
- train/test examples `10000/1000`, four key/value pairs, seed123;
- one serialized native initialization loaded into both arms;
- one identical warmup batch and data order;
- run order: native FutureSeed control, then certificate candidate.

## 5. Integrity Contract

Before formal training require the sole visible registered GPU, clean detached
pushed/read-back source, pinned FLA and Zoology hashes, Triton short convolution,
and no fallback. Candidate parameters must be `661,592` versus `661,584` native,
with exactly 8 new scalar logits. Main persistent state remains 4,096 values per
layer; certificate state is exactly another transient 4,096 values; there are
exactly two official scans per layer and four `ChunkGDN2FunctionBackward` nodes.

At zero logits require bit-exact full logits and, for finite nonzero incoming
main state, bit-exact layer output and main terminal state. Parent tensors and
parent-gradient topology must match the native model, using a byte-identical
native replay to calibrate official BF16/Triton gradient nondeterminism. Parent
gradient max-absolute and aggregate relative-RMS differences may not exceed the
larger of fixed `0.125/0.01` limits and `2.0x` native replay noise. Every
certificate gate must receive finite nonzero gradient. With opened synthetic
gates, certificate output must affect logits, certificate state must depend on
token-ID permutation, same-token payload error must be `<=1e-6`, and simultaneous
head permutation of Q/K/g/b/w/tag/gate must commute within `3e-3` BF16 tolerance.

## 6. Fixed Falsifiers And Gate

Endpoint activation requires both layers and all eight gates active with mean
absolute gate `>=1e-4`; finite nonzero certificate output/state RMS; token and
board variation `>=1e-4/1e-6`; exact repeated-token payload identity; and native
FutureSeed active on the receiving layer.

All quality checks are conjunctive:

- balanced accuracy `>=0.55` and `>=control+0.10`;
- future and past accuracy each `>=0.50` and `>=control+0.08`;
- joint exact `>=0.06` and `>=control+0.04`;
- fewer total query errors;
- wrong-key valid-value swap fraction reduced by at least `0.10`.

Any miss closes certificate payload source, gate map/scale, normalization,
per-layer versus shared gate, certificate FutureSeed/transport, tag width,
seed, data, LR, loss, batch, width, depth and duration rescue.

## 7. Cost And Scaling

The second official scan is deliberate. Fixed ceilings are `<2.25x` for elapsed,
post-warm wall and independently warmed step, and `<1.60x` peak allocated CUDA
memory. These limits cannot be relaxed after observing the result.

## 8. Results And Decision

Exact pushed/read-back source `abb8427c` passed the strict CUDA contract. It
adds exactly 8 parameters, preserves bit-exact zero-logit output and nonzero
incoming main-state behavior, exposes four official backward paths, gives all
eight certificate gates finite nonzero gradients, and passes token-ID,
head-permutation, pinned-kernel and parent-gradient-noise checks.

The fixed endpoint strongly improves retrieval but fails binding closure:

| metric | control | certificate | delta |
|---|---:|---:|---:|
| balanced accuracy | 0.174750 | 0.483000 | +0.308250 |
| future accuracy | 0.161000 | 0.486500 | +0.325500 |
| past accuracy | 0.188500 | 0.479500 | +0.291000 |
| joint exact | 0.000000 | 0.041000 | +0.041000 |
| total query errors | 3,301 | 2,068 | -1,233 |
| wrong-key valid-value swaps | 770 | 1,973 | +1,203 |
| wrong-key fraction of errors | 0.233263 | 0.954062 | +0.720799 |

The paired audit is diagnostic rather than merely aggregate. The certificate
repairs 1,580 previously wrong predictions (`1214` other-wrong and `366`
wrong-key) while breaking 347 previously correct predictions (`334` become
wrong-key). Of the 2,068 remaining errors, 1,973 are valid values assigned to
the wrong key. The auxiliary memory therefore learns the value set and lowers
CE (`2.91->0.92`), but does not preserve ownership.

All eight gates and both certificate states activate. Mean absolute gates are
`.00935/.01134`; certificate output RMS is `.1962/.2220`; certificate state RMS
is `.3182/.3531`; and native FutureSeed remains active. The registered raw
payload board-variation check misses (`3.88e-7 < 1e-6`) even though downstream
certificate state/output board variation is nonzero. This does not rescue the
run because the absolute balanced/directional/joint floors and the required
swap-fraction reduction independently fail.

Cost gates pass: warmed-step and peak-allocation ratios are `1.3207x/1.2430x`.
The decision is **discarded**. Do not tune tag source/width, gate map/scale,
normalization, layer sharing, certificate transport, seed, data, optimizer or
duration, and do not transfer this mechanism to Sudoku. The result localizes
the remaining problem: extra semantic payload can solve value-set retrieval,
but a useful GDN3 must preserve key ownership inside the memory organization
rather than append another post-read evidence channel.

## 9. Provenance

- source branch: `codex/gdn3-binding-certificate-20260815`
- exact source SHA: `abb8427ce91b155c660b115ec3c3e07e7955716f`
- clean detached worktree: `/huyang2/double-loop/worktrees/p-gdn3-047-abb8427`
- run: `/huyang2/double-loop/runs/p-gdn3-047-binding-certificate-l1024-20260815T013135Z-abb8427`
- score/comparison SHA256: `d593a01f651136795dbcbbdc9d1b981a2ce942cfb623011ed00b5796f6166426`
- contract JSON SHA256: `a1b0581e53ce9e9d83fdbc17386a7c27cebdfa71212b514406704e793b619675`
- formal log SHA256: `4ab0ea1dc350db28f6b4a09e2e4faa19a3ab21339fca545edd8192cf23495066`
- control/candidate checkpoint SHA256:
  `4d96b91f9278d38bd92e6f6cf35559ad067a8d5f3ea96d5cd519c558b8a346aa` /
  `ea65805078c282767b4ecd4634fc16991dfe1f822c7d11d1c4da1dd055f8741a`
- source snapshot SHA256: `424469d2d7679365ad20927e0075d90dedb652a81480f163aa87ce5231e40be2`
- completed: `2026-08-15T01:41:37Z`; formal status `2` is the registered
  science miss, not an integrity or launcher failure
