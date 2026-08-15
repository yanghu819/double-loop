# P-GDN3-047: Binding-Certificate Companion State

## 1. Metainfo

- Status: preregistered; implementation and GPU contract pending
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

Pending exact pushed source, strict CUDA contract and the one fixed matched
endpoint. A pass admits one Sudoku-scale transfer; any integrity, activation,
quality or cost miss closes the mechanism without a nearby run.

## 9. Provenance

Pending source SHA, contract, run, score, checkpoints and artifact hashes.
