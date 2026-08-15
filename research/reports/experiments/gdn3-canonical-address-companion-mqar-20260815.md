# P-GDN3-049: Shared Canonical-Address Companion State

## 1. Metainfo

- Status: preregistered; implementation complete, no formal score yet
- Task: directional MQAR, sequence length 1024, four future and four past queries
- Carrier: D128/L2/H4/K32/V32 pinned-official GDN2 plus native FutureSeed
- Fixed endpoint: 10 epochs, batch32, seed123, contemporaneous control then candidate
- Decision target: preserve value ownership before any hard-Sudoku transfer

## 2. Mechanism Hypothesis

P-GDN3-047 is the strongest recent mechanistic signal. Its semantic certificate
state raises balanced accuracy `.17475->.48300` and reduces errors `3301->2068`,
but 1,973 of 2,068 remaining errors are valid values assigned to another key.
The model therefore knows which values exist but lacks a stable ownership index.
P-GDN3-048 then shows that encoding ownership directly into the native payload
damages the parent retrieval path.

The falsifiable hypothesis is that ownership needs an independent address domain,
not another payload code or modification of the native state. A smaller companion
state can preserve the complete native value while learning a shared canonical
address. If this is the missing organization, it must improve both directions,
joint exact, total errors and wrong-key fraction. Merely recovering more legal
values while leaving the swap fraction high is a failure.

This is not P025's compressed committed-edit bank, P038's content router, P046's
residual added to the main address, or P047's stable-token payload under the same
native address. The primary state remains unchanged and the companion changes
only the organization of address ownership.

## 3. Exact Intervention

For each layer, compute the unchanged native tensors and parent scan:

```text
(o_main, S_main) = GDN2(q, k, v, g, b, w, S_in)
```

One learned matrix `P in R^(16x32)` is shared across layers and heads:

```text
a = normalize(P(normalize(q + k)))
g_aux = mean_K(g), broadcast to K16
b_aux = mean_K(b), broadcast to K16
(o_aux, C) = GDN2(a, a, v, g_aux, b_aux, w, 0)
o = o_main + tanh(r_layer,head) * normalize_V(o_aux)
```

The original output norm/projection consumes `o`. The projection contributes
512 parameters and two four-head zero-initialized read gates contribute eight,
for exactly 520 new parameters. Each layer adds one transient H4xK16xV32 state,
2,048 values, and one official scan. The companion state is rebuilt per layer;
native FutureSeed transports only the unchanged main state.

No token labels, lag rules, cache selector, task rule, search, repair, oracle,
extra target supervision or recurrence fallback is introduced.

## 4. Independent Data Contract

- frozen directional-MQAR L1024 train/test generator and hashes;
- train/test examples `10000/1000`, four key/value pairs, seed123;
- one serialized native initialization loaded into both arms;
- identical warmup batch, data order, optimizer and scheduler;
- run order: native FutureSeed control, then P049 candidate.

## 5. Integrity Contract

Before formal training require the sole visible registered GPU, clean detached
pushed/read-back source, pinned FLA and Zoology hashes, Triton short convolution
and no fallback. Candidate parameters must be `662,104` versus `661,584` native.
Main state remains 4,096 values/layer; companion state is exactly 2,048 values;
there are two official scans/layer and four `ChunkGDN2FunctionBackward` nodes.

At zero read gates require bit-exact full logits and, with finite nonzero incoming
main state, bit-exact layer output and main terminal state. Parent tensors and
parent-gradient topology must match native, calibrated against a byte-identical
native replay. All eight read gates must receive finite nonzero gradient. After
opening the gates, the shared projection must receive finite nonzero gradient,
rank must be 16, companion output/state must vary across tokens and boards, token
order must affect state, and simultaneous head permutation must commute within
`3e-3` BF16 tolerance.

## 6. Fixed Falsifiers And Gate

Activation requires both layers, all eight read gates, the shared rank-16
projection, finite nonzero canonical-address/output/state RMS, token and board
variation, nonzero decay/erase, and native FutureSeed.

All quality checks are conjunctive:

- balanced accuracy `>=0.55` and `>=control+0.10`;
- future and past accuracy each `>=0.50` and `>=control+0.08`;
- joint exact `>=0.06` and `>=control+0.04`;
- fewer total query errors;
- wrong-key valid-value swap fraction reduced by at least `0.10`.

Any miss closes projection source/map, K16 width, gate scale, mean gate reuse,
per-layer/shared projection, companion FutureSeed/transport, seed, data, LR,
loss, batch, model width/depth and duration rescue.

## 7. Cost And Scaling

The extra K16xV32 scan is the intended cost. Fixed ceilings are `<2.00x` for
elapsed, post-warm wall time and independent warmed-step time, and `<1.45x` for
peak allocated memory. The companion is linear in sequence length and has half
the key rows of the native state.

## 8. Results

Pending strict CUDA contract and the single fixed endpoint.

## 9. Decision

Pass all integrity, activation, quality and cost checks: admit one hard-Sudoku
scale transfer. Any miss: close P-GDN3-049 and retain the diagnosis that a naive
secondary address bank is insufficient; do not rescue nearby settings.
