# P-GDN3-048: Reciprocal Address-Conditioned Payload Gauge

## 1. Metainfo

- Status: preregistered; implementation and GPU contract pending
- Task: directional MQAR, sequence length 1024, four future and four past queries
- Carrier: D128/L2/H4/K32/V32 pinned-official GDN2 plus native FutureSeed
- Fixed endpoint: 10 epochs, batch32, seed123, contemporaneous control then candidate
- Decision target: reduce correct-value/wrong-key binding errors without another state bank

## 2. Mechanism Hypothesis

P-GDN3-047 sharply separates value retrieval from ownership. A companion state
raises balanced accuracy `.17475->.48300` and reduces errors `3301->2068`, but
1,973 remaining errors are valid values assigned to the wrong key. A semantic
side channel therefore teaches the value set without binding each value to its
owner.

The falsifiable hypothesis is that ownership must be encoded in the primary V
payload itself. Native GDN2 already provides one coherent address `k` for erase
and write and a query `q` for read. A bounded address-conditioned code applied
to every written payload and its reciprocal applied to every read should cancel
for a matching Q/K address while leaving a mismatch when a wrong address
retrieves another valid value.

This is not another decoupled key, Q/K metric/frame, extra V capacity, product
address, correction bank, cache or certificate readout. It preserves one native
Q/K pair, one K32xV32 state, one official scan and the original FutureSeed edge.

## 3. Exact Intervention

For native pre-kernel Q/K and V32 payload, define per-layer/head scalar
`s=log(2)*tanh(a)` and address unit vectors `q_hat`, `k_hat`. The write and read
codes are

```text
d_k = exp(s * k_hat)              in [0.5, 2]
d_q^-1 = exp(-s * q_hat)          in [0.5, 2]
v_code = v * d_k
(o_raw, S) = GDN2(q, k, v_code, g, b, w, S_in)
o = o_raw * d_q^-1
```

The diagonal code commutes with GDN2's elementwise V write gate. For an exactly
matching address, `d_k*d_k^-1=1`; for a different query, the mismatch remains
in V space before the unchanged output norm/projection. At `a=0` the candidate
is bit-exact native GDN2 including nonzero incoming state and FutureSeed.

There are four logits per layer, exactly 8 new parameters in the D128/L2 model,
zero new state values, zero new tokens and zero new scans. The factor is bounded
by construction and introduces no task rule, selector, search or target access.

## 4. Independent Data Contract

- frozen directional-MQAR L1024 train/test generator and hashes;
- train/test examples `10000/1000`, four key/value pairs, seed123;
- one serialized native initialization loaded into both arms;
- identical warmup batch and data order;
- run order: native FutureSeed control, then payload-gauge candidate.

## 5. Integrity Contract

Before formal training require the sole visible registered GPU, clean detached
pushed/read-back source, pinned FLA/Zoology hashes, Triton short convolution and
no fallback. Candidate parameters must be `661,592` versus native `661,584`.
Both models retain exactly 4,096 recurrent state values per layer and one
official scan/layer, with exactly two `ChunkGDN2FunctionBackward` nodes/model.

At zero logits require bit-exact full logits, layer outputs, terminal states and
native FutureSeed for finite nonzero incoming state. Parent tensors must be
identical; parent-gradient differences must remain within `2x` a byte-identical
native BF16/Triton replay noise calibration. All eight logits require finite
nonzero gradients. With opened logits require factor bounds, reciprocal closure
within `2e-3`, exact production replay, head equivariance, nonzero Q/K mismatch,
and nonzero changes to both written state and decoded output.

## 6. Fixed Falsifiers And Gate

Endpoint activation is conjunctive:

- both layers and all eight scalar codes have `abs(s)>=1e-4`;
- factors remain in `[0.5-2e-3, 2+2e-3]` and same-address reciprocal error
  remains `<=2e-3`;
- factor token variation, Q/K code mismatch, coded-V change and decoded-output
  change are each `>=1e-4`;
- terminal state is finite with nonzero board variation;
- native FutureSeed remains active.

All quality checks are conjunctive:

- balanced accuracy `>=0.55` and `>=control+0.10`;
- future and past accuracy each `>=0.50` and `>=control+0.08`;
- joint exact `>=0.06` and `>=control+0.04`;
- fewer total query errors;
- wrong-key valid-value swap fraction reduced by at least `0.10`.

Any miss closes payload-code radius, scalar/vector code, Q/K source, diagonal
map, normalization, layer/head sharing, initialization, seed, data, optimizer,
loss, batch, width, depth and duration rescue.

## 7. Cost And Scaling

The mechanism adds only pointwise normalize/exp/multiply operations around the
unchanged official scan. Fixed ceilings are `<1.35x` for endpoint elapsed,
post-warm wall and independently warmed step, and `<1.12x` peak allocation.
They cannot be relaxed after observing the result.

## 8. Results And Decision

Pending exact pushed source, strict CUDA contract and the one fixed matched
endpoint. A pass admits one Sudoku transfer. Any integrity, activation, quality
or cost miss closes this final ownership-coding test without a nearby run.

## 9. Provenance

Pending source SHA, contract, run, score, checkpoints and artifact hashes.
