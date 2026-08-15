# P-GDN3-048: Reciprocal Address-Conditioned Payload Gauge

## 1. Metainfo

- Status: complete; discarded at the fixed activation/quality/cost gate
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

Exact pushed/read-back source `97cc8f57` passes the strict CUDA contract. The
candidate adds exactly 8 parameters, keeps one pinned-official scan/layer and
4,096 recurrent state values/layer, and is bit-exact to the parent at zero for
full output, FutureSeed transport and finite nonzero incoming state. Both
layers expose `ChunkGDN2FunctionBackward`; all eight logits have finite nonzero
gradient. Opened FP32 factors stay bounded, close same-address reciprocity to
`2.38e-7`, commute exactly with head permutation and change both terminal state
and decoded output.

The fixed endpoint rejects the mechanism:

| metric | control | payload gauge | delta |
|---|---:|---:|---:|
| balanced accuracy | 0.174750 | 0.075000 | -0.099750 |
| future accuracy | 0.161000 | 0.086000 | -0.075000 |
| past accuracy | 0.188500 | 0.064000 | -0.124500 |
| joint exact | 0.000000 | 0.000000 | 0.000000 |
| total query errors | 3,301 | 3,700 | +399 |
| wrong-key valid-value swaps | 770 | 337 | -433 |
| wrong-key fraction of errors | 0.233263 | 0.091081 | -0.142182 |

The lower conditional swap fraction is broad retrieval damage, not ownership
closure. Paired predictions make this explicit: only 228 previously wrong
queries become correct, while 627 previously correct queries break. Candidate
future/past CE rises from `2.9178/2.9002` to `3.5845/3.6341`. The gauge therefore
removes many recognizable valid-value errors by making the value itself less
retrievable.

All eight learned strengths activate. Per-layer strength RMS is
`.02565/.02307`; coded-V relative change is `.00350/.00143`; decoded-output
relative change is `.00229/.00156`; Q/K code mismatch is `.00473/.00410`; and
native FutureSeed remains active. Production BF16 factors remain narrowly
bounded (`.9766..1.0234`), but same-address reciprocal error is `.004211`, above
the fixed `.002` activation tolerance even though the FP32 contract closes.

Independent warmed-step cost is `1.2047x`, but peak allocation is `1.2593x`
and misses the fixed `1.12x` ceiling. Arm-order elapsed and post-warm ratios are
`.8181/.8212x` and are retained only as timing context, not evidence against
the independent benchmark.

The decision is **discarded**. Do not tune radius, scalar/vector map, Q/K
source, normalization, precision path, layer/head sharing, seed, data,
optimizer or duration. There is no Sudoku transfer. Together with direct
decoupled-key, coherent-key, gauge/frame and certificate tests, this closes the
nearby address/payload reparameterization route. The useful surviving evidence
is P047's value-set recovery; the unresolved problem remains ownership in a
learnable recurrent state organization, not a reversible wrapper around the
native payload.

## 9. Provenance

- source branch: `codex/gdn3-address-payload-gauge-20260815`
- exact source SHA: `97cc8f57a5c767c21466a345a6bf51bb83adb847`
- clean detached worktree: `/huyang2/double-loop/worktrees/p-gdn3-048-97cc8f5`
- run: `/huyang2/double-loop/runs/p-gdn3-048-address-payload-gauge-l1024-20260815T021156Z-97cc8f5`
- score/comparison SHA256: `793f31de2e0f36ec14d4675295390938c75c841e847b8caa726b2ad9d8df1d51`
- contract JSON SHA256: `fdd375a57eca5fb2a09257a665e201fe195ffd8e79eaa7658738594b01d821d2`
- formal log SHA256: `8fd86c1047407c67b4811a47435df39f57a7c68bb7a62bc8052d4a67f111c8ba`
- control/candidate checkpoint SHA256:
  `4d96b91f9278d38bd92e6f6cf35559ad067a8d5f3ea96d5cd519c558b8a346aa` /
  `cb7dcfa5c468f2e257ec74eb85dfd87ab78f6097f79e7ba2914628305afac763`
- source snapshot SHA256: `20d6fe1273b55b0ac37aca1a04b80024615da49e50778ebbcf01a947db0529b4`
- completed: `2026-08-15T02:20:39Z`; formal status `2` is the registered
  science miss, not an integrity or launcher failure
