# P-FS2-016 Momentum-Only FutureSeed

## 1. Research Question

Can P059 retain its verified long-context retrieval while transporting only
the useful Momentum plane across layers, cutting the FutureSeed interface in
half?

## 2. Evidence And Hypothesis

P059 Momentum DeltaNet plus native `[S,M]` FutureSeed is the strongest verified
GDN3 carrier: balanced/future/past/joint accuracy is
`.94425/.95150/.93700/.82400`, versus official/native GDN2's
`.494/.454/.534/.041` on directional MQAR L1024. A frozen same-weight component
audit then measured:

| transported component | balanced | future | past | joint | errors | swaps |
|---|---:|---:|---:|---:|---:|---:|
| native `[S,M]` | .94425 | .95150 | .93700 | .824 | 223 | 151 |
| `M` only | .94375 | .95000 | .93750 | .822 | 225 | 153 |
| `S` only | .48450 | .03100 | .93800 | 0 | 2,062 | 434 |

This closed component masking as a quality repair: it does not remove the
remaining owner swaps. It separately opens one efficiency question. Almost all
cross-layer future information is in `M`; carrying `S` may be unnecessary
interface width. A model trained end to end with only `M` can co-adapt to the
smaller contract and should preserve P059 within a tight equivalence budget.

## 3. Fixed Mechanism

Keep P059's exact external Momentum DeltaNet at SHA `c6e77fa`, both native
chunk scans, D128/L2/H4/K32/V32 recurrence, Q/K/V projections, width-four short
convolutions, output correction, optimizer, data and training schedule. Each
layer still owns the complete recurrent state `[S,M]` with 8,192 values.

At the sole cross-layer FutureSeed edge, the producer packs only terminal `M`
with 4,096 values. The receiver applies the unchanged native normalization and
learned head gate to `M`, then constructs its initial recurrent state as
`[zeros_like(M), M_seed]`. No `S` tensor participates in that cross-layer
gradient. The candidate adds no parameter, persistent state, scan, cache,
selector, rule or task-specific logic.

## 4. Strict CUDA Contract

The exact pushed SHA in a clean detached worktree must prove:

- one visible registered A800 and exact external Momentum/host FLA hashes;
- two native Momentum chunk backward paths and no fallback;
- exactly 599,672 parameters, with all names and initial values identical to
  P059;
- exact layer-0 output and terminal-state identity;
- unchanged 8,192-value recurrent state and exact 4,096-value transported
  payload;
- candidate receiver `M` seed exactly equals P059's `M` seed while receiver
  `S` is exactly zero;
- cross-layer gradient to producer `S` is exactly zero and the gradient to
  producer `M` is finite and nonzero;
- complete Q/K/V/Momentum and FutureSeed gradients and full-stack activation.

Any contract or provenance miss closes the implementation before training.

## 5. Prediction And Gates

Run one candidate-only from-scratch directional MQAR L1024 endpoint for 10
epochs, batch32, seed123 from P059's exact matched initialization and data. Do
not rerun the frozen P059 control. A Pareto FS2 pass requires every item:

- transported values are exactly `4,096/8,192 = .5`, with unchanged
  parameters, persistent state and scans;
- both Momentum layers and the one M-only FutureSeed route are active; receiver
  S-seed RMS is exactly zero and M-seed RMS is finite and at least `1e-4`;
- balanced accuracy is at least `.94` and no more than `.005` below P059;
- future and past accuracy are each no more than `.005` below P059;
- joint exact is at least `.81` and no more than `.01` below P059;
- errors are at most 243 and wrong-key valid-value swaps at most 171;
- elapsed, post-warm and independently warmed-step ratios are each below
  `1.05x`, and peak allocation is below `1.02x` P059.

The claim is interface efficiency at matched quality, not repair of P059's
residual 151 adjacent-owner swaps. Any miss closes Momentum-only FutureSeed.
There is no component, gate, normalization, scale, seed, LR, loss, batch,
width, depth or duration rescue.

## 6. Result

Exact pushed/read-back source `6ad6dbad73c51407f93d094183c0a592fded02e0`
passed the strict A800 contract. It proved 599,672 identical parameters, two
native Momentum backwards, exact producer identity, exact M-seed parity,
zero receiver S, nonzero M transport gradient, unchanged recurrent state and
scans, and the exact `4096/8192 = .5` transport ratio.

The from-scratch endpoint then failed decisively:

| metric | frozen P059 `[S,M]` | trained M-only |
| --- | ---: | ---: |
| balanced | .94425 | .01325 |
| future | .95150 | .01350 |
| past | .93700 | .01300 |
| joint exact | .82400 | 0 |
| errors | 223 | 3,947 |
| wrong-key swaps | 151 | 157 |

Validation stayed near chance through epoch 7 and reached only `.01325` at
epoch 10. All integrity and activation checks pass, so this is not a wiring,
kernel, source or data failure. Removing S during optimization prevents the
second-order solution from forming even though masking S after P059 has
converged is almost lossless. Cost also misses the cold matched wall gates
(`1.1490x` elapsed, `1.1432x` post-warm), while warmed-step compute is
`0.6071x` and allocation is `0.9995x`.

Formal active-window GPU utilization averages `43.62%` including evaluation
gaps and `66.71%` over nonzero samples, peaks at `78%`, and reaches `2,270
MiB`; contract compilation peaks at `3,024 MiB`. There is one A800 compute
process and no NaN, OOM or fallback.

Artifacts:

- run: `/huyang2/double-loop/runs/p-fs2-016-momentum-only-20260816T143919Z-6ad6dba`
- contract SHA256: `1ef8b8bf0fc80a9445bce70126b616c8041bf3cdc298beeca3a63c7055cb74d2`
- comparison SHA256: `a4473c2e9c8c7fd55fe9cf19937e5225901e413d639356e8b99e80a23b3d6426`
- checkpoint SHA256: `5e919122fe6af8321af652a953694e4b518fac5807e8b70f53ae8530adbca3db`
- formal log SHA256: `4ab97df0eff5ef8b0bba53ea4f806119a3f1bf8c3c276f5f02442d60928d1765`
- GPU samples SHA256: `f507c82ede8445500bdc1faf2e1e28f0a705219f955952ff388ac2b8d4f271ea`

## 7. Decision

Close M-only end-to-end training with no component, gate, normalization,
scale, seed, LR, loss, batch, width, depth or duration rescue. The new
mechanistic result is phase dependence: S is dispensable after convergence
but necessary while learning. Open only the separately registered
train-full/serve-M-only deployment test P-FS2-017.
