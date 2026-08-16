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

Pending the registered A800 contract and formal endpoint.

## 7. Decision

Pending.
