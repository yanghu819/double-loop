# P-GDN3-045: Coherent Key-Spectrum Credit

## 1. Metainfo

- Status: complete; discarded at the fixed L1024 science and allocation gate
- Task: directional MQAR, sequence length 1024, four future and four past queries
- Carrier: D128/L2/H4/K32/V32 pinned-official GDN2 plus native FutureSeed
- Fixed endpoint: 10 epochs, batch32, seed123, contemporaneous control then candidate
- External idea boundary: `yanghu819/GDN_decouple_k` at
  `c7667fd11d95d3d147f59bb3d4492989909b69ae`

## 2. Mechanism Hypothesis

GDN2 already decouples read queries from the coherent write/erase key. The
external proposal adds a second erase key. P-GDN3-031 tests that transfer
directly and P-GDN3-036 tests its strongest ownership-preserving tangent form;
both destroy retrieval. P042-P044 additionally reject forced Q/K agreement,
dual gauges and a dynamic address frame. Functional key splitting is closed.

The underlying interference hypothesis remains plausible in a narrower form:
native coherent keys may occupy an anisotropic, low-effective-rank trajectory,
causing unrelated bindings to share address rows. P020's positive private
Log-SPD result is evidence that address geometry matters. The untested minimal
intervention is to preserve one exact read/write/erase ownership key at
inference while training its sequence-level spectrum to use the available K32
space more evenly.

Prediction: a scale-free per-board/head key-covariance credit should increase
effective rank, lower anisotropy and reduce total and wrong-key errors without
changing the recurrent equation. If accuracy falls while conditional swaps
fall, the mechanism is another broad retrieval collapse and must close.

## 3. Exact Intervention

For each native layer during training only, recompute the exact K projection
and pinned Triton short convolution from a detached copy of that layer's
native input. Let normalized keys be `k[b,t,h,:]` and define

```text
C_bh = (K/T) * sum_t k_bth k_bth^T
L_spectrum = mean_bh ||C_bh - I_K||_F^2 / K^2
L = L_CE + 0.10 * mean_layers(L_spectrum)
```

The detached input confines auxiliary gradients to each layer's native K
projection and K short convolution. The actual official scan, Q/K/V/g/b/w,
FutureSeed transfer, logits and inference graph are untouched. There are zero
new parameters, recurrent values and inference scans. Training adds exactly
one K projection/short-conv recomputation per layer. The loss uses no target,
query position, task rule, selector, search or repair signal.

## 4. Independent Data Contract

- frozen directional-MQAR generator and existing L1024 cache;
- train/test examples `10000/1000`, four key/value pairs, seed123;
- exact fixed train/test hashes;
- one saved matched initialization and one warmup-batch hash shared by arms;
- order is native FutureSeed control, then key-spectrum candidate.

## 5. Integrity Contract

Before formal training, require the sole visible GPU UUID, clean detached
pushed/read-back source, pinned FLA/Zoology hashes, two official
`GatedDeltaNet2`/`ChunkGDN2FunctionBackward` layers and Triton convolutions.
Control and candidate must have exactly 661,584 parameters and 4,096 recurrent
values/layer. Eval logits, train logits, and nonzero-incoming outputs/states
must be bit exact at initialization.

The auxiliary graph must contain zero official recurrent backward nodes and
finite nonzero gradients in both K projections and K convolutions, with no
gradient outside those modules. The combined objective must retain exactly two
official recurrent backward nodes and the native full gradient topology. Any
fallback, target use, unexpected gradient, parameter/state/scan delta or
identity miss is an integrity failure and authorizes no rescue.

## 6. Fixed Falsifiers And Gate

Activation requires two active layers, finite unweighted credit in `[1e-4,2]`,
weighted credit at least `1e-5`, finite noncollapsed effective rank in `(1,32]`,
finite anisotropy at least one, and nonzero board-to-board spectrum variation.

All quality checks are conjunctive:

- balanced accuracy `>=0.55` and `>=control+0.10`;
- future and past accuracy each `>=0.50` and `>=control+0.08`;
- joint exact `>=0.06` and `>=control+0.04`;
- fewer total query errors;
- wrong-key valid-value swap fraction reduced by at least `0.05`.

Any miss closes this exact key-spectrum credit. Do not tune coefficient,
covariance scope/normalization, token/layer mask, detach boundary, seed, data,
loss, width, depth or duration.

## 7. Cost And Scaling

Inference cost is exactly native. Fixed training cost ceilings are `<1.20x`
for elapsed, post-warm wall and independently warmed step, and `<1.12x` peak
allocated CUDA memory. These cannot be relaxed after results.

## 8. Results And Decision

The strict CUDA contract passed. Control and candidate were bit exact at
initialization for eval, training, and nonzero incoming-state output/state;
both had 661,584 parameters and 4,096 recurrent values per layer. The combined
candidate objective retained two official `ChunkGDN2FunctionBackward` paths,
while the auxiliary graph contained none. Auxiliary gradients were finite and
nonzero only in both native K projections and K convolutions.

| Metric | Control | Candidate |
|---|---:|---:|
| Balanced accuracy | 0.4830 | 0.0130 |
| Future accuracy | 0.4830 | 0.0115 |
| Past accuracy | 0.4830 | 0.0145 |
| Joint exact | 0.0370 | 0 |
| Total query errors | 2,068 | 3,948 |
| Wrong-key valid-value swaps | 1,959 | 144 |
| Swap fraction among errors | 0.947292 | 0.036474 |

The candidate's endpoint key spectrum moved opposite to the prediction.
Layer effective rank ended at `25.423/12.352`, anisotropy at
`3.006/4.779`, and unweighted/weighted credit at
`0.040736/0.004074`. The second layer therefore became substantially more
anisotropic despite the active credit. The large conditional swap reduction
is broad retrieval collapse, not repaired binding.

Elapsed, post-warm wall, independently warmed step, and peak-allocation ratios
were `0.91199/0.91238/1.01868/1.13064x`. The first three cost checks passed;
allocation missed the fixed `1.12x` ceiling. There was no NaN, OOM, fallback,
data drift, source drift, or infrastructure failure. Endpoint exit status 2 is
the registered science miss.

P-GDN3-045 is discarded and has no Sudoku transfer. Close coefficient,
covariance scope/normalization, token/layer mask, detach boundary, seed, data,
loss, width, depth and duration rescue. Key whitening/isotropy credit is not a
viable refinement of decoupled keys. A successor must alter a scalable state
organization or recurrent transition while preserving native coherent
ownership and the useful learned Q/K differential.

## 9. Provenance

- Pushed/read-back source: `89f3ec31d3b9616576cfd64534f4535be2891638`
- Formal run:
  `/huyang2/double-loop/runs/p-gdn3-045-coherent-key-spectrum-l1024-20260814T235135Z-89f3ec3`
- Contract score SHA256:
  `fb05a09f2040e9ee94b69d71832aeb5d3f40fc672a3a82c1b8d1425a0100650b`
- Endpoint score SHA256:
  `7fa489eecb9a40c1ffaac7298da6cf9bc937269d7d5231632dc6a2875320ae27`
- Control checkpoint SHA256:
  `d1eea0ce82329f17be4bd1193a5be93a4e164779a8d1bf952f54fac57bc36fae`
- Candidate checkpoint SHA256:
  `2bad1d91b84a1886fa526407cae6af53881698267b531a267b4c57f5faca75c5`
- Formal log SHA256:
  `bcd664bacc94cd0294cf1bc502ecb37c7f3c052faf73fcae2eafacea036823d5`
- Source snapshot SHA256:
  `05cc54442e899c601ff77f3c53b13683bf293c659a75953ddd92d6da5cabaf56`
