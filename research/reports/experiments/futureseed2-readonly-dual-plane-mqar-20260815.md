# P-FS2-012: Read-Only Dual-Plane FutureSeed

## 1. Metainfo

- Status: complete; discarded at the fixed quality and cost gate
- Formal run:
  `p-fs2-012-readonly-dual-plane-l1024-20260815T074456Z-b5da2ac`
- Exact pushed/read-back source:
  `b5da2ac543adfeff180b94445ed0062d12cd5b78`
- GPU: A100-SXM4-80GB index 0, UUID
  `GPU-573c7ed1-1c51-8334-299b-edf2ff3440e6`
- Task: validated directional MQAR L1024 wrong-key binding regime
- Parent: frozen deterministic P-REPRO-001 initialization
- Candidate: D128/L2/H4/K32/V32 official GDN2, 10 epochs, batch 32,
  seed 123
- Control: frozen P-REPRO-001 replay B; do not rerun it

## 2. Mechanism Hypothesis

Native FutureSeed uses the producer terminal matrix as the receiving layer's
live initial state. The receiver then decays, erases and writes into that same
matrix. Future evidence and current-layer causal evidence therefore compete
for the same mutable addresses. This can explain the validated endpoint where
76.38% of errors are correct values assigned to the wrong key.

The candidate separates those roles. The producer terminal matrix is retained
as a bounded read-only future plane. The receiving layer starts its ordinary
official GDN2 state from zero and uses it only for current-layer causal writes.
The receiver's own normalized query reads both planes, and their outputs are
combined before the unchanged official output norm/gate/projection.

This mechanism is not covered by P-FS2-010, which decoded with the producer's
query/output projections and added a hidden residual while still injecting the
state live. It is not P-FS2-011's training-only credit, P-GDN3-007's write
controller, or P-FS3-004's basis rotation. P-FS2-006 described a related
read-only contrast but failed its FP32 replay integrity gate before any
intervention or quality result. P-FS2-012 is the first admitted from-scratch
quality test of separated inherited and live state planes.

## 3. Exact Intervention

For adjacent layers `l -> l+1`, normalize the producer terminal state exactly
as native FutureSeed already does:

```text
F = sigmoid(a_receiver) * S_l / RMS_KV(S_l)
```

The receiving layer runs the unchanged pinned official recurrence with zero
live initial state:

```text
O_live, S_live = official_chunk_gdn2(Q, K, V, g, b, w, initial_state=0)
O_future[t] = (normalize(Q[t]) / sqrt(K)) @ F
O = O_live + O_future
```

`O` then enters the exact official `o_norm`, hidden gate and output projection.
The existing FutureSeed gate is reused, so parameter delta is exactly zero.
There is one official scan per layer, no reverse scan, selector, cache policy,
extra recurrence or task logic. The retained side matrix is one K32xV32 state
per receiving route (4,096 values in the fixed H4 model), but it is read-only
and adds no recurrent state transition.

## 4. Independent Data Contract

Use the exact P-REPRO-001 directional MQAR dataset and initialization:

- train/test examples: 10,000/1,000;
- sequence length: 1,024; four mixed-direction KV pairs;
- train epochs/batch/seed: 10/32/123;
- initialization SHA256:
  `7402e46c48cbd47070d65262d1a1b62a32ac55a4d644a6fb9644b96b0914850f`;
- frozen control score SHA256:
  `df8ae1c212590666440f12029d60c74a6d09cbc059b23e58bcf864c00ae98854`.

The candidate loads every parent tensor strictly from that initialization.
Optimizer, RNG reset, data order and warmup batch remain those of the validated
length-scaling runner. No repeated control, second seed or hyperparameter arm
is authorized.

## 5. Integrity Contract

Formal source must be an exact pushed/read-back SHA in a clean detached
worktree. Only CUDA index 0 on the registered A10080 may be visible. Require
pinned FLA SHA `9c8e42e762fce087c27b673af4922795d9edb85e`, clean Zoology SHA
`1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb`, two exact official GDN2
modules and exactly two `ChunkGDN2FunctionBackward` nodes.

The strict checker must prove identical parameter names/count/hash, exact
4,096 recurrent values per layer, zero parameter and recurrent-state delta,
one official scan per layer, edge-off bit identity with causal official GDN2,
an exact receiver-read formula, read-only state immutability, head permutation
equivariance, state-shuffle dependency, finite nonzero producer/receiver/gate
gradients, and no fallback.

## 6. Fixed Falsifiers And Gate

Activation requires exactly one receiving route; read, live output, terminal
state and inherited state RMS each at least `1e-4`; read board/token variation
and live board variation each at least `1e-5`; finite read/live cosine with
absolute value below `.995`; and a live native seed gate.

All quality checks are conjunctive:

- balanced accuracy at least `.60` and at least `+.10` over frozen control;
- future accuracy at least `.60` and at least `+.14` over control;
- past accuracy regression no worse than `.03`;
- joint exact at least `.10`;
- total query errors reduced by at least 15%;
- wrong-key valid-value swap count reduced by at least 20%;
- enabled versus the same trained model with the read-only edge disabled gains
  at least `.10` future accuracy and `.05` balanced accuracy, with fewer
  errors.

Any miss closes dual-plane FutureSeed without gate/normalization/scale,
seed/LR/loss/batch/width/depth/duration or nearby architectural rescue.

## 7. Cost And Scaling

The candidate adds one `QF` contraction per receiving edge, O(TKV), while
retaining linear sequence complexity. There are zero new parameters, zero new
recurrent transitions and one bounded 4,096-value read-only side matrix.
Against frozen control, elapsed, post-warm wall and independent warmed-step
ratios must each be below `1.25x`; peak allocation must be below `1.10x`.
These limits are fixed before launch and cannot be relaxed after results.

## 8. Results And Decision

The strict CUDA contract passed before training. It proves zero parameter,
recurrent-state and scan delta; exact edge-off identity with causal official
GDN2; exactly two `ChunkGDN2FunctionBackward` nodes; exact read-only state
immutability; head equivariance and state-shuffle dependency; and finite
nonzero producer, receiver and FutureSeed-gate gradients. Contract JSON SHA256
is `68e3bd2a090d1c98ff76d4ba100f1b31b50215d866fe6c4445f7724bd77eeaf7`.

The fixed endpoint rejects the mechanism:

| metric | frozen native control | read-only dual plane | delta |
|---|---:|---:|---:|
| balanced accuracy | 0.494000 | 0.444500 | -0.049500 |
| future accuracy | 0.454000 | 0.451000 | -0.003000 |
| past accuracy | 0.534000 | 0.438000 | -0.096000 |
| joint exact | 0.041000 | 0.009000 | -0.032000 |
| total query errors | 2,024 | 2,222 | +198 |
| wrong-key valid-value swaps | 1,546 | 1,749 | +203 |
| wrong-key fraction of errors | 0.763834 | 0.787129 | +0.023295 |

The path is causally active rather than ignored. Disabling only the read-only
edge in the trained candidate drops balanced/future accuracy to
`.211/.011`, joint exact to zero, and raises total errors to `3,156`. With the
edge enabled, receiver read RMS/relative RMS is `.157005/.018889`, board/token
variation is `.007413/.007188`, inherited state remains exactly unchanged,
and read/live cosine is `.230676`. The model therefore relies on the protected
future plane, but that plane protects wrong associations as readily as correct
ones.

Cost also rejects the realization. Candidate/control elapsed, post-warm wall,
independently warmed step and peak-allocation ratios are
`1.98898/1.97700/1.33450/1.02418x`; all three time ceilings miss while the
allocation ceiling passes. Formal training completed naturally with status 0;
the registered endpoint returned the science-close status and wrote no
integrity abort.

Decision: discard P-FS2-012 and close read-only/live plane separation without
gate, normalization, scale, seed, optimizer, data, width, depth or duration
rescue. Overwrite is not the dominant causal bottleneck. The next experiment
must change how a value is committed to its owner inside the live recurrent
transition, not merely preserve or reread the inherited state.

Provenance SHA256:

- comparison: `8be0f00c9b648d2ffa8356c6c777c5581113565990d91f08014a6b47036e311a`;
- candidate score: `ccf85ace85bb9f406a781f6abe0154640f07c2c87fda3d37e0dee87ca0cf2384`;
- checkpoint: `24f3da66bdbbb753b74858624d482515cbf4d07462264438747ac22a616f21d9`;
- formal log: `1b41fb6f09726be8555e3a24bee8de8c258b46a865d3f91bae07968d4b1c5d36`;
- source snapshot:
  `2d91353ba7113104ed07e8e407950452e1081b83deb1a353003955dd3cffe59e`.

## 9. Submission Record

Not applicable.
