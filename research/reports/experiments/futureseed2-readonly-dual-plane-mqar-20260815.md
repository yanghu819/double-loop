# P-FS2-012: Read-Only Dual-Plane FutureSeed

## 1. Metainfo

- Status: registered, not yet run
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

## 8. Next Decision

If every gate passes, freeze the checkpoint and allow one hard-Sudoku transfer
that tests the same state separation without changing training settings. If
quality or cost fails, close the dual-plane family and use the result to choose
one GDN3 live-transition mechanism; do not tune this interface. The endpoint
must archive score, cases, edge-off cases, checkpoint, config, logs, source and
hashes before that decision.

## 9. Submission Record

Not applicable.
