# P-GDN3-033: Cross-Layer Shared Log-SPD Address Geometry

## 1. Question

Can one bounded address metric shared by every GDN2 layer stabilize the
long-sequence key namespace while retaining native coherent erase/write/read
ownership and the pinned official GDN2 recurrence?

## 2. Motivation

P-GDN3-031 shows that independent erase and write/read keys destroy retrieval:
the branches become almost orthogonal and balanced L1024 accuracy falls to
`0.011`. P-GDN3-032 shows that simply doubling coherent K rows also fails.
P-GDN3-020 is the only positive address intervention: a learned bounded
Log-SPD transform on Q/K raised balanced accuracy by `0.30875` in its fixed
run without adding state or scans. Its metric, however, was independently
parameterized in each layer even though native FutureSeed transports a
producer state into the receiver.

This test imposes one global address geometry across layers. It is not direct
key decoupling, a K expansion, a cache, a second state, a write controller or
a producer-basis transport. P020 did not test cross-layer parameter sharing,
and P023 changed head/state factorization at the same time.

P-DIAG-CARRIER-002 retired historical `.7475` as an absolute gate. This run
therefore includes one contemporaneous native control followed by one
candidate in the same source, process, task and fixed order.

## 3. Intervention

Use D128/L2/H4/K32/V32 official-FLA GDN2 plus native FutureSeed on directional
MQAR L1024. Before each unchanged official `chunk_gdn2` call, transform both Q
and K with the same per-head factor

```text
C = exp(0.5 * log(2) * A / (1 + ||A||_F))
q' = q C
k' = k C
```

where `A` is symmetric and trace-free. The same `C` parameter object is used
by both layers. Consequently its metric eigenvalues stay in `[0.5, 2]`, its
determinant is one, and erase/write/read continue to use one normalized key.
At zero parameters `C=I`, giving exact native output, state, gradients and
FutureSeed transport.

The candidate adds exactly 2,108 parameters total, no recurrent state, no
scan, no cache and no task-specific operation. Fixed run order is native
FutureSeed control then shared-Log-SPD candidate; both use 10,000 training and
1,000 validation examples, ten epochs, batch32 and seed123.

## 4. Falsifiable Prediction

If per-layer metric drift is part of the FutureSeed binding failure, the shared
metric should activate in all four heads, receive nonzero gradient from each
layer, remain bounded, and improve candidate balanced accuracy to at least
`max(0.50, control + 0.15)`. Future and past accuracy must each improve by at
least `0.10`, joint exact must reach at least `max(0.10, control + 0.10)`, total
errors must fall, and wrong-key valid-value swap fraction among errors must
fall by at least `0.10` versus the contemporaneous control.

If the shared metric is active but misses any quality requirement, then a
single global smooth address geometry is insufficient. Close cross-layer
Log-SPD sharing without metric scale/rank/head/layer/seed/LR/epoch rescue.

## 5. Integrity Contract

Require exactly one visible registered A100, pinned FLA source and wheel
hashes, official `GatedDeltaNet2` and `ChunkGDN2FunctionBackward` paths,
Triton short convolution, exact data hashes, and a clean detached pushed SHA.
The checker must prove:

- exact 2,108-parameter delta and zero state/scan delta;
- the two layers reference the same metric parameter storage;
- zero-init full-model and nonzero-incoming-state output/state identity;
- exact native FutureSeed transport identity;
- finite nonzero per-layer contributions to the shared metric gradient;
- head-permutation equivariance, bounded eigenvalues/condition/logdet, and a
  nonzero output dependency when opened;
- no fallback, NaN or OOM.

## 6. Budget And Kill Rule

One strict GPU contract followed by one fixed two-arm endpoint. Candidate
training elapsed, post-warm wall and warmed-step time must each be below
`1.15x` control; peak allocation must be below `1.10x`. Any integrity,
activation, quality or cost miss closes the mechanism. No rescue run.

## 7. Results

Pending.

## 8. Decision

Pending the fixed contract and endpoint. A pass authorizes one matched Sudoku
transfer; a miss returns to a different recurrent state-transition family.

## 9. Provenance

- Carrier parent decision: P-DIAG-CARRIER-002, score SHA256
  `f26dd46bef652f8f88136fcd11a2a52b4dcd1f86b07563f8750424263031e255`.
- P020 exploratory mechanism signal:
  `/huyang2/double-loop/runs/p-gdn3-020-log-spd-exploratory-a100-20260812T025949Z-2af49c5`.
- Source SHA, contract and endpoint hashes: pending.
