# P-GDN3-028: Atomic Shared-Payload Paired-Address GDN

## 1. Metainfo

- Status: preregistered, implementation complete, GPU contract pending
- Decision field: directional MQAR L1024 binding closure before Sudoku transfer
- Fixed setting: D128/L2/H4/K32/V32, native FutureSeed, 10 epochs, batch32, seed123
- Resource: one A100-SXM4-80GB, CUDA index0

## 2. Evidence And Hypothesis

P-FS2-007 raises balanced accuracy from `0.30625` to `0.48825`, but `2044/2047`
remaining errors are correct-value/wrong-key swaps. More cached values do not
close binding. P020 proves address geometry matters, while P021 shows that two
independent serial payload edits destructively overwrite one state.

The falsifiable hypothesis is that one payload bound under two jointly
committed, erase-orthogonal addresses improves key separability without a
second payload, state bank, cache, selector, or dense inverse state.

## 3. Mechanism

For each logical token, retain the official parent Q/K/V/decay/erase/write
paths and add only a second Q/K projection. Stack the two K addresses and apply
the analytic symmetric inverse square root of their `2x2` Gram under the
positive channel-wise erase metric. This gives `k1^T diag(b) k2 = 0`.

Both updates share exactly `v/sqrt(2)`, `w`, `b`, and the one parent decay.
They are encoded as adjacent microsteps in one pinned official `chunk_gdn2`
call; the second microstep has zero extra decay. Because cross erase is zero,
the two edits commute and equal one simultaneous rank-two block update. The
retained query is the symmetric sum of the two Q views. Added parameters are
exactly 65,536; persistent recurrent state remains 4,096 values per layer.

## 4. Novel Boundary

P021 learned an independent auxiliary K/V/erase/write path. Its auxiliary
payload reached `1.71x/3.62x` the parent and erased earlier writes. P028 has no
auxiliary payload or gate stream, preserves aggregate payload energy, and
makes column order algebraically irrelevant. P022 banks state, P024 replaces
linear addressing with a product hash, P025 adds correction state, P027 adds a
dense KxK inverse recurrence, and P-FS2-007 adds sparse replay. None tests this
atomic shared-payload live transition.

## 5. Prediction And Gates

The strict CUDA contract must prove the exact A100 identity, pinned FLA and
Zoology hashes, fixed data hashes, exact parent initialization, 65,536
parameters, zero state delta, two official `ChunkGDN2FunctionBackward` nodes,
Triton short convolution, nonzero auxiliary Q/K gradients, symmetric polar
identity/error `<=2e-4`, and official BF16 output/state column-swap relative
RMS `<=5e-3`. No fallback is allowed.

The single candidate passes only if all of the following hold:

- balanced accuracy `>=0.85`, future and past each `>=0.85`, joint exact
  `>=0.60`;
- balanced improves at least `0.10` over historical `0.7475` and current
  runtime `0.30625`;
- wrong-key swap fraction is at least `0.10` below historical `0.806931`, and
  total errors are lower than both references;
- both layers remain conditioned (mean condition `<=4`, maximum `<=100`),
  weighted cross erase `<=1e-3` mean/`5e-3` max, FutureSeed active, and state
  finite/nonzero with receiver raw state RMS `<=2x` current control;
- fit, post-warm wall and warmed-step ratios are each `<=2.0x`; peak allocation
  is `<=1.5x`.

Any contract, activation, quality, stability, or cost miss closes atomic
paired-address updates without rank, projection, polar epsilon, payload scale,
query aggregation, gate, seed, LR, loss, batch, epoch, width, or depth rescue.

## 6. Results

Pending.

## 7. Decision

Pending the one fixed A100 run. A pass admits one hard-Sudoku transfer gate; a
miss returns to a fundamentally different scalable recurrence rather than a
nearby variant.
