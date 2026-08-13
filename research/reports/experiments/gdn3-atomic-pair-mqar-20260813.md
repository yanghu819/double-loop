# P-GDN3-028: Atomic Shared-Payload Paired-Address GDN

## 1. Metainfo

- Status: completed, rejected at the registered activation/quality/cost gate
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

The strict A100 contract passed before training. It verified the exact target
UUID, pinned official FLA/Triton path, two `ChunkGDN2FunctionBackward` nodes,
exact parent initialization, exactly 65,536 added parameters, zero recurrent
state delta, finite nonzero auxiliary-Q/K gradients, and BF16 column-swap
output/state relative RMS `6.85e-5/1.01e-5`.

The fixed 10-epoch endpoint completed without NaN, OOM, fallback, source, or
data drift. Candidate balanced/future/past/joint accuracy was
`0.04975/0.0470/0.0525/0`, versus the same-runtime native-FutureSeed control's
`0.30625/0.3115/0.3010/0` and the locked historical reference's
`0.7475/0.7415/0.7535/0.339`. Total candidate errors increased to `3,801`,
from `2,775` in the same-runtime control and `1,010` historically.

The mechanism nevertheless produced a sharp diagnostic separation. The
wrong-key valid-value swap fraction among errors fell from `0.458018` in the
same-runtime control and `0.806931` historically to `0.074980`; wrong-key
swaps fell to `285`, versus `1,271/815`. Atomic addressing therefore suppresses
cross-key confusion, but the full model no longer learns a useful value/read
map. This is not closure: it trades binding errors for a much larger set of
non-swap errors.

The trained geometry also missed its preregistered stability bounds. Layer
mean conditions were `8.29/10.50` (required `<=4`), maximum conditions were
`170.62/100.15`, polar diagonal errors were `0.01953/0.02344`, and the second
layer's maximum weighted cross term was `0.00550`. Peak allocation was
`1.739x` control, above the `1.5x` ceiling; warmed-step, fit, and post-warm wall
ratios were `1.832x/1.535x/1.535x` and stayed within their `2x` ceilings.

Provenance hashes: decision
`fb60b9e51f07bc2a4e700acf0b8df527dd352fed21d7fa015c61e5e8daa9be96`,
contract `5210d0b7fd5f8055befc83c330d5ef7817f0db4aee709f044349835e6e66594c`,
checkpoint `e04e95dec6d91b7aa95fac5926ca126467ad0705f738df849f32278e4f888d38`,
formal log `207d02b11bd81c8ca370bc1d0e39746c9f807f6c3a5b79cd8deeea82a3a5e8cf`,
and source snapshot
`6bd580cdaeee641a47a83896371fc8c2d5721de2102fea8c16ea7b104862335a`.

## 7. Decision

Reject P-GDN3-028 and close the atomic paired-address family. Do not rescue
rank, auxiliary projection, whitening, epsilon, query aggregation, payload
scale, gate, order, seed, LR, loss, batch, duration, width, or depth. The useful
result is the decomposition: suppressing wrong-key swaps alone is insufficient
when the address transformation damages value/read learnability. The next
mechanism must organize interference without duplicating or whitening each
token's address inside the same update.
