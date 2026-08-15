# P-GDN3-059: Momentum-State FutureSeed

## 1. Research Question

Can a second-order live recurrent transition preserve key/value ownership at
L1024 better than GDN2 while retaining native cross-layer FutureSeed?

## 2. Mechanism Hypothesis

The local failure is not lack of value capacity. Native replay retains many
valid values but assigns them to adjacent wrong keys. P058 further shows that
adding an ownership reread changes the whole learned retrieval path, so an
inference-time edge-off cannot recover it. The candidate therefore changes the
primary token-scan update itself. Alongside matrix state `S`, it maintains a
momentum state `M`; committed residuals accumulate in `M` before updating `S`.
The hypothesis is that consistent ownership corrections reinforce while
single-token interference is damped. Native FutureSeed transports the stacked
`[S, M]` terminal state to the next layer.

This is not covered by the failed families. P025/P056 add auxiliary memory to
the GDN2 transition, P037 scalarizes its edit, and P058 adds a reverse sweep and
reread. P059 instead trains a distinct second-order primary recurrence end to
end. It adds no selector, cache admission rule, search, repair, reverse scan or
task-specific logic.

## 3. Fixed Implementation And Provenance

- External Momentum DeltaNet source:
  `HuuYuLong/MomentumDeltaNet@c6e77fa261fb0c002fae1a14b6209a5b28d2edc9`.
- The external repository has no declared license in this checkout. It is used
  read-only as a pinned experimental dependency and is not redistributed or
  copied into this MIT repository.
- External source hashes are fixed in the checker for the layer, chunk kernel
  and fused recurrent kernel.
- Current pinned FLA remains the host package for shared modules and cache;
  only the external momentum layer and operator namespace is added.
- Candidate: D128/L2/H4/K32/V32, native state-and-momentum FutureSeed, chunk
  training, directional MQAR L1024, 10 epochs, batch32, seed123.
- The frozen P-REPRO initialization is loaded exactly into every same-name,
  same-shape shared parameter. Momentum-only gates use their deterministic
  architecture initialization. This is one from-scratch architecture run,
  not a short zero-init continuation.
- Candidate has 8,192 recurrent values per layer, exactly twice the GDN2
  matrix state, and 61,912 fewer total parameters than the frozen carrier.

## 4. Falsifiable Prediction

If second-order state dynamics address binding interference, balanced accuracy
must rise by at least `.10`, both future and past retrieval must improve rather
than trade off, and valid-value/wrong-key swaps must fall as a share of all
errors. Merely activating `M`, reducing one direction's errors, or improving
loss is not sufficient.

## 5. Registered Gates

Integrity and activation:

- exactly one visible registered GPU at CUDA index 0;
- exact pushed clean detached source and frozen artifact hashes;
- exact external Git SHA and three external source hashes;
- exact shared-parent tensor mapping, parameter count and 2x state geometry;
- two chunk momentum backward paths, gradients through Q/K/V and all four
  momentum gates, one active FutureSeed route;
- chunk/recurrent output and final-state relative RMS difference at most `.05`;
- finite nonzero `S` and `M`, with `M/S` RMS ratio in `[.01, 20]`.

Science:

- balanced accuracy `>=.65` and gain `>=.10` over frozen `.494`;
- future and past accuracy each `>=.62`;
- joint exact `>=.15` and gain `>=.10`;
- total query errors reduced at least 20%;
- wrong-key swap fraction of errors reduced by at least `.10`.

Cost:

- elapsed, post-warm wall and warmed-step ratios each `<2.0x`;
- peak allocation ratio `<1.5x`.

Any miss closes this mechanism. There is no momentum coefficient, gate,
state-size, width, seed, LR, loss, batch, depth or duration rescue. A complete
pass opens one hard-Sudoku transfer; otherwise the next decision returns to a
different scalable ownership transition.

## 6. Result

Pending strict CUDA contract and the single fixed endpoint.

## 7. Decision

Pending.

## 8. Artifacts

Pending exact source, run, score, checkpoint and GPU-sampler hashes.

## 9. Lessons

Pending.
