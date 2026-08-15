# Native GDN2 Head-Ownership Diagnostic

## Decision question

P-REPRO-001 makes the endpoint failure precise: `1546/2024` errors return a
valid value for the wrong key, and almost all choose the adjacent owner in the
same direction class. The remaining ambiguity is where ownership is lost:

1. different heads retain different owners, but the final multi-head fusion
   chooses the wrong one; or
2. each head's recurrent state already binds the value to the wrong key.

This distinction changes the next experiment. The first case supports a
query-dependent head-confidence fusion. The second requires changing the
within-head live recurrent transition. Another generic wrapper is not an
informative intervention until this boundary is known.

## Frozen protocol

`P-DIAG-OWN-003` is a zero-parameter, inference-only audit of exact
P-REPRO-001 replay-B:

- L1024 directional MQAR, four distinct bindings, 1,000 examples and 4,000
  queries;
- D128/L2/H4/K32/V32 native FutureSeed with pinned official FLA GDN2;
- exact frozen checkpoint, cases and trained-parameter hashes;
- one full replay plus, for each of two layers and four heads, one `drop-head`
  and one `only-head` replay, for 17 total variants;
- masks are applied to normalized/gated head outputs immediately before the
  official GDN2 output projection; weights, states and logits outside that
  diagnostic intervention are unchanged.

P-DIAG-OWN-002 originally required exact fresh-process replay. It was closed
before reading any mask score because a hook-free replay matched `3990/4000`
frozen predictions. The checkpoint, trained-parameter and dataset hashes were
exact, so P-DIAG-OWN-003 registers the observed official-kernel reload boundary
before rerunning masks: prediction agreement must be at least `99.5%`, each of
balanced/future/past/joint may drift at most `.005`, and error/swap counts may
drift by at most 20. Every mask and derived comparison uses this same-process
fresh baseline. This diagnostic is not a deployable selector and cannot be
reported as a new quality score.

## Registered prediction and gate

If ownership survives in a subset of final-layer heads, a final-layer
`only-head` oracle should repair at least 25% of the 1,546 baseline swaps. A
label-free top-1-margin fusion proxy over the full output and those four
final-layer variants must also improve balanced accuracy by at least `.02` and
reduce absolute wrong-key swaps by at least 10%. All three conditions are
conjunctive. First-layer masks are sensitivity evidence only because native
FutureSeed still carries the complete first-layer terminal state downstream.

- **Pass:** open exactly one query-dependent head-confidence mechanism that
  preserves the native recurrence and trains end to end.
- **Fail:** close head fusion/readout as the main bottleneck and make the next
  formal GDN3 candidate alter ownership inside each head's live transition.

No head-count, mask, confidence-temperature, seed, LR, loss or duration sweep
is authorized. Oracle results only locate available information; they are not
an allowed inference rule.

## Budget and kill criteria

The audit performs no backward pass and allocates no new parameters or state.
Wall budget is 30 minutes on one A800 80GB, with a five-second GPU sampler.
Wrong GPU identity, a second compute process, dirty or non-detached source,
unpushed source, FLA/Zoology drift, hash mismatch, OOM, NaN or incomplete
output kills the run as an integrity failure. A clean diagnostic decision is
archived before any successor is launched.

## Endpoint evidence

`P-DIAG-OWN-003` completed with `status=0` from exact pushed/read-back source
`e535977a89048880cb467bbd4d64bb634e244056`. The fresh full replay passed the
registered reload boundary: it agreed with `3990/4000` frozen predictions
(`99.75%`), balanced/future/past/joint were
`.4945/.4545/.5345/.041`, and errors/swaps were `2022/1546`. The frozen score
was `.494/.454/.534/.041` with `2024/1546`; every accuracy drift was at most
`.0005` and the error-count drift was two.

The information-localization result is real but not deployable. Across the
four final-layer `only-head` variants, a label oracle repairs `516/1546`
baseline swaps (`33.38%`) and reaches `.64825` balanced accuracy. Across every
masked variant, the oracle reaches `.8500`. However, the registered label-free
max-margin final-only proxy falls to `.42325` balanced accuracy and `1300`
swaps. It reduces swaps by `15.91%`, but does so while creating 285 additional
errors. Majority and drop-head proxies likewise fail to improve the full
model. Moreover, `441/1546` swaps (`28.53%`) keep the same wrong owner across
all four final-only variants.

The conjunctive gate is therefore:

- final-only oracle repairs at least 25% of swaps: **pass**;
- label-free proxy reduces swaps by at least 10%: **pass**;
- label-free proxy improves balanced accuracy by at least `.02`: **fail**.

Decision: **close head fusion, confidence routing and readout selection as the
main mechanism**. Correct evidence is distributed across heads, but the model
does not expose an answer-free confidence signal that identifies it. The
dominant ownership error is already present inside each head's recurrent
state. A successor must change the live within-head binding transition rather
than add another selector or output router.

## Systems and provenance

The 17 variants used `55.82 s` of measured inference and peaked at
`513,454,592` CUDA bytes allocated. A five-second external sampler recorded 31
samples: whole-launch SM utilization averaged `14.16%` because shared-data and
cold Triton startup dominate this inference-only audit; the ten active samples
averaged `43.90%`, peaked at `80%`, and used at most `1226 MiB`.

- run: `/huyang2/double-loop/runs/p-diag-own-003-head-ownership-20260815T111256Z-e535977`
- result SHA256: `03dd65df8807e65d6def00eebbf34ed44531efa2d2a5f858948b4d150a9650ae`
- diagnostic-log SHA256: `6ae8b110feea20c0322276fac86ae9151eec8cc7eefdf876b2cb1622d1596883`
- GPU-sampler SHA256: `52d071a12be70dcff3b82d745638c331aef85496f1d251086d3d3407808d450c`
- source-snapshot SHA256: `0798ee55a37cf218e8d3429124d712ada7824cceb452454951d89576c07f6a65`

The two P-DIAG-OWN-002 attempts remain preserved as pre-science evidence. The
first all-one hook path was rejected as an invasive baseline; the second
hook-free path stopped immediately after observing `3990/4000` fresh-process
agreement. No P002 mask score was used. The ten-prediction reload jitter is a
pinned-BF16/Triton process boundary, not checkpoint, data or parameter drift;
all P003 comparisons therefore use one same-process full baseline.
