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

`P-DIAG-OWN-002` is a zero-parameter, inference-only audit of exact
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

The full replay must match all 4,000 frozen predictions and the registered
`.494/.454/.534/.041` endpoint exactly. Any mismatch is an integrity failure.
This diagnostic is not a deployable selector and cannot be reported as a new
quality score.

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
