# Native GDN2 Edit-Component Causal Diagnostic

## Decision question

P-REPRO-001 shows that 76.38% of endpoint errors return a valid value owned by
the adjacent same-direction key. P-DIAG-OWN-003 then closes head fusion: useful
evidence is distributed across heads, but no answer-free confidence reliably
selects it and 28.53% of swaps retain the same wrong owner in every final-head
variant. The remaining high-value ambiguity is inside the live per-head edit:

1. a neighboring binding destructively erases the true owner's evidence; or
2. its write remains superposed as a competing valid payload.

That distinction decides whether a successor should preserve coherent erase
ownership or replace the shared state organization. It is not answered by the
failed decoupled erase-key, write-controller, side-memory or readout families.

## Frozen intervention

`P-DIAG-EDIT-001` is a zero-parameter, inference-only causal audit of the exact
P-REPRO-001 replay-B checkpoint and cases. For each example, its four MQAR
bindings are recovered and ranked by write position. At the value token
(`write_position+1`) of one fixed owner rank, a hook sets either the erase
projection logit or write projection logit to negative infinity, making the
corresponding sigmoid gate exactly zero before the unchanged pinned official
GDN2 kernel. The audit replays:

- one hook-free baseline;
- erase-off and write-off for four fixed owner ranks;
- each intervention in layer 0 only, layer 1 only and both layers.

This yields 25 same-process variants. There are no weights, gradients, new
parameters, new state, alternate kernels, labels inside the model, or training
steps. Labels and the baseline wrong prediction are used only after inference
to associate a swap with its competing owner. Such selected comparisons are
diagnostic oracles and are forbidden as deployable inference rules.

The hook-free baseline must agree with at least 99.5% of frozen predictions;
balanced/future/past/joint drift is limited to .005 and error/swap-count drift
to 20. Hash, source, GPU and official-kernel provenance remain exact.

## Registered gates

The decision uses the preregistered both-layer intervention. Layer-only rows
are localization evidence and cannot be selected after seeing quality.

Open `coherent erase ownership` only when all conditions hold:

- erase-off repairs at least 20% of baseline wrong-key swaps;
- at least 15% of swaps are repaired while the competing owner's own
  originally-correct query remains correct;
- competing-owner query retention is at least 80%;
- an originally-correct query retains at least 80% accuracy when its own
  owner's erase is disabled.

Otherwise, open `separable state topology` only when write-off repairs at
least 25% of swaps, exceeds erase repair by at least five points, and retains
at most 60% of the competing owners' originally-correct queries. This signature
means removing the collision fixes one association by sacrificing another;
it does not authorize a learned selector or another write controller.

If neither gate passes, close local erase/write edits as the primary cause and
treat the failure as distributed binding/decoder entanglement. No owner rank,
token position, layer subset, threshold, gate scale, seed, LR, loss, width,
depth or duration rescue is authorized.

## Budget and next decision

The wall budget is 30 minutes on exactly one A800 80GB with a five-second GPU
sampler. Wrong GPU identity, a second compute process, dirty or attached source,
unpushed source, hash/FLA/Zoology drift, OOM, NaN or incomplete output kills the
run as an integrity failure. A clean result is archived before one successor
is implemented:

- erase gate pass: a bounded, coherent ownership-aware live erase transition;
- write collision pass: a differentiable separable state topology that keeps
  both associations, not a write selector;
- neither: stop local gate/state wrapper iteration and revisit the binding
  representation at a more foundational level.

## Endpoint evidence

Pending.
