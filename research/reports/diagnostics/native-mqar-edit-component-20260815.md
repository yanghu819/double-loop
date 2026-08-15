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

The exact pushed/read-back source
`2b7fef7562c0a7968719b24885579cf4e1a3c83e` completed with status zero on
the registered A800. Its hook-free replay passed the frozen boundary at
`3990/4000` prediction agreement. Balanced/future/past/joint remained
`.4945/.4545/.5345/.041`, with `2022` errors and `1546` wrong-key swaps.

The causal separation is large. Selecting the fixed owner-rank variant that
corresponds to each baseline swap's wrong value gives:

| intervention | swap repairs | repair fraction | wrong-owner query retained |
|---|---:|---:|---:|
| erase off, layer 0 | 400/1546 | .258732 | 835/1141 (.731814) |
| erase off, layer 1 | 9/1546 | .005821 | 1138/1141 (.997371) |
| erase off, both | 402/1546 | .260026 | 834/1141 (.730938) |
| write off, layer 0 | 1519/1546 | .982536 | 5/1141 (.004382) |
| write off, layer 1 | 48/1546 | .031048 | 1118/1141 (.979842) |
| write off, both | 1519/1546 | .982536 | 5/1141 (.004382) |

Erase-off passes the raw 20% repair threshold but fails every preservation
condition: only `78/1546` swaps are repaired while retaining the competing
owner, and self-owner correct-query retention is `.717391`. The coherent
erase-ownership route is closed.

Write-off passes all three collision conditions. It repairs `98.25%` of swaps,
beats erase by more than 72 points, and almost always destroys the competing
owner's own answer. Layer 0 alone produces the full effect; layer 1 contributes
little. The label-mediated selected summary reaches `.87425` balanced and
`.705` joint only because an oracle identifies which valid competing write to
remove. This is causal localization, not a valid quality score or inference
algorithm.

Decision: **open one separable-state topology, not a write selector**. The
dominant error is first-layer superposition: both payloads are useful, but one
shared trajectory makes them mutually destructive at readout. Another erase
key, gate controller, owner classifier or label-aware suppression is closed.
The successor must train end to end from scratch on this MQAR regime, store
each binding redundantly under independently learned addresses, and combine
reads without assigning a token to one global slot.

The 25 variants used `55.316 s` of measured inference and peaked at
`513,454,592/620,756,992` allocated/reserved CUDA bytes. The five-second
sampler recorded 30 samples: whole-launch SM utilization averaged `17.70%`
because source archival, NFS loading and cold Triton dominate; ten active
samples averaged `53.10%`, peaked at `80%`, and used at most `1226 MiB`.

- run: `/huyang2/double-loop/runs/p-diag-edit-001-erase-write-20260815T114809Z-2b7fef7`;
- result SHA256: `e0d524705d6032b2557b2706a3d0b4c9abdb30dd4ca434b1b26423add30a7b80`;
- diagnostic-log SHA256: `ba7f3e4f16855f7bc5165e076e341a3b7d024d4cfb99f372b81103c389cfff52`;
- GPU-sampler SHA256: `32f4b0cb8eb1f159fa520daf6b0f9de80d430ea176bb6ac83d5b54ade1809a7c`;
- source-snapshot SHA256: `7ad99e006df8de7dff2c0521fcf1e0c42d72e38bf0553ddfd848c3a189105224`.
