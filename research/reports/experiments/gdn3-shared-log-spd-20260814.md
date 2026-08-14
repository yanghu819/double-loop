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

The strict A10080 contract and fixed endpoint completed status0. The contract
proved exact parent identity, one shared 2,108-parameter storage object, two
official GDN2 backward paths, finite nonzero gradient from each layer, native
FutureSeed identity and bounded head-equivariant activation.

The contemporaneous native control reached balanced/future/past/joint
`0.17325/0.1820/0.1645/0`. The shared-metric candidate reached only
`0.1250/0.1465/0.1035/0`. Total query errors increased `3307->3500`.
Wrong-key valid-value swaps fell `769->587`, but their fraction among errors
improved only `0.232537->0.167714`, below the registered `0.10` reduction and
accompanied by broader retrieval failure.

The shared metric was active and stable: applied metric delta Frobenius norm
was `0.285376`, eigenvalues stayed in `[0.894175, 1.182312]`, condition was
`1.322237`, and both layers referenced the same raw CUDA address. Quality
therefore failed despite production activation rather than because the module
was inert.

Elapsed/post-warm-wall/peak-allocation ratios were
`0.9381/0.9418/1.0321`, but warmed-step ratio was `1.2738`, missing the
registered `1.15` limit. No NaN, OOM or fallback occurred.

## 8. Decision

Reject P-GDN3-033. A single smooth cross-layer metric suppresses layer-specific
address adaptation and delays the validation rise; tying geometry is not a
valid way to make FutureSeed state namespaces coherent. Close shared Log-SPD
without metric scale/rank/head/layer/training rescue and do not transfer this
candidate to Sudoku.

The contemporaneous control (`0.17325`) independently reproduces P020's
control (`0.1735`). This removes the carrier ambiguity for P020's per-layer
candidate (`0.48225`): its relative `+0.30875` signal is credible even though
the retired historical absolute gate rejected it. The next decision may test
that already-completed per-layer mechanism on Sudoku; it must not rerun or tune
the MQAR arm.

## 9. Provenance

- Carrier parent decision: P-DIAG-CARRIER-002, score SHA256
  `f26dd46bef652f8f88136fcd11a2a52b4dcd1f86b07563f8750424263031e255`.
- P020 exploratory mechanism signal:
  `/huyang2/double-loop/runs/p-gdn3-020-log-spd-exploratory-a100-20260812T025949Z-2af49c5`.
- Source SHA: `f357af4061bee624b7ffe771f7269f602bf0ff28`.
- Run:
  `/huyang2/double-loop/runs/p-gdn3-033-shared-log-spd-l1024-20260814T0706Z-f357af4`.
- Score SHA256:
  `b7e166268c2a4dad288064f6583c9784392e7793d1300839e82ce02ce0030d30`.
- Contract SHA256:
  `701d3c101b1ae72f76a6dd1442bf17279869de5a0e577acebfb8cdeb560a8e29`.
- Formal log SHA256:
  `d273fb9434c272113dd5538ea273704f7f4b4d3b860ca0923cb4acadb99da48f`.
- Candidate checkpoint SHA256:
  `d8a215cc11939b57510153f88597372ae7409cd7963f12836bd2e6424a4c4064`.
