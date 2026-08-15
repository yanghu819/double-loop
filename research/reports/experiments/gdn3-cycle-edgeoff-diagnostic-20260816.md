# P-DIAG-CYCLE-001: Same-Weight Cycle Edge-Off Diagnostic

## Purpose

P-GDN3-058 lowers wrong-key swaps but collapses total retrieval. This one-shot
diagnostic distinguishes two causes without training or changing any parent
weight: did the trained cycle reread itself break inference, or did learning
with the opened cycle path already move the native GDN2 base into a bad basin?

## Fixed Intervention

Load the exact P-GDN3-058 model-state SHA256
`64bb48df94f0e7baf2102e47ee5ca0df8d032fb172e234815cf91a28d5d5e1fe`,
rebuild the identical L1024 test set, set exactly eight `cycle_gate` parameters
to zero, and evaluate once. All embeddings, native GDN2 projections/gates,
reverse-state computation, FutureSeed and output weights remain bit-identical.
There is no optimizer, gradient, checkpoint write, threshold search or quality
claim.

## Interpretation Registered Before Execution

- balanced `>=.40` with at most 2,400 errors: read-time cycle reread is the
  primary damage;
- balanced `<=.15` with at least 3,200 errors: training co-adaptation already
  collapsed the native path;
- otherwise: both effects are material.

The diagnostic must use the sole registered A100 CUDA index 0 from an exact
pushed SHA in a clean detached worktree. Any source, checkpoint, test hash,
GPU, model-arm or parameter-hash mismatch invalidates it. The result cannot
rescue P-GDN3-058; it only chooses the next architecture boundary.

## Result

Complete. Exact pushed/read-back source
`086e25f77716b5eaab7aa218c22df70545c8ac68` ran in a clean detached
worktree on the sole A100-SXM4-80GB at CUDA index 0. The frozen model, formal
score, cases and test data all matched their registered hashes. The process
finished with status 0 and no NaN, OOM or fallback.

| condition | balanced | future | past | joint | errors | wrong-key swaps |
|---|---:|---:|---:|---:|---:|---:|
| trained cycle path | .07425 | .07450 | .07400 | 0 | 3703 | 380 |
| same weights, 8 gates zero | .07575 | .07600 | .07550 | 0 | 3697 | 388 |

The edge-off intervention repairs only 23 formerly wrong queries while
breaking 17 formerly correct queries. Of the 380 original wrong-key events,
362 remain wrong-key, 15 become another wrong value and only three become
correct. Balanced accuracy remains far below `.15` and total errors remain
well above 3,200, so the preregistered diagnosis is
`training_coadaptation_already_collapsed_native_path`.

This rules out the interpretation that the final cycle reread is the main
source of P058's collapse. Learning with the opened cycle route already moved
the native embeddings, projections, gates and readout into a poor joint
solution; disabling only the new route at inference cannot restore the frozen
native control. P058 remains closed. The next architecture must preserve or
isolate the native retrieval trajectory during learning, or train a genuinely
ownership-aware recurrent transition from scratch. Another gate, reverse
decay, reread scale or short continuation is not authorized.

## Checkpoint Drift Follow-Up

A tensor-only comparison against the exact shared initialization and frozen
native replay checkpoint confirms that the co-adaptation is broad rather than
localized to the two new cycle gates. All 57 parent tensors are shape-compatible;
the candidate has only the two expected additional `cycle_gate` tensors. The
cosine between the native and P058 training updates is only `.0138` for decay,
`.0798` for the write gate, `.1383` for erase, `.2797` for value payload,
`.2906` for write address and `.3264` for query address. Even embeddings and
the output path diverge materially (`.4167` and `.5353`). Norm updates remain
the most aligned family at `.9349`, so the failure is not a single malformed
normalization parameter.

The implication is stricter than "freeze Q/K": cycle training changes the
whole native ownership trajectory. A viable successor must either isolate all
native retrieval parameters during a graft, or replace the primary live
transition and train that architecture end to end. The latter is the higher
information next test because the former would test a constrained adapter,
not a stronger recurrent memory system.

- Tensor-only diagnostic script:
  `experiments/zoology_mqar/diagnose_checkpoint_drift.py`
- Diagnostic JSON SHA256:
  `d370d348030ac12901d598c7140ac106bd3fdffd67e9dcd5589b98efdf3099e4`

## Runtime And Provenance

- Run: `p-diag-cycle-001-edgeoff-20260815T214500Z-086e25f`
- Wall interval: `2026-08-15T21:38:50Z` to `21:41:19Z` (149 seconds,
  including imports and shared-filesystem cache reads)
- GPU sampler: 30 five-second samples; active-sample utilization mean
  `39.875%`, sampled peak `62%`, separately observed instantaneous peak
  `69%`, peak observed memory `1,235 MiB`, peak power `321.9 W`
- Diagnostic JSON SHA256:
  `c2174644173bdf2e757936cbb41f556b70b2372120b31f8ccb7537e805904720`
- Diagnostic log SHA256:
  `55b31e31340550acbec041ff289eeef4910c2e01807d6c8b198d9586fa01cd8f`
- GPU samples SHA256:
  `fa11c6df434f24fad6e66777c5482fc1393eef34afb43da2127702a9bc2ca14e`
