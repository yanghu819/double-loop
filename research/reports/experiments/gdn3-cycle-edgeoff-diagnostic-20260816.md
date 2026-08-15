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

Pending.
