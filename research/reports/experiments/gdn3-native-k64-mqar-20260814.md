# P-GDN3-032: Native Coherent K64/V32 Main-State Expansion

## 1. Question

Does doubling the native GDN2 address-row dimension from K32 to K64, while
holding D128/L2/H4/V32, the official recurrence, the single scan, and native
FutureSeed fixed, reduce directional-MQAR binding interference enough to close
the L1024 task?

## 2. Motivation

The direct `GDN_decouple_k` test P-GDN3-031 falsified unconstrained separate
erase and write/read keys: the branches became nearly orthogonal and retrieval
collapsed. P-DIAG-ADDR-001 then found severe K32 producer/receiver key-Gram
collapse but no evidence that cache admission, write survival, or receiver
reprojection was the causal repair. This leaves one bitter-lesson hypothesis:
the learned main state may simply need more coherent address rows.

P-CAUSAL-014 is not this test. It held K32 fixed and expanded V32 to V64;
quality regressed, so payload-axis capacity is closed. P-GDN3-014 is also not
this test. It tried to graft zero rows onto a trained Sudoku parent and failed
the exact parent-function migration gate before training. No native K64/V32
GDN2 has been trained from scratch on the validated L1024 binding regime.

## 3. Mechanism

Instantiate the unmodified pinned official FLA `GatedDeltaNet2` with
`hidden_size=128`, four heads, `head_dim=64`, and `expand_v=0.5`. Per head the
state is K64xV32. The native recurrence remains

```text
S_t = (I - k_t (b_t * k_t)^T) Diag(exp(g_t)) S_{t-1}
      + k_t (w_t * v_t)^T
```

Read, erase, and write therefore retain one learned normalized key namespace.
There is one official chunk scan per layer, one native terminal-state
FutureSeed edge, no cache, replay, selector, auxiliary state, second write,
second scan, or task rule.

## 4. Falsifiable Prediction

If K32 address-row capacity is the binding bottleneck, the extra learned rows
should reduce correct-value/wrong-key swaps and raise both directional
accuracies, not merely change the composition of failures. The fixed endpoint
must reach balanced/future/past accuracy at least 0.85, joint exact at least
0.60, balanced accuracy at least 0.10 above both historical and current-runtime
native K32 references, at least 0.10 lower wrong-key swap fraction than both,
and fewer total errors than both.

If those conditions fail, K-axis main-state expansion is closed. There is no
K48/K96, head-count, state-size, seed, LR, loss, batch, epoch, width, depth, or
duration rescue.

## 5. Fixed Protocol

- Directional MQAR L1024, four key/value pairs, 10,000 train and 1,000 test
  examples with the locked data hashes.
- D128/L2/H4/K64/V32, batch32, ten epochs, seed123, native FutureSeed.
- One candidate only. Reuse the locked historical K32 and contemporaneous
  P-FS2-007 native K32 references; do not rerun or select a control.
- Candidate parameters: exactly 770,384. Persistent recurrent state: exactly
  8,192 values per layer, twice K32. Logical scan count remains one.
- Fit, post-warm wall, and warmed-step ratios must each be at most 2.00; peak
  allocation ratio must be at most 1.50 versus the current-runtime control.

## 6. CUDA Contract

Before science, require exactly the registered task-mode A100 at CUDA index0,
the pinned FLA wheel/source/ops hashes, exact Zoology SHA, two official
`GatedDeltaNet2` layers, two `ChunkGDN2FunctionBackward` nodes, Triton Q/K/V
short convolutions, and no fallback. Assert exact K64/V32 geometry, 8,192 state
values per layer, 770,384 parameters, finite nonzero gradients in the added
Q/K/decay/erase rows plus V/write and native FutureSeed, a finite nonzero-state
path, fixed data hashes, and genuine future dependency.

Any contract or systems-integrity miss aborts before training. Any completed
quality or cost miss rejects the mechanism without rescue.

## 7. Results

R1 source `e80baf7e` exited while sourcing the launch environment because the
GPU name containing spaces was not quoted. It occurred before contract model
construction and allocated zero GPU memory, so it is retained as a non-science
orchestration abort. R2 changes only that quoting; mechanism, data, budget and
all gates remain frozen.

## 8. Decision

Pending.

## 9. Provenance

Pending exact pushed source, clean detached worktree, run, checkpoint, and
artifact hashes.
