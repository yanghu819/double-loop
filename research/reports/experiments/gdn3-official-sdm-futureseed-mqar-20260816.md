# P-GDN3-064: Official Sparse Delta Memory FutureSeed

## 1. Research Question

Can a genuinely sparse product-key delta state remove the residual wrong-owner
errors left by Momentum DeltaNet while retaining native cross-layer future
information transfer?

## 2. Mechanism Hypothesis

P059 established that changing the live recurrence, rather than adding a
readout or controller, makes directional L1024 learnable. Its remaining 223
errors are still dominated by 151 valid values retrieved under adjacent wrong
keys. A fixed K32xV32 matrix forces every owner into one dense address basis.
The candidate instead makes a sparse product-key slot bank the primary state:
each token selects eight of 1,024 slots and applies exact gated delta updates
only there. The hypothesis is that discrete sparse ownership reduces
cross-key interference while preserving a scalable recurrent update.

The complete layer-0 terminal slot bank is transported to layer 1. It is
RMS-normalized per board and head and multiplied by one learned FutureSeed
gate before the receiving sparse scan. There is no side state, dense GDN2
scan, replay admission rule, selector, reverse scan, search, repair, or
task-specific logic.

This is not covered by P-GDN3-050. P050 kept dense GDN2 as the primary
recurrence and added a 16-slot GSA sidecar over detached committed edits. It
collapsed balanced accuracy from `.494` to `.04425`. P064 replaces the primary
state itself with the official sparse delta recurrence and trains it from
scratch.

## 3. Fixed Implementation And Provenance

- External source:
  `facebookresearch/sparse-delta-memory@183e7df809131b80ad4393741029d0f20fc3640b`.
- The external source is CC-BY-NC-4.0 and is used only as a pinned read-only
  dependency. No external implementation is copied into this repository.
- Exact external tree and core-file hashes are asserted before import.
- Candidate: D128/L2, one sparse head, 1,024 slots/head, eight reads, eight
  writes, block size 64, output gate enabled, normalized readings, no learned
  initial memory, BF16 training.
- Native FutureSeed transports `[B,1,1024,128]` terminal memory from layer 0
  to layer 1 through one scalar gate. State size is exactly 131,072 values per
  layer, 32x the dense GDN2 state and 16x the P059 `[S,M]` state.
- Directional MQAR L1024/K4, 10 epochs, batch32, seed123, identical data and
  post-warm determinism reset to P059. Shared embedding, MLP, norm and readout
  tensors are loaded exactly from the frozen matched initialization; the new
  sparse recurrence uses its deterministic architecture initialization.
- This is one configuration and one seed. There is no slots/read/write/head,
  gate, activation, LR, loss, batch, width, depth, or duration sweep.

## 4. Falsifiable Prediction

If residual errors are caused by dense address collisions, sparse slot
ownership should preserve P059's high bidirectional retrieval while reducing
wrong-key valid-value swaps. A result that only beats the old `.494` GDN2
carrier, activates many slots, or lowers swap share by destroying recognizable
values is a failure.

## 5. Registered Gates

Integrity and activation:

- exactly one A800 at CUDA index 0 with the registered UUID;
- exact pushed clean detached source, external Git SHA, external source hashes,
  frozen artifact hashes, data hashes and warmup batch;
- PyTorch 2.8 and Triton at least 3.4 in an isolated environment;
- two official sparse delta memory layers and backward paths, exact parameter
  and state accounting, no fallback or dense GDN2 scan;
- finite nonzero gradients for read/write/value/decay/input/output projections
  in both layers and the one FutureSeed gate;
- both layers use more than 64 distinct slots on the diagnostic batch, finite
  nonzero state/board variation, finite read/write entropy, and one active
  FutureSeed route with gate in `[.05,.95]`.

Quality against frozen P059 (`.94425/.95150/.93700/.82400`, 223 errors, 151
wrong-key swaps):

- balanced/future/past/joint at least `.94/.93/.93/.82`;
- total errors at most 223;
- wrong-key swaps at most 100 and at most `.60` of all errors;
- at least one strict Pareto gain: balanced `>.94425`, joint `>.824`, or total
  errors `<223`.

Cost against P059:

- elapsed, post-warm wall and warmed-step ratios each at most `3.0x`;
- peak allocation ratio at most `4.0x`.

Any miss closes this exact configuration. No rescue is authorized.

## 6. Result

Pending.

## 7. Decision

Pending the strict CUDA contract and single formal endpoint.

## 8. Artifacts

Pending.

## 9. Lessons

Pending.
