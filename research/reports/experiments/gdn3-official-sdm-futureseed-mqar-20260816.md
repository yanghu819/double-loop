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
- The CUDA contract reports whether the official training operator exposes the
  committed layer-0 bank as an autograd-connected terminal output. Forward-only
  bank transport is allowed to answer the sparse-state quality question, but a
  pass without producer-side terminal credit is not evidence of a better FS2
  learning rule and will be labeled accordingly; the external operator is not
  patched to manufacture that credit.
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

R1 stopped before CUDA because the isolated environment's `LD_LIBRARY_PATH`
made system `curl` load an incompatible `libffi`; the launcher now runs that
single network preflight with the library path unset. R2 reached the exact
official training operator, then Triton rejected a BF16/FP32 mixed dot in the
upstream fused dual-matmul kernel. The mechanism had not produced logits or a
quality score. Root cause was the host adapter's outer autocast: it recast the
official operator's deliberately FP32 `QB = QK @ B_matrix` intermediate to
BF16 while `retrieved` remained FP32. R3 keeps the external source and all
science settings fixed, but disables autocast only at the official
`gated_write_read` boundary. The strict contract now also asserts this
precision boundary and a BF16 memory input.

R3 then completed that official CUDA forward but the newly written graph
checker falsely reported only the two loss nodes. Unlike the repository's
established graph walkers, it retained only `id(node)` rather than the autograd
node objects; Python wrapper-id reuse truncated traversal before the sparse
backward nodes. R4 changes only this contract harness bookkeeping. The model,
external source, precision boundary and all science settings remain fixed.

R4 source `cbec8008095ef40551c749626d785184b34788ba` passed the strict
contract and completed the fixed endpoint on A800 index 0, UUID
`GPU-c1d7c624-a393-befa-3807-7e00602d65ca`. Both official sparse layers and
their backward paths were present. The one FutureSeed route had finite
nonzero gradient RMS `.0025482`; layer 0 and layer 1 touched 66 and 214 slots
on the diagnostic batch, and the trained receiving gate was `.4765625`.
Integrity, activation and every cost check passed.

The quality result is decisive and negative:

| arm | balanced | future | past | joint | errors | wrong-key swaps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P059 Momentum + native FutureSeed | .94425 | .95150 | .93700 | .82400 | 223 | 151 |
| official SDM + full-bank FutureSeed | .26825 | .02850 | .50800 | 0 | 2,927 | 1,126 |

Validation accuracy rises from `.02450` to `.26825`, so the sparse recurrence
does learn a real causal retrieval path. It does not learn the future path:
future exact is `.002`, while past exact reaches `.181`. The candidate's
wrong-key share among errors is lower (`.38469`), but only because broad
retrieval failure replaces the small P059 tail. The paired audit records 2,770
P059-correct queries becoming wrong and only 66 P059-wrong queries becoming
correct.

Elapsed/post-warm/warmed-step/peak-allocation ratios versus P059 are
`1.5064/1.5094/1.8086/1.0327x`. Formal training takes `188.91 s`; the warmed
benchmark reaches `539.67` examples/s and peak allocated memory is
`1,031,413,760` bytes. The 100 five-second telemetry samples have mean/peak
utilization `30.64/97%`; 81 active-memory samples average `37.83%`, observed
memory peaks at `1,876 MiB`, and mean/peak power is `94.50/179.55 W`.

## 7. Decision

Discard P-GDN3-064. R1-R3 remain non-science integration or harness failures;
R4 is the sole quality result. Close this slots/read/write/head/block and
full-bank transport configuration without rescue. The result is not evidence
for improved FS2 credit because the official terminal bank is detached from
the producer graph, and it is not a better GDN3 because every quality gate
fails.

The next decision must not tune sparse capacity. The useful distinction is
that P064 learns causal storage but cannot make a huge terminal slot bank
receiver-readable as future evidence. Test one complete closed-loop primary
transition in which prediction and committed owner coordinates remain
coherent inside every token update.

## 8. Artifacts

- run:
  `/huyang2/double-loop/runs/p-gdn3-064-official-sdm-fs-r4-20260816T065300Z-cbec800`;
- source SHA: `cbec8008095ef40551c749626d785184b34788ba`;
- external SDM SHA: `183e7df809131b80ad4393741029d0f20fc3640b`;
- contract/score SHA256:
  `87ea54d129f640cba895d1100cc4cdc2a6c3ad24f3fd29ac267de7f94d3d59c1` /
  `cb22182e62cf5123fff35fe87e30802fa45570384e810d5aa26c5c46d1712469`;
- checkpoint/cases SHA256:
  `ca5955e8a1e40f440847d728401227e96481a34900d07d0f959e51d1f866fe48` /
  `737029a8a67391d1304ccbe05a506cd466215794278e8e31d82df5e87f122daf`;
- formal log/GPU samples/source snapshot SHA256:
  `03c8ca71cc0f92d4664da5c3c29d891d834c82d0cffe55f2ef2e62b9fc487d85` /
  `15a0fce8687440263468a37f328a94cc5fec2b05550e9382d84f91b9701180a1` /
  `36fa1e7dba438516102376d75d9e3dfa2076584c1d917840d5489fa04a92aff3`.

## 9. Lessons

More address slots are not a substitute for a learnable ownership transition.
P064 has 16x P059's recurrent values, activates hundreds of sparse slots and
fits the cost budget, yet it cannot align future writes with receiver reads.
Its direction asymmetry is especially informative: causal past retrieval is
partially learned, while cross-layer future retrieval remains near chance.

Native FutureSeed works when the transported state has the same semantics as
the receiving transition. P059's compact momentum component carries a learned
update trajectory; P064's full sparse bank is a large endpoint snapshot with
weak receiver-native indexing. Continue with a complete end-to-end recurrence,
not another cache, slot-count sweep or producer-state codec.
