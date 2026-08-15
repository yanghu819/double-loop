# P-GDN3-053: Redundant Independent-Address GDN2

## 1. Metainfo

- Status: preregistered; implementation pending strict CUDA contract
- Date: 2026-08-15
- Branch: `codex/gdn3-redundant-address-20260815`
- Benchmark: validated directional MQAR L1024 binding-error regime
- Model: D128/L2/H4x2/K32/V32, native FutureSeed
- Compute: one AIStation A800-SXM4-80GB, CUDA index 0 only
- Seed: 123 only
- Frozen control: P-REPRO-001 native replay B
- Control balanced/future/past/joint: `.494/.454/.534/.041`
- Control errors/wrong-key swaps: `2024/1546`

## 2. Evidence Boundary

P-DIAG-EDIT-001 shows that `1519/1546` wrong-key swaps disappear when the
competing owner's layer-0 write is deleted, but only `5/1141` of that owner's
own correct queries survive. The values are present; two nearby bindings are
mutually destructive in one first-layer state trajectory. Erase-off repairs
only 402 swaps and fails preservation, so erase-key and gate-control families
stay closed.

This experiment is not P-GDN3-013 restarted. P013 was a function-preserving
Sudoku graft: a duplicated address plus zero read gate and zero Q/K residual
gave the companion address no production gradient at step3001. P053 is a
foundational from-scratch topology. Both address banks are independently
initialized, both participate through a fixed read from the first update, and
the validated MQAR regime trains the architecture for all ten epochs. It does
not tune P013's gate or initialization after its result.

It is also distinct from P038's token router, P022's same-byte K16 split,
P049's smaller transient companion scan, and P006's zero-read auxiliary core.
Every token writes to both full K32 banks; there is no discrete assignment,
side scan, subordinate payload or residual readout.

## 3. Mechanism

Each native H4 head receives two physical K32xV32 state banks inside one
pinned official `chunk_gdn2` call:

- bank A uses the parent's Q/K projection and short convolution;
- bank B uses independently learned Q/K projection and short convolution;
- V, decay, erase and write tensors are shared and duplicated to both banks;
- both official bank outputs are combined by the fixed mean `(o_A+o_B)/2`;
- native FutureSeed transports the full H8 terminal state with duplicated H4
  seed gates.

The model adds 67,592 parameters and doubles recurrent state values per layer
from 4,096 to 8,192. It adds no token, scan, core call, router, selector,
temperature, task rule or fallback. The main edit stream is unchanged; only
the learned address organization is redundant.

## 4. Falsifiable Prediction

If first-layer ownership interference is caused by forcing two useful
bindings through one address trajectory, independently learned banks should
remain non-collinear and preserve both reads. Balanced accuracy should rise by
at least `.15`, total errors by at least 20%, and wrong-key swaps by at least
25%, with meaningful gains in both temporal directions.

If the banks diversify but quality remains flat, raw address redundancy is
not sufficient and this topology closes. If they stay collinear, the proposed
symmetry did not break under end-to-end learning and also closes. No bank
count, read mix, companion width, projection source, seed or training rescue
is authorized.

## 5. CUDA Contract

Before formal training, exact pushed source in a clean detached worktree must
prove on the registered GPU:

1. CUDA index0 and the exact A800 UUID are the sole visible device;
2. pinned official FLA SHA is
   `9c8e42e762fce087c27b673af4922795d9edb85e`;
3. both layers wrap official `GatedDeltaNet2` and expose exactly two total
   `ChunkGDN2FunctionBackward` nodes, one per layer;
4. frozen parent tensors map to the exact P-REPRO initialization hash while
   the secondary Q/K banks remain independently initialized;
5. parameter delta is exactly 67,592 and state geometry is exactly
   H8xK32xV32, or 8,192 values per layer and board;
6. gradients are finite and nonzero for both primary and secondary Q/K plus
   shared V/erase/write paths in both layers;
7. swapping both address banks and their state leaves the fixed-mean output
   equivariant within `3e-6`;
8. native FutureSeed carries the full H8 state, all outputs/states are finite,
   and there is no router or fallback.

This architecture is not parent-function preserving, so a zero-initialized
output identity would be the wrong contract. Exact common-parent parameter
mapping and deterministic data order are the matched boundary.

## 6. Fixed Endpoint

Run one candidate only from the serialized P-REPRO initialization on the exact
same L1024 data, warmup batch, 10 epochs, batch32 and seed123. Do not repeat the
reproducible control. Evaluate the same 1,000 boards and compare paired query
transitions against frozen replay B.

Activation requires two layers/two full banks, finite independent Q/K
geometry, token and board variation, both read RMS values `>=1e-4`, output and
state disagreement `>=1e-3`, maximum bank-state cosine `<.999`, and one active
native FutureSeed route.

## 7. Quality And Cost Gates

All quality checks are binding:

- balanced accuracy `>=.65` and gain over control `>=.15`;
- future and past accuracy each `>=.60`;
- joint exact `>=.15` and gain `>=.10`;
- total errors reduce by at least 20%;
- wrong-key valid-value swaps reduce by at least 25%;
- wrong-key fraction among errors falls by at least `.10`.

Elapsed, post-warm wall and independently warmed step must each stay below
`2.0x` frozen control; peak CUDA allocation must stay below `1.75x`. Any
integrity, activation, quality or cost miss closes P053 without bank/read/QK,
seed/LR/loss/batch/width/depth/duration rescue.

## 8. Required Readout

Report balanced/future/past/joint accuracy and CE, total errors, wrong-key
swaps and paired transitions; per-layer Q/K cosine and contrast, address
variation, bank read RMS/disagreement, state RMS/difference/cosine, native
FutureSeed activation; warmed throughput, elapsed, peak memory and sampled GPU
utilization; exact config/source/init/checkpoint/score/log hashes.

## 9. Decision

Pending strict CUDA contract and the single fixed candidate endpoint. A pass
admits one Sudoku transfer. A miss closes this topology and returns the next
decision to the causal evidence rather than a nearby parameter sweep.
