# P-GDN3-039: Raven Recurrent Address Composer

## 1. Metainfo

- Status: preregistered and implemented, pending strict CUDA contract
- Date: 2026-08-14
- Benchmark: directional MQAR L1024 wrong-key regime
- Fixed setting: D128/L2, main H4/K32/V32 GDN2, native FutureSeed,
  10 epochs, batch32, seed123
- Arms: contemporaneous native GDN2 control, then one Raven-address candidate
- Resource: one task-mode GPU1, no concurrent model process

## 2. Evidence Boundary

P031/P036 close decoupled erase keys: erase, write and read need coherent row
ownership. P037 shows native K-wise erase and V-wise write control cannot be
scalarized. P038 creates two full live states and lowers wrong-key swap
fraction, but one global content hash delays learning and loses general
retrieval. The remaining issue is not raw state count; it is whether a key
occurrence can recover enough of its own binding history to construct a more
separable coherent address.

Existing Raven negatives do not answer this question. P017 is a stateless
softmax allocator over fixed GDN K rows. P019 runs a persistent Raven beside a
mature Sudoku GDN and injects its retrieved content into V; it never changes
the main address. Full Raven replacement tests Raven as the data plane. P039
instead keeps GDN2 as the only data plane and uses Raven only to provide a
learned recurrent context to the single Q/K address interface. It trains the
foundational composition from scratch on the validated L1024 binding regime,
not as a 100-step Sudoku graft.

## 3. Mechanism

Each layer first runs one pinned-official D128/H4/S16/top1 Raven over the same
hidden sequence. A zero-initialized bias-free D128-to-D128 adapter maps the
Raven retrieval to an address residual `r_t`. The main GDN2 then computes both
Q and K, including their unchanged Triton short convolutions, from

`x_address,t = x_t + r_t`.

Main V, decay, K-wise erase, V-wise write, output gate/projection, K32xV32
state and native FutureSeed remain functions of the original `x_t` exactly as
in the control. Thus every committed GDN edit still uses one coherent K for
write and erase, while query and key share the same recurrently enriched input.
Raven has no adjacent-layer state transport and does not write a payload into
GDN. It is an intra-layer address context, not a second data plane.

The candidate adds exactly 90,824 parameters/layer: 74,440 for official Raven
and 16,384 for the address adapter, or 181,648 total. Raven adds 4,096 state
values/layer, bringing total recurrent state to 8,192. It adds one official
Raven scan/layer and retains one official GDN2 scan/layer. Slots/top-k are fixed
at S16/top1 by the pinned kernel; there is no width, temperature, auxiliary
loss, selector, cache or task logic.

## 4. Falsifiable Prediction

If wrong-key swaps arise because repeated key occurrences cannot reconstruct a
stable binding context from token content alone, Raven retrieval should become
active, vary by token and board, alter both Q and K projections, and improve
balanced accuracy by at least 0.08 while reducing wrong-key swap fraction by
at least 0.05 and total errors. Future and past directions must both improve;
otherwise the controller learned an order shortcut rather than a generic
address mechanism.

If Raven and the adapter activate but quality misses, close recurrent address
composition. Do not tune slots, top-k, Raven width, adapter scale/target,
cross-layer Raven transport, seed, LR, loss, batch, main width/depth or
duration. If the adapter remains dead or the sparse state collapses, the
composition is not learnable under the fixed budget and is also closed.

## 5. Contract And Matched Initialization

Before formal training, exact pushed source in a clean detached worktree must
prove on GPU1:

1. exact CUDA index0/UUID, pinned FLA SHA and clean Zoology SHA;
2. exactly two official `ChunkGDN2FunctionBackward` paths and two official GSA
   backward paths, with six main Triton short convolutions and no fallback;
3. exact +181,648 parameters, 8,192 state values/layer and one GDN plus one
   Raven scan/layer;
4. every parent tensor byte-matches one serialized control initialization;
5. zero adapter gives bit-exact full output and, with finite nonzero incoming
   main state, bit-exact output and main terminal state;
6. stage1 gives finite nonzero adapter gradients; after one synthetic adapter
   step, both layers give finite nonzero Raven Q/K/V/decay/router and adapter
   gradients plus nonzero Q/K projection changes;
7. changing Raven recurrent state changes its output, and permuting Raven heads
   with the matching Raven output-projection blocks commutes with full output;
8. Raven slot state, main state and address residual are finite and vary by
   token/board; train/test and warmup-batch hashes match.

## 6. Activation And Quality Gates

At the fixed endpoint both layers must execute one Raven and one GDN2 scan.
Raven output and terminal RMS must be at least `1e-4`, terminal board variation
at least `1e-4`, normalized slot entropy in `[0.10,0.95]`, and mean maximum
slot mass below `0.80`. Address residual relative RMS and Q/K projection change
must each be at least `1e-3`, with token/board variation at least `1e-4`.
Main state must remain finite in `[1e-4,1e4]` RMS and native FutureSeed active.

All quality checks are binding:

- balanced accuracy at least `0.55` and at least control `+0.08`;
- future and past accuracy each at least `0.50` and control `+0.07`;
- joint exact at least `0.08` and control `+0.04`;
- fewer total errors;
- wrong-key valid-value swap fraction among errors lower by at least `0.05`.

## 7. Cost And Kill Gates

Elapsed, post-warm wall and independently warmed-step ratios must each stay
below `2.75x`; peak allocation must stay below `1.75x`. Any integrity,
activation, quality or cost miss closes P-GDN3-039 and writes a structured
abort. No nearby rescue is authorized. Only a full pass permits one matched
hard-Sudoku transfer.

## 8. Required Readout

Archive future/past/balanced/joint metrics, CE curves, total errors, wrong-key
swaps, Raven output/state/slot geometry, address residual and Q/K changes,
native FutureSeed diagnostics, parameter/state/scan counts, elapsed/wall/warm/
allocation ratios and source/config/checkpoint/score/log hashes.

## 9. Decision

Pending strict CUDA contract and the one fixed matched endpoint.
