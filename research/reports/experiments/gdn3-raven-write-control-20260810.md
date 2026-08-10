# P-GDN3-018: Persistent Raven Write Control for GDN3

## 1. Metainfo

- Status: failed at strict CUDA production-fit contract; no model score
- Date: 2026-08-10
- Branch: `codex/gdn3-raven-write-control-20260810`
- Benchmark: official/full-diversity hard 9x9 Sudoku 51-64 blanks
- Seed: 52 only
- Parent: D256/L12/H8/K32/V32 position-QK GDN3 with native terminal
  FutureSeed
- Parent checkpoint:
  `/huyang2/double-loop/models/gdn3-position-qk-d256l12-s12000-20260806T131527Z-9f2ee8d/checkpoints/train_state_step003000.pt`
- Parent checkpoint SHA256:
  `6339c3cb2b5fc5230a581d6633716483e35ff8e4522f06a9d7aaf26512f023da`
- Frozen control: `p-fs3-001-terminal-s3100-20260806T200859Z-3e167b6`

## 2. Hypothesis

Full Raven is a weak replacement for dense GDN2 on Sudoku, but it exhibits
real sparse recurrent correction. P017 retained GDN2 and borrowed only a
stateless soft allocation; that allocation approached one-slot collapse and
did not improve exact closure. The missing composition may be Raven's actual
persistent sparse key/value slot state rather than its router in isolation.

The falsifiable claim is that a compact Raven memory can retrieve selected
long-lived content while the stronger position-addressed GDN2 transition
performs dense constraint propagation. The Raven result must enter the live
GDN write payload, not merely add another output residual. If the controller
and all write adapters activate but exact closure stays flat, persistent sparse
retrieval is not the missing complement at this parent and budget.

This mechanism is not P006: it does not add a second dense GDN state or a
parallel residual expert. It is not P007: the controller owns independent
sparse slot state instead of rereading the incoming main GDN state. It is not
P017: allocation is realized by the official Raven recurrence and state bank,
not by rescaling existing GDN K rows.

## 3. Mechanism and Fixed Configuration

For every one of the 12 main blocks:

1. project normalized D256 hidden content to D64;
2. run one official FLA Raven with H4, K16, V16, eight slots and top1 routing;
3. normalize and pass the preceding block's Raven terminal state as the next
   controller's initial state;
4. project the retrieved D64 controller output through a zero-initialized
   bias-free D64-to-V256 adapter;
5. add that residual to the main GDN2 V payload before the unchanged official
   position-QK GDN2 chunk call.

Main Q/K, key decay, erase/write gates, output gate, main K32xV32 state and
native terminal FutureSeed are unchanged. The adapter's zero initialization
makes the full parent output and every main terminal state exact at migration,
including arbitrary finite nonzero main incoming states.

The fixed candidate has no tunable slot or width choice: D64/H4/K16/V16,
S8/top1. Expected additions are 618,768 trainable parameters and 12,288 Raven
controller-state values per board across 12 blocks. The contract must recompute
and assert both exact deltas.

## 4. Prediction and Decision Value

The one-step production probe should activate all 12 write adapters and all 11
cross-layer controller-state paths. Controller output/state and write
residuals must vary across boards and tokens. Sparse slot usage must remain
distributed enough to be a real memory bank rather than another collapsed
selector.

The experiment decides whether to retain a two-plane Raven/GDN architecture.
A quality pass authorizes a longer registered trajectory and mechanism
analysis. A miss closes compact Raven write control and redirects work to a
different stable recurrent state topology; it does not authorize slots,
top-k, controller width, injection target or scale sweeps.

## 5. Strict CUDA Contract

Before any training continuation, exact pushed source in a clean detached
worktree must prove:

1. one visible target GPU at CUDA index0 and no competing compute process;
2. pinned official FLA provenance and no fallback;
3. exactly 12 official main `GatedDeltaNet2` layers and 12 official Raven
   controllers, with their official chunk backward paths;
4. exact full-output and all-main-state parent identity at zero adapters,
   including finite nonzero incoming main state;
5. exact +618,768 parameters and +12,288 controller-state values;
6. finite nonzero first-stage gradients for all 12 V adapters, then finite
   nonzero Raven Q/K/V/router gradients after a synthetic adapter opening;
7. controller-state shuffling changes the controller output and main update;
8. packed Raven state round-trips losslessly and every state remains finite;
9. no Sudoku logic, search, repair, selector, reverse traversal, CPU model path
   or silent fallback.

Any miss closes before a production probe.

## 6. Exact Step3001 Production Gate

Resume the exact parent for one step with the semantic state-expert-upgrade flag,
preserving optimizer/RNG/data order. The formal metrics and checkpoint must be
complete and hashed. Require:

- 12/12 controller and V-adapter paths active;
- 11/11 incoming controller-state paths with RMS at least `1e-4`;
- V residual relative RMS in `[1e-4,0.5]` with finite board/token variation;
- controller slot-state normalized entropy at least `0.50` and maximum slot
  mass share below `0.80`;
- main terminal-state RMS no more than `4x` the parent;
- no NaN, OOM, fallback, SHA, data, GPU or optimizer migration drift.

The probe is migration and activation evidence only, not a benchmark result.

## 7. Matched Science and Cost Gates

Only after the contract and step3001 gate pass may one candidate-only exact
step3000-to3100 continuation run. Do not repeat the frozen control.

- Primary: hard51-64 macro loop5 exact improves by at least `+0.02`, with each
  official hard-range blank regression no worse than `0.01`.
- Alternate: mixed loop5 exact improves by at least `+0.03`, 61-64 does not
  regress, and same-board loop3-to5 wrong-cell correction is stronger.
- Activation/stability: all controller/write paths remain active, slot-state
  entropy is at least `0.50`, maximum slot mass share is below `0.80`, V
  residual relative RMS is at most `0.5`, and main state RMS is within `4x`.
- Cost: independent warmed elapsed overhead is below `60%`; peak allocated
  memory overhead is below `30%` versus the frozen control.

Any integrity, activation, stability, quality or cost miss discards P-GDN3-018.
There is no controller width, slot/top-k, injection target/scale, gate,
initialization, seed, LR, loss, batch, main width/depth or duration rescue.

## 8. Results

The exact pushed source `10cf31bf6cbcab7cda08ca3a36fad0f637308717`
reached the strict A10040 contract on CUDA index0, UUID
`GPU-93aad99c-9d1c-f2fb-1f10-fed39dde185c`. R1 exited before model
construction because an explicit `FLA_SOURCE_ROOT` pointed at a source tree
nested below another repository and therefore resolved the wrong outer Git
SHA. R2 used the immutable pinned-FLA marker and passed provenance, source,
GPU and exclusivity checks.

R2 then failed while compiling the official Raven `chunk_gsa` kernel. Its
eight-slot axis reached a Triton `tl.dot` whose matrix dimensions must each be
at least 16. The compiler raised `Input shapes should have M >= 16, N >= 16
and K >= 16` before the first full model output. Contract log SHA256 values
are `957583164566dc47658e27488879c71461270ba5a7e9c806ff07a8a55ad4f0e1`
for R1 and
`8990a167f084c67e36230484ac55ce8227005c66b693be788cd89dd3ba5f82cc`
for R2. The structured abort SHA256 is
`794473fb98b8fbbca336d5ed6b4d3706d1f982f774e9838c2cdf6ce2159561fd`.
No checkpoint, benchmark metric or science claim was produced.

## 9. Decision

Mark P-GDN3-018 `failed`, not `discarded`: the registered S8 mechanism is not
executable by the pinned official chunk kernel. Do not pad, patch Triton,
switch to recurrent fallback or reinterpret this as a score. Register one
successor with the kernel-minimum S16 fixed before execution; no other slot
count is authorized.
