# P-GDN3-014: Coupled Address-Row Expansion

## 1. Metainfo

- Status: discarded at strict CUDA contract; no production probe or formal run
- Date: 2026-08-07
- Branch: `codex/gdn3-coupled-address-rows-20260807`
- Source SHA: `078fe7e21bc1f4c77b37900b8a2bc318fad86759`
- Benchmark: official/full-diversity hard 9x9 Sudoku 51-64 blanks
- Compute: AIStation task-mode GPU1 only
- GPU UUID: `GPU-53e9f3b4-2966-65d3-6614-09c540921519`
- Seed: 52 only
- Parent: D256/L12/H8/K32/V32 position-QK GDN3 plus native terminal FutureSeed
- Parent checkpoint:
  `/huyang2/double-loop/models/gdn3-position-qk-d256l12-s12000-20260806T131527Z-9f2ee8d/checkpoints/train_state_step003000.pt`
- Parent checkpoint SHA256:
  `6339c3cb2b5fc5230a581d6633716483e35ff8e4522f06a9d7aaf26512f023da`
- Parent source SHA: `9f2ee8d1738032bc5f09b55db0b81d507780b376`
- Frozen control: `p-fs3-001-terminal-s3100-20260806T200859Z-3e167b6`

## 2. Evidence Boundary

P005 changes erase/write coherence, P007 reads inherited state before the scan,
P008 adds a post-scan consolidation transition, P009/P010 alter transition
composition, P011 routes payload across heads, and P012 opens a signed erase
spectrum. P006 adds a separate smaller expert. All either activate without
closing hard boards or fail a registered systems boundary.

P013 is the closest capacity test, but it duplicates H8 into two independent
H16 trajectories and hides the companion output behind a zero read gate. Its
real step3001 updates only that gate; the zero companion Q/K projections have
no first-order production gradient. P014 does not change that gate, its map or
its initialization. It removes the independent companion trajectory entirely
and expands the K row axis inside each original head's single live state.

Raw V/state-width expansion is also closed: P-SCALE-035 expanded payload
columns and gave only a transient checkpoint gain before endpoint regression.
That does not test address-row capacity. P-CAUSAL-014 explicitly left K/address
scaling open after rejecting payload-axis scaling.

## 3. Mechanism

Keep the parent D256/L12/H8/V32 architecture and its original K32 projections.
For each layer, reproduce the audited parent FLA Q/K normalization outside the
kernel, then form one K64 official transition:

`q64 = concat(q32_unit, q32_unit)`

`k64 = concat(k32_unit, soft_unit_ball(W_extra * address))`

`g64 = concat(g32, g32)`

`b64 = concat(b32, b32)`

where `W_extra: D256 -> H8*K32` is bias-free and zero-initialized, and
`soft_unit_ball(z) = z / sqrt(1 + ||z||^2)`. The operation is the original
pinned-official `chunk_gdn2`, called once with explicit parent scale
`1/sqrt(32)` and kernel-side Q/K normalization disabled because both banks are
already in their registered geometry.

Incoming K32xV32 state is padded with one zero K32xV32 row bank. Native
FutureSeed normalizes the base and extra K banks separately before transfer;
therefore a zero extra bank cannot alter the parent base RMS or gate. At
`W_extra=0`, the extra rows remain exactly zero, the base recurrence is exactly
the parent recurrence, and full output/base terminal state are bit-exact.

Unlike P013, the extra state has a fixed nonzero read basis from the first
forward pass and no read gate. The derivative of the official write term with
respect to `W_extra` is therefore visible to the task loss in the first real
optimizer step. Once active, base and extra rows participate in the same
rank-one erase/write transition, so this is coupled address capacity rather
than another independent head or expert.

The candidate adds exactly `12 * 256 * 256 = 786,432` parameters. Persistent
state increases by `12 * 8 * 32 * 32 = 98,304` values per board. V width, head
count, tokens, scans, official core count, data, traversal, loss and task logic
do not change.

## 4. Falsifiable Prediction

If address-row capacity is a bottleneck, all 12 extra K projections and state
banks should activate at step3001 with finite board/token/head variation, then
produce stronger same-board late-loop correction and at least `+0.02`
hard51-64 macro loop5 exact. If the rows activate but exact remains flat, one
additional coupled address bank is insufficient. If they do not activate in
one real step, the claimed first-order route is false. Either result closes the
mechanism; no address map, row count, normalization, scale or training rescue.

## 5. Strict CUDA Contract

Before any continuation, exact pushed source in a clean detached worktree must
prove all of:

1. CUDA index0 and the registered UUID are the only visible GPU and compute app;
2. pinned FLA source SHA is `9c8e42e762fce087c27b673af4922795d9edb85e`;
3. all 12 cores are exact official `GatedDeltaNet2`, each candidate terminal
   state exposes exactly one `ChunkGDN2FunctionBackward`, and no fallback exists;
4. parameter/state deltas are exactly 786,432/98,304, with no token, scan,
   head, V-width or core-count delta;
5. zero-init full output and every base terminal state are bit-exact to control,
   every extra state row is exactly zero, including finite nonzero parent
   incoming states padded with zero extra rows;
6. external parent-bank normalization is bit-exact to the untouched
   kernel-normalized parent and explicit scale remains `1/sqrt(32)`;
7. all 12 extra K projections receive finite nonzero first-stage gradients
   without opening any synthetic read gate;
8. an opened synthetic projection changes extra state/output while preserving
   finite state geometry and obeying K-row permutation equivariance to max error
   `<=3e-6`;
9. split-row FutureSeed preserves exact base normalization/gating when the
   extra bank is zero;
10. state tensors are finite K64xV32 and terminal RMS is at most `4x` control;
11. exact-resume model/optimizer/RNG/data-order migration writes complete
    step3001 checkpoint and metrics;
12. no NaN, OOM, source/data drift, CPU model path, concurrent GPU process,
    task dependency or silent fallback occurs.

## 6. Step3001 Production Gate

The one-step probe is migration and production-fit evidence only. It must show
all 12 paths enabled; extra K weight and residual relative RMS, extra terminal
state relative RMS, and board/token/head variation all finite and `>=1e-4`;
one official backward per layer; terminal RMS `<=4x` the matching parent; and
complete checkpoint/config/source hashes. Any miss closes P014 before formal
training. The probe score is not science evidence.

## 7. Matched Science And Cost Gates

Only after the contract and production probe pass may one candidate-only exact
step3000->3100 continuation run. Do not repeat the frozen control.

- Primary: hard51-64 macro loop5 exact improves by at least `+0.02`, and each
  official hard range blank accuracy regresses by no more than `0.01`.
- Alternate: mixed loop5 exact improves by at least `+0.03`, 61-64 does not
  regress, and same-board loop3->5 wrong-cell correction is stronger.
- Stability: every bank remains finite; terminal RMS stays `<=4x` control and
  extra/base state RMS ratio stays finite and `<=4`.
- Cost: independent warmed elapsed overhead `<80%` and peak allocated-memory
  overhead `<70%` versus the frozen control.

Any activation, stability, quality, integrity or cost miss discards P014.
There is no row-count, map, normalization, scale, init, precision, seed, LR,
loss, batch, width/depth or duration rescue.

## 8. Results

The implementation, checker and preregistration were committed, pushed and
read back at exact SHA `078fe7e21bc1f4c77b37900b8a2bc318fad86759`.
The detached worktree remained clean and GPU1 exposed only CUDA index0 with
UUID `GPU-53e9f3b4-2966-65d3-6614-09c540921519`.

R1-R3 exited before model construction or CUDA execution on explicit launch-
environment assertions: R1 used an interpreter without PyTorch, R2 treated the
non-git FLA snapshot as a git root and therefore read the parent repository
SHA, and R3 omitted one required persistent cache variable. Their logs and
non-science abort records are retained. None produced a model result or used a
GPU compute application.

R4 used the established Python 3.10/PyTorch 2.7 environment, pinned FLA marker
`9c8e42e762fce087c27b673af4922795d9edb85e`, all required persistent cache
paths, the exact pushed source and the exclusive target GPU. It reached the
binding identity check and failed with:

`external K32 normalization plus K64 zero bank changed full output`

Thus zero extra K projection and zero extra incoming state do not make this
K32-to-K64 invocation bit-exact to the parent function. The strict contract
exited status1 and released GPU1 naturally to 0 MiB with no surviving compute
application. The R4 log SHA256 is
`d0cb54ed2feed88bedc7f0145572e79a51324e195f0917a8fc2edc19595836eb`;
the abort JSON SHA256 is
`bbc04bd09d963dd1ebe27ad85c6e9ff372fcf1706f00155c0aa54642e9ce7caf`.
No step3001 metrics, checkpoint, benchmark score or visualization exists.

## 9. Decision

Discard P-GDN3-014 at the preregistered parent-identity gate. Do not change the
normalization placement, scale, row count, address map, initialization,
precision or tolerance after observing the miss. Do not run a production
probe, formal continuation, seed, LR, loss, batch, width/depth or duration
rescue. The result closes this exact external-normalization K-row embedding;
it does not establish that address-row capacity is irrelevant.
