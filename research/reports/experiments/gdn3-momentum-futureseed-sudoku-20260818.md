# P-GDN3-071 Momentum GDN3 + P-FS2-018 Phase-Asymmetric FutureSeed on Sudoku

## 1. Research Question

Does the second-order Momentum DeltaNet recurrence that solved the validated
long-range binding regime also improve native FutureSeed on the fixed hard
9x9 Sudoku scaling cliff, and can the converged Sudoku model later transport
only its Momentum plane without losing quality?

## 2. Evidence And Hypothesis

The canonical official-FLA GDN2 + native FutureSeed D192/L10 endpoint reaches
official 51-55/56-60/61-64 exact `.4492/.1543/.2637`, mean `.2891`, and mixed
loop5 exact `.3379`. A sequence of small pre-scan controllers, readout
residuals, parallel experts and state wrappers activated but did not close the
hard boards. They altered signals around the same first-order KxV recurrence.

In contrast, P-GDN3-059 changed the recurrent transition itself. Its pinned
Momentum DeltaNet keeps a primary state `S` and a velocity-like state `M`, and
improved directional MQAR L1024 balanced/future/past/joint from
`.494/.454/.534/.041` to `.94425/.95150/.93700/.82400`. P-FS2-017 then showed
that a model trained with full `[S,M]` can serve with `[0,M]` at nearly identical
quality and exactly half the cross-layer payload. P-FS2-016 also showed that
training M-only from scratch collapses, so the training intervention here is
fixed to full `[S,M]`.

The falsifiable hypothesis is that hard Sudoku is limited partly by a
first-order memory transition that cannot carry a stable correction direction
across tokens and layers. A learned second-order state should improve the
51-64 exact frontier and late-loop correction. If it does, the converged
Momentum plane may provide a cheaper receiver-facing FutureSeed summary.

## 3. Fixed Mechanism

Replace each of the ten official GDN2 layers with the exact pinned external
Momentum DeltaNet layer at commit
`c6e77fa261fb0c002fae1a14b6209a5b28d2edc9`. Each layer uses the native chunk
operator and returns `[S,M]` with shape `[2,B,H,K,V]`. Adjacent-layer
FutureSeed RMS-normalizes each plane, applies the existing learned per-head
gate, and imports both planes into the next layer. No Sudoku rules, search,
repair, selector, reverse traversal, side cache, second pass, extra loss or
silent kernel fallback is allowed.

The sole formal trajectory is from scratch with the existing canonical
D192/L10/H6/K32/V32, full-diversity 12k curriculum, effective batch128,
BF16, seed52, random traversal and loop5 equal CE. At the frozen endpoint,
P-FS2-018 changes only the inference edge to `[0,M]`; it does not train a
second model.

## 4. Registered Gates

Before formal training:

- source SHA256 for the external layer, chunk and recurrent files must match
  the frozen values, and host FLA must match
  `9c8e42e762fce087c27b673af4922795d9edb85e`;
- CUDA index0 must be the single requested A100 and all ten layers must be the
  exact external class with ten `Chunkmode_ruleFunctionBackward` paths;
- every layer must return `[2,B,6,32,32]`, all Q/K/V/alpha/beta/Momentum/eta
  gradients must be finite and nonzero, and all nine FS gate gradients must be
  finite and nonzero;
- state and Momentum RMS plus board variation must be finite and nonzero;
  same-weight full versus M-only transport must change the computation while
  transport fraction changes exactly `1.0 -> .5`;
- a two-step run through the production Sudoku trainer must produce a complete
  metrics JSON and checkpoint with no NaN, OOM or fallback.

Training gates are fixed before launch:

- step500: h50 loop5 exact at least `.60`, h53 blank accuracy at least `.48`,
  ten active Momentum layers and genuine loop-wise wrong-cell reduction;
- step3000: h53 loop5 exact at least `.02` or blank accuracy at least `.60`,
  positive h53 blank slope from step1000, and genuine loop correction;
- step6000: official 51-55 exact at least `.10` or fixed h58 exact nonzero,
  with finite CE and no hard-range blank collapse;
- step9000: positive hard exact or blank slope from step6000 and no stability
  regression;
- endpoint primary: mean official 51-64 loop5 exact at least `.3191`, at least
  `.03` above the canonical mean, with no individual official range more than
  `.02` below its baseline and genuine same-board late-loop correction;
- endpoint alternate: mixed loop5 exact at least `.3779`, `.04` above the
  canonical result, with 61-64 not below baseline and stronger same-board
  loop3-to-loop5 correction.

The formal architecture cost ceiling is at most `2x` matched GDN2 wall time
and `1.5x` peak allocation, with an absolute peak below 72 GiB. The endpoint
M-only replay passes only if it halves logical FS transport, changes no
parameters/state/scans, stays within `.01` mixed and mean hard exact, and no
official range regresses by more than `.015` from the full `[S,M]` replay.

Any integrity, stability, scheduled science or cost miss closes this exact
mechanism. There is no learning-rate, state scale, seed, loss, batch, width,
depth, duration, transport or nearby recurrence rescue.

## 5. Result

Closed at the formal stability gate.

The strict CUDA contract passed on the requested single A100. It verified ten
exact external Momentum layers, ten native Momentum backward paths, complete
Q/K/V/alpha/beta/Momentum/eta and FutureSeed-gate gradients, finite nonzero
`S`/`M` geometry, and the exact `1.0 -> .5` logical transport change between
full and M-only FutureSeed. The contract JSON SHA256 is
`5f729d879dd13286a9e20734fa778bdbbdc1a0679c3567f6246df20daae03150`.

The two-step production probe also passed and produced both a complete metrics
JSON and checkpoint. Its metrics/checkpoint SHA256 values are
`64e137750b305ecc27256b1a39f6b8fb9834e72b5b724c943e5fd42d4601f2c0` and
`e9cb6993bc359e29e062fc3aedbc7b5e92f6ff78d3e4349061388a763c189651`.
The eight-board probe exists only to prove the production path; its loop scores
are not a quality result.

The formal run
`p-gdn3-071-momentum-sudoku-d192l10-s12000-20260818T032813Z-733fa14`
failed before step100 with `RuntimeError: Momentum terminal state is
nonfinite`. No formal checkpoint or scheduled evaluation was written. The
orchestrator recorded exit status 1 and `rescue_authorized=false`. Consequently
there is no h50/h53/h58/h64, official-range, mixed, throughput-at-steady-state
or endpoint M-only quality result to report.

## 6. Mechanistic Interpretation

The exact second-order carrier that learns directional MQAR does not transfer
unchanged to five-loop hard Sudoku training. The important distinction is
stability, not task quality: a finite forward/backward contract and a two-step
probe did not predict finite recurrent state under the production trajectory.
The complete `[S,M]` edge repeatedly re-injects a second-order state across
layers and macro loops; the resulting dynamical system can leave its finite
region before the first science checkpoint.

This does not falsify Momentum as a sequence-memory mechanism, because there
is no converged Sudoku model to evaluate. It does falsify the exact naive
composition of the external Momentum recurrence with full-state native
FutureSeed under the registered canonical training setup. Any successor must
make bounded loop stability part of the recurrence design itself. Changing a
state scale or optimizer after observing this failure would be a rescue, not a
new mechanism.

## 7. Cost And GPU

The complete contract/probe/formal launch window ran from `03:28:32Z` to
`03:53:53Z` on one `NVIDIA A100-SXM4-80GB`, CUDA index0, UUID
`GPU-0da20a4f-5e67-e47d-7aab-8c6efa2864ad`. Across 301 five-second samples,
161 were compute-active; active utilization averaged `28.87%`, peaked at
`79%`, and observed memory peaked at `10,967 MiB`. These samples include
compilation, contract and probe activity and are not a steady-state training
throughput measurement. Because formal training failed before step100, the
registered wall-time and allocation ratios cannot be evaluated.

## 8. Provenance

- GitHub source/read-back SHA: `733fa141b8213ec6c345266caf31c6ad235867d3`.
- Source tree: `dad1b8eeaf88b3350a59d8824a820fd084eeb88d`.
- Clean detached worktree:
  `/huyang2/double-loop/worktrees/p-gdn3-071-4b3bdef`.
- Launch directory:
  `/huyang2/double-loop/artifacts/launch/p-gdn3-071/20260818T032813Z-733fa14`.
- Formal log SHA256:
  `52b76be1126bbe7aaf3817e8b5df56867f36b3b27a417cf0256414c3f3097f1a`.
- Abort JSON SHA256:
  `d7b8db92de571853bb0e3446fffcc45a6ecb05d44a931ce6619ee8ea2ab4e363`.
- Source snapshot SHA256:
  `5806ba53ac8f259945e49d3da8967e05258e44e2c7befbc666b9fcc033c9641a`.
- Pinned external Momentum SHA:
  `c6e77fa261fb0c002fae1a14b6209a5b28d2edc9`.
- Pinned host FLA SHA:
  `9c8e42e762fce087c27b673af4922795d9edb85e`.

## 9. Decision

Discard and close the exact P-GDN3-071 mechanism. Do not run state-scale,
learning-rate, seed, loss, batch, width, depth, duration, full-versus-M-only
transport or nearby Momentum rescue. `P-FS2-018` is not launched because its
required frozen endpoint does not exist. The next Sudoku experiment must use a
distinct bounded recurrent/FS mechanism with a pre-registered long-horizon
state-stability contract, not reinterpret the tiny production probe.
