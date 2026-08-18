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

Pending strict CUDA contract, production probe and formal training.

## 6. Mechanistic Interpretation

Pending. A pass would show that a general second-order linear recurrent
transition transfers from binding to iterative constraint solving. A failure
would bound the strong MQAR result as task-specific and close Momentum-to-
Sudoku transfer without implying that another wrapper around first-order GDN2
is useful.

## 7. Cost And GPU

Pending. Report effective boards/s after compilation warmup, step timing
variation, peak allocated/reserved memory, sampled utilization and transport
bytes for both endpoint FS modes.

## 8. Provenance

Pending pushed source SHA, clean detached worktree, GPU UUID, run names,
checkpoint/config/metrics/source hashes and launch logs.

## 9. Decision

Approved as one high-information from-scratch Sudoku trajectory. Do not start
another arm until a registered gate yields a decision.
