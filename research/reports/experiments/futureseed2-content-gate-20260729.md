# FutureSeed2 Content-Adaptive Trust Gate

## Decision question

Does FutureSeed fail on the hardest boards because seed quality varies by
sample, while the current model applies one fixed trust value per layer/head?

## Mechanism hypothesis

The rejected static KxV gate assumes that one recurrent-state coordinate is
always useful or harmful. GDN2 state coordinates are content-dependent, so
that assumption is too rigid. FutureSeed2 instead reads five generic summary
statistics from each sample/head state and predicts one bounded trust offset
for that sample/head:

- log RMS;
- signed mean divided by RMS;
- mean absolute value divided by RMS;
- row-energy spread;
- column-energy spread.

The offset only changes the existing FutureSeed head gate. It does not rotate
the state, add recurrent memory, change the official FLA kernel, or use Sudoku
rules. Zero-initialized weights make it exactly FutureSeed1 at launch.

## Prediction

If seed reliability is sample-dependent, the learned gate should show nonzero
between-sample variation and improve the 61-64 blank exact score without
reducing the 51-55 score. The intervention adds only 270 parameters.

## Budget and kill criteria

- GPU1 only, strict official FLA GDN2, no fallback and no CPU model smoke.
- Resume the frozen step9000 checkpoint and train exactly 100 optimizer steps.
- Compare against the already completed identity continuation with identical
  data, RNG, optimizer, evaluation, and checkpoint.
- Stop if hard-range mean exact does not improve, 61-64 exact falls, or the
  gate remains effectively sample-invariant.
- Do not sweep features, scale, seed, learning rate, loss, or run length.

## Paper claim if successful

A content-adaptive trust decision improves FutureSeed while preserving its
cheap cross-layer state reuse: future context is useful, but recurrent memory
quality must be judged per sample rather than by a globally fixed mask.

## Integrity gates

- CUDA zero-init gate is bit-exact to FutureSeed1.
- Activated gate matches the direct Torch formula with max absolute error 0.
- State, base FutureSeed logit, and content-weight gradients are finite and
  nonzero.
- The full-stack checkpoint migration has exactly one missing tensor:
  `reasoner.future_seed_selector.content_weight`; optimizer groups expand.
- Exact official `fla.layers.gdn2.GatedDeltaNet2`, chunk backward, Triton
  convolution, clean detached SHA, and no-fallback contracts pass.
- Parameter count is `5,461,958`, exactly 270 above the control.

## Result

Run:
`futureseed2-content-s9100-20260728T1647Z-b0e2f0a`

| metric | FutureSeed1 control | content FutureSeed2 | delta |
|---|---:|---:|---:|
| train CE | 0.6435 | 0.6480 | +0.0045 |
| mixed loop5 exact | 0.2520 | 0.2227 | -0.0293 |
| official 51-55 exact | 0.3672 | 0.3613 | -0.0059 |
| official 56-60 exact | 0.1309 | 0.1445 | +0.0137 |
| official 61-64 exact | 0.1973 | 0.1699 | -0.0273 |
| three-range mean exact | 0.2318 | 0.2253 | -0.0065 |

The mechanism is active:

- gate delta RMS: `0.03646`;
- seed relative change: `0.01513`;
- content-feature batch standard deviation: `0.07143`;
- resulting gate batch standard deviation: only `0.00065`.

On 61-64 blanks, content gating partially recovers the static-gate damage but
does not reach the control:

```text
FutureSeed1:       0 -> 0 -> .0566 -> .1855 -> .1973
static FS2:        0 -> 0 -> .0469 -> .1172 -> .1387
content FS2:       0 -> 0 -> .0430 -> .1484 -> .1699
```

Shared 64-blank case `b0048` is diagnostic. FutureSeed1 reduces wrong cells
`32 -> 12 -> 3 -> 0 -> 0`; content FutureSeed2 reaches
`32 -> 14 -> 5 -> 4 -> 5`. It finds a useful early direction but disrupts the
last two loops.

## Decision

Discard this content-adaptive head-trust gate. It fails the preregistered hard
mean and 61-64 criteria, and its output is nearly sample-invariant despite
nontrivial input-state variation. Do not sweep features, gate scale, learning
rate, seed, loss, or continuation length.

Together with P-FS2-001 and P-FS2-002, this closes simple state transport,
static state-entry selection, and scalar trust modulation. The next
decision-changing FutureSeed2 hypothesis must improve the generated seed
content itself, for example with an identity-initialized low-rank residual,
rather than place another gate around the existing seed.

Local comparison:
`runs/futureseed2-content-20260729/index.html`.
