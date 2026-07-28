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
