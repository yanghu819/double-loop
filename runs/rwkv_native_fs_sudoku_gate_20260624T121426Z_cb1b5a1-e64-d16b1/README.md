# Native RWKV FutureSeed Sudoku Gate

This run tests the corrected FutureSeed definition: forward RWKV token mixer, where the terminal recurrent state of RWKV layer `i` seeds layer `i+1`. It is not a right-to-left scan.

## Decision

Discard as positive evidence. The integration is real and CUDA statepassing runs, but the E64 matched gate is essentially tied with no-FS and slower at eval. Both arms have zero exact accuracy.

## Metrics

| arm | train final loss | eval acc | exact | eval lm loss | residual16 | eval time |
|---|---:|---:|---:|---:|---:|---:|
| RWKV no-FS | 1.592871 | 0.165159628 | 0.000000 | 2.370409 | 222.312012 | 74.67s |
| RWKV native-FS | 1.590624 | 0.165147573 | 0.000000 | 2.369608 | 222.399414 | 124.47s |

## Delta: FS - no-FS

- accuracy: `-0.000012055`
- lm_loss: `-0.000801325`
- total_loss: `-0.000808001`
- residual16: `+0.087402`
- final train loss: `-0.002247`
- eval time ratio: `1.667x`

## Interpretation

The corrected native FutureSeed mechanism is not broken: it trains to the same final loss as no-FS. But it also does not open an accuracy/exact/residual gap at this budget, and its eval is slower. This says the current cross-layer terminal-state seed is not yet a strong replacement for the official EqR token mixer.

Next high-ROI move is not a seed/gate sweep. If we keep this direction, change the generic FutureSeed state dynamics so the seed changes later recurrent computation more materially, then rerun one gate.
