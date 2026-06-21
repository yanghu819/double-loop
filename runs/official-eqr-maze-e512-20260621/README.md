# Official EqR Maze e512 Evidence

Matched official EqR Maze e512 pair on GPU1. Clean EqR baseline uses official upstream SHA `aba94e9`; FutureSeed applies only the FutureSeed patch plus runtime SDPA fallback. No selector, search, repair, or maze-specific postprocessing.

## Training Readout

- Base final step3500: token acc `0.868423`, exact `0.0`, residual16 `8.001`.
- FutureSeed final step3500: token acc `0.868430`, exact `0.0`, residual16 `7.920`.
- FutureSeed step500 has early optimization signal, but final official scalar metrics are effectively tied.

## Path-Aware Case Readout

- Base loop16 path F1 `0.001273`, pred path frac `0.000226`.
- FutureSeed loop16 path F1 `0.000000`, pred path frac `0.000000`.
- True path cells are about 13% of valid cells, so token accuracy around 0.87 can still mean the model predicts almost no PATH.

Decision: Both official EqR variants mostly fail to emit PATH under this budget; token accuracy is not a valid Maze success signal.
