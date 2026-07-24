# Native GDN geometry diagnostic

- Integrity: **PASS**
- Hypothesis supported: **false**
- Decision: NOT SUPPORTED: native K24/V48 adds state and parameters but misses both gates; CE improvement is -0.0089 (need +0.03) and 46-50 exact delta is +0.0254 (need +0.15). Hard 51-64 exact remains zero.
- Native run: `sudoku-backbone-gdn-native-k24v48-s500-20260724T194300Z-8662e3a` at `8662e3ade15f27cba53171fb9c18129f1df08c39`
- Only substantive argument changes: `head_dim 32 -> 24` and `gdn_expand_v 1 -> 2`.
- No silent fallback, no backend dispatch, official FLA GDN and chunk backward verified.
