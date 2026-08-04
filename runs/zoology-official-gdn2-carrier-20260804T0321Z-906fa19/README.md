# P-CAUSAL-006: strict GDN2 in the validated Zoology shell

- Decision: passed
- Final validation accuracy: `0.9965`
- Final validation loss: `0.0222`
- Early stop: epoch 2, official threshold `>0.99`
- Model: strict official FLA `GatedDeltaNet2`, 538,696 total parameters
- Source SHA: `906fa19be2236baac25824196a973205902594e2`
- Zoology SHA: `1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb`
- FLA SHA: `9c8e42e762fce087c27b673af4922795d9edb85e`

With the exact validated Zoology Trainer/data/metric shell, GDN2 opened from
validation accuracy `0.242` at epoch0 to `0.984` at epoch1 and `0.9965` at
epoch2. The upstream MHA baseline needed epoch25 to cross the same threshold.

This validates GDN2 as the causal carrier. It is not yet FutureSeed evidence.
The next controlled experiment may now hold this entire shell fixed and compare
matched native FutureSeed versus no FutureSeed on explicit future queries.
