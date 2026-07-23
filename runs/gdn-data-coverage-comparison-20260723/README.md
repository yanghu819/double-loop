# GDN data-coverage comparison

Decision: **continue independent-data coverage**.

- mixed loop5 unseen-control: `-0.0039`
- official 56-64 unseen-control: `+0.0625`
- official 51-55 unseen-control: `+0.0273`

- second-batch 56-64 unseen-control: `+0.0703`
- second-batch 51-55 unseen-control: `+0.0508`

Open `index.html` for loop curves, official blank-range results, provenance, and hard-case links.

The result is a hard-tail preservation win, not a new scalar best: mixed exact
is `-0.0039` versus control, and official 56-64 remains `-0.0137` below the
step30000 parent. Across two different hard held-out batches, unseen solves
`287/768` boards versus control's `237/768`.

Visual examples:

- `hardcase-loop-correction.png`: wrong hidden cells
  `36 -> 28 -> 10 -> 0 -> 0`.
- `hardcase-loop-regression.png`: wrong hidden cells
  `27 -> 9 -> 3 -> 4 -> 6`, exposing late-loop instability.
