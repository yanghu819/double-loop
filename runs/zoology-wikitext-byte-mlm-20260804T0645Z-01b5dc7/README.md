# P-CAUSAL-008 WikiText Byte MLM Gate

- Status: discarded because the registered bidirectional carrier gate failed.
- Source SHA: `01b5dc70aef9af3c47f638785df57173e84723d5`.
- Data: fixed WikiText-103 byte windows, 20,000 train and 2,000 validation,
  sequence length 256, exactly 38 masked positions per window.
- Causal GDN2: masked accuracy `0.424737`, CE `2.018437`.
- FutureSeed GDN2: masked accuracy `0.421829`, CE `2.031740`.
- Bidirectional attention: masked accuracy `0.187855`, CE `3.167591`.

The bidirectional attention ceiling was much worse than causal GDN2, so this
run cannot support a positive or negative language claim about FutureSeed. The
FutureSeed path was active and correctly isolated, but it was slightly worse
than causal GDN2 in this invalid carrier. Do not rescue this setup with tuning.

See `visualizations/index.html` for 12 same-window comparisons and
`output/comparison.json` plus `paired_diagnostics.json` for complete metrics.
The full source snapshot and omitted large per-example outputs remain in the
AIStation copy of this run.
