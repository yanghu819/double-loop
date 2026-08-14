# P-GDN3-031 Compact Evidence

This directory is the tracked, lightweight evidence subset for the rejected
function-preserving decoupled-key GDN2 endpoint. The authoritative decision is
`output/decision.json`; `score.json` is an identical copy used by the run
launcher. `contract.json`, the candidate config, epoch metrics, score, launch
metadata, GPU identity, logs, and the original remote hash manifest are kept.
`formal.log` has only terminal carriage returns expanded to line breaks and
trailing progress-bar spaces removed for a clean textual Git diff; the raw
remote log hash remains recorded in `artifacts.sha256`.

Heavy but hash-anchored files are intentionally omitted from Git:

- source snapshot tarball: SHA256
  `2ae98911b65f69811af838e0a208f1347d95769fbca40112dd059fcc52a3eda0`;
- model checkpoint: SHA256
  `02f61cfbae4fc88625df3acd68f9c75a2eda06a7ab711ce06bb6df0bec3db802`;
- 1,000-case prediction export: SHA256
  `f8ac152cc4e9fedb2181b1b70658b4a584a49d127a787254447e0be3ac1ae5d6`.

The full run remains at
`/huyang2/double-loop/runs/p-gdn3-031-decoupled-key-l1024-20260814T0338Z-004dd05`.
Source SHA and GitHub readback are both
`004dd0553a160ea397bbbe6cc51b237ec62dbb89`.
