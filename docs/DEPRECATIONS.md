# Code Status and Deprecations

This document is the maintenance boundary for the final baseline. Deprecated
does not mean deleted: old code and failed runs remain in Git because they
explain why the active line is narrow.

## Canonical

These paths receive fixes and define reproducible claims:

| Path | Status | Purpose |
|---|---|---|
| `experiments/rwkv_fs_sudoku/study_rwkv_futureseed_loop.py` | maintained | Shared FutureSeed + loop runner |
| `experiments/rwkv_fs_sudoku/rwkv7_cuda/` | maintained | RWKV state-passing CUDA |
| `configs/sudoku/gdn2_scale.env` | candidate | Active official GDN2 quality-scaling recipe |
| `scripts/run_canonical_gdn2_scale.sh` | candidate | Strict GPU1 GDN2 scale launcher |
| `experiments/rwkv_fs_sudoku/check_fla_delta_backbones.py` | maintained | Official FLA kernel/provenance gate |
| `configs/sudoku/backbone_benchmark.env` | canonical | Fair four-backbone contract |
| `scripts/run_sudoku_backbone_*.sh` | canonical | Strict GPU1 benchmark launch |
| `scripts/summarize_sudoku_backbone_benchmark.py` | canonical | Fairness validation and report |
| `scripts/check_futureseed_causal_contract.py` | canonical | FS-on/off constructor and trained-weight functional contract |
| `scripts/summarize_futureseed_causal_carriers.py` | canonical | Strict four-carrier causal on/off report |

## Supported but Not the Scale Default

- Official FLA GDN2 is the active first quality candidate. Its D256/L12
  compound-scale gate must open hard exact before replacing the completed GDN
  result as evidence.
- RWKV is supported in the benchmark and remains the speed/mechanism reference.
- Local GDN-Triton, `configs/sudoku/gdn_scale.env`, and
  `scripts/run_canonical_gdn_scale.sh` remain supported as the strongest
  completed long-run reference, available through `./run.sh gdn_legacy`.
- Official FLA KDA remains a supported benchmark backbone, not a scale line.
- Official FLA GDN is the public `gdn` benchmark arm. It is not silently mixed
  with the local `BACKBONE=gdn` scale implementation.

The accepted shared-initialization FutureSeed-on carrier table is archived at
`runs/sudoku-backbone-benchmark-sharedinit-20260724T154020Z-6e51f06/`.
Its paired causal on/off result is archived at
`runs/futureseed-causal-four-carrier-20260726T022718Z-6e51f06/`.
At 500 steps, GDN2 is the strongest 46-50-blank opener and RWKV7 is the fastest
FS carrier, but all four are zero exact at 51-64 blanks. More importantly,
FutureSeed raises 46-50 exact from zero without FS to
`0.7285/0.3633/0.7969/0.6465` for RWKV7/GDN/GDN2/KDA. Use the causal report
for the mechanism claim. `P-SCALE-037` now tests whether GDN2 can replace the
older long GDN line on hard-data quality; do not claim that replacement before
the gate completes.

## Deprecated Mechanisms

Do not launch new experiments from these paths without a new mechanism-level
reason:

| Mechanism/path | Status | Reason |
|---|---|---|
| feature-difference or hidden-aggregate noise | deprecated | Sensitive cost, no stable exact gain |
| scratch state and Gaussian projection objectives | deprecated | Did not unlock global closure |
| loop feedback/corruption attractor | deprecated | Trainable but no reliable late-loop correction |
| exact-margin, ranking, mass, threshold, boundary losses | deprecated | Mostly moved soft scores without stable exact closure |
| selector, best-of-K, rollout oracle as a method | forbidden | Diagnostic only; not the claimed model |
| Sudoku repair, search, or rule-specific state | forbidden | Violates the general mechanism claim |
| progressive split-bank GDN state expansion | deprecated | Numerically valid, temporary gain, failed strong endpoint |
| right-to-left scan called FutureSeed | forbidden | It is bidirectional scanning, not FutureSeed |
| EqR mixer replacement/transplant patches | archived | Confounded small adapters; not the standalone backbone |
| Maze-specific training objectives and visual tools | archived | Useful history, not the final Sudoku baseline |
| old 6x6 feature-noise mainline | archived | Superseded by official 9x9 full-diversity data |

The old flags remain parsable so checkpoints and source snapshots can still be
read. Canonical configs explicitly set every deprecated mechanism to zero or
`none`.

## Superseded Entry Points

- `scripts/run_fla_official_futureseed_compare.sh`
- `scripts/run_fla_delta_sudoku_gate.sh`
- `scripts/official_eqr_compare/`
- `scripts/apply_eqr_patch.sh`

Use `./run.sh baseline`, `./run.sh baseline_preflight`, or
`./run.sh benchmark_suite` instead.

## Historical Evidence

`runs/`, `research/reports/experiments/`, `plans.md`, and `leaderboard.csv` are
append-only evidence stores. A row marked `discarded`, `failed`, or `aborted`
must not be promoted back to the main line just because one metric looked good.
