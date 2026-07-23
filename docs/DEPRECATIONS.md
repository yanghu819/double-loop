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
| `experiments/rwkv_fs_sudoku/gdn_triton.py` | maintained | Local GDN-Triton scale carrier |
| `experiments/rwkv_fs_sudoku/check_fla_delta_backbones.py` | maintained | Official FLA kernel/provenance gate |
| `configs/sudoku/gdn_scale.env` | canonical | Strong clean scaling recipe |
| `configs/sudoku/backbone_benchmark.env` | canonical | Fair four-backbone contract |
| `scripts/run_sudoku_backbone_*.sh` | canonical | Strict GPU1 benchmark launch |
| `scripts/summarize_sudoku_backbone_benchmark.py` | canonical | Fairness validation and report |

## Supported but Not the Scale Default

- RWKV is supported in the benchmark and remains important mechanism evidence.
  GDN-Triton is the current scale default because the long independent-data run
  is strongest and cheaper to continue.
- Official FLA GDN2 and KDA remain supported benchmark backbones. Their finite
  budget results do not justify separate scaling lines.
- Official FLA GDN is the public `gdn` benchmark arm. It is not silently mixed
  with the local `BACKBONE=gdn` scale implementation.

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

