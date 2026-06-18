# Maze31 PATH Decision-Margin Probe

Recorded: 2026-06-18T05:12:04Z

## Hypothesis

The previous loop-pair self-correction probe showed that soft probability
regularizers can move calibration but do not reliably change the hard PATH mask.
This run tested a more direct, generic decision-boundary pressure: after loop 4,
true PATH cells should have PATH logit above the best non-PATH logit, while
non-PATH cells should push PATH logit below the best non-PATH logit.

This does not encode maze rules, search, repair, or a selector. It only changes
the recurrent training signal around the generic PATH-vs-not-PATH decision.

## Config

- Remote row: GPU1 only, A100 80GB, `CUDA_VISIBLE_DEVICES=0`
- Git SHA: `2cea7266a403d375ea8bf6c391a820f460fed05a`
- Task: 31x31 perfect mazes, path length 160-260
- Model: EqR maze runner with FutureSeed scale 1, hidden 256, layers 4, heads 8
- State update: `state_compete_boundary`
- Train/eval loops: 6 / 12
- Steps/batch/eval: 800 / 16 / 512
- New objective: `EQR_PATH_MARGIN_WEIGHT=1.0`, start loop 4, positive/negative margin 0.25

## Primary Result

The objective learned a softer margin change, but it did not create loop-time
hard-mask correction.

| metric | loop1 | loop12 | delta |
|---|---:|---:|---:|
| path F1 | 0.583447 | 0.583447 | 0.000000 |
| precision | 0.413946 | 0.413946 | 0.000000 |
| recall | 1.000000 | 1.000000 | 0.000000 |
| predicted PATH fraction | 0.466732 | 0.466732 | 0.000000 |

Soft diagnostics did move:

- PATH margin loss: loop1 `0.4714` to loop12 `0.2348`
- false-positive PATH margin mean: loop1 `1.1403` to loop12 `0.1463`
- non-PATH PATH probability: loop1 `0.2551` to loop12 `0.1813`
- calibrated soft PATH fraction: loop1 `0.3533` to loop12 `0.2507`

But no hard decision flipped:

- `state_boundary_prune_flip_frac = 0.0`
- `state_boundary_add_flip_frac = 0.0`
- visual hard cases average FP/FN: `290.0/0.0 -> 290.0/0.0`

## Decision

Discard this direction as a standalone loss-only fix. It is useful negative
evidence: the model can lower soft false-positive confidence without changing
the actual PATH decision boundary. The bottleneck is not just "give a harder
margin loss"; it is how recurrent state and readout convert soft uncertainty
into discrete pruning while preserving the true path.

Do not sweep path-margin weight, margin, seed, start loop, or temperature. The
next high-ROI step should change the generic recurrent decision mechanism, or
return to clean scaling where loop compute has already shown real gains.

## Artifacts

- `config.json`, `metadata.json`, `score.json`
- `logs/run.log`, `logs/launch.log`
- `output/eqr_maze_probe_fs1_seed52.json`
- `output/eqr_maze_probe_fs1_seed52.md`
- `output/visualizations/index.html`
- `output/visualizations/casebook.md`
- `output/visualizations/cases.json`
