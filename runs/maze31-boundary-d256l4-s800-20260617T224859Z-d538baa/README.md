# Maze31 Recurrent Decision-Boundary Probe

Run: `maze31-boundary-d256l4-s800-20260617T224859Z-d538baa`

## Question

Can a direct learned decision-boundary state convert hidden/logit score gaps
into actual PATH mask pruning on hard Maze31?

The previous `state_compete_conf` run showed that candidate diversity was not
the bottleneck: keep/proposed/context weights were non-collapsed, but the output
mask did not prune false positives. This run tests the next simplest mechanism:
let the recurrent loop learn a per-cell threshold for the PATH decision itself.

## Mechanism

`state_compete_boundary` keeps the generic keep/proposed/context state
competition, then applies a learned per-cell threshold to the configured decision
token logit.

The threshold module uses only generic readout and state-comparison features:

- current, previous, and context PATH-vs-non-PATH margins,
- entropy and candidate disagreement,
- keep/proposed/context weights,
- loop index.

It does not encode maze topology, shortest-path search, repair, selector logic,
or a hand-written rule for corridors.

## Prediction

If the missing piece was score-to-mask conversion, then loop12 should show:

- lower predicted PATH fraction than loop1,
- higher precision with recall mostly preserved,
- nonzero threshold magnitude,
- nonzero prune flips from PATH to non-PATH,
- hard-case false positives decreasing in the visualization casebook.

## Budget And Kill Criteria

- GPU: GPU1 only.
- Source SHA: `d538baaa312abcc6fd61f72b4a209e3d2ccf10e4`.
- Steps: 800.
- Model: Maze31, D256, L4, train loops 6, eval loops 12.
- Kill if step100 exceeded 15 minutes or step300 F1 fell below 0.50.

Step100 arrived in about 2 minutes and step300 F1 was `0.5845`, so the run
completed.

## Results

| metric | loop1 | loop12 | delta |
|---|---:|---:|---:|
| path F1 | 0.5857 | 0.5854 | -0.0003 |
| precision | 0.4177 | 0.4173 | -0.0004 |
| recall | 0.9917 | 0.9921 | +0.0005 |
| predicted PATH fraction | 0.4587 | 0.4593 | +0.0007 |
| path exact | 0.0000 | 0.0000 | +0.0000 |

Candidate weights still did not collapse:

| diagnostic | loop1 | loop12 |
|---|---:|---:|
| keep mean | 0.1498 | 0.3602 |
| proposed mean | 0.5369 | 0.4343 |
| context mean | 0.3133 | 0.2054 |
| context minus proposed RMS | 0.8870 | 1.0601 |
| candidate entropy | 0.9616 | 1.0472 |

The boundary module did move the decision boundary, but in the wrong direction:

| diagnostic | loop1 | loop12 |
|---|---:|---:|
| threshold abs mean | 0.2747 | 0.2978 |
| threshold mean | -0.2730 | -0.2731 |
| raw PATH fraction | 0.4380 | 0.4339 |
| calibrated PATH fraction | 0.4593 | 0.4606 |
| prune flip fraction | 0.0000 | 0.0000 |
| add flip fraction | 0.0213 | 0.0267 |

The raw recurrent state is actually sparser than the calibrated prediction. The
learned threshold mostly lowers the PATH boundary, adds PATH cells, and preserves
recall. It does not prune false-positive corridors.

Hard-case visualization confirms the same pattern. Across the ten selected
failure cases, false positives moved from `283.5` to `285.1` on average, misses
from `2.5` to `2.7`, and F1 from `0.5230` to `0.5211`.

Artifacts:

- JSON: `output/eqr_maze_probe_fs1_seed52.json`
- Summary: `output/eqr_maze_probe_fs1_seed52.md`
- Visual casebook: `output/visualizations/index.html`
- Case metrics: `output/visualizations/cases.json`
- Logs: `logs/run.log`, `logs/launch.log`

## Decision

Discard this exact mechanism as a pruning fix, but keep the mechanism lesson.

The important result is not just negative. The learned threshold is trainable
and large enough to change predictions. The failure is directional: under the
current CE plus path-weight training pressure, the model learns to spend the
threshold on adding high-recall PATH mass rather than removing false positives.

This means the bottleneck has moved from "can the boundary move?" to "what
training pressure makes later loops prefer pruning when the true path is already
covered?"

Do not respond with seed, gate-bias, temperature, threshold-scale, or ranking
loss sweeps. The next high-ROI probe should change the generic training pressure
on the recurrent boundary, for example by making path-mass calibration part of
the objective or by training later loops against a target that rewards reducing
excess predicted mass without encoding maze rules.

## Paper Claim Status

This strengthens the paper story:

FutureSeed can give a broad direction, loop compute can maintain candidate
states, and decision-boundary dynamics can mechanically move the mask. The
remaining missing ingredient is not a stronger candidate generator, but a
training signal/state dynamic that makes the learned boundary move toward
correction rather than high-recall expansion.
