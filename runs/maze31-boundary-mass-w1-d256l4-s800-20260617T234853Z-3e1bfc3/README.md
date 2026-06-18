# Maze31 Late-Loop Path-Mass Pressure Probe

Run: `maze31-boundary-mass-w1-d256l4-s800-20260617T234853Z-3e1bfc3`

## Question

Can a generic late-loop training pressure reverse the learned boundary's
high-recall expansion bias and make later loops prune extra PATH mass?

The previous boundary probe showed that the threshold module could move the
decision boundary, but CE plus path-weight pressure trained it to add PATH cells
instead of pruning. This run keeps the same state/threshold mechanism and changes
only the training objective.

## Mechanism

The added objective starts at loop4. It minimizes soft PATH probability on
non-PATH cells while guarding true-PATH probability relative to loop1.

This is generic supervised segmentation pressure:

- it uses logits and token labels,
- it does not use maze topology,
- it does not search, repair, or select trajectories,
- it does not encode corridor or shortest-path rules.

## Prediction

If training pressure was the missing piece, loop12 should show:

- lower predicted PATH fraction than loop1,
- higher precision,
- recall mostly preserved,
- lower soft PATH mass on non-PATH cells,
- visual cases with fewer false-positive corridors and few new misses.

## Budget And Kill Criteria

- GPU: GPU1 only.
- Source SHA: `3e1bfc372ea472753c4e583011f9eff4f7187677`.
- Steps: 800.
- Model: Maze31, D256, L4, train loops 6, eval loops 12.
- Kill if step100 exceeded 15 minutes, step300 F1 fell below 0.50, or GPU
  utilization was abnormal.

Step100 arrived in about 2.5 minutes and step300 F1 was `0.5896`, so the run
completed.

## Results

| metric | loop1 | loop12 | delta |
|---|---:|---:|---:|
| path F1 | 0.5849 | 0.5879 | +0.0031 |
| precision | 0.4154 | 0.4235 | +0.0081 |
| recall | 1.0000 | 0.9727 | -0.0273 |
| predicted PATH fraction | 0.4651 | 0.4437 | -0.0214 |
| path exact | 0.0000 | 0.0000 | +0.0000 |

Soft mass moved in the intended direction:

| diagnostic | loop1 | loop12 |
|---|---:|---:|
| PATH prob fraction | 0.3649 | 0.2983 |
| excess prob fraction over label mass | 0.1717 | 0.1051 |
| non-PATH PATH probability | 0.2630 | 0.2137 |
| true-PATH PATH probability | 0.7880 | 0.6500 |
| recall guard loss | 0.0000 | 0.0191 |

Candidate weights remained non-collapsed:

| diagnostic | loop1 | loop12 |
|---|---:|---:|
| keep mean | 0.1635 | 0.3499 |
| proposed mean | 0.5004 | 0.4220 |
| context mean | 0.3361 | 0.2281 |
| context minus proposed RMS | 0.8479 | 0.9077 |

Boundary behavior also changed from the previous expansion-only run:

| diagnostic | loop1 | loop12 |
|---|---:|---:|
| raw PATH fraction | 0.4618 | 0.4433 |
| calibrated PATH fraction | 0.4651 | 0.4447 |
| threshold abs mean | 0.2347 | 0.4499 |
| threshold mean | -0.2013 | +0.0425 |
| add flip fraction | 0.0034 | 0.0013 |
| prune flip fraction | 0.0000 | 0.0000 |

Visual hard cases reveal the tradeoff. Average false positives dropped from
`285.5` to `276.5`, but false negatives rose from `0.0` to `11.7`, so selected
failure cases got lower F1 despite aggregate precision improving.

Artifacts:

- JSON: `output/eqr_maze_probe_fs1_seed52.json`
- Summary: `output/eqr_maze_probe_fs1_seed52.md`
- Visual casebook: `output/visualizations/index.html`
- Case metrics: `output/visualizations/cases.json`
- Logs: `logs/run.log`, `logs/launch.log`

## Decision

Keep the mechanism lesson, but do not claim this as a solved pruning dynamic.

This is the first probe in this sequence where the training pressure clearly
reverses the expansion tendency: predicted mask mass drops and precision rises.
However, the recall guard is too weak or too indirect. The model buys precision
by removing some true path cells, not by cleanly removing only false-positive
corridors.

The next step should not be a weight sweep. The useful follow-up is a better
generic recall-preserving pruning objective: one that applies pruning pressure
inside the currently predicted PATH set while strongly protecting high-confidence
true-path cells, or one that trains loop-to-loop improvement on excess mass
without letting true-path probability fall.

## Paper Claim Status

This advances the story:

FutureSeed gives broad path direction, loop compute can revise the mask, and
training pressure can decide whether the recurrent boundary expands or prunes.
The current missing piece is precise recall-preserving pruning, not candidate
diversity or another readout module.
