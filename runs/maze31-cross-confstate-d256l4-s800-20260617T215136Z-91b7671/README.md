# Maze31 Confidence-Aware State Competition

Run: `maze31-cross-confstate-d256l4-s800-20260617T215136Z-91b7671`

## Question

Can a generic uncertainty / candidate-contrast signal inside the recurrent state
update turn the existing true-path vs false-positive score gap into real mask
pruning?

This was run as a single high-information probe, not a sweep. The motivation was
the previous dense-ranking result: it learned a positive average true-vs-false
score gap, but did not change the predicted PATH mask. This run tests whether
putting margin, entropy, and candidate disagreement directly into the state
competition gate can make later loops actually remove false-positive corridors.

## Mechanism

`state_compete_conf` keeps the same generic FutureSeed-loop setup:

- FutureSeed provides a recurrent state.
- Looping provides repeated compute.
- Candidate competition chooses among keep, proposed, and context states.
- A confidence bias is computed from readout-derived generic features:
  proposed/context margins, entropies, margin delta, and distribution
  disagreement.

No maze rule, search, repair, selector, or PATH-specific hand prior was added.

## Prediction

If score-to-mask conversion was the missing piece, loop12 should show:

- lower `path_pred_frac` than loop1,
- higher precision with recall roughly preserved,
- non-collapsed keep/proposed/context weights,
- confidence-bias diagnostics with enough magnitude to affect candidate choice.

## Budget And Kill Criteria

- GPU: GPU1 only.
- Steps: 800.
- Model: Maze31, D256, L4, train loops 6, eval loops 12.
- Kill if step100 exceeded 15 minutes or step300 F1 fell below the useful
  regime. Step100 arrived in about 2 minutes and step300 F1 was `0.5859`, so
  the run completed.

## Results

| metric | loop1 | loop12 | delta |
|---|---:|---:|---:|
| path F1 | 0.5852 | 0.5849 | -0.0003 |
| precision | 0.4158 | 0.4154 | -0.0004 |
| recall | 0.9993 | 1.0000 | +0.0007 |
| predicted path fraction | 0.4643 | 0.4651 | +0.0008 |
| path exact | 0.0000 | 0.0000 | +0.0000 |

Candidate weights did not collapse:

| diagnostic | loop1 | loop12 |
|---|---:|---:|
| keep mean | 0.1266 | 0.3486 |
| proposed mean | 0.5126 | 0.4412 |
| context mean | 0.3608 | 0.2102 |
| context minus proposed RMS | 0.7892 | 0.9635 |
| candidate entropy | 0.9665 | 1.0556 |

Confidence diagnostics were present but small:

| diagnostic | loop1 | loop12 |
|---|---:|---:|
| confidence bias abs mean | 0.0355 | 0.0349 |
| confidence bias std | 0.0434 | 0.0427 |
| proposed margin | 7.1167 | 7.1911 |
| context margin | 7.0641 | 5.5359 |
| margin delta | -0.0526 | -1.6552 |
| disagreement | 0.0498 | 0.0468 |

The visualization casebook confirms the same failure mode. Across the ten
selected hard cases, false negatives were already zero at loop1 and stayed zero,
while false positives moved from `287.4` to `288.0` on average. The model covers
the true path but does not prune extra corridors.

Artifacts:

- JSON: `output/eqr_maze_probe_fs1_seed52.json`
- Summary: `output/eqr_maze_probe_fs1_seed52.md`
- Visual casebook: `output/visualizations/index.html`
- Case metrics: `output/visualizations/cases.json`
- Logs: `logs/run.log`, `logs/launch.log`

## Decision

Discard this specific mechanism. It answered the intended question:

The bottleneck is not candidate collapse. The alternative candidate is different
and the gate uses it. The bottleneck is that the learned confidence bias is too
weak and too indirect to move the PATH decision boundary. It changes hidden-state
mixing, but the output remains a high-recall broad mask.

Next high-ROI direction should not be another ranking, bias, temperature, seed,
or gate-width sweep. The next mechanism needs a stronger generic way to make
later loops change the decision boundary, such as a learned recurrent threshold
or normalization state that directly controls mask sparsity while remaining
task-agnostic.

## Paper Claim Status

This supports a negative but useful claim:

FutureSeed can give direction and loops can provide compute, but state dynamics
must contain an effective decision-boundary update. Simply adding more candidate
diversity or weak confidence features is not enough for sustained correction on
hard Maze31.
