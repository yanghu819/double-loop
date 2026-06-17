# Maze31 Frontier Probe

Run: `maze31-frontier-fs-s1800-20260617T072102Z-4e9fd34`

Source SHA: `4e9fd341eca88c04e3e2f6dca3d68f476b72acdc`

Date: 2026-06-17 UTC

## Question

Does the Maze21 FutureSeed + loop behavior scale directly to a larger maze, or
does the loop lose value once the path propagation problem crosses the current
model's capacity?

This run intentionally does not compare against base. Maze21 already showed a
clear positive loop story; here the higher-value question is where that story
breaks.

## Configuration

- GPU: GPU1 only
- Task: 31x31 perfect mazes
- Path length range: 160-260
- Model: hidden 192, 6 heads, 2 layers
- Train loops: 6
- Eval loops: 12
- Train steps: 1800
- Batch: 32
- Eval N: 512
- FutureSeed scale: 1
- Path loss weight: 1.5
- Visualization cases: 16, failure-first selection

## Result

| loop | exact | path F1 | precision | recall | predicted path fraction |
|---|---:|---:|---:|---:|---:|
| 1 | 0.0000 | 0.5203 | 0.4501 | 0.6251 | 0.2681 |
| 2 | 0.0000 | 0.5086 | 0.4516 | 0.5911 | 0.2525 |
| 4 | 0.0000 | 0.5092 | 0.4519 | 0.5919 | 0.2527 |
| 6 | 0.0000 | 0.5093 | 0.4518 | 0.5924 | 0.2530 |
| 8 | 0.0000 | 0.5094 | 0.4521 | 0.5923 | 0.2528 |
| 10 | 0.0000 | 0.5093 | 0.4520 | 0.5922 | 0.2528 |
| 12 | 0.0000 | 0.5094 | 0.4520 | 0.5924 | 0.2529 |

The loop gain is `-0.0109`, so the loop is not refining the answer. It mostly
keeps the same broad, wrong path mask.

Training never entered the Maze21-style opening phase:

- step 100: F1 0.4839, exact 0.0000
- step 200: F1 0.5791, exact 0.0000
- step 800: F1 0.5691, exact 0.0000
- step 1200: F1 0.5716, exact 0.0000
- step 1800: F1 0.5533, exact 0.0000

## Visualization

Open:

`output/visualizations/index.html`

The casebook is also available at:

`output/visualizations/casebook.md`

The failure-first selector worked. The selected examples are real final
failures. Case 71 is the clearest example:

- loop1 F1 `0.3478`, with 183 false positives and 87 misses
- loop12 F1 `0.1763`, with 172 false positives and 127 misses

That means extra loops did not clean up a rough path. The model missed large
parts of the true path and kept many wrong corridors.

## Interpretation

Maze31 is not just "Maze21 but needs more loops" under this configuration. The
hidden-192 two-layer model does not build a useful enough intermediate state for
the loop to revise.

The important distinction is:

- Maze21: loop1 is broad but contains enough of the true path; later loops prune
  false branches and reach exact path masks.
- Maze31 D192: loop1 is already missing too much of the true path; later loops
  cannot recover it and sometimes make F1 worse.

So the next high-value question is capacity/effective scale, not selector,
repair, or handcrafted maze rules.

## Decision

Do not extend this exact D192 Maze31 run. It has low marginal ROI.

The next run should test an effective scaling axis. A D320 Maze31 capacity probe
is the cleanest immediate question: if D320 makes loop gain positive, the
frontier is representation capacity; if D320 still stays flat, the next issue is
curriculum or recurrent state dynamics.
