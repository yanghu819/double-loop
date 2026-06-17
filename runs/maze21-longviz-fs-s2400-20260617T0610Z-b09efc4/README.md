# Maze21 FutureSeed Long Visualization

Run: `maze21-longviz-fs-s2400-20260617T0610Z-b09efc4`

Source SHA: `b09efc43c7ec26b4ef006feaf3ac6bfdc61c5e48`

Date: 2026-06-17 06:29:49 UTC

## Question

Does the FutureSeed + loop recipe only make a fuzzy path mask on Maze21, or can
plain scale and recurrent refinement turn it into exact path solving?

This is the right Maze proxy question because Maze21 path propagation is not a
Sudoku-specific constraint trick. The model must spread information through a
long grid structure and then remove wrong branches.

## Configuration

- GPU: GPU1 only
- Task: 21x21 perfect mazes
- Path length range: 80-140
- Model: hidden 192, 6 heads, 2 layers
- Train loops: 6
- Eval loops: 10
- Train steps: 2400
- Batch: 64
- Eval N: 512
- FutureSeed scale: 1
- Path loss weight: 1.5
- Visualization loops: 1, 2, 4, 6, 8, 10

## Result

| loop | exact | path F1 | precision | recall | predicted path fraction |
|---|---:|---:|---:|---:|---:|
| 1 | 0.0000 | 0.7817 | 0.6706 | 0.9466 | 0.2928 |
| 2 | 0.1426 | 0.9341 | 0.8995 | 0.9774 | 0.2258 |
| 3 | 0.6934 | 0.9842 | 0.9766 | 0.9934 | 0.2108 |
| 4 | 0.9434 | 0.9977 | 0.9963 | 0.9995 | 0.2078 |
| 5 | 0.9902 | 0.9990 | 0.9984 | 0.9998 | 0.2074 |
| 10 | 0.9961 | 0.9997 | 0.9995 | 1.0000 | 0.2072 |

The label path fraction is `0.2070`, so loop1 over-predicts path cells by about
41%. By loop10, predicted path mass matches the label almost exactly.

Training exact opened late:

- step 100: path F1 0.5864, exact 0.0000
- step 1000: path F1 0.8124, exact 0.0000
- step 1400: path F1 0.8681, exact 0.0781
- step 1800: path F1 0.9276, exact 0.6250
- step 2000: path F1 0.9838, exact 0.9062
- step 2400: path F1 0.9966, exact 0.9531

## Visualization

Open:

`output/visualizations/index.html`

Legend:

- `#`: wall
- `S`: start
- `G`: goal
- `T`: predicted path cell that is truly on the path
- `F`: false positive path cell
- `M`: missed true path cell
- `.`: open non-path cell

The selected cases are largest loop-gain examples. They are not final failures;
all selected examples end solved at loop10. This makes them good for seeing what
the loop fixes, but not for analyzing the rare remaining `0.4%` failures.

Representative selected cases:

| case | loop1 F1 | loop1 false positives | loop1 misses | loop10 F1 | loop10 false positives | loop10 misses |
|---:|---:|---:|---:|---:|---:|---:|
| 191 | 0.5972 | 69 | 16 | 1.0000 | 0 | 0 |
| 133 | 0.6032 | 53 | 22 | 1.0000 | 0 | 0 |
| 124 | 0.6201 | 79 | 8 | 1.0000 | 0 | 0 |
| 82 | 0.6311 | 61 | 15 | 1.0000 | 0 | 0 |

## Interpretation

The useful behavior is visible:

1. Loop1 usually finds a broad connected region that includes much of the true
   path, but it also lights up many wrong corridors.
2. Loops 2-4 mostly delete those wrong branches while preserving the true path.
3. Later loops polish the remaining local mistakes; loop10 reaches almost exact
   path masks.

So the loop is doing iterative global cleanup. It is not just averaging logits
or doing a cosmetic confidence pass.

The strongest insight is that the earlier Maze21 1200-step result was not a
hard ceiling. At 1200 steps, FutureSeed had path F1 0.8051 and exact 0.0. At
2400 steps, the same task reaches path F1 0.9997 and exact 0.9961. This supports
the bitter-lesson direction: the scalable path is more compute, stronger
recurrent state dynamics, and better proxy tasks, not hand-coded repair.

## Decision

Keep Maze21 as a serious non-Sudoku proxy for FutureSeed + loop. The next
high-value step is either:

- compare a full-budget base run only if we need attribution against FutureSeed,
  or
- scale the maze frontier to a harder setting where loop10 is no longer solved,
  then visualize the first real failure mode.

Do not spend the next budget on selector, path repair, or handcrafted maze
priors.
