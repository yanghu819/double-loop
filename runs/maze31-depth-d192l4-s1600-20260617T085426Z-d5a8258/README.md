# Maze31 Depth Scale Probe

Run: `maze31-depth-d192l4-s1600-20260617T085426Z-d5a8258`

Source SHA: `d5a82583fd3e4f160cf86731c05f35a1bc3fc445`

## Question

Is the Maze31 failure mainly interaction depth? If the hidden-192 two-layer
model lacks enough composition depth, four layers should improve the hard
160-260 path setting and make later loops useful again.

## Configuration

- GPU row: GPU1 only
- Task: 31x31 perfect mazes
- Path range: 160-260 from step 1
- Model: hidden 192, 6 heads, 4 layers
- Parameters: 1,845,892
- Train loops/eval loops: 6/12
- Train steps: 1600
- Batch/eval N: 32/512
- FutureSeed scale: 1
- Path loss weight: 1.5

## Result

| loop | exact | path F1 | precision | recall | predicted path fraction |
|---|---:|---:|---:|---:|---:|
| 1 | 0.0000 | 0.5372 | 0.4478 | 0.6788 | 0.2927 |
| 12 | 0.0000 | 0.5364 | 0.4477 | 0.6767 | 0.2918 |

Loop gain is `-0.0008`; extra recurrent steps are effectively static. Training
stays near the same plateau seen in the two-layer run:

- step100: F1 0.5274
- step300: F1 0.5644
- step800: F1 0.5671
- step1200: F1 0.5687
- step1600: F1 0.5552

## Visualization

Open `output/visualizations/index.html` or read
`output/visualizations/casebook.md`.

Case 349 is representative: loop1 F1 is `0.2982`; loop12 F1 drops to `0.2751`.
False positives remain around 206 and misses rise from 99 to 105.

## Decision

Do not continue simple depth scaling on this small Maze31 model. Width, depth,
and naive path curriculum have now all failed to make Maze31 loops revise the
state. The next useful direction is a simple recurrent state-dynamics change or
a genuinely larger effective-compute run, not another one-axis table entry.
