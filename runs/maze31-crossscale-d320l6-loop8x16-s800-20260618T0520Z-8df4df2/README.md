# Maze31 Cross-Candidate Scale Abort

Recorded: 2026-06-18T05:35:55Z

## Hypothesis

The previous scalar loss probes showed that margin, mass, floor, and
self-correction losses mostly change soft confidence rather than hard loop-time
repair. This run returned to a bitter-lesson direction: spend more generic
model capacity and recurrent compute, without maze rules, repair, search,
selector, seed sweep, or extra pruning losses.

Prediction: if the FutureSeed + loop + generic state dynamics path scales,
D320/L6 with train8/eval16 and `state_compete_cross` should at least open PATH
prediction and then show loop16 pruning relative to loop1.

## Config

- GPU row: GPU1 only, A100 80GB, `CUDA_VISIBLE_DEVICES=0`
- Source SHA: `8df4df222b946363e5c854b4e6765e7efa851eb1`
- Task: 31x31 perfect mazes, path length 160-260
- Model: hidden 320, layers 6, heads 8, 9.38M parameters
- Train/eval loops: 8 / 16
- Planned steps: 800
- Batch/eval: 12 / 512
- FutureSeed scale: 1
- State update: `state_compete_cross`
- Path loss weight: 1.5
- Extra pruning/correction losses: all disabled

## Abort Result

The run was stopped after step400 because it never opened PATH prediction.
GPU utilization was healthy, so this was not CPU fallback or an idle-GPU
artifact.

| step | CE | path F1 | exact |
|---:|---:|---:|---:|
| 100 | 1.0625 | 0.0000 | 0.0000 |
| 200 | 1.0625 | 0.0000 | 0.0000 |
| 300 | 1.0625 | 0.0000 | 0.0000 |
| 400 | 1.0547 | 0.0000 | 0.0000 |

GPU memory was about 51GB and utilization was 98-100% while running. The
process was killed by exact PID before burning the remaining lease.

## Decision

Do not continue this exact scale direction. It is not evidence that scaling is
bad; it is evidence that this particular larger cross-candidate state update,
trained directly on hard Maze31 paths with batch 12, falls into a no-PATH
optimization attractor.

The next bitter-lesson-compliant move should not be another scalar loss around
this failure. Higher-ROI options are:

- use a data/compute schedule that first opens PATH prediction and then scales
  hard paths, without hand-written maze rules;
- return to the clean Maze21-to-Maze31 frontier and scale only axes that keep
  path prediction alive;
- improve the generic state update so it is easier to optimize, not by adding
  maze-specific pruning logic.

## Artifacts

- `abort.json`
- `config.json`, `metadata.json`, `score.json`
- `logs/run.log`, `logs/launch.log`
- `source_HEAD.txt`, `source.patch`
