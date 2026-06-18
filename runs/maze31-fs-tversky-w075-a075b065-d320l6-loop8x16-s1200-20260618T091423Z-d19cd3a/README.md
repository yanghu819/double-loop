# Maze31 FutureSeed Tversky Abort

Recorded: 2026-06-18T09:39Z

## Hypothesis

Clean FutureSeed+loop opens hard Maze31, but later loops mostly add coverage.
A generic late-loop Tversky objective might directly reward the desired behavior:
keep true path cells while reducing false-positive PATH mass.

This was a single high-information probe, not a sweep. It used no maze rules,
repair, search, selector, feature noise, state competition, or seed table.

## Config

- Remote row: GPU1 only, A800 80GB, `CUDA_VISIBLE_DEVICES=0`
- Git SHA: `d19cd3a265ce0b0d275f4925d2e0f6705fce786b`
- Task: Maze31 perfect mazes, path length `160-260`
- Model: FutureSeed+loop, hidden `320`, layers `6`, heads `8`
- State update: `none`
- Train/eval loops: `8 / 16`
- Planned steps/batch/eval: `1200 / 16 / 512`
- Tversky: `weight=0.75`, `start_loop=4`, `alpha=0.75`, `beta=0.65`

## Result

The run was stopped at step600 because it never opened.

| step | CE | train path F1 | Tversky loss |
|---:|---:|---:|---:|
| 100 | 1.0625 | 0.0000 | 0.8190 |
| 200 | 1.0781 | 0.0000 | 0.8227 |
| 300 | 1.0625 | 0.0000 | 0.8212 |
| 400 | 1.0625 | 0.0000 | 0.8256 |
| 500 | 1.0703 | 0.0000 | 0.8281 |
| 600 | 1.0703 | 0.0000 | 0.8241 |

The matched clean FutureSeed run opened at step400. This probe therefore failed
before it could test loop pruning.

## Decision

Discard loss-only Tversky pressure as a mainline next step. The negative result
is useful: direct FP/FN pressure can damage the FutureSeed opening dynamics.

The next useful experiment should preserve opening first, then change a generic
recurrent state or decision dynamic so later loops can revise the mask. Another
weight/alpha/beta sweep would be low ROI.

## Artifacts

- `abort.json`
- `config.json`
- `hypothesis.md`
- `launch.env`
- `logs/run.log`
- `logs/launch.outer.log`
- `source_HEAD.txt`, `source.patch`, `source_snapshot.ls.txt`
