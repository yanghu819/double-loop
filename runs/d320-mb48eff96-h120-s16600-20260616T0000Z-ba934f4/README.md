# D320 Packed H120 Long Continuation Abort

- Timestamp: 2026-06-15T21:46Z launch, 2026-06-15T22:10Z abort record
- Source SHA: `ba934f4f717dfb7ce987d3384b03147ba2df72c9`
- GPU: GPU1 only

## What Happened

The first step16600 continuation attempt reached only step13700 before GPU1 was
halted by the platform. It had no train checkpoint and no quality eval, so it is
not a model result.

The retry run `d320-mb48eff96-h120-s16600-r2-20260615T2210Z-ba934f4` resumed
from the previous step13600 checkpoint and completed the experiment.

## Lesson

Treat short pre-checkpoint platform halts as discarded segments. They can be
archived for provenance, but they should not affect model conclusions.

