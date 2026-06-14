# Activation-Checkpoint D320 Batch72 h120 Abort

- Run: `actckpt-d320-b72-h120-step6800-20260614T0648Z-d3ddc3b`
- Machine: AIStation GPU1 only
- Source SHA: `d3ddc3bd73c2bf3f9ad603ebed880279c6af0a25`
- Status: aborted by exact process cleanup before first h120 checkpoint

## Hypothesis

The earlier D320 h120 run was a weak capacity test because batch fell to 48.
After validating activation checkpointing with a D320/B72 smoke, this run asked:

Can D320/L12/loop6 with batch72 reach the same h120 first-checkpoint curriculum
inside one GPU1 lease, making width a fair next scale axis?

This is a single feasibility and capacity-path test, not a width table.

## Configuration

- Board: 12x12, random holes, 3x4 boxes
- Model: `D_MODEL=320`, `LAYERS=12`, `HEADS=10`, `HEAD_DIM=32`, `CHANNEL_MULT=4`
- Loop: `MAX_LOOPS=6`, `LOOP_LOSS=all`
- FutureSeed: fixed
- Activation checkpoint: `ACTIVATION_CHECKPOINT=1`
- Batch/eval: `FULL_BATCH=72`, `FULL_EVAL_N=512`
- Curriculum target: `16-36:200,36-60:300,60-72:500,72-84:1200,84-96:1800,96-108:1800,108-120:1000`
- First planned checkpoint: step `6800`, holes `96,108,120,132`

## Observed Early Run

| step | CE |
| ---: | ---: |
| 100 | 1.8375 |
| 200 | 1.8053 |

The run used about `2.65` seconds per step between step100 and step200. At
step200, reaching the first planned h120 checkpoint still required about `6600`
more steps, or roughly `17490` seconds. That projected beyond the available
GPU1 lease, so the run was stopped before it could produce the intended
checkpoint evidence.

## Decision

Activation checkpointing solves the memory side, but D320/B72 full h120
curriculum is too slow for a normal 4-hour lease. Letting it run would produce
a partial, hard-to-compare result rather than the intended first-checkpoint
answer.

Do not repeat this exact configuration under the same lease length. The next
scale options should be one of:

- Use D320 with a smaller reachable target only if the checkpoint still answers
  a concrete question.
- Use a true longer/resumable training setup before attempting this fair
  D320/B72 h120 checkpoint.
- Prefer a simpler state-update mechanism if the goal is h120 progress inside
  one normal lease.

## Artifacts

- `config.json`
- `abort.json`
- `source_HEAD.txt`
- `source.patch`
- `source_snapshot.tar.gz.sha256`
- `source_snapshot.ls.txt`
- `logs/run.log`
- `logs/driver.log`

The full `source_snapshot.tar.gz` exists on GPU1 under the run directory but was
not committed because it is about 150 MB; the committed SHA, patch, snapshot
hash, and file list identify the source state.
