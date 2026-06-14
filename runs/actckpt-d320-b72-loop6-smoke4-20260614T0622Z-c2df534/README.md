# Activation Checkpoint D320 Batch72 Smoke

- Run: `actckpt-d320-b72-loop6-smoke4-20260614T0622Z-c2df534`
- Machine: AIStation GPU1 only
- Source SHA: `c2df53407b17b8efee037f41ce960874897a9fb5`
- Status: completed GPU-only smoke with clean provenance
- Recorded score: `0.0`; not a quality signal

## Purpose

This smoke validates the new activation-checkpoint switch as a scale enabler.
The question is not whether a 2-step model solves h120. The question is whether
the D320/L12/loop6 shape can run with batch72 without CPU fallback or OOM.

This matters because the previous same-budget D320 result was not a fair pure
capacity test: it used batch48 and lost effective training budget. Activation
checkpointing trades compute for memory so the next D320/D384 scale run can
keep batch/effective batch closer to the D256 baseline.

## Configuration

- Board: 12x12, random holes, 3x4 boxes
- Model: `D_MODEL=320`, `LAYERS=12`, `HEADS=10`, `HEAD_DIM=32`, `CHANNEL_MULT=4`
- Loop: `MAX_LOOPS=6`, `LOOP_LOSS=all`
- FutureSeed: fixed
- Activation checkpoint: `ACTIVATION_CHECKPOINT=1`
- Batch/eval: `SMOKE_BATCH=72`, `SMOKE_EVAL_N=16`
- Kernel/dtype: `RWKV_KERNEL=statepassing`, `FORWARD_DTYPE=bfloat16`

## Result

The smoke completed on GPU1 with `git_dirty=false`.

| readout | value |
| --- | ---: |
| train steps | 2 |
| train sec | 5.24 |
| peak CUDA allocated MB | 5473.15 |
| peak CUDA reserved MB | 6078.00 |
| final train CE | 3.8850 |

The score is expected to be zero because this is only a two-step fit test. The
useful result is that the D320/B72/loop6 forward-backward path works with CUDA
statepassing and activation checkpointing.

## Decision

Activation checkpointing is validated enough to use in the next real scale run.
The next useful experiment should not be another loop-count run. It should be a
fair capacity/effective-compute test, for example D320 with batch72 and a real
h120 curriculum, or D384 if a short fit probe confirms enough memory headroom.

Keep this as an enabling smoke only; do not use the score for model comparison.

## Artifacts

- `config.json`
- `metadata.json`
- `score.json`
- `source_HEAD.txt`
- `source.patch`
- `source_snapshot.tar.gz.sha256`
- `source_snapshot.ls.txt`
- `logs/run.log`
- `logs/driver.log`
- `output/futureseed_loop_seed52.json`
- `output/futureseed_loop_seed52.md`
- `output/futureseed_loop_case_seed52.html`

The full `source_snapshot.tar.gz` exists on GPU1 under the run directory but was
not committed because it is about 149 MB; the committed SHA, patch, snapshot
hash, and file list identify the source state.
