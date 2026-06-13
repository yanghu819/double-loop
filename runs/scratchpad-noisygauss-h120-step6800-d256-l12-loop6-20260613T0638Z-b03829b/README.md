# Noisy Gaussian Scratchpad h120 Checkpoint

- Run: `scratchpad-noisygauss-h120-step6800-d256-l12-loop6-20260613T0638Z-b03829b`
- Machine: AIStation GPU1 only
- Source SHA: `b03829be2b4ec387f4bf772e5144b6c5dab7af74`
- Status: completed the first h120 hard-stage checkpoint at step `6800`
- Primary recorded score: `0.0000` from `metrics.eval_clean.loop6.label_exact`

## Hypothesis

The previous gated scratchpad run showed that the mutable scratch path writes, but the writes do not become useful h120 corrections. This run tested the next narrow mechanism: add train-time noise only to mutable scratch and weak Gaussian-lite regularization only to scratch update residuals, so the loop learns a less collapsed and more recoverable latent workspace.

This was not a table-filling ablation. The expected useful signature was specific: h120 loop4-6 should gain exact solves without destroying h96/h108 transfer.

## Configuration

- Board: 12x12, 3x4 boxes
- Model: `D_MODEL=256`, `LAYERS=12`, `HEADS=8`, `HEAD_DIM=32`, `CHANNEL_MULT=4`
- Loop: `MAX_LOOPS=6`, `LOOP_LOSS=all`
- FutureSeed: `FUTURE_SEED_UPDATE=fixed`, `FUTURE_SEED_DECAY=0.0`
- Scratchpad: `SCRATCH_MODE=gated`, `SCRATCH_NOISE_SCALE=0.05`, `SCRATCH_GAUSS_WEIGHT=0.0001`, `SCRATCH_GAUSS_PROJECTIONS=64`
- Curriculum: `16-36:200,36-60:300,60-72:500,72-84:1200,84-96:1800,96-108:1800,108-120:1000`
- Batch/eval: `FULL_BATCH=72`, `FULL_EVAL_N=512`, `FORWARD_DTYPE=bfloat16`, `RWKV_KERNEL=statepassing`

## Checkpoint Result

Step `6800`, after about `8518.3s`; train CE was `0.5760`.

| holes | loop1 | loop2 | loop3 | loop4 | loop5 | loop6 | loop6-loop3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| h96 exact | 0.0078 | 0.4668 | 0.5938 | 0.5996 | 0.5977 | 0.5977 | +0.0039 |
| h108 exact | 0.0000 | 0.0449 | 0.1465 | 0.1562 | 0.1562 | 0.1582 | +0.0117 |
| h120 exact | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| h132 exact | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |

Blank accuracy at loop6 was h96 `0.9761`, h108 `0.9152`, h120 `0.6056`, h132 `0.1649`.

The anti-collapse pressure worked in the literal diagnostic sense: final `scratch_proj_rank=63.0` out of 64 projections, `scratch_proj_var=0.8086`, `scratch_gate=0.2559`, and `scratch_residual=13.2968`. But higher-rank residuals did not become better board-level reasoning. h120 blank accuracy improved versus gated scratch without noise/reg, while exact and validity fell to zero.

## Decision

Stop this strength of noise and Gaussian residual regularization. It made the scratch workspace more diverse but less useful: h96 exact fell from `0.9883` in the no-noise scratch run to `0.5977`, h108 fell from `0.8164` to `0.1582`, and h120 fell from tiny nonzero to `0.0`.

The useful insight is precise: the h120 issue is not simply low-rank scratch collapse. Forcing update residual diversity can raise local fill accuracy at h120 but still destroy global consistency. A weaker setting, such as `SCRATCH_NOISE_SCALE=0.02` and `SCRATCH_GAUSS_WEIGHT=0.00003`, is only worth testing with a full lease if the next question is "was this pressure just too strong?" Do not run another `0.05 / 1e-4` variant.

## Artifacts

- `config.json`
- `metadata.json`
- `score.json`
- `source_HEAD.txt`
- `source.patch`
- `source_snapshot.tar.gz.sha256`
- `source_snapshot.ls.txt`
- `logs/run.log`
- `output/checkpoint_eval_step006800.json`
- `output/futureseed_loop_seed52.json`
- `output/futureseed_loop_seed52.md`
- `output/futureseed_loop_case_seed52.html`

The full `source_snapshot.tar.gz` exists on GPU1 under the run directory but was not committed because it is about 149 MB; the committed SHA, patch, and snapshot hash identify the source state.
