# Weak Denoising Scratchpad h120 Checkpoint

- Run: `scratchpad-weakdenoise-h120-step6800-d256-l12-loop6-20260613T1119Z-b03829b`
- Machine: AIStation GPU1 only
- Source SHA: `b03829be2b4ec387f4bf772e5144b6c5dab7af74`
- Status: completed the first h120 hard-stage checkpoint at step `6800`
- Primary recorded score: `0.0000` from `metrics.eval_clean.loop6.label_exact`

## Hypothesis

The strong noisy/Gaussian scratchpad run proved that `SCRATCH_NOISE_SCALE=0.05` plus `SCRATCH_GAUSS_WEIGHT=1e-4` was too disruptive. This run asked the only remaining high-value question for this mechanism: was the idea wrong, or was the pressure simply too strong?

The intervention kept the same gated mutable scratchpad but weakened the denoising and anti-collapse pressure to `SCRATCH_NOISE_SCALE=0.02` and `SCRATCH_GAUSS_WEIGHT=3e-5`.

## Configuration

- Board: 12x12, 3x4 boxes
- Model: `D_MODEL=256`, `LAYERS=12`, `HEADS=8`, `HEAD_DIM=32`, `CHANNEL_MULT=4`
- Loop: `MAX_LOOPS=6`, `LOOP_LOSS=all`
- FutureSeed: `FUTURE_SEED_UPDATE=fixed`, `FUTURE_SEED_DECAY=0.0`
- Scratchpad: `SCRATCH_MODE=gated`, `SCRATCH_NOISE_SCALE=0.02`, `SCRATCH_GAUSS_WEIGHT=0.00003`, `SCRATCH_GAUSS_PROJECTIONS=64`
- Curriculum: `16-36:200,36-60:300,60-72:500,72-84:1200,84-96:1800,96-108:1800,108-120:1000`
- Batch/eval: `FULL_BATCH=72`, `FULL_EVAL_N=512`, `FORWARD_DTYPE=bfloat16`, `RWKV_KERNEL=statepassing`

## Checkpoint Result

Step `6800`, after about `7927.0s`; train CE was `0.3505`.

| holes | loop1 | loop2 | loop3 | loop4 | loop5 | loop6 | loop6-loop3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| h96 exact | 0.0059 | 0.6426 | 0.6758 | 0.6855 | 0.6797 | 0.6816 | +0.0059 |
| h108 exact | 0.0000 | 0.1191 | 0.2246 | 0.2422 | 0.2422 | 0.2441 | +0.0195 |
| h120 exact | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| h132 exact | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |

Blank accuracy at loop6 was h96 `0.9801`, h108 `0.9320`, h120 `0.6575`, h132 `0.1746`.

The scratch diagnostics show real anti-collapse pressure: final `scratch_proj_rank=62.5` out of 64 projections, `scratch_proj_var=0.8047`, `scratch_gate=0.2236`, and `scratch_residual=13.2492`.

## Decision

Stop the scratch denoising/Gaussian branch for now. Weakening the pressure reduced the damage relative to the strong run, but it did not recover the no-noise scratchpad baseline:

- no-noise gated scratch h96/h108/h120 loop6 exact: `0.9883 / 0.8164 / 0.0039`
- strong noisy/Gaussian scratch: `0.5977 / 0.1582 / 0.0000`
- weak denoising scratch: `0.6816 / 0.2441 / 0.0000`

The important mechanism lesson is that h120 CE and blank accuracy can improve while exact and validity remain zero. This is a global consistency failure, not a local fill failure. Adding latent diversity to scratch makes the model more active, but not more coherent.

The next high-ROI move should not be more scratch noise/reg sweeps. Pivot back to clean scaling or a structurally simpler state update that directly improves late-loop correction without forcing residual diversity.

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
