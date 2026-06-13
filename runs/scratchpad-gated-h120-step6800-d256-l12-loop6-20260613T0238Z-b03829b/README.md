# Gated Scratchpad h120 Checkpoint

- Run: `scratchpad-gated-h120-step6800-d256-l12-loop6-20260613T0238Z-b03829b`
- Machine: AIStation GPU1 only
- Source SHA: `b03829be2b4ec387f4bf772e5144b6c5dab7af74`
- Status: completed the first h120 hard-stage checkpoint at step `6800`
- Primary recorded score: `0.0059` from `metrics.eval_clean.loop6.label_exact`

## Hypothesis

This run tested whether separating FutureSeed into stable context plus a gated mutable scratchpad is enough to make late loops revise h120 puzzles. The intended signal was not a small CE improvement; the question was whether loop4-6 keep adding exact solves after loop3, which would mean the scratchpad is functioning as a useful latent workspace rather than just another active hidden state.

## Configuration

- Board: 12x12, 3x4 boxes
- Model: `D_MODEL=256`, `LAYERS=12`, `HEADS=8`, `HEAD_DIM=32`, `CHANNEL_MULT=4`
- Loop: `MAX_LOOPS=6`, `LOOP_LOSS=all`
- FutureSeed: `FUTURE_SEED_UPDATE=fixed`, `FUTURE_SEED_DECAY=0.0`
- Scratchpad: `SCRATCH_MODE=gated`, `SCRATCH_NOISE_SCALE=0.0`, `SCRATCH_GAUSS_WEIGHT=0.0`
- Curriculum: `16-36:200,36-60:300,60-72:500,72-84:1200,84-96:1800,96-108:1800,108-120:1000`
- Batch/eval: `FULL_BATCH=72`, `FULL_EVAL_N=512`, `FORWARD_DTYPE=bfloat16`, `RWKV_KERNEL=statepassing`

## Checkpoint Result

Step `6800`, after about `8534.8s`; train CE was `0.5035`.

| holes | loop1 | loop2 | loop3 | loop4 | loop5 | loop6 | loop6-loop3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| h96 exact | 0.0000 | 0.8770 | 0.9863 | 0.9883 | 0.9883 | 0.9883 | +0.0020 |
| h108 exact | 0.0000 | 0.2383 | 0.7617 | 0.8066 | 0.8164 | 0.8164 | +0.0547 |
| h120 exact | 0.0000 | 0.0000 | 0.0020 | 0.0039 | 0.0039 | 0.0039 | +0.0020 |
| h132 exact | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 |

Blank accuracy at loop6 was h96 `0.9996`, h108 `0.9768`, h120 `0.5460`, h132 `0.1524`.

The scratchpad was active at the end of training: `scratch_gate=0.2363`, `scratch_decay=0.2295`, `scratch_delta=5.4339`, and `scratch_residual=13.6786`. So the negative result is not "scratch did not write"; it is "scratch wrote, but did not become useful h120 correction."

## Decision

Gated mutable scratch alone is not enough. It improves over the loop-residual failure at h120, but it is weaker than the clean h120 hard-stage curve at the comparable first h120 checkpoint. The key failure signature is that h120 exact is already stuck by loop4; loop5 and loop6 add blank accuracy but no extra exact solves.

The only high-ROI follow-up is the paired mechanism test from the refined plan: add train-time noise only to the mutable scratchpad and weak Gaussian regularization only to the scratch update residual. The decision it should answer is narrow: can anti-collapse plus denoising turn active scratch writes into late-loop corrections? Do not run another longer no-noise scratchpad variant.

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
