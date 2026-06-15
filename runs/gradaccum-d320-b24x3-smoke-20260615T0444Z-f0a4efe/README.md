# Gradient Accumulation D320 Effective-Batch Smoke

Run: `gradaccum-d320-b24x3-smoke-20260615T0444Z-f0a4efe`

Source SHA: `f0a4efe561b2123dafce7cc9d5b06a92131afe12`

Recorded: `2026-06-15T04:46:31.847092+00:00`

## Hypothesis

The last clean h120 continuation showed low marginal return from simply training the same D256/L12/loop6 model longer: h120 blank accuracy improved, but exact solve did not. The next scale axis should therefore be general compute capacity rather than another identical hard-stage extension.

This smoke tests whether gradient accumulation can restore a fair effective batch for a wider D320 clean FutureSeed+loop model without lowering the batch to the weak D320/B48 regime. It is a scaling-infrastructure test, not a model-quality result.

The bitter-lesson interpretation is deliberately narrow: avoid Sudoku-specific repair, selector search, or extra human prior. Spend engineering effort on making larger general models and more effective compute practical.

## Configuration

- Board: 12x12, random holes, implicit 3x4 boxes.
- Model: D320/L12, heads10, head_dim32, channel_mult4, loop6, `LOOP_LOSS=all`.
- FutureSeed: fixed update, decay `0.0`, no scratch state, no loop feedback, no loop-time conditioning.
- Noise: `NOISE_SCALE=0.0`, `SCRATCH_NOISE_SCALE=0.0`, `ROLLOUT_NOISE_SCALE=0.0`.
- Runtime: GPU1 only, CUDA visible device 0, bf16, RWKV statepassing CUDA kernel.
- Batch: microbatch `24`, `GRAD_ACCUM_STEPS=3`, effective batch `72`.
- Smoke scope: two optimizer steps, eval_n `8`, rollout K `1`.

Command shape:

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 \
SUDOKU_SIZE=12 D_MODEL=320 LAYERS=12 HEADS=10 HEAD_DIM=32 \
CHANNEL_MULT=4 L_CYCLES=2 MAX_LOOPS=6 LOOP_LOSS=all \
FUTURE_SEED_UPDATE=fixed FUTURE_SEED_DECAY=0.0 \
NOISE_SCALE=0.0 ROLLOUT_NOISE_SCALE=0.0 SCRATCH_MODE=none \
FORWARD_DTYPE=bfloat16 RWKV_KERNEL=statepassing \
SMOKE_BATCH=24 GRAD_ACCUM_STEPS=3 SMOKE_STEPS=2 \
SMOKE_EVAL_N=8 SMOKE_ROLLOUT_KS=1 \
RUN_NAME=gradaccum-d320-b24x3-smoke-20260615T0444Z-f0a4efe \
./run.sh smoke
```

## Results

The GPU-only smoke completed with `git_dirty=false`. Score is `0.0` as expected after only two train steps and must not be compared to trained runs.

| item | value |
| --- | ---: |
| train steps | 2 |
| final train CE | 2.3856 |
| train time | 28.73s |
| microbatch | 24 |
| grad accumulation | 3 |
| effective batch | 72 |
| CUDA max allocated | 27025.29 MB |
| CUDA max reserved | 28600.00 MB |

## Insight

Gradient accumulation gives a cleaner scaling path than the earlier D320/B48 full run. The earlier run fit, but the lower batch and same-step budget collapsed the h96/h108 foundation and gave no h120 signal. This smoke shows that D320 can keep an effective batch of 72 without activation checkpointing and without leaving the 80GB card close to full.

This does not prove D320 quality. It proves the next fair capacity experiment should not be forced into the weak B48 setting. The next real question is whether D320 with effective batch 72 and resumable long training improves the h120/h132 frontier relative to the best D256 clean run.

## Decision

Keep gradient accumulation as standard scaling infrastructure. Use it for the next full capacity run only if the run is resumable and has reachable quality checkpoints. Do not tag this smoke.

Recommended next experiment: D320/L12/loop6, microbatch 24, grad accumulation 3, effective batch 72, clean fixed FutureSeed, no scratch/noise/selector/repair, h120 curriculum with train checkpoints. Kill or redirect if step time makes the first h120 checkpoint unreachable inside the current lease.
