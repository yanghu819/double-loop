# P-CAUSAL-016: GDN2 FutureSeed Context-Length Curve

## 1. Metainfo

- Plan: `P-CAUSAL-016`
- Date: 2026-08-04 CST
- Machine: AIStation task-mode GPU1, one A100-SXM4-80GB
- Status: in progress
- Run/source SHA: recorded by the exact-SHA launcher

## 2. Hypothesis

P-CAUSAL-010 and P-CAUSAL-012 already fix the endpoints. With four key/value
associations, native FutureSeed future accuracy changes from `0.9920` at length
64 to `0.7415` at length 1024, while the strict-causal future control remains
near chance. If a previous layer's terminal recurrent state is a scalable but
finite summary of future context, intermediate lengths should produce a
gradual degradation curve rather than an early directionality cliff.

This experiment characterizes one mechanism axis. It does not add a new model,
claim Transformer superiority, or mix context length with memory load.

## 3. Configuration

- Frozen endpoints: exact P-CAUSAL-010 L64 and P-CAUSAL-012 L1024 score/source
  hashes; no retraining.
- New lengths: 128, 256 and 512. Four associations remain fixed.
- Data: deterministic directional MQAR, 10,000 train and 1,000 validation
  examples per length, mixed past/future queries, vocabulary 256.
- Model: official-FLA GDN2 D128, two layers, four heads, K/V head dimension 32,
  `expand_v=1`, chunk mode and Triton short convolution.
- Arms: strict causal `future_seed_scale=0` and native terminal-state seeding
  `future_seed_scale=1`; same parameter tensors and initialization.
- Training: batch 32, ten epochs, AdamW LR 1e-3, weight decay 0.1, cosine
  schedule and seed 123.
- Systems measurement: every arm at every length is reconstructed in its own
  fresh Python process, warmed before measurement, and reports tokens/s and
  peak allocated CUDA memory. This avoids the known same-process Triton
  autotune ordering confound, including at the frozen endpoints.

Registered FutureSeed future-accuracy floors are `0.95/0.90/0.80` at
L128/L256/L512. At every length, causal future accuracy must be at most `0.10`
and the FutureSeed-minus-causal future delta at least `+0.70`. Frozen L64/L1024
floors are `0.90/0.70`.

## 4. Environment

- SSH alias: `aistation-task-gpu1`
- Persistent root: `/huyang2/double-loop`
- CUDA visibility: only container index 0, physical GPU1 UUID
  `GPU-53e9f3b4-2966-65d3-6614-09c540921519`
- Python: `/opt/conda/bin/python`
- Zoology SHA: `1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb`
- FLA SHA: `9c8e42e762fce087c27b673af4922795d9edb85e`
- CPU model smoke and GPU2 are forbidden.

## 5. Commands

```bash
CUDA_VISIBLE_DEVICES=0 RUN_NAME=<generated> ./run.sh gdn2_length_curve
```

The launcher first validates GPU/source/wheel/data hashes, official
FLA/Triton mode, matched initialization and parameters, exact scale-zero
identity, zero causal future leakage, active FutureSeed dependence and finite
CUDA backward at the largest newly trained length.

Budget: two wall-clock hours. Kill after L128 if FutureSeed future accuracy is
below `0.80`, or immediately on any integrity, fallback, OOM, NaN or timeout
failure. No epoch, LR, seed, width, depth, loss or kernel rescue is allowed.

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Conclusions

Pending.

## 9. Publication Record

No tag before the registered gate and artifact audit complete.
