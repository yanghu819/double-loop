# P-CAUSAL-016: GDN2 FutureSeed Context-Length Curve

## 1. Metainfo

- Plan: `P-CAUSAL-016`
- Date: 2026-08-04 CST
- Machine: AIStation task-mode GPU1, one A100-SXM4-80GB
- Status: completed; all preregistered gates passed
- Run: `zoology-gdn2-fs-length-curve-20260804T125520Z-cbe7060`
- Source SHA: `cbe7060d1962ef4fd6f9c7df6e9556b34ec9779b`
- Start/end: `2026-08-04T12:55:30Z` / `2026-08-04T13:19:24Z`

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

- Run root:
  `runs/zoology-gdn2-fs-length-curve-20260804T125520Z-cbe7060/`
- Primary score: `score.json`; SHA256
  `a07b4103dee02e10a7c848614b1c2545c6dc6213e0199a02d415e3dbe1e582d8`.
- Integrity: `preflight.json`, `git_sha.txt`, `git_status.txt`, exact data
  hashes, FLA/Zoology/source hashes and the remote source-snapshot hash.
- Raw evidence: per-length/arm scores, cases and fresh-process benchmark
  JSON under `output/`, plus `formal.log`.
- Visualization: `visualizations/index.html`, `hardest_cases.json`, desktop
  and mobile screenshots.

## 7. Results

### 7.1 Quality curve

| Length | Causal past | Causal future | FS past | FS future | FS joint exact |
|---:|---:|---:|---:|---:|---:|
| 64 | 0.9965 | 0.0100 | 0.9985 | 0.9920 | 0.9810 |
| 128 | 0.0490 | 0.0085 | 0.9965 | 0.9810 | 0.9550 |
| 256 | 0.0315 | 0.0090 | 0.9845 | 0.9780 | 0.9320 |
| 512 | 0.0345 | 0.0105 | 0.9860 | 0.9850 | 0.9420 |
| 1024 | 0.0110 | 0.0085 | 0.7535 | 0.7415 | 0.3390 |

Every preregistered check passed. FutureSeed-minus-causal future-accuracy
deltas are `+0.9820/+0.9725/+0.9690/+0.9745/+0.7330` from L64 through
L1024. The L128 early-kill was not triggered. The newly trained L128/L256/L512
FutureSeed arms all exceed their absolute floors, and the strict-causal future
direction remains at chance.

The curve is not a gradual dilution from L64. FutureSeed future accuracy stays
between `0.9780` and `0.9920` through L512, then drops by `0.2435` at L1024.
Retention from L64 to L1024 is `0.7475`. This localizes the capacity/optimization
boundary between 512 and 1024 tokens under the fixed ten-epoch recipe.

### 7.2 Mechanism and errors

The full-size CUDA preflight used L512 and passed all controls: one physical
GPU1, official FLA GDN2 in chunk mode with Triton convolutions, identical
parameters and initialization, exact `scale=0` output identity, causal future
dependency `0`, FutureSeed future dependency `0.01557`, nonzero FutureSeed-gate
gradient `0.00219`, and finite backward.

At L128/L256/L512, FutureSeed makes only `38/44/30` future-query errors out of
2,000. At L1024 it makes 517. Of those L1024 errors, 426 (`82.4%`) select the
value attached to another key in the same sequence. Thus the 1024-token limit
is predominantly key/value binding interference after future content has been
transported, not loss of all future information.

### 7.3 Systems evidence

Fresh-process peak allocated memory differs by only about `1.50 MiB` between
the matched arms at every length. At L1024 this is `1028.0 MiB` causal versus
`1029.5 MiB` FutureSeed, consistent with no sequence-sized backward state being
added by the seed itself.

The isolated warmed throughput measurements are useful diagnostics, but not a
paper-ready speed curve. They used only 20 measured steps (0.30--1.01 seconds),
and the causal/FS ratio changes implausibly across lengths, including an FS
speedup at L256 and a slowdown to `0.513x` at L1024. Do not claim a precise
compute advantage from these timings. A longer repeated timing protocol and a
separately opened bidirectional baseline remain required for the final
cost-versus-quality figure.

## 8. Conclusions

The mechanism claim passes strongly: terminal-state FutureSeed converts an
otherwise chance future direction into near-perfect retrieval through 512
tokens without changing parameter count, data, optimizer, training tokens or
official FLA kernel. It also rescues past retrieval once irrelevant context
makes the same shallow causal stack fail at L128 and above.

The correct narrow claim is therefore:

> A previous recurrent layer's terminal state can serve as a compact
> future-context route for a deeper causal GDN2 layer, preserving high
> bidirectional retrieval quality through 512 tokens and substantial quality
> at 1024 tokens.

This experiment does not establish superiority over Transformers. P-CAUSAL-015
closed the attempted bidirectional-attention carrier because it never learned
the binding task; using its roughly 0.49 accuracy as a ceiling would be
misleading. The next paper gate is an independently validated bidirectional
carrier and a robust repeated timing frontier. Do not run more MQAR seeds,
middle lengths, LR/epoch rescues or width/value-state tables.

## 9. Publication Record

The run is eligible for a highlighted annotated experiment tag after the
result commit because all registered gates passed and L512 FutureSeed future
accuracy is `0.9850`. The tag must describe the narrow mechanism result and
must not claim a Transformer comparison.
