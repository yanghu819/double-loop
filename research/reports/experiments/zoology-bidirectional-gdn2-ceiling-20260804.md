# P-CAUSAL-017: Explicit Bidirectional GDN2 Ceiling

## 1. Metainfo

- Plan: `P-CAUSAL-017`
- Date: 2026-08-04 CST
- Machine: AIStation task-mode GPU1, one A100-SXM4-80GB
- Status: discarded by preregistered carrier gate
- Parent evidence: P-CAUSAL-016 result commit `d6747fbd`

## 2. Mechanism Question

Can native FutureSeed match a conventional bidirectional recurrent network at
lower systems cost, or has it only beaten a strict causal control with no
valid bidirectional ceiling?

The explicit baseline is deliberately strong and easy to interpret. At every
layer it runs two independent official-FLA GDN2 streams: one in the original
sequence order and one on the reversed sequence. The reverse output is flipped
back, concatenated with the forward output, and passed through a learned linear
fusion. This is a baseline only. It is never called FutureSeed.

FutureSeed remains exactly the P-CAUSAL-016 algorithm: the previous layer's
terminal recurrent state is RMS-normalized and gated into the next layer's
initial state. It never reverses or concatenates the sequence.

## 3. Fixed Experiment

- Task: exact frozen L512 directional MQAR with four associations, 10,000 train
  and 1,000 validation examples.
- Common model geometry: D128, two layers, four heads, K/V head dimension 32.
- Common optimization: batch 32, ten epochs, AdamW LR 1e-3, weight decay 0.1,
  cosine schedule and seed 123.
- Causal and FutureSeed quality are reused from the hash-pinned P-CAUSAL-016
  run; only the explicit bidirectional arm is newly trained.
- The explicit bidirectional model is a quality ceiling, not a parameter-matched
  arm. Each stream keeps the common D128/L2 geometry, while the second stream
  and fusion parameters are charged explicitly in parameter, throughput and
  memory accounting.
- Systems comparison: three fresh processes per arm, rotated order, three
  compile warmups followed by five benchmark warmups and 200 measured
  forward/backward steps per process. Use medians and report the full range.
- Budget: 40 minutes, GPU1 only. No CPU model smoke.

## 4. Prediction And Gates

Carrier gate:

- explicit bidirectional past and future accuracy each `>= 0.95`;
- explicit bidirectional joint exact `>= 0.90`.

Approximation gate:

- FutureSeed future accuracy no more than `0.03` below explicit bidirectional;
- FutureSeed joint exact no more than `0.05` below explicit bidirectional.

Efficiency gate:

- FutureSeed median throughput at least `1.25x` explicit bidirectional;
- FutureSeed median peak memory at most `0.80x` explicit bidirectional;
- each arm's fresh-process throughput relative range at most `0.15`.

If the bidirectional carrier misses its quality gate, stop without tuning
fusion, epochs, LR, width, depth, seed or loss. If it opens but FutureSeed
misses quality, the cheap-bidirectionality claim is weakened. If quality
matches but efficiency does not, FutureSeed is an alternative route rather
than a cheap one. Only a full pass permits a later three-way length curve.

## 5. Integrity

The CUDA preflight must prove one visible physical GPU1, exact Zoology and FLA
source hashes, official chunk/Triton execution in all four bidirectional GDN2
streams, exact data hashes, zero causal future dependency, nonzero explicit
bidirectional future dependency, finite backward, and nonzero gradients in the
reverse stream and fusion projection. The baseline must contain no FutureSeed
module or fallback.

Forbidden: selector, search, repair, task rule, oracle, reverse scan mislabeled
as FutureSeed, multi-seed runs, and post-hoc optimization rescue.

## 6. Results

The first detached launch at `75567b1b` stopped before model execution because
the runner imported an undefined constant. It is archived as an engineering
failure with `scientific_failure=false`. Commit `6a39f0f6` fixed only that
import and preserved every scientific setting.

The formal run was
`zoology-bidirectional-gdn2-ceiling-20260804T140052Z-6a39f0f`, source SHA
`6a39f0f6b9da32401abdbc4eb225c2c01ef46dde`, launcher PID/PGID
`67585/67585`. The strict preflight passed:

- exactly one visible A100 80GB with the registered GPU1 UUID;
- exact Zoology, FLA, P-CAUSAL-016 score and L512 train/test hashes;
- four official-FLA GDN2 streams, all `chunk` mode with Triton convolution;
- causal future dependency exactly `0`; bidirectional future dependency
  `0.123501`;
- finite loss, reverse-stream gradient `0.185547`, fusion gradient `0.089844`;
- no FutureSeed module inside the explicit bidirectional baseline.

| arm | parameters | past acc | future acc | joint exact | past CE | future CE |
|---|---:|---:|---:|---:|---:|---:|
| causal GDN2 | 596,048 | 0.0345 | 0.0105 | 0.0000 | 4.1991 | 4.5961 |
| GDN2 + FutureSeed | 596,048 | 0.9860 | 0.9850 | 0.9420 | 0.0481 | 0.0461 |
| explicit forward + reverse GDN2 | 894,608 | 0.0345 | 0.0120 | 0.0000 | 4.0001 | 4.6667 |

The explicit bidirectional model's aggregate validation accuracy was
`0.01425/0.01425/0.01375/0.01100/0.01075/0.01075/0.01900/0.02500/0.02525/0.02325`
across epochs 0--9. Its best value was only `0.02525`; it never opened. The
registered `0.95/0.95/0.90` past/future/joint carrier gate therefore failed by
a large margin.

## 7. Kill And Cost Accounting

Training completed in `126.48` seconds with peak allocated CUDA memory
`815,277,056` bytes. The runner had entered its first fresh-process benchmark,
but this was stopped by exact process groups after the carrier failure became
known. Robust throughput and memory ratios are intentionally not reported:
cost comparisons against a chance-level quality baseline do not answer the
paper question.

The final `abort.json` records `scientific_failure=true`,
`training_completed=true`, and
`benchmarks_skipped_after_carrier_failure=true`. GPU1 returned to zero memory
and utilization; GPU2 and CPU model execution were never used.

## 8. Decision And Lesson

Discard this explicit per-layer forward/reverse fusion as a paper ceiling. Do
not tune its fusion, LR, epochs, width, depth, seed or loss. FutureSeed is about
`+0.9623` balanced accuracy above it at L512, but this is not evidence that
FutureSeed beats bidirectional recurrent models in general because the
baseline did not solve the carrier.

The mechanistic lesson is sharper: making information visible from both
directions is not sufficient for long-range random key/value binding. At L512,
each plain directional GDN2 stream still has to preserve associations over a
long causal distance, and both streams remain near chance. FutureSeed's
terminal-state transfer is doing more than replacing a reverse scan; it also
creates a trainable cross-layer long-range memory route. A future cost-quality
claim still requires an independently established bidirectional model/task
pair that opens before comparison.

## 9. Artifacts

- Remote run: `/huyang2/double-loop/runs/zoology-bidirectional-gdn2-ceiling-20260804T140052Z-6a39f0f`
- Frozen P-CAUSAL-016 reference:
  `/huyang2/double-loop/runs/zoology-gdn2-fs-length-curve-20260804T125520Z-cbe7060`
- Formal outputs: `config.json`, `preflight.json`, `abort.json`, model
  `score.json`, `metrics.jsonl`, `cases.json`, logs and source snapshot.
- Visualization: `visualizations/index.html` with the quality table and 12
  hardest matched cases. Cost cells are explicitly marked not measured.
- No experiment tag: the carrier failed and no new positive score is claimed.
