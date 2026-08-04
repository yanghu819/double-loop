# P-CAUSAL-017: Explicit Bidirectional GDN2 Ceiling

## 1. Metainfo

- Plan: `P-CAUSAL-017`
- Date: 2026-08-04 CST
- Machine: AIStation task-mode GPU1, one A100-SXM4-80GB
- Status: preregistered
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

Pending formal GPU1 run.
