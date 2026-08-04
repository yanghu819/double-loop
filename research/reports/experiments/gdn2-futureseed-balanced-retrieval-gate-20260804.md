# GDN2 FutureSeed balanced retrieval gate

## Metainfo

- Plan: `P-CAUSAL-003`
- Status: preregistered; strict CUDA preflight pending
- Date: 2026-08-04
- Machine: AIStation task-mode GPU1 only
- Seed: 52 only

## Mechanism Hypothesis

Native FutureSeed passes a recurrent layer's terminal memory into the next
layer's initial memory. This should let a causal GDN2 answer an early query
using a key-value write that occurs later in the sequence, without a reverse
scan, attention, search, or task rule.

The task contains equal numbers of:

- past queries: write first, query later;
- future queries: query first, matching write later.

Values are uniformly random and unique within each sequence. A no-FutureSeed
causal model can learn past queries but is information-theoretically limited to
chance on future queries. Query-only CE is balanced, so there is no Maze-style
background-class shortcut.

## Prediction

- The no-FutureSeed arm reaches past-query accuracy at least `0.80`, proving
  that the GDN2 carrier and optimizer can perform ordinary retrieval.
- No-FutureSeed future-query accuracy remains near chance (`1/32 = 0.03125`).
- Native FutureSeed reaches future-query accuracy at least `0.70` and improves
  it by at least `+0.50`, while preserving past-query accuracy within `0.05`.
- Increasing loops should preserve or improve future-query accuracy rather
  than merely alter an imbalanced output threshold.

## Experiment

One matched pair only:

- strict official FLA `GatedDeltaNet2` SHA `9c8e42e`, chunk/Triton, no fallback;
- D128/L4/H4/D32, expand-v1, short-conv4;
- sequence length 128, eight unique key-value pairs, four past and four future
  queries, key vocabulary 64, value vocabulary 32;
- train loops2, eval loops4, query-only CE at every loop;
- batch128, 300 optimizer steps, fixed eval bank of 1024 sequences, seed52;
- arms differ only by native FutureSeed scale `0` versus `1`.

Report past/future/query accuracy and exact, CE, loop1-to-loop4 changes,
initialization hash, fixed-eval hash, parameters, wall time, peak VRAM, strict
runtime provenance, and same-sequence event visualizations.

## Kill Criteria

- Stop on anything other than exactly one visible GPU, wrong source, wrong
  official FLA class/SHA, backend dispatch, non-Triton short convolution,
  fallback, failed CUDA backward, NaN, or OOM.
- If past-query accuracy is below `0.50` at step200, mark the carrier invalid
  and stop rather than tuning it.
- After one pair, reject cross-task transfer if FutureSeed future accuracy is
  below `0.50` or its delta is below `+0.20`.
- No second seed, width/depth/length table, loss change, reverse scan, selector,
  search, repair, or oracle state.

## Claim Enabled

A strong result supports the narrow, general claim that FutureSeed gives a
causal linear-attention recurrence a cheap route to future key-value evidence.
It does not establish language-model quality; that would require a later
natural-language benchmark. A failed or invalid carrier ends this proxy without
rescue tuning.
