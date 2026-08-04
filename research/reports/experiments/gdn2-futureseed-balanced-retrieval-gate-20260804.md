# GDN2 FutureSeed balanced retrieval gate

## Metainfo

- Plan: `P-CAUSAL-003`
- Status: discarded; no-FutureSeed carrier failed the preregistered validity gate
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

## Execution

- Clean detached GitHub truth:
  `1260e0934d933999ea2fda817edf7f2e3e90a270`.
- The full-size FutureSeed CUDA preflight completed forward, backward, four-loop
  evaluation, and visualization on exactly one A100 80GB GPU. It used 1,026,208
  parameters and `2.01 GiB` peak allocated memory.
- Runtime provenance was strict official FLA `GatedDeltaNet2` SHA `9c8e42e`,
  exact official classes in all four layers, chunk recurrence, Triton short
  convolutions, backend dispatch disabled, and no fallback.
- Preflight initialization SHA256 was
  `7615a9e693b432fb0d54b835f118debda92f59b7a3e5ef18be4c67ac2f407a7b`.
  The formal no-FutureSeed arm reproduced the same initialization hash.
- Formal fixed-eval SHA256 was
  `6d0dcd5c3422fbf2e3cc121b6c3b1b970235a647d084a32fc19774455fdf26af`.

## Step-200 Carrier Gate

| Readout | Step 1 | Step 100 | Step 200 |
|---|---:|---:|---:|
| Train query CE | 3.6128 | 3.4718 | 3.4721 |
| Loop-4 eval CE | 3.6237 | 3.4687 | 3.4699 |
| Loop-4 past accuracy | 0.0327 | 0.0347 | 0.0286 |
| Loop-4 future accuracy | 0.0334 | 0.0317 | 0.0273 |
| Loop-4 past exact | 0 | 0 | 0 |
| Loop-4 future exact | 0 | 0 | 0 |

Chance is `1/32 = 0.03125`. The no-FutureSeed model never learned even the
causally available write-before-query half. Loop 1 to loop 4 changed past and
future accuracy by only `+0.00049` and `+0.00024` at step200. The registered
past-accuracy threshold was `0.50`, so the carrier gate failed by a wide margin.

The event visualization is archived under
`runs/gdn2-future-retrieval-nofs-s300-20260804T0151Z-1260e09/output/visualizations/`.
It shows both past and future query predictions remaining random across loops.

## Decision And Lesson

The exact launcher PGID `23494` was stopped before the formal FutureSeed arm
entered model execution; GPU memory returned to zero. This is not evidence
against FutureSeed. As with the Maze carrier, the control behavior was not
established, so a scale-0/scale-1 delta would not identify the mechanism.

Do not rescue this custom proxy with more steps, a second seed, easier length,
fewer pairs, larger width, or a changed loss. The next cross-task attempt must
first reproduce an upstream, established GDN2 associative-recall or language
baseline using its official training/evaluation semantics. FutureSeed may be
added only after that no-FutureSeed baseline is verified. Until then, leaving
GPU1 idle is higher ROI than running another invented proxy.
