# Native FutureSeed L64 to L1024 Endpoint

## 1. Metainfo

- Plan: `P-CAUSAL-012`
- Status: completed; registered strong and partial endpoint gates missed
- Date: 2026-08-04 CST
- Machine: AIStation task-mode GPU1 only
- Branch: `codex/p-causal-012-gdn2-length1024`
- Source SHA: `77e5539fc0ef74231cab658bd610b24254fdcfa7`
- Run: `zoology-gdn2-fs-length1024-20260804T095300Z-77e5539`
- Exact launcher PID/PGID: `47365/47365`

## 2. Hypothesis

Native FutureSeed gives layer two the normalized terminal recurrent state from
layer one. P-CAUSAL-007 and P-CAUSAL-010 show that this route solves future
retrieval at sequence length64. The unresolved core scaling question is
whether one fixed-size terminal state remains informative after 16x more
irrelevant context.

Holding the number of associations at four isolates context dilution from
memory load. A scalable route should preserve high future accuracy at L1024;
strict causal GDN2 should still solve past queries and remain at chance on
future queries.

## 3. Configuration

- Frozen reference: P-CAUSAL-010 L64 no-FS/FS scores and cases from source SHA
  `4b909629ee3cc32ab1c57e9d3ac1f26ecaa8b273`; no L64 retraining.
- New data: exact directional generator, vocab256, sequence1024, four unique
  associations, 10,000 train and 1,000 fixed validation examples.
- Model: strict official-FLA GDN2 D128/L2/H4/D32, expand-v1, short-conv4,
  chunk recurrence and Triton convolution; upstream Zoology positions, MLP,
  residual/norm shell and tied readout.
- Training: batch32, 10 epochs, AdamW LR `1e-3`, weight decay `0.1`, cosine,
  seed123 and the same query-only CE as L64.
- Arms: FutureSeed scale0 versus scale1 with identical initialization, data,
  parameters, optimizer, steps and kernel.
- FutureSeed remains cross-layer terminal-state seeding. There is no reverse
  scan, attention arm, outer loop, selector, search, repair or task rule.

## 4. Environment

- Remote alias: `aistation-task-gpu1`.
- Root: `/huyang2/double-loop`; all cache/run/artifact paths remain below it.
- GPU: exactly one A100-SXM4-80GB UUID
  `GPU-53e9f3b4-2966-65d3-6614-09c540921519`, exposed as CUDA index0.
- Pinned Zoology SHA `1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb`.
- Pinned FLA semantic SHA
  `9c8e42e762fce087c27b673af4922795d9edb85e` and wheel/source hashes.
- No CPU model smoke and no backend fallback.

## 5. Commands

```bash
CUDA_VISIBLE_DEVICES=0 PERSIST_ROOT=/huyang2/double-loop \
  ./run.sh gdn2_length1024
```

The launcher first repeats the strict full-batch L1024 CUDA/provenance test.

## 6. Artifacts

- Run directory:
  `runs/zoology-gdn2-fs-length1024-20260804T095300Z-77e5539`
- Main metrics: `score.json` and `output/comparison.json`
- Registered scientific stop: `abort.json`
- Same-sequence visualization: `visualizations/index.html`
- Source snapshot hash, config, logs, GPU metadata, per-arm cases and
  preflight results are archived with the run. The 66 MB source tar remains
  on persistent remote storage and is represented in Git by its SHA256.

## 7. Registered Readouts

- past/future accuracy, exact and CE;
- balanced accuracy and joint exact;
- FutureSeed future-accuracy delta over causal GDN2;
- FutureSeed future-accuracy retention from L64 to L1024;
- independently warmed tokens/s and peak memory at both lengths;
- hardest same-sequence query/write/target/prediction cases.

Strong support requires L1024 causal past `>=0.90`, causal future `<=0.10`,
FutureSeed past `>=0.90`, FutureSeed future `>=0.80`, future delta `>=+0.70`
and at least80% retention from the frozen L64 FutureSeed accuracy. Partial
signal requires FutureSeed past `>=0.90`, future `>=0.50` and delta `>=+0.40`.

## 8. Kill Criteria And Decision

Stop on wrong GPU/SHA/wheel/source/data hash, scale0 mismatch, causal leakage,
dead FutureSeed dependency, non-finite backward, fallback, OOM/NaN or the
two-hour wall limit. An endpoint miss is archived without seed, LR, width,
depth, epoch or loss rescue.

A strong pass authorizes filling the middle sequence lengths for the paper
curve, then one separate K4 to K32 memory-load axis. A miss says the current
fixed-size terminal state loses too much information with context and directs
the next mechanism work toward generic state capacity/compression, not task
hacks.

## 9. Result And Submission

The strict CUDA and provenance preflight passed. Exactly one A100 GPU was
visible; the official FLA GDN2 source and Triton convolution were used; the two
arms had `661,584` parameters, identical initial tensors and identical train
and validation data. The no-FutureSeed path had zero future dependency, the
FutureSeed path had nonzero future dependency and gate gradient, and the
scale-zero output difference was exactly zero.

### Quality

| Length | Model | Past acc | Future acc | Past exact | Future exact | Joint exact | Past CE | Future CE |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 64 | causal GDN2 | 0.9965 | 0.0100 | 0.9930 | 0 | 0 | 0.0236 | 4.7085 |
| 64 | GDN2 + FutureSeed | 0.9985 | 0.9920 | 0.9970 | 0.9840 | 0.9810 | 0.0083 | 0.0280 |
| 1024 | causal GDN2 | 0.0110 | 0.0085 | 0 | 0 | 0 | 4.8056 | 4.8181 |
| 1024 | GDN2 + FutureSeed | 0.7535 | 0.7415 | 0.5900 | 0.5720 | 0.3390 | 0.6143 | 0.6167 |

At length1024, FutureSeed improves future accuracy by `+0.7330`, past accuracy
by `+0.7425`, and joint exact by `+0.339`. This is a large long-context
optimization and information-routing effect. It is not a strong scaling pass:
FutureSeed future accuracy is below the registered `0.80`, past accuracy is
below `0.90`, and future-accuracy retention from L64 is `0.7475`, below
`0.80`. The causal past carrier also failed to open, so this endpoint cannot
support the narrower claim that only future directionality fails at L1024.

The learning curves separate the arms sharply. Causal validation accuracy
stays around chance for all ten epochs. FutureSeed is also near chance through
epoch2, then moves `0.0588 -> 0.1965 -> 0.3450 -> 0.7158` at epochs3-6 and
ends at `0.7475`. The last two epochs are nearly flat, so extending this exact
run would be an unregistered rescue rather than evidence-based scaling.

### Error mechanism

FutureSeed usually retains the possible values but confuses which key owns
which value. Among its 517 wrong future-query predictions, 426 (`82.4%`) are
the correct value for another key in the same sequence. The same is true for
389 of 493 (`78.9%`) wrong past-query predictions. Hardest-case visualization
shows direct value swaps between the two future associations. Error distance
is not strongly separated: mean absolute distance is about 518 for errors and
509 for correct predictions. This points to address/binding capacity or
interference, not simple inability to transport any value across 1024 tokens.

### Systems boundary

Peak allocated memory in the warmed L1024 step is nearly identical:
`1033.0/1034.5 MiB` for causal/FutureSeed. The archived raw throughput is
`0.727/1.422M tokens/s`, but the faster FutureSeed number is not accepted as a
paper speed claim: the arms ran sequentially in one process and likely shared
Triton compilation/autotuning cache. A separate fresh-process, alternating-
order systems benchmark is required before quoting runtime.

### Decision

The launcher wrote `abort.json` with `scientific_failure=true` because the
registered endpoint gate missed; the process itself completed normally and
GPU memory returned to zero. Do not run middle lengths, another seed, more
epochs, or tune LR/loss. The next high-information test should change exactly
one general scaling axis: recurrent address/state capacity at L1024. If a
larger state specifically reduces same-case value swaps while preserving the
FutureSeed gain, the limiting factor is state interference. If it does not,
the bottleneck is the terminal-state compression/update mechanism rather than
raw capacity. No experiment tag is created for this boundary result.
