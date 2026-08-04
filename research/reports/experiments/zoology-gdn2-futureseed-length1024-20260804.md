# Native FutureSeed L64 to L1024 Endpoint

## 1. Metainfo

- Plan: `P-CAUSAL-012`
- Status: in progress; full-L1024 CUDA preflight precedes the matched pair
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

Pending. The run will archive config, scores, logs, source hash, GPU/PID
metadata, endpoint gate and same-sequence L64/L1024 HTML visualization.

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

Pending. No tag before a completed strong endpoint.
