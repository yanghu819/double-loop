# Established BERT MLM Carrier Gate

- Plan: `P-CAUSAL-009`
- Status: discarded by the registered endpoint carrier gate
- Date: 2026-08-04 CST
- Machine: AIStation task-mode GPU1 only

## Question

Can a standard, published bidirectional masked-language model open on the fixed
WikiText carrier and clearly outperform an otherwise identical strict-causal
control before GDN2 or FutureSeed is introduced?

## Insight And Mechanism Hypothesis

P-CAUSAL-008 did not fail because right context was useless. Its custom
attention ceiling omitted core parts of an established masked-language model:
standard WordPiece tokenization, learned absolute position embeddings, the
BERT residual/normalization stack, and the BERT MLM prediction head. A bare
noncausal attention matrix is not a validated BERT baseline.

This gate restores the complete published BERT miniature architecture and the
official Transformers MLM pipeline. If the carrier is valid, full
bidirectional BERT should learn masked recovery and beat the same model with a
strict causal attention mask. If it does not, no FutureSeed language result is
interpretable on this setup.

## Pinned Upstream Components

- Model architecture: Google BERT miniature `L-2_H-128_A-2`, associated with
  *Well-Read Students Learn Better*; Apache-2.0 model assets at revision
  `12e6feda7b005864b03bd5f7f5b6a7d00f24c82e`.
- Trainer/data pipeline: Hugging Face Transformers tag `v4.46.3`, commit
  `052e652d6d53c2b26ffde87e039b723949a53493`, exact `run_mlm.py` SHA256
  `b5017fb36ffacbe81dc53a0dd77dc3420369a010f5a37fc5263228d7a8811cb1`.
- Installed `BertForMaskedLM` must match the same tag's source SHA256
  `3493bff5da90fdcce98dad5c84aafe4d3ce1c550dcd93bc99289309953559eca`.
- The only upstream-script patch replaces the online `evaluate.load` name with
  a pinned local accuracy-metric path. It cannot change model, data, loss,
  optimizer, masking, or metric semantics.
- Corpus: the same locally downloaded and hash-pinned WikiText-103 raw subset
  used by P-CAUSAL-008, converted losslessly to JSON Lines. No server download.
- Prepared train/validation JSON SHA256 values are
  `6e22a5a233e6f7901db6714ceabfc71f4e9e99aa4769f0c9bb8d7c3fd5fdd0cd`
  and `7f24169df1b7683363682a3d90f0a4c513b941eabaa2c121843b322f29b77658`;
  manifest SHA256 is
  `e6fc677fa5c663f79e4c44d2d15be0e1255f7d5379ef8afe8af1d76b30157b9b`.

## Registered Pair

Both arms are BERT-for-MLM initialized from scratch with hidden size 128, two
layers, two attention heads, intermediate size 512, 30,522-token uncased
WordPiece vocabulary, learned absolute positions, tied embedding/readout, and
the standard BERT MLM transform. The only model semantic difference is:

```text
bidirectional: config.is_decoder = false
strict causal: config.is_decoder = true
```

`add_cross_attention=false` and `use_cache=false` in both. A CUDA preflight
must prove identical parameter keys, counts, and initialization; zero causal
future dependency; nonzero bidirectional future dependency; finite backward;
and one exact GPU1.

Training is fixed at sequence length 128, 15% standard dynamic MLM corruption,
batch 128, 1,250 optimizer steps, BF16, AdamW LR `1e-4`, weight decay `0.01`,
linear schedule with 10% warmup, and seed 123. Each arm therefore consumes
exactly 20.48M input tokens. Validation runs every 250 steps. Run the
bidirectional arm first; start causal only if bidirectional opens.

## Prediction, Budget, And Kill Criteria

- Opening requirement: by step 250, bidirectional validation accuracy should
  exceed `0.02` or its CE should be clearly falling. At the endpoint it should
  exceed `0.10` accuracy with a continuing or converged positive curve.
- Carrier success: bidirectional must beat strict causal by at least `+0.03`
  masked accuracy or `-0.20` masked CE at equal tokens and parameters.
- Stop immediately on wrong GPU/UUID, online fallback, source/data/tokenizer
  hash mismatch, unequal initialization/parameters, causal future leakage,
  missing bidirectional future dependency, non-finite backward, NaN, or OOM.
- If bidirectional does not open, stop before causal and audit the upstream
  reproduction. Do not rescue with epochs, LR, mask rate, width, depth,
  tokenizer, or seed sweeps.
- One bidirectional arm plus one conditional causal arm; maximum one GPU hour.

## Claim Boundary

A pass validates only the established natural-language carrier and its
right-context sensitivity. It does not establish FutureSeed language quality.
Only after this gate passes may a separate preregistered experiment replace
the mixer with matched official-FLA GDN2 scale 0/1. No reverse scan, extra
layer, extra step, oracle, selector, search, repair, or text-specific rule is
allowed.

## Launch Audit

- `established-bert-mlm-bidirectional-20260804T073834Z-97c1af7` exited before
  the first optimizer step because the offline accuracy metric path was sourced
  as a shell variable but not exported to the exact upstream Python child.
- CUDA preflight had passed, no training result was produced, GPU memory
  returned to zero, and the run contains `abort.json` with
  `scientific_failure=false`. The launcher now exports the entire pinned env
  file before materializing and executing `run_mlm.py`; no experimental
  setting changed.

## Result

The corrected exact-upstream bidirectional arm completed on source SHA
`a86805732a1a238039bcecb8c74224ad15eeaee0`. Preflight established one exact
GPU1, `4,416,698` parameters in both registered configs, identical
initialization, zero causal future dependency, bidirectional future dependency
`0.003185`, finite CUDA backward, and exact Transformers source provenance.

| Step | masked validation CE | masked validation accuracy |
|---:|---:|---:|
| 250 | 8.06919 | 0.04586 |
| 500 | 7.25322 | 0.06869 |
| 750 | 7.25350 | 0.06934 |
| 1000 | 7.21631 | 0.06966 |
| 1250 | 7.19014 | 0.07438 |
| final independent eval | 7.21371 | 0.07162 |

The step-250 weak opening condition passed, but the preregistered endpoint
accuracy requirement `>0.10` did not. Accuracy improved only about `+0.0057`
from step 500 to the independent final evaluation, so the carrier was not
merely interrupted on a steep slope. The strict-causal control was therefore
not started, and GDN2/FutureSeed were never introduced.

Training consumed exactly 20.48M registered input tokens, took `72.83` seconds
at `2,196.9` samples/s (`17.16` optimizer steps/s), and an external training
sample observed `12,255 MiB` GPU memory. The model checkpoint remains outside
Git with SHA256
`11a56b051035d4364b072302be4cba693b6092b5b86cc15ce791367cd4a170c8`.

A deterministic 64-window audit using the same grouping and standard 15%
corruption produced accuracy `0.05321` and CE `7.39540` over 1,184 masked
tokens. Its hardest cases show mostly high-frequency generic predictions with
low target probability, consistent with an undertrained language model rather
than a directional masking bug.

## Decision

Discard P-CAUSAL-009 as an invalid language carrier at this fixed budget. It is
neither positive nor negative evidence about FutureSeed. Do not rescue it by
changing epochs, learning rate, model width/depth, mask rate, tokenizer, or
seed. The next paper experiment returns to the validated Zoology + official-FLA
GDN2 shell and tests the already-positive mechanism along a genuine scaling
axis with an explicit bidirectional-attention ceiling.

## Artifacts

- Completed run:
  `runs/established-bert-mlm-bidirectional-20260804T074331Z-a868057`
- Fixed-mask visualization:
  `runs/established-bert-mlm-bidirectional-20260804T074331Z-a868057/visualizations/index.html`
- Non-scientific launcher abort:
  `runs/established-bert-mlm-bidirectional-20260804T073834Z-97c1af7`
- Source snapshot SHA256:
  `10d5fe30a15c65626a278ffbc191b2819d9540e10824e7d5e2bef7d88d9093e3`
