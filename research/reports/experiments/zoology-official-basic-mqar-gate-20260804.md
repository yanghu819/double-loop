# Zoology Official Basic MQAR Reproduction Gate

- Plan: `P-CAUSAL-005`
- Status: in progress
- Date: 2026-08-04 CST
- Machine: AIStation task-mode GPU1 only

## Question

Can the exact upstream Zoology code, model, trainer, data generator, and metric
reproduce its documented basic MQAR result in the current CUDA environment?

## Mechanism Hypothesis

P-CAUSAL-004 used the upstream MQAR generator but a project-local recurrent
training shell. Its failure therefore did not identify whether the problem was
MQAR itself, the shell, or GDN2. The unmodified README baseline is a two-layer
causal Transformer with learned positions. If this exact upstream carrier
opens, the generator/trainer/metric path is valid and the failed custom shell
is the remaining difference.

## Fixed Contract

- Exact Zoology commit:
  `1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb`, clean worktree.
- Exact documented config:
  `zoology/experiments/basic_examples/basic.py`, with no semantic override.
- Data: official MQAR, vocab256, sequence64, four K/V pairs, 10,000 train and
  1,000 test examples.
- Model: official two-layer D128, one-head causal MHA, MLP state mixer, tied
  embeddings/readout, learned positional embeddings.
- Training: exact upstream Trainer defaults: batch32, AdamW LR1e-3, weight
  decay0.1, cosine schedule, seed123, up to100 epochs, early stop above0.99.
- Command: upstream `python -m zoology.launch ...basic.py` on one visible GPU.
- No CPU model smoke and no second seed, LR sweep, rescue, or code patch.

## Prediction

The official baseline should exceed `0.99` query accuracy and trigger its own
early stop. This would establish an executable upstream carrier, not a
FutureSeed result.

## Budget And Kill Criteria

- One GPU1 run, at most 30 wall-clock minutes.
- Stop for wrong SHA, more than one visible GPU, CPU execution, NaN/OOM,
  dependency/fallback error, or no validation signal after the official run.
- Do not simplify the vocabulary/task or tune model/training settings if it
  fails. Archive the exact failure and inspect upstream/environment drift.

## Decision

- If validation accuracy exceeds `0.99`, next port strict official FLA GDN2
  into this exact upstream shell on ordinary write-before-query MQAR. Only
  after that carrier opens may native FutureSeed and future-query data enter.
- If the exact upstream baseline fails, stop cross-task mechanism claims until
  the official reproduction discrepancy is resolved.

## Claim Boundary

Success proves only that the official Zoology benchmark pipeline is correctly
reproduced on GPU1. It does not support GDN2 or FutureSeed by itself.
