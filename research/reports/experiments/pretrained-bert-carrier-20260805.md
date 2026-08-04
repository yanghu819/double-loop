# Official Pretrained BERT-Tiny Carrier Gate

- Plan: `P-CAUSAL-018`
- Status: preregistered; blocked only on local checkpoint transfer
- Date: 2026-08-05 CST
- Machine: AIStation task-mode GPU1 only

## Question

Does an established, officially pretrained bidirectional masked-language model
actually use right context and open on the exact fixed WikiText-103 carrier
before GDN2 or FutureSeed is introduced?

## Hypothesis

P-CAUSAL-009 trained the published BERT-Tiny architecture from scratch on only
20.48M input tokens. Its 7.16% endpoint accuracy did not validate the carrier,
and extending that run would be rescue tuning. The official Google checkpoint
was pretrained with the published BERT recipe and therefore provides a clean,
independently established carrier test without another training budget choice.

The same pretrained tensors are evaluated twice: once with their native full
bidirectional mask and once after changing only the attention semantics to a
strict causal mask. This is not a fair trained causal baseline and will never
be reported as one. It asks only whether the fixed task and checkpoint expose a
measurable right-context benefit.

## Fixed Protocol

- Model: `google/bert_uncased_L-2_H-128_A-2`, revision
  `12e6feda7b005864b03bd5f7f5b6a7d00f24c82e`.
- Runtime: pinned Transformers 4.46.3 `BertForMaskedLM` source already verified
  by P-CAUSAL-009.
- Data: exact hash-pinned P-CAUSAL-009 WikiText-103 validation JSON and official
  uncased WordPiece tokenizer.
- Corruption: one deterministic standard 15% MLM mask over 256 grouped
  length-128 windows, seed 123.
- Arms: byte-identical checkpoint tensors; only full versus strict-causal
  attention masking differs.
- Evaluation: GPU1 BF16, masked-token CE and top-1 accuracy, future-perturbation
  dependency, finite backward, independently warmed inference throughput and
  peak allocated memory, and 12 same-window cases.
- No training, FutureSeed, GDN2, reverse scan, selector, search, repair, or
  task-specific rule is present.

The checkpoint must be downloaded on the local Mac, SHA256-recorded, and then
uploaded to `/huyang2/double-loop/data`; server-side Hugging Face fallback is
forbidden. Formal launch is blocked until the placeholder checkpoint hash in
`configs/retrieval/pretrained_bert_carrier.env` is replaced and committed.

## Prediction And Kill Criteria

Carrier opening requires:

- native bidirectional masked accuracy at least `0.10`;
- bidirectional minus strict-causal accuracy at least `+0.03`, or strict-causal
  minus bidirectional CE at least `+0.20`;
- exact shared pretrained tensors and parameter count;
- zero strict-causal future dependency and nonzero bidirectional dependency;
- finite CUDA backward on the exact single GPU1.

Any hash, source, GPU, state, masking, dependency, finite-value, or offline
gate failure stops before interpretation. A quality miss discards this carrier
without changing checkpoint, corpus, mask rate, sequence length, sample count,
or seed.

## Claim Boundary And Next Decision

A pass proves only that the real-text carrier rewards right context. It then
authorizes a separate preregistered common-lexical-initialization experiment in
which matched official-FLA GDN2 scale 0/1 replaces the sequence mixer. It does
not itself support a FutureSeed quality or efficiency claim.

A miss ends this BERT-Tiny carrier. It does not justify another BERT size,
checkpoint, tokenizer, corpus, mask, or budget table.
