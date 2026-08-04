# Official Pretrained BERT-Tiny Carrier Gate

- Plan: `P-CAUSAL-018`
- Status: done; carrier opened
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
  dependency, finite backward, five warmup plus 50 measured inference steps per
  arm, peak allocated memory, and 12 same-window cases.
- No training, FutureSeed, GDN2, reverse scan, selector, search, repair, or
  task-specific rule is present.

The checkpoint was downloaded on the local Mac through Kimi WebBridge from the
registered Hugging Face revision, then uploaded to `/huyang2/double-loop/data`.
Its size is `17,743,809` bytes and its SHA256 is
`dd152f8450c0579bd271ac0ccb4a88fa4f6a67d8035b7799dbf3a0fb7156d9d0`.
The local and remote hashes match. No server-side Hugging Face fallback was used.

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

## GPU Implementation Preflight

The detached GPU1 worktree at `1b2db2af1dd6293a1bc5c4edceccc60fcdb314f4`
ran the complete evaluation path on 2026-08-05 CST using the already archived
P-CAUSAL-009 scratch checkpoint and 16 windows. This is an implementation smoke,
not P-CAUSAL-018 evidence, because it is not the registered official pretrained
checkpoint.

- Artifact: `/huyang2/double-loop/artifacts/p018-implementation-smoke-20260804T173405Z-1b2db2a`
- Exact shared tensor difference: `0.0`.
- Strict-causal future dependency: `0.0`.
- Bidirectional future dependency: `0.0285289`.
- Finite CUDA forward/backward, metric serialization, case JSON, and HTML all
  completed on the single registered GPU1.
- Five-warmup/50-step throughput was `795,299` input tokens/s bidirectional and
  `780,691` input tokens/s causal; each arm peaked at `161,513,472` allocated
  bytes. These measurements validate stable instrumentation only.
- Scratch-checkpoint accuracy was `0.0559` for both masks, so the scientific
  quality gate correctly returned exit status 2. It is deliberately excluded
  from the paper result and does not alter the formal preregistration.

## Formal Result

Run `pretrained-bert-carrier-20260804T2129Z-4bfd24c` evaluated the registered
official checkpoint on detached source `4bfd24cffdf3c4539c54fb76455e053f248d050f`.
All asset, source, GPU, tensor, dependency, and finite-backward gates passed.

| metric | native bidirectional | strict causal | bidirectional gain |
|---|---:|---:|---:|
| masked accuracy | 0.355546 | 0.215732 | +0.139814 |
| masked CE | 4.033269 | 5.453952 | 1.420684 lower |
| future dependency | 3.567824 | 0.000000 | nonzero vs exact zero |
| input tokens/s | 2,462,778 | 2,519,247 | diagnostic only |
| peak allocated bytes | 353,021,952 | 353,021,952 | 0 |

The evaluation contains 4,742 fixed masked targets. Shared checkpoint tensors
have maximum difference `0.0` and identical state hashes. Bidirectional accuracy
exceeds the registered `0.10` floor; both the `+0.03` accuracy and `+0.20` CE
right-context gates pass by wide margins. The carrier is therefore open.

This does not compare a trained causal model with a trained bidirectional model:
it compares attention semantics at evaluation under one bidirectionally
pretrained checkpoint. Its only valid claim is that this fixed real-text task,
tokenizer, corruption, and checkpoint expose a substantial right-context signal.
That result authorizes a separate matched GDN2 scale-0/scale-1 training test.
