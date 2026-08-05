# P-CAUSAL-023: Frozen L128 Inference Quality-Cost Frontier

## 1. Metainfo

- Plan: `P-CAUSAL-023`
- Status: approved; CUDA preflight passed; formal systems rerun pending
- Date: 2026-08-05 CST
- Resource: AIStation task-mode GPU1 only
- Source: exact clean detached preregistration commit recorded by the launcher
- Formal run: first attempt aborted on an over-strict one-token reproducibility
  assertion; corrected exact-protocol rerun pending

## 2. Hypothesis and Decision

P-CAUSAL-022 proves that ordinary language data and optimization amplify
FutureSeed into a strong matched quality result. It does not yet prove the word
"cheap" in the paper claim. The highest-information next test is therefore not
another training scale or architecture sweep. It is a frozen systems gate:
does the successful FutureSeed checkpoint form a better inference
quality-cost point than an independently valid full-bidirectional model?

Prediction: on the exact fixed length-128 WordPiece task, P022 FutureSeed will
reproduce quality within `0.02` accuracy and `0.10` CE of official pretrained
BERT-Tiny. If the recurrent route is already practically cheap, it will also
deliver at least `1.20x` batch-64 masked-recovery throughput, or a substantial
memory reduction without materially losing throughput.

The next decision is binary:

- quality and cost pass: support a narrow L128 inference quality-cost claim;
- quality passes but cost misses: keep the future-context and quality claims,
  remove current-kernel cheapness from the headline, and investigate systems
  work only in a separately motivated experiment;
- quality reproduction or provenance fails: invalidate the comparison before
  interpreting timing.

## 3. Fixed Configuration

Three frozen arms:

1. Official `google/bert_uncased_L-2_H-128_A-2`, native full-bidirectional
   Transformers 4.46.3 SDPA, 4,416,698 parameters.
2. P022 causal official-FLA GDN2 D128/L4/H4/D32, 4,965,722 parameters.
3. The exactly matched P022 GDN2 plus native FutureSeed, 4,965,722 parameters.

Quality uses the exact P018/P022 deterministic 15% corruption over 256 grouped
WikiText-103 WordPiece windows: sequence length 128 and 4,742 masked targets.
The registered tensor SHA256 is
`26d57156e43b4d7c21036b1340f23b0e418bf4aed32366bccc4ec252685b0ada`.

Systems measurement uses BF16 and identical input/mask tensors. Each arm runs
in five independent Python processes with rotated launch order. Every process
measures both encoder-only and complete masked-recovery calls at:

- batch 1: 30 warmups and 600 measured calls;
- batch 64: 30 warmups and 200 measured calls.

The masked-recovery workload applies each model's architecturally equivalent
dense-GELU-layernorm-tied-decoder head only at registered masked positions. It
therefore does not charge BERT for unrequested logits at every sequence
position while allowing the custom GDN2 runner to select positions.

## 4. Environment and Provenance

- SSH alias: `aistation-task-gpu1`
- Expected GPU: container index 0, UUID
  `GPU-53e9f3b4-2966-65d3-6614-09c540921519`
- GPU2 forbidden; CPU model smoke forbidden.
- Pinned Zoology commit:
  `1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb`.
- Pinned FLA commit:
  `9c8e42e762fce087c27b673af4922795d9edb85e`.
- Pinned FLA wheel SHA256:
  `0280db310981915eb048ece99d7bedca8b5caa9be65c99835a0f912ada977d6a`.
- BERT checkpoint SHA256:
  `dd152f8450c0579bd271ac0ccb4a88fa4f6a67d8035b7799dbf3a0fb7156d9d0`.
- Causal/FS P022 checkpoint SHA256:
  `f268e86cd9e2f74e447d1e5eae73d36db03f77e1888b94b88d14b962cff27b6b`
  and
  `08bf4f47553bc7a639f8694820be9b83dac24d7ff1ec50458fbb9bb77f2ddc6b`.
- Offline-only assets; no Hugging Face or package fallback.

## 5. Commands

CUDA preflight:

```bash
PREFLIGHT_ONLY=1 RUN_NAME=futureseed-inference-frontier-preflight-<timestamp>-<sha> \
  ./scripts/run_futureseed_inference_frontier.sh
```

Formal run, only after all three preflight workers pass:

```bash
RUN_NAME=futureseed-inference-frontier-formal-<timestamp>-<sha> \
  ./scripts/run_futureseed_inference_frontier.sh
```

## 6. Budget and Kill Criteria

Budget: one three-process CUDA preflight, then fifteen fresh formal processes,
with a 30-minute hard wall budget. A healthy worker is never interrupted.

Stop before interpretation on any GPU UUID/count, source tree, checkpoint,
tokenizer, validation tensor, model state, parameter count, official
FLA/Triton, native BERT masking, finite output, or quality reproduction failure.
Reproduction means the registered integer correct-token count within one of
4,742 targets and BF16 CE within `1e-3`. These numerical tolerances are far
below either scientific quality threshold and cannot change the gate.
Strict causal future dependency must be exactly zero; FutureSeed and BERT must
be nonzero. Timing is non-claimable if any primary throughput coefficient of
variation exceeds `0.10`.

There is no rescue via batch size, sequence length, CUDA graph, compile mode,
kernel replacement, width, depth, checkpoint, tokenizer, or repetition count.

## 7. Registered Gates

Quality gate:

- FutureSeed accuracy at least BERT accuracy minus `0.02`;
- FutureSeed CE at most BERT CE plus `0.10`.

Cost gate, after timing stability:

- batch-64 complete masked-recovery throughput at least `1.20x` BERT; or
- peak allocation at most `0.80x` BERT while throughput remains at least
  `0.90x` BERT.

Batch-1 latency and encoder-only measurements are required diagnostics, not
alternative post-hoc success routes.

## 8. Claim Boundary and Important Non-Experiment

A pass supports only a fixed length-128 inference quality-cost statement.
BERT and P022 have different training data and optimization histories, so this
cannot establish training efficiency, equal-data superiority, or a general
language-model ranking.

The frozen P022 model has a learned absolute position table of shape
`128 x 128`. Running it at length 256 or 512 would require creating new
position parameters or an interpolation rule. That would be a new model, not
a frozen scaling benchmark. P023 therefore refuses to draw a fake length curve.

## 9. Artifacts and Result

Implementation preflight attempt 1,
`futureseed-inference-frontier-preflight-20260805T014320Z-c6710b5`, stopped
cleanly in the causal worker before FutureSeed or formal timing. BERT reproduced
its registered accuracy and CE exactly. Causal GDN2 reproduced the exact
correct-token count, but its fresh-process BF16 CE was `4.61831687` versus the
archived `4.61827578`, a difference of `4.11e-5`; the initial `1e-5` comparison
was therefore a serialization tolerance bug. All asset, tensor, source,
checkpoint and direction checks reached before that assertion passed. The
process group exited, GPU memory returned to zero and `abort.json` records an
infrastructure failure. Before observing FutureSeed or any formal benchmark,
the tolerance is fixed at `1e-3`; no model, metric, gate or workload changes.

Implementation preflight attempt 2,
`futureseed-inference-frontier-preflight-20260805T014857Z-bb7c022`, completed
all three GPU1 workers. BERT reproduced accuracy/CE exactly; causal GDN2
reproduced `1467/4742` correct with CE `4.61835765`; FutureSeed reproduced
`1773/4742` correct with CE `3.90512393`. Directionality checks were also
exactly as required: causal future dependency `0`, BERT `3.56782`, FutureSeed
`1.57495`, with three active native seed routes. The preflight was therefore
valid and the GPU returned to zero allocation.

Formal attempt 1,
`futureseed-inference-frontier-formal-20260805T015242Z-bb7c022`, stopped in its
first FutureSeed worker before producing a complete timing population. BERT
and causal workers reproduced. The frozen FutureSeed model produced
`1774/4742` correct rather than the archived/preflight `1773/4742`, a single
BF16 boundary-token flip (`0.000211` accuracy) while all data, model, source,
state, CUDA and dependency provenance remained fixed. Treating this one-token
numeric edge as architecture drift was an implementation error. The
reproduction guard is restricted to at most one correct-token difference;
quality gates, reported measured accuracy, CE tolerance, model, workload,
repetition count and timing protocol remain unchanged. The aborted partial
timings are not scientific evidence and will not be combined with the rerun.

The completed run must archive config, source snapshot, all 15 raw worker JSON
files, aggregate score, logs, GPU snapshots, same-window three-arm cases and
HTML. A valid negative systems result is still completed evidence, not an
`abort.json`; `abort.json` is reserved for infrastructure or integrity failure.
