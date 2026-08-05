# P-CAUSAL-024: ModernBERT L1024 Carrier Audit

## 1. Metainfo

- Plan: `P-CAUSAL-024`
- Status: discarded by scope correction; read-only audit only, no assets or GPU
- Audit time: 2026-08-05 10:47 CST
- Resource for the conditional run: AIStation task-mode GPU1 only
- Audit source: `f255601753a8660a71b9079262b72c29329ee22d`
- Experiment branch: `codex/p-causal-024-long-context-carrier`

No model ran during this audit. GPU1 was reachable and idle at zero allocated
MiB; the container exposed exactly CUDA index 0, UUID
`GPU-53e9f3b4-2966-65d3-6614-09c540921519`.

## 2. Scientific Question

P023 showed that the current FutureSeed implementation has useful masked-text
quality but is not practically cheap at length 128. Its checkpoint cannot be
extended honestly: it learned absolute positions only through index 127.

The next question is therefore narrower than a model comparison:

> Can an established bidirectional masked-language model validate one native
> L1024 text task with measurable right-context utility and no position-table
> extension?

This carrier must open before another GDN2/FutureSeed model is trained. A failed
carrier would make any recurrent comparison uninterpretable.

## 3. Candidate Decision

Select official
[`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base)
at revision `9e1cd05c51ed0a9a9a94ac6e51ccb76ed6d6f3ae`.

Reasons:

- it is an established bidirectional masked-language model, not a custom
  attention ceiling;
- its native positional mechanism is RoPE and its configured maximum length is
  8192, so L1024 needs no interpolation or copied position table;
- it alternates 128-token local attention with global attention every three
  layers, giving a published long-context implementation rather than a local
  approximation;
- the official checkpoint is Apache-2.0, 149M parameters, 22 layers, hidden
  size 768 and 12 heads;
- the official model card exposes it directly through
  `AutoModelForMaskedLM` and states that Transformers 4.48.0 or newer is
  required.

Rejected alternatives:

- Longformer supports long input, but it retains a learned absolute position
  table and its official research stack uses historical TVM/custom-CUDA paths.
  That would mix position and reproduction questions into this carrier gate.
- BigBird's official repository is archived and centered on an old
  TensorFlow/TPU/static-shape stack, making it a poor single-A100 reproducible
  baseline.

ModernBERT is only the carrier ceiling. It was pretrained on 2T tokens and has
149M parameters, while P022 GDN2 has about 5M parameters and saw 81.92M input
tokens. A direct quality or speed win between them would not be a fair method
claim.

## 4. Fixed Asset and Environment Contract

Pinned assets before CUDA execution:

- ModernBERT model revision:
  `9e1cd05c51ed0a9a9a94ac6e51ccb76ed6d6f3ae`;
- `model.safetensors` SHA256:
  `340ac08b74eef0d7bdec2d7981a6a3d4249bf0e6aab60634b72ad02c2b8023a9`;
- Transformers 4.48.3 wheel SHA256:
  `78697f990f5ef350c23b46bf86d5081ce96b49479ab180b2de7687267de8fd36`;
- tokenizers 0.21.1 Linux x86-64 wheel SHA256:
  `2dd9a0061e403546f7377df940e866c3e678d7d4e9643d0461ea442b4f89e61a`;
- WikiText-103 raw validation revision:
  `f776294184f13b8ff2337b3841cf9269a6216d1e`;
- raw validation parquet SHA256:
  `4bf7077b7e8b0bcf98ba9ad4058019e5dda719f8932ac7f323865bd9a52389db`.

The dataset was already downloaded locally and uploaded under
`/huyang2/double-loop/data/source/wikitext-103-raw-v1`; its manifest records
CC-BY-SA-3.0 and GFDL. New model/tokenizer/package assets must also be downloaded
locally, hashed, then uploaded under `/huyang2/double-loop/data` and
`/huyang2/double-loop/wheelhouse`. Server-side Hugging Face or package download
is forbidden.

Remote audit found Python 3.10.11, Torch 2.7.0, Transformers 4.46.3,
tokenizers 0.20.3, safetensors 0.5.3 and huggingface-hub 0.30.2. The installed
Transformers lacks `ModernBertForMaskedLM`, and Transformers 4.48.3 requires
tokenizers at least 0.21. The conditional run must therefore install only the
pinned local wheels into a project-local dependency path and run offline while
reusing the existing CUDA Torch. It must assert the native ModernBERT class and
the SDPA implementation; an eager/custom fallback is a hard stop.

## 5. Evaluation Contract

Build one deterministic evaluation artifact with the native ModernBERT
tokenizer from the full hash-pinned WikiText validation parquet:

1. Concatenate nonempty source rows without changing text.
2. Tokenize once, then form 256 nonoverlapping length-1024 windows.
3. Apply one fixed standard 15% BERT corruption using seed 123.
4. Compute cross-entropy and top-1 accuracy only at selected targets.

A separate generic directionality bank isolates right-context utility without
changing the model:

1. Select two deterministic target positions from the interior first half of
   every window, for 512 total cases.
2. Mask only that target in both paired inputs.
3. In the full arm, keep all other tokens and attention positions visible.
4. In the suffix-removed arm, set the attention mask to zero strictly after the
   target while leaving its position and entire prefix byte-identical.

This intervention does not encode a language rule or select favorable cases.
It asks whether the same target distribution changes when only later context
is removed.

## 6. Hypothesis, Prediction, and Claim

Hypothesis: a valid native-L1024 bidirectional MLM carrier will retain useful
masked-token quality and will measurably depend on the visible suffix for early
targets.

Registered carrier gate:

- full-context masked accuracy at least `0.20`;
- full-context masked CE at most `5.0`;
- suffix removal worsens paired accuracy by at least `0.03` or paired CE by at
  least `0.20`;
- full-versus-removed target-logit RMS difference is finite and nonzero.

A pass supports only this claim: the fixed L1024 task is a valid established
bidirectional carrier with right-context signal. It authorizes a new,
separately preregistered experiment comparing strict-causal official-FLA GDN2
scale 0 and native FutureSeed scale 1 on the same tokenizer/data semantics.

It does not support parameter efficiency, training efficiency, quality parity,
or practical cheapness.

## 7. Budget and Kill Criteria

Budget after assets are present: one minimal CUDA preflight and one carrier
evaluation process, at most 15 GPU minutes. No training and no repeated seed.

Kill before interpretation on any of the following:

- visible GPU count is not one or UUID differs from the registered GPU1;
- model, tokenizer, wheel, data or generated tensor hash differs;
- remote network access or cache fallback occurs;
- the loaded class is not official `ModernBertForMaskedLM` from pinned
  Transformers 4.48.3;
- attention does not remain native SDPA, or an eager/custom fallback appears;
- CUDA forward/backward is nonfinite;
- native configuration is not max length 8192 with RoPE/local-global attention;
- either quality threshold misses or suffix removal has less than the
  registered right-context effect.

No rescue by model revision, backend, mask rate, target placement, context
length, tokenizer, seed, width, depth, fine-tuning or threshold change.

## 8. Required Artifacts and Diagnostics

The conditional run must archive:

- `config.json`, asset manifest and generated tensor manifest;
- full command, log, GPU snapshot, source snapshot and exact package sources;
- masked accuracy/CE and target counts;
- paired suffix-removal accuracy/CE deltas and target-logit RMS dependency;
- independently warmed throughput, peak allocation and timing CV as diagnostics
  only, not a cost claim;
- same-window HTML showing target, full-context top predictions and
  suffix-removed top predictions.

## 9. Current Decision

Discarded at 2026-08-05 11:12 CST before any asset download, model load, CUDA
preflight, or GPU run. The audit itself was reproducible, but it moved the active
program away from the user-locked objective: use hard Sudoku scaling to jointly
iterate native FutureSeed and the next GDN generation. It therefore contributes
no positive or negative model evidence. Preserve this report as a scope-control
lesson; do not resume it unless the user explicitly changes the research goal.
