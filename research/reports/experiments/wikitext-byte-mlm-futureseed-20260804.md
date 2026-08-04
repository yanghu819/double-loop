# WikiText-103 Masked Recovery: Native FutureSeed Gate

- Plan: `P-CAUSAL-008`
- Status: discarded (registered carrier-validity gate failed)
- Date: 2026-08-04 CST
- Machine: AIStation task-mode GPU1 only

## Question

Does native FutureSeed provide useful right-context information on established
language data, rather than only on Sudoku and fixed-layout synthetic retrieval?

## Mechanism Hypothesis

A strictly causal GDN2 prediction at a masked position cannot depend on bytes
to its right. Native FutureSeed passes the previous layer's terminal recurrent
state into the next layer, so a later layer can use a compressed summary of
the full input without a reverse scan. If that summary is useful for language,
it should improve masked-token recovery and close part of the gap to full
bidirectional attention.

The FutureSeed equation is unchanged from `P-CAUSAL-007`:

```text
rms_h = sqrt(mean(S_prev_terminal[h]^2))
S_next_initial[h] = scale * sigmoid(gate_h) * S_prev_terminal[h] / rms_h
```

`scale=0` is exactly the normal causal GDN2 stack. `scale=1` is the native
FutureSeed arm. There is no reverse scan, bidirectional concatenation, oracle,
selector, search, repair, or text-specific rule.

## Fixed Data Protocol

- Established corpus: `WikiText-103-raw-v1`.
- License: CC BY-SA 3.0 and GFDL.
- Local-download provenance: Oxen mirror `Salesforce/wikitext`, commit
  `430edf94f37286d0`, schema hash
  `d930b7eee9d7f11db9ee0fa4ba45117d`.
- The first 20,000 train rows and complete validation parquet were downloaded
  locally before upload; the server performs no HuggingFace/network fallback.
- Train source SHA256:
  `ef7ee3e125df2f9beea8ea566650bbd8082dc7a0baa105adddf789737e6ba1ab`.
- Validation source SHA256:
  `4bf7077b7e8b0bcf98ba9ad4058019e5dda719f8932ac7f323865bd9a52389db`.
- Encoding: UTF-8 bytes, token IDs 0-255, one mask token 256.
- Sequence length 256; deterministic, non-overlapping contiguous windows.
- Exactly 38 of 256 positions are masked in every example (15% rounded),
  always replaced by the mask token; CE and accuracy use masked positions only.
- Train/validation: 20,000/2,000 fixed windows, mask seeds 123/124.
- Prepared NPZ SHA256:
  `3987c057693c67e0b2de5bd249c5ef84391d39bb10c23a563f8b187f5122ffa2`.

Byte tokens avoid importing a pretrained tokenizer or vocabulary as an
uncontrolled source of knowledge. This is a language masked-recovery gate, not
a claim that byte tokenization is the best production language setup.

## One Registered Three-Arm Gate

1. strict causal official-FLA GDN2 with FutureSeed scale 0;
2. exactly matched official-FLA GDN2 with native FutureSeed scale 1;
3. full noncausal SDPA attention at the same D128/L2/H4 width/depth and token
   budget, used as a quality ceiling rather than a parameter-matched RNN arm.

Both GDN2 arms have the same model class, parameters, initialization, data,
seed, optimizer, four epochs, batch 64, and official chunk/Triton kernel. All
arms use the upstream Zoology embedding, MLP, normalization, trainer, masked
cross entropy, AdamW LR `1e-3`, weight decay `0.1`, cosine schedule, seed 123,
20.48M training tokens, and no outer reasoning loop.

## Prediction

- The causal arm should learn byte statistics but remain limited because it
  cannot inspect right context at a mask.
- Bidirectional attention must clearly beat causal GDN2; otherwise this budget
  is not a valid test of future-context value.
- FutureSeed should lower masked validation CE by at least 0.05 or raise masked
  accuracy by at least 0.03, with a positive validation slope, and close a
  measurable portion of the causal-to-bidirectional gap.

## Budget And Kill Criteria

- One strict CUDA preflight and one sequential three-arm run, at most two hours.
- Stop before training on wrong GPU/UUID, multiple visible GPUs, source SHA,
  official class/path, chunk/Triton provenance, dataset hash, scale-0 identity,
  non-finite backward, dead gate gradient, or causal leakage.
- Stop interpretation if bidirectional attention does not clearly outperform
  causal GDN2; the carrier/budget then lacks discrimination.
- Reject this FutureSeed language setting if CE improvement is below 0.05,
  accuracy improvement is below 0.03, and the registered curve has no positive
  slope. Do not rescue with epochs, mask rate, width, depth, LR, loss, seed, or
  tokenizer sweeps.
- One clean scale continuation is allowed only if FutureSeed remains on a clear
  positive validation slope at the endpoint. Its configuration must be
  preregistered before launch.

## Claim Boundary

A pass supports: native cross-layer terminal-state seeding carries useful
future context on established text while retaining linear recurrent state.
It does not by itself establish language-model pretraining quality, asymptotic
scaling, or a speed win over attention. Systems comparisons are accepted only
from independent per-arm warmup benchmarks.

## Required Output

Archive masked validation CE/accuracy/exact, opening epoch, parameter count,
training tokens, warmed throughput, peak memory, FutureSeed gap closure, data
and source hashes, strict preflight, logs, and same-window masked-token
visualizations for all three arms. Update this report only after the fixed gate.

## Result

The fixed three-arm run completed on source SHA
`01b5dc70aef9af3c47f638785df57173e84723d5`. The registered carrier-validity
condition failed: the full noncausal attention arm was substantially worse than
the causal GDN2 arm. Therefore this run cannot measure how much of a valid
causal-to-bidirectional language gap FutureSeed closes.

| arm | masked accuracy | masked CE | exact windows | parameters | warmed tokens/s | warmed peak CUDA memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| causal GDN2 | `0.424737` | `2.018437` | `0/2000` | `563,408` | `981,294` | `553,473,536` B |
| FutureSeed GDN2 | `0.421829` | `2.031740` | `0/2000` | `563,408` | `921,913` | `556,623,872` B |
| bidirectional attention | `0.187855` | `3.167591` | `0/2000` | `461,440` | `2,556,030` | `378,702,336` B |

All arms saw the same 20,000 training windows for four epochs, or 20.48M input
tokens, and the same 2,000 validation windows with 76,000 masked targets. The
causal and FutureSeed arms had identical initialization hashes and parameter
counts. Their validation accuracy curves were:

```text
epoch                 0        1        2        3
causal GDN2       0.3486   0.3890   0.4101   0.4247
FutureSeed GDN2   0.3388   0.3820   0.4055   0.4218
bidirectional     0.1879   0.1879   0.1879   0.1879
```

FutureSeed was slightly worse than the matched causal arm at the registered
endpoint: accuracy delta `-0.002908`, CE delta `+0.013303`. It made 3,999
wrong-to-right masked-token repairs but 4,220 right-to-wrong regressions, for a
net `-221` correct tokens. The difference was not isolated to one sequence
region: early/middle/late accuracy was `0.4119/0.4226/0.4310` for FutureSeed
versus `0.4150/0.4276/0.4316` for causal GDN2.

These values are a boundary for this exact byte-level setup, not a negative
language conclusion. The bidirectional arm achieved only `0.187855` accuracy,
was worse than causal GDN2 on 1,991 of 2,000 windows, and never moved from its
epoch-0 accuracy. Because the supposed upper bound did not learn, the
registered kill rule invalidates the carrier before a FutureSeed claim can be
made. The computed numeric "gap closure" is meaningless when the ceiling is
below the baseline and must not be reported as scientific evidence.

## Integrity Checks

- GPU1 was the only visible device: A100-SXM4-80GB, UUID
  `GPU-53e9f3b4-2966-65d3-6614-09c540921519` (PyTorch reports the UUID without
  the `GPU-` prefix).
- Both recurrent arms used pinned official FLA `GatedDeltaNet2` at commit
  `9c8e42e`, chunk mode, and the Triton short-convolution path. No fallback ran.
- Scale 0 was output-identical to the ordinary causal path and had zero future
  perturbation dependency. FutureSeed dependency was `0.176870`, bidirectional
  attention dependency was `0.048228`, and the FutureSeed gate gradient was
  finite and nonzero (`0.001521`).
- The prepared dataset hash was
  `3987c057693c67e0b2de5bd249c5ef84391d39bb10c23a563f8b187f5122ffa2`.
- The source snapshot archive hash was
  `58b25411eb771f518e934e1f6a387bfb5d6929679b23f9437ae2fc688583c094`.
- Only independently warmed per-step benchmarks are retained as systems
  evidence. Raw sequential arm times include compilation and validation and
  are not compared.

## Visualization And Failure Shape

The archived HTML shows 12 identical validation windows for all three arms,
with each masked byte marked correct or wrong. It confirms that FutureSeed
produces different predictions but has balanced repairs and regressions, while
the attention arm remains near a frequent-byte solution rather than using
right context effectively.

Visualization:
`runs/zoology-wikitext-byte-mlm-20260804T0645Z-01b5dc7/visualizations/index.html`.

## Decision

Stop this exact gate. Do not rescue it by extending epochs or tuning tokenizer,
mask rate, LR, width, depth, loss, or seed. The next high-information step is
to reproduce an established bidirectional masked-language baseline with its
validated model, tokenizer, data processing, and training recipe. Only after
that ceiling clearly beats a strict causal control should GDN2 and native
FutureSeed replace its mixer under matched conditions. This mirrors the
successful P-CAUSAL-005 strategy: validate the carrier first, then test the
mechanism.

Full artifacts remain on AIStation at
`/huyang2/double-loop/runs/zoology-wikitext-byte-mlm-20260804T0645Z-01b5dc7`.
