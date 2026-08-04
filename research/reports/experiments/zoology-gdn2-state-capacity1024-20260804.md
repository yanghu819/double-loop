# GDN2 FutureSeed L1024 State-Capacity Gate

## 1. Metainfo

- Plan: `P-CAUSAL-013`
- Status: preregistered, not launched
- Date: 2026-08-04 CST
- Machine: AIStation task-mode GPU1 only
- Branch: `codex/p-causal-013-gdn2-capacity1024`
- Parent result: `P-CAUSAL-012`

## 2. Mechanism Hypothesis

At length1024, D128 FutureSeed reaches past/future accuracy `0.7535/0.7415`,
but most remaining errors choose the correct value for a different key in the
same sample. That means the model usually transports the value set; the weak
point is preserving distinct key-value bindings in a fixed-size recurrent
state.

Increase only generic model and recurrent address capacity. The reference is
official-FLA GDN2 D128/L2/H4/D32, whose recurrent state has
`4 * 32 * 32 = 4,096` values per layer. The candidate is
D256/L2/H4/D64, with `4 * 64 * 64 = 16,384` values per layer. Depth,
FutureSeed rule, data, task, optimizer, epochs and seed stay fixed. This is a
plain scaling intervention, not a task-specific mechanism.

## 3. Prediction

- D256 causal GDN2 should recover past retrieval to at least `0.80` while
  future retrieval remains at or below `0.10`.
- D256 FutureSeed should reach at least `0.90` on both directions and at least
  `0.70` joint exact.
- FutureSeed balanced accuracy should improve by at least `+0.10` over the
  frozen D128 value `0.7475`.
- Wrong predictions that select another key's value should fall by at least
  `0.08` of all future queries and `0.07` of all past queries.

If these predictions hold, raw recurrent address capacity is the dominant
L1024 bottleneck and the larger model becomes the paper scaling candidate. If
they do not, increasing state size is insufficient; the next work must change
generic state compression/update rather than continue width or duration
sweeps.

## 4. Fixed Configuration

- Data: exact frozen directional MQAR L1024 hashes from P-CAUSAL-012, vocab256,
  four unique associations, 10,000 train and 1,000 validation examples.
- Candidate model: strict official-FLA GDN2 D256/L2/H4/D64, expand-v1,
  short-conv4, chunk recurrence and Triton convolution.
- Training: batch32, 10 epochs, AdamW LR `1e-3`, weight decay `0.1`, cosine,
  seed123, upstream Zoology positions/MLP/residual/norm and query-only CE.
- Arms: matched no-FutureSeed scale0 and native FutureSeed scale1.
- The frozen D128 result is copied, not retrained.
- No attention, reverse scan, outer loop, selector, search, repair, task rule,
  second seed, LR/loss/epoch/head/width sweep or fallback.

## 5. Budget And Kill Criteria

- One strict full-batch D256 CUDA preflight plus two ten-epoch arms.
- Maximum wall time: two hours; expected runtime is minutes on the single A100.
- Stop before training on wrong GPU/SHA/wheel/source/data hash, fallback,
  non-finite backward, scale0 mismatch, causal leakage, dead FutureSeed path,
  unequal matched initialization/parameters, or wrong state size.
- Stop after the fixed endpoint if the strong gate misses. No rescue.
- Low-information stop: if FutureSeed balanced-accuracy gain is below `+0.05`
  or same-case binding-error reduction is below `0.04`, do not scale this
  geometry further.

## 6. Required Readouts

- past/future accuracy, exact and CE; balanced accuracy and joint exact;
- FutureSeed delta over matched causal D256;
- recurrent-state and parameter scaling ratios;
- same-case wrong-value count as a fraction of all queries and of all errors;
- independently recorded training curves, warmed throughput and peak memory;
- same-sequence D128-FS/D256-causal/D256-FS hardest-case visualization.

Raw sequential timing remains diagnostic only. A separate fresh-process,
alternating-order benchmark is required for a paper cost claim.

## 7. Success Claim

A strong pass supports: increasing generic recurrent address/state capacity
reduces long-context binding interference and lets native FutureSeed preserve
both past and future associations at length1024. It authorizes the full
64/128/256/512/1024 curve at the selected scale and a separate cost benchmark.

A miss supports only: FutureSeed still provides a strong route at L1024, but
raw model/state scaling does not remove its binding limit under the fixed
budget. It does not authorize duration or hyperparameter rescue.
