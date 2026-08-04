# GDN2 FutureSeed L1024 State-Capacity Gate

## 1. Metainfo

- Plan: `P-CAUSAL-013`
- Status: discarded after fixed endpoint; raw width/state scaling failed
- Date: 2026-08-04 CST
- Machine: AIStation task-mode GPU1 only
- Branch: `codex/p-causal-013-gdn2-capacity1024`
- Source SHA: `d0a2cf69fa30105205e4d8487e622015953d450f`
- Run: `zoology-gdn2-state-capacity1024-20260804T105038Z-d0a2cf6`
- Exact launcher PID/PGID: `53624/53624`
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
D256/L2/H8/D32, with `8 * 32 * 32 = 8,192` values per layer. Depth,
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
- Candidate model: strict official-FLA GDN2 D256/L2/H8/D32, expand-v1,
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

The first preflight-only launch at source `87827f0a`, run
`zoology-gdn2-state-capacity1024-20260804T102407Z-87827f0`, stopped before
training. Official FLA `chunk_gdn2` hit an illegal memory access while
autotuning the `head_dim=64` backward kernel. `abort.json` records this as
`scientific_failure=false`, and GPU memory returned to zero. No fallback was
used. Before observing any model-quality result, the candidate was revised to
the official already-validated `head_dim=32` geometry: D256/H8/D32. It still
scales total recurrent state 2x and model width 2x while preserving the
scientific question and all quality gates.

The second preflight-only launch at source `9bdae8f4`, run
`zoology-gdn2-state-capacity1024-20260804T103146Z-9bdae8f`, also stopped
before training and is likewise infrastructure-only (`scientific_failure=false`).
Crucially, the supported D256/H8/D32 official-FLA CUDA forward and backward had
already completed. The stop came from our final audit reading the nonexistent
generic attribute `layer.conv1d` instead of official GDN2's
`layer.q_conv1d`. The audit now uses the same explicit `q_conv1d.backend`
check as the previously validated P-CAUSAL-012 preflight. This changes no
model, optimizer, data, budget, or registered quality gate.

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

## 8. Result

The final detached-SHA launch passed every integrity gate before training:

- one visible A100 GPU, expected UUID, official FLA SHA/source and Triton
  convolution;
- exact frozen train/test hashes;
- `2,240,608` parameters in each D256 arm, identical initialization and
  parameter hashes;
- scale-0 output difference exactly zero, causal future dependency exactly
  zero, FutureSeed dependency `0.007921`, nonzero FutureSeed gate gradient and
  finite CUDA backward;
- `8,192` recurrent-state values per layer, exactly 2x the D128 reference.

The registered quality gate then failed decisively:

| model | past accuracy | future accuracy | past/future CE | joint exact |
|---|---:|---:|---:|---:|
| frozen D128 FutureSeed | 0.7535 | 0.7415 | 0.6143 / 0.6167 | 0.339 |
| D256 causal | 0.0145 | 0.0120 | 5.1385 / 5.1279 | 0 |
| D256 FutureSeed | 0.0090 | 0.0115 | 5.4761 / 5.4032 | 0 |

D256 FutureSeed balanced accuracy is `0.01025`, a `-0.73725` change from the
frozen D128 FutureSeed endpoint. Its learned seed gate remained active at
`0.5106`; this is not a dead or bypassed FutureSeed path.

The learning curves distinguish failure to generalize from a merely delayed
opening. D128 FutureSeed starts opening at epoch3 (`0.0588` validation
accuracy), reaches `0.7158` by epoch6 and ends at `0.7475` with CE `0.6179`.
Both D256 arms remain near 1% accuracy for all ten epochs, while validation CE
turns upward after epoch5 and finishes at `5.1341/5.4374`. Training loss falls
below validation loss, so the larger model is fitting the finite train set
without learning the transferable key-value rule.

The reported raw same-case swap reductions (`0.182` future, `0.165` past) are
not evidence of improved binding. A random prediction rarely equals any of the
four values in its sequence, so the swap fraction necessarily falls when total
accuracy collapses. The hardest-case view confirms this: D128 often solves two
of four queries, while both D256 arms usually miss all four with unrelated
values.

Warmed diagnostic throughput is about `1.05M` tokens/s for both D256 arms and
peak allocated memory is about `2.09GB`. These numbers show matched systems
cost between scale0/1, but are not used as a paper speed claim.

## 9. Decision And Lesson

Reject whole-model D128-to-D256 width scaling under this fixed recipe. Doubling
state values by tripling total parameters changed the optimization regime and
destroyed the sharp FutureSeed opening; it did not test state capacity in
isolation. Do not extend epochs, tune LR, change seed or sweep width/head
geometry to rescue this endpoint.

The next single decision-changing probe should preserve the D128/L2/H4/D32
geometry that demonstrably opens, and increase only generic recurrent value
state through official GDN2 `expand_v=2`. That separates address/state
capacity from whole-network width. A pass would support state interference as
the L1024 limit; a miss would redirect work to state compression/update rather
than more raw capacity.

Artifacts include `score.json`, full logs, preflight/source hashes, output
cases, an HTML report with validation curves and twelve hardest same-sequence
cases, plus `overview.png` and `overview-full.png` screenshots.
