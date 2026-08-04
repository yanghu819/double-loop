# GDN2 FutureSeed L1024 Value-State Isolation

## 1. Metainfo

- Plan: `P-CAUSAL-014`
- Status: discarded after fixed endpoint; value-axis expansion failed
- Date: 2026-08-04 CST
- Machine: AIStation task-mode GPU1 only
- Branch: `codex/p-causal-014-gdn2-value-state`
- Parent result: `P-CAUSAL-013`
- Source SHA: `b2db9d2e8e29f3df0cd3bd3588312e402c27fad7`
- Run: `zoology-gdn2-value-capacity1024-20260804T112629Z-b2db9d2`
- Started: `2026-08-04T11:26:29Z`
- Remote detached worktree: `/huyang2/double-loop/worktrees/p-causal-014-b2db9d2`
- Launcher PID/PGID: `55495/55495`
- Formal timeout PID/PGID and Python PID: `57207/57207`, `57209`
- Launcher log: `/huyang2/double-loop/artifacts/launch/zoology-gdn2-value-capacity1024-20260804T112629Z-b2db9d2.launcher.log`

## 2. Mechanism Hypothesis

P-CAUSAL-012 showed a working but capacity-limited D128 FutureSeed model at
length1024: past/future accuracy `0.7535/0.7415`, with most errors selecting the
right value for the wrong key. P-CAUSAL-013 then doubled recurrent-state values
by widening the entire model to D256, but both arms collapsed to chance. That
intervention mixed two variables: recurrent memory grew 2x while total model
parameters grew 3.39x and the sharp opening disappeared.

The high-information question is now narrower:

> Is the L1024 limit caused by insufficient recurrent value-state capacity, or
> does it survive when state grows without changing the proven D128 backbone
> width and key-address geometry?

Use official FLA GDN2's native `expand_v=2`. D128/L2/H4/D32 and key dimension
stay fixed; only value dimension changes from 32 to 64. Recurrent state grows
from `4 * 32 * 32 = 4,096` to `4 * 32 * 64 = 8,192` values per layer. This is
a generic linear-memory scaling axis available to language and retrieval
models, not a task-specific rule.

## 3. Prediction

- The D128 expand-v2 FutureSeed arm should preserve the known epoch3 opening,
  unlike the D256 collapse.
- If value-state interference is the dominant limit, past and future accuracy
  should both reach at least `0.85`, joint exact at least `0.60`, and balanced
  accuracy should improve by at least `+0.10` over frozen expand-v1.
- At comparable or better total accuracy, wrong predictions that select another
  key's value should fall by at least `0.08` of future queries and `0.07` of
  past queries.
- The matched expand-v2 causal arm must remain at or below `0.10` future
  accuracy; extra state must not create a future-information leak.

A stronger paper-candidate gate is past/future accuracy at least `0.90` and
joint exact at least `0.70`.

## 4. Fixed Configuration

- Exact frozen P-CAUSAL-012 L1024/K4 train/test hashes, 10,000/1,000 examples.
- Strict official-FLA GDN2 D128/L2/H4/D32, short-conv4, chunk recurrence,
  Triton convolution; only `expand_v: 1 -> 2` changes.
- Two exactly matched candidate arms: native FutureSeed scale0 and scale1.
- Batch32, ten epochs, AdamW LR `1e-3`, weight decay `0.1`, cosine, seed123,
  upstream Zoology positions/MLP/residual/norm and query-only CE.
- Frozen expand-v1 results are copied, not retrained.
- No attention, reverse scan, outer loop, selector, search, repair, task rule,
  second seed, LR/loss/epoch/width/head sweep or fallback.

## 5. Budget And Kill Criteria

- One strict full-L1024 CUDA preflight plus two ten-epoch arms; maximum one hour.
- Stop before training on wrong GPU/SHA/wheel/source/data hash, fallback,
  non-finite backward, scale0 mismatch, causal leakage, dead FutureSeed path,
  unequal matched initialization/parameters, wrong H4/K32/V64 geometry or
  wrong 8,192-value state.
- Run the fixed endpoint because the known D128 opening is discontinuous.
- Stop this axis after the endpoint if balanced gain is below `+0.05`, or if
  quality-comparable binding reduction is below `0.04` in either direction.
- No duration, LR, seed, width, expand-v or loss rescue after observing result.

## 6. Required Readouts

- past/future accuracy, exact and CE; balanced accuracy and joint exact;
- FutureSeed delta over matched expand-v2 causal;
- opening epoch at validation accuracy `0.05` and `0.50`;
- state, parameter and model-width ratios;
- quality-conditioned same-case binding errors;
- validation accuracy/CE curves, warmed diagnostic throughput and peak memory;
- same-sequence expand-v1-FS/expand-v2-causal/expand-v2-FS hardest-case HTML
  and PNG visualization.

## 7. Decision

A pass supports a simple scaling claim: increasing generic GDN2 value-state
capacity at fixed backbone width reduces long-context binding interference and
raises FutureSeed quality. It authorizes one complete length curve at the
selected geometry and a separate fresh-process cost benchmark.

A miss means raw state quantity is not enough. The next mechanism work must
improve how FutureSeed compresses or updates associations, not add more width,
epochs or hand-written retrieval logic.

## 8. Result

The detached run passed every integrity gate before training:

- exactly one visible A100 GPU with the registered UUID;
- pinned official Zoology and FLA source/wheel hashes, official chunk GDN2 and
  Triton short convolution;
- exact frozen L1024 train/test hashes;
- H4/K32/V64 geometry and `8,192` recurrent-state values per layer;
- `810,384` parameters in both candidate arms, identical initialization and
  parameter hashes;
- scale-0 output difference exactly zero, causal future dependency exactly
  zero, FutureSeed dependency `0.005461`, nonzero gate gradient `0.001078`,
  and finite CUDA forward/backward.

The quality gate then missed decisively:

| model | past acc / exact / CE | future acc / exact / CE | balanced acc | joint exact |
|---|---:|---:|---:|---:|
| frozen expand-v1 FutureSeed | 0.7535 / 0.590 / 0.6143 | 0.7415 / 0.572 / 0.6167 | 0.7475 | 0.339 |
| expand-v2 causal | 0.0405 / 0 / 3.7517 | 0.0125 / 0.001 / 4.6622 | 0.0265 | 0 |
| expand-v2 FutureSeed | 0.2555 / 0.011 / 2.4199 | 0.2980 / 0.024 / 2.3058 | 0.27675 | 0 |

FutureSeed still beats its exactly matched expand-v2 causal control by
`+0.25025` balanced accuracy, so the directional mechanism remains active.
However, it is `-0.47075` below the frozen expand-v1 FutureSeed endpoint and
does not solve any complete four-query sample. The candidate reaches 5%
validation accuracy at epoch5 instead of epoch3 and never reaches 50%; the
reference reaches 50% at epoch6. This is a slower, much lower opening, not a
hidden late win.

The binding statistics also move in the wrong direction, but they are not used
as an isolated claim because overall quality is not comparable. Wrong values
from another association in the same sample rise from `0.2130` to `0.3185` of
future queries and from `0.1945` to `0.2640` of past queries. Hardest-case
tables show expand-v1 solving two or three of four queries while both expand-v2
models frequently miss all four.

The intervention doubled state values but increased parameters by `22.49%`.
Warmed diagnostic FutureSeed throughput fell from `1.422M` to `1.202M`
tokens/s (`-15.48%`), and peak allocated training memory rose from `0.98` to
`1.08` GiB (`+10.72%`). These fixed-order figures are diagnostics, not an
independent paper speed benchmark.

## 9. Decision And Lesson

Reject official GDN2 `expand_v=2` as the answer to the observed long-context
binding limit. Do not extend epochs, tune LR, run another seed, sweep
`expand_v`, or fill middle lengths as a rescue for this model.

The sharper lesson is about which state dimension was scaled. GDN2 stores a
`K x V` recurrent matrix. This experiment kept the key/address dimension at
`K=32` and doubled only the value/payload dimension from `V=32` to `V=64`.
That doubles the number of state scalars but leaves the maximum key-address
rank unchanged. P-CAUSAL-012's dominant failure is selecting the correct value
for the wrong key, so value width was not the dimension most directly tied to
the observed error. The negative result rejects payload-axis expansion, not
all state scaling.

Any next capacity mechanism must increase or preserve generic address
separability at fixed residual width, or improve FutureSeed's generic state
compression/update. It must not encode MQAR keys, reverse the scan, or add a
retrieval rule. Before another training run, audit whether official FLA exposes
such a K-axis independently; if not, a new kernel/adapter must first prove
exact scale-0 equivalence and CUDA backward.

Archived artifacts include config, score, preflight, full logs, source SHA and
remote source-snapshot hash, all per-arm outputs/cases, twelve hardest same-
sequence cases, HTML, and Playwright-verified overview screenshots. The remote
run retains the complete source snapshot.
