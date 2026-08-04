# GDN2 FutureSeed L1024 Value-State Isolation

## 1. Metainfo

- Plan: `P-CAUSAL-014`
- Status: in progress after strict detached-SHA GPU1 launch
- Date: 2026-08-04 CST
- Machine: AIStation task-mode GPU1 only
- Branch: `codex/p-causal-014-gdn2-value-state`
- Parent result: `P-CAUSAL-013`
- Source SHA: `b2db9d2e8e29f3df0cd3bd3588312e402c27fad7`
- Run: `zoology-gdn2-value-capacity1024-20260804T112629Z-b2db9d2`
- Started: `2026-08-04T11:26:29Z`
- Remote detached worktree: `/huyang2/double-loop/worktrees/p-causal-014-b2db9d2`
- Launcher PID/PGID: `55495/55495`
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
