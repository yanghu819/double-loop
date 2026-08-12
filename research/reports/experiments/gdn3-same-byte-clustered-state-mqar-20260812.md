# P-GDN3-022: Same-Byte Clustered GDN2 State

## 1. Metainfo

- Status: in progress on one A100
- Source: exact pushed SHA `18ad10fee1efa531fb970dff92efc7d985cfe07f`
- Decision field: directional MQAR L1024 from scratch
- Fixed setting: D128/L2, native FutureSeed, 10 epochs, batch32, seed123
- Parent state: H4/K32/V32 = 4,096 values per layer
- Candidate state: H8/K16/V32 = 4,096 values per layer

## 2. Evidence And Hypothesis

P020 proves learned address geometry can raise same-runtime balanced accuracy
from `0.1735` to `0.48225`, but wrong predictions are still overwhelmingly
valid values bound to another key. P021 proves two independent writes into one
state are destructive: active auxiliary payload reaches up to `3.62x` parent
RMS and balanced accuracy collapses to `0.00975`.

The falsifiable hypothesis is that address interference requires separate
bounded domains, not another write into the same rows. Splitting the same
state budget into eight K16 banks doubles the independently normalized address
domains while preserving total recurrent-state values and a single official
scan.

## 3. Mechanism

Use the existing pinned-official GDN2 with H8/K16 and V32. D remains 128, so
Q/K projection width is unchanged. `expand_v=2` keeps V32 and total state
exactly 8x16x32=4,096 values. Native FutureSeed passes the full H8/K16/V32
terminal state to the receiving layer. There is no router, selector, cache,
second scan, custom kernel, task rule, or additional state byte.

## 4. Prediction And Gate

Pass requires balanced accuracy at least `0.70`, joint exact at least `0.25`,
both directions at least `0.68`, gain over P020 at least `0.10`, and wrong-key
swap fraction among errors down at least `0.10`. Fit elapsed must remain at
most `1.5x` and peak allocation at most `1.5x` the same-runtime control. Any
miss closes this geometry without head count, K/V geometry, seed, LR, loss,
width, depth or duration rescue.

## 5. Decision

Pending the single registered endpoint.
