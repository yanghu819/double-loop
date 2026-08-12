# P-GDN3-022: Same-Byte Clustered GDN2 State

## 1. Metainfo

- Status: discarded after one complete fixed A100 endpoint
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

Discard. Exact pushed source `18ad10f` completed all ten epochs with status 0.
Balanced/future/past accuracy is `0.27975/0.3000/0.2595`, and joint exact is
`0.007`. The curve opens only at epoch6 and ends
`0.1525 -> 0.24775 -> 0.2670 -> 0.27975`; it is a real improvement over the
same-current-A100 FutureSeed replay (`0.1735`) but remains below P020 Log-SPD
(`0.48225`) and misses every fixed quality threshold.

Among 2,881 wrong queries, 1,081 are another valid value from the same sample,
for a wrong-key swap fraction of `0.37522` among errors. This is below P020's
`0.94157` but does not translate into enough correct bindings. Fit elapsed is
`375.25s` versus control `146.97s` (`2.55x`), so the fixed `1.5x` fit gate also
fails; peak allocation is `1.281GB` (`1.22x`) and the independently warmed step
is `0.46168s`.

Score and checkpoint SHA256 values are
`adbfc19a392db895516c1f7044d0e46a1dc3d37fe9dde5cff54dd59514ee9781`
and `0ee0b3ff7ac213d877c19ebfcb3b7283fd935dbf772871ff8dae8a17b3f44c6b`.
Close isolated H8/K16/V32 splitting without a head/K/V geometry, seed, LR,
loss, width, depth or duration rescue. P023 is the one fixed complementarity
test with P020's bounded metric.
