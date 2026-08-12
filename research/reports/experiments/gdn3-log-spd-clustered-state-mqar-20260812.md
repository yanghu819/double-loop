# P-GDN3-023: Log-SPD Same-Byte Clustered GDN2

## 1. Metainfo

- Status: in progress on one A100 80GB
- Source: exact pushed SHA `18ad10fee1efa531fb970dff92efc7d985cfe07f`
- Fixed field: directional MQAR L1024, D128/L2, 10 epochs, batch32, seed123
- Geometry: H8/K16/V32, exactly 4,096 recurrent-state values per layer
- Transition: one pinned-official GDN2 chunk scan plus native FutureSeed

## 2. Evidence And Hypothesis

On the current A100 runtime, native FutureSeed reaches balanced `0.1735`.
P020's bounded H4/K32 Log-SPD metric raises this to `0.48225`, but `94.16%`
of its remaining errors are correct values bound to another key. P022's
same-byte H8/K16 banks reach only `0.27975`, but their remaining-error swap
fraction is much lower at `37.52%`.

The falsifiable hypothesis is that these mechanisms address different parts
of binding interference: Log-SPD improves geometry within an address domain,
while independent H8/K16 domains reduce cross-address collisions. If they are
complementary, their fixed combination should outperform P020 materially. If
not, both are alternate views of the same insufficient static address change.

## 3. Mechanism

Use the existing bounded Log-SPD Q/K transform inside the existing official
GDN2 wrapper, configured as H8/K16 with `expand_v=2` so V remains 32 and total
state remains `8*16*32=4,096` values per layer. D, layers, data, optimizer,
seed, budget, FutureSeed rule and single official scan are unchanged. There is
no new cache, selector, second state, second scan, task rule or custom kernel.

## 4. Registered Decision

The single endpoint passes only if all of the following hold:

- balanced accuracy at least `0.60` and at least `+0.08` over P020 `0.48225`;
- joint exact at least `0.15`;
- past and future accuracy each at least `0.58`;
- state remains exactly 4,096 values per layer and execution uses one official
  chunk scan;
- fit elapsed and peak allocation are each at most `1.5x` the current-A100
  native FutureSeed control.

Any miss closes the combination. There is no metric scale/cap, head count,
K/V shape, epoch, LR, loss, seed, width, depth or duration rescue.

## 5. Artifacts

- Run: `/huyang2/double-loop/runs/p-gdn3-023-log-spd-clustered-h8k16v32-l1024-20260812T035700Z-18ad10f`
- Launch log: `/huyang2/double-loop/artifacts/launch/p-gdn3-023/formal-r3-20260812T035700Z-18ad10f.log`
- Formal PGID: `11725`
- GPU: CUDA index0, A100-SXM4-80GB, UUID
  `GPU-d2877fe4-641c-fe64-2a74-8abca47c292f`

## 6. Decision

Pending the one fixed endpoint.
