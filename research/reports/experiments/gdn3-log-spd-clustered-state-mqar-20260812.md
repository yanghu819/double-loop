# P-GDN3-023: Log-SPD Same-Byte Clustered GDN2

## 1. Metainfo

- Status: discarded after the single registered endpoint
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

The run completed normally with status `0` and preserved the registered one
official scan and 4,096 recurrent-state values per layer. Its endpoint is:

| Metric | Result | Gate |
|---|---:|---:|
| balanced accuracy | `0.3090` | `>=0.60` and `>=0.56225` |
| future accuracy | `0.3225` | `>=0.58` |
| past accuracy | `0.2955` | `>=0.58` |
| joint exact | `0.004` | `>=0.15` |
| fit elapsed | `245.390s` (`1.6697x` control) | `<=1.5x` |
| peak allocation | `1,314,409,984` bytes (`1.2500x`) | `<=1.5x` |

The training curve ends at balanced `0.3090` after
`0.00975/0.00975/0.00975/0.0130/0.0200/0.03575/0.11775/0.2315/0.2910/0.3090`.
The bounded metric remains active: layer condition numbers are
`1.46498/1.30475`, with eigenvalue ranges `0.73690..1.29130` and
`0.84858..1.22366`. Thus the negative result is not a dead metric or an
unbounded transform.

The combination improves only `+0.02925` over P022 and falls `-0.17325` below
P020. It leaves `2,764/4,000` records wrong; `1,332` are wrong-key
valid-value swaps (`0.48191` of errors). The changed swap fraction does not
represent closure because total errors increase substantially relative to
P020. Static within-bank geometry and same-byte bank factorization are not
complementary enough under the fixed budget.

Two earlier launch attempts failed before model execution: one lacked the
Zoology import path and one lacked `FLA_EXPECTED_SOURCE_SHA`. Both are retained
as orchestration evidence and did not consume a science arm. The `formal-r3`
run above is the sole completed endpoint.

## 7. Systems

- Parameters: `771,536`, including `2,160` Log-SPD parameters for H8/K16.
- Post-warm wall time: `247.203s`; cold wall time: `665.869s`.
- Independent warm benchmark: `0.622636s`, `1,027.888` examples/s and
  `1,052,557` tokens/s.
- Training peak allocation: `1,314,409,984` bytes; benchmark peak allocation:
  `1,347,702,272` bytes.

## 8. Integrity

- Score and candidate-summary SHA256:
  `20483f5ddb9ca895c279274f6b789a7cfd17fa227726c92d46c9c4428310a006`
- Cases SHA256:
  `226c7ea8371015280b380ae7425a9aaebc24617644358a2676c5e75cf1252faf`
- Checkpoint SHA256:
  `8c80f92b0c27e0889edeeddab8ef0ff4fd708b86e3acf5630f573f078f8e37af`
- Source remained the exact pushed SHA above. No NaN, OOM, fallback, second
  model process or GPU identity drift occurred.

## 9. Next Decision

Discard P023 and close the fixed static-address combination line. Do not tune
metric strength, head count, K/V geometry, epochs, LR, loss or seed. P020 is
evidence that learned address geometry can matter; P021-P023 show that an
extra write, a same-byte bank split, and their static combination do not solve
binding. A future GDN3 proposal must introduce a stable, scalable
address-binding state organization and first establish a reproducible carrier.
