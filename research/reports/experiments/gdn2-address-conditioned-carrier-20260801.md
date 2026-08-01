# P-CARRIER-001: Address-conditioned write carrier

## 1. Metainfo

- Status: discarded after the single preregistered matched probe
- Date: 2026-08-01 CST
- Machine: AIStation GPU1 only, NVIDIA A800 80GB
- Branch: `codex/gdn2-address-carrier-20260801`
- Parent: `c191bdbbfa0486a6d4f7e23c3f3593d56533097a`
- Source implementation SHA: `e547694b736858bbf69830bcc8eea89ce6c1726f`
- Source checkpoint: shared-address GDN2 step9100, SHA256
  `503debf94a40294e151f72335157fafc58d9a1d88004af3d07bd9030653315f2`

## 2. Hypothesis

Shared stable Q/K addresses improve hard Sudoku, but every activated key row
still receives the same delta-rule correction implied by the key. Treating the
GDN2 state as an online `Linear(K -> V)` suggests a missing independent
decision: which K rows should accept the current error. A generic carrier
conditioned on the already learned stable address may reduce memory
interference without changing erase semantics, adding state, or encoding any
Sudoku rule.

Prediction: relative to an exact shared-address continuation from the same
step9100 checkpoint, the carrier arm should lower train CE and improve mean
official 51-64 blank accuracy by at least `+0.03`, while preserving or
increasing loop1-to-loop5 gain. The carrier must vary across addresses rather
than acting as one global write shrinkage.

## 3. Configuration

- Backbone: official FLA GDN2, D192/L10/H6/K32/V32.
- FutureSeed: native layer-scope state, scale 1, five loops.
- Supervision: CE at every loop; blank weight 8.
- Data: official Sudoku-Extreme train/test, random traversal, 51-64 hard stage.
- Control: `GDN2_ADDRESS_MODE=anchor_residual`.
- Candidate: `GDN2_ADDRESS_MODE=anchor_carrier`.
- Both arms resume the same exact shared-address step9100 checkpoint and run
  exactly 100 matched optimizer steps to step9200.
- The completed historical `51-64:1100` stage is unchanged; continuation is
  a new same-distribution `51-64:100` stage so the strict resume contract holds.
- One seed only. No carrier bias, scale, rank, LR, loss, or duration sweep.

The candidate uses the shared stable address residual for Q/K, then computes a
per-token/head/K-channel carrier initialized near one. Because the official
kernel internally L2-normalizes K, the carrier is folded into `k/b/w` as:

```text
r = ||carrier * k|| / ||k||
k_official = carrier * k
b_official = r^2 * b / carrier
w_official = r * w
```

This leaves the official FLA chunk/Triton recurrence unchanged while making
its update equal to `outer(carrier * k_unit, delta)` with the original
`(b * k_unit)` erase address.

## 4. Environment

- CUDA only; CPU model smoke forbidden.
- `CUDA_VISIBLE_DEVICES=0`; current GPU1 UUID must match the launcher input.
- Pinned FLA source: `9c8e42e762fce087c27b673af4922795d9edb85e`.
- `FLA_DISABLE_BACKEND_DISPATCH=1`, `FLA_CONV_BACKEND=triton`.
- All cache, environment, data, model, run, and artifact paths stay below
  `/huyang2/double-loop`.

## 5. Commands

```bash
CUDA_VISIBLE_DEVICES=0 GPU1_UUID=<current-gpu1-uuid> \
  python experiments/rwkv_fs_sudoku/check_gdn2_address_carrier_cuda.py

CUDA_VISIBLE_DEVICES=0 GPU1_UUID=<current-gpu1-uuid> \
  scripts/run_gdn2_address_carrier_probe.sh carrier smoke

CUDA_VISIBLE_DEVICES=0 GPU1_UUID=<current-gpu1-uuid> \
  scripts/run_gdn2_address_carrier_probe.sh control formal

CUDA_VISIBLE_DEVICES=0 GPU1_UUID=<current-gpu1-uuid> \
  scripts/run_gdn2_address_carrier_probe.sh carrier formal
```

## 6. Artifacts

- Control run:
  `/huyang2/double-loop/runs/gdn2-address-carrier-control-s9200-20260801T045008Z-e547694`
- Carrier run:
  `/huyang2/double-loop/runs/gdn2-address-carrier-candidate-s9200-20260801T050525Z-e547694`
- Comparison and visualizations:
  `/huyang2/double-loop/runs/gdn2-address-carrier-comparison-20260801-e547694`
- Launch and CUDA-contract logs:
  `/huyang2/double-loop/artifacts/launch/gdn2-address-carrier-e547694`
- Both formal source snapshots have SHA256
  `6e542fabf9506459607546124c0049054bf06b51e0d15e0dcb26602a37dc6cc0`.
- Local mirror:
  `.codex-transfer/localrepo/runs/gdn2-address-carrier-{control,candidate,comparison}*`

## 7. Results

### CUDA and kernel contract

- Physical device: `NVIDIA A800-SXM4-80GB`, UUID
  `GPU-7db97dd1-77e8-df7f-570e-09e7059ba5a3`; GPU1 only.
- Pinned FLA source SHA matched exactly.
- Direct Torch recurrence versus the algebraically folded official chunk kernel:
  output max-abs `0.004539`, terminal-state max-abs `0.012373`, both below
  the BF16 gate `0.025`.
- At `carrier=1`, output and terminal-state max-abs errors are exactly `0`.
- The autograd graph contains `ChunkGDN2FunctionBackward`; input, initial
  state, address projection, carrier scale, and carrier bias all have finite,
  nonzero CUDA gradients. No fallback path was used.

### Matched quality

| Official range | Control loop1 | Control loop5 | Carrier loop1 | Carrier loop5 | Carrier loop5 delta |
|---|---:|---:|---:|---:|---:|
| 51-55 blanks | 0.4803 | 0.5286 | 0.4794 | 0.5284 | -0.0002 |
| 56-60 blanks | 0.4373 | 0.4726 | 0.4394 | 0.4723 | -0.0002 |
| 61-64 blanks | 0.4065 | 0.5043 | 0.4062 | 0.5043 | +0.0000 |
| Mean | 0.4414 | 0.5018 | 0.4417 | 0.5017 | -0.0001 |

- Mean loop1-to-loop5 blank gain is `+0.06044` for control and `+0.06001`
  for carrier, a carrier delta of `-0.00044`.
- Hard-range full-board exact is `0` for every arm and range. On the separate
  mixed evaluation, both arms have exact `0.02344`; blank accuracy changes
  `0.51047 -> 0.50963`.
- Train CE changes `1.055997 -> 1.056259` (`+0.000262`, worse).

### Capacity, cost, and learned behavior

| Diagnostic | Control | Carrier | Delta/ratio |
|---|---:|---:|---:|
| Parameters | 5,830,328 | 5,834,168 | +3,840 (+0.066%) |
| Train wall time | 769.61 s | 857.48 s | +11.42% |
| Peak allocated VRAM | 8,390 MB | 11,051 MB | +31.72% |
| Terminal state RMS | 2.9512 | 2.7679 | 0.938x |

- Carrier mean/std/token-std/min are
  `0.997525 / 0.0000591 / 0.0000105 / 0.996791`.
- No carrier value is below `0.95`. Effective write norm is `0.997535x` the
  unmodulated norm. `b` and `w` kernel inputs move only `0.353%` and `0.355%`.
- The carrier therefore learned an almost uniform `0.25%` write shrink, not
  an address-specific edit policy. Its token standard deviation misses the
  preregistered `0.002` specialization gate by about `191x`.

### Matched cases

All case comparisons use the same 256 boards per range, with mechanically
selected visualization indices `250`, `80`, and `28`.

| Range | Better / equal / worse final boards | Mean carrier wrong-cell reduction | Mean carrier loop-correction advantage |
|---|---:|---:|---:|
| 51-55 | 84 / 79 / 93 | -0.0117 | -0.3516 |
| 56-60 | 92 / 71 / 93 | +0.0195 | -0.0742 |
| 61-64 | 81 / 75 / 100 | -0.0938 | 0.0000 |

The visualized boards show local swaps in which arm is better, but no
systematic reduction of late-loop errors and no hard-range closure.

## 8. Conclusions And Kill Criteria

- **Decision: discard this carrier parameterization.** It fails both quality
  gates and the address-specialization gate. State is stable, so this is not a
  numerical failure.
- The algebra is valid and the official CUDA training path is real. The failed
  assumption is optimization: under task loss, the near-one sigmoid takes the
  easier global-shrink solution instead of learning token-specific routing.
- This closes `sigmoid(6 + bias + scale * address)` carrier gating. Do not run
  a second seed or sweep bias, scale, LR, loss, rank, width, or duration.
- This does not rule out address-specific online learning rates in general.
  A future revisit must change the optimization geometry, for example an
  exact-identity but nonsaturated signed/log carrier, and first prove selective
  writes on a generic interference/retrieval task rather than another Sudoku
  score probe.
- The retained strongest modification remains the shared Euclidean stable
  address residual. The new carrier is not part of the baseline.

## 9. Publication/Tag Record

No tag: the preregistered mechanism gate failed. Configs, metrics, logs, source
snapshots, comparison JSON, and visualizations are archived; model,
checkpoint, and dataset files are not committed to Git.
