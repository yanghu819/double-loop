# P-CARRIER-001: Address-conditioned write carrier

## 1. Metainfo

- Status: in progress
- Date: 2026-08-01 CST
- Machine: AIStation GPU1 only, NVIDIA A800 80GB
- Branch: `codex/gdn2-address-carrier-20260801`
- Parent: `c191bdbbfa0486a6d4f7e23c3f3593d56533097a`
- Source checkpoint: shared-address GDN2 step9100

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

Pending GPU1 launch.

## 7. Results

Pending.

Required readouts:

- train CE and wall time;
- official 51-55, 56-60, 61-64 loop1/loop5 blank and exact;
- loop1-to-loop5 gain;
- carrier mean/std/token-std/min and fraction below 0.95;
- effective write-norm ratio and `b/w` kernel-input reparameterization changes;
- terminal state RMS and peak VRAM;
- matched case errors at loops 1/2/3/4/5.

## 8. Conclusions And Kill Criteria

- Continue only if candidate mean hard blank delta is at least `+0.03`, or CE
  is at least `0.05` lower with stronger loop correction, without state
  instability.
- Discard if address-conditioned token std stays below `0.002`, if improvement
  is below the gate, or if runtime overhead exceeds 25% without `+0.05` gain.
- Abort immediately on NaN, non-finite gradients/state, wrong GPU, FLA source
  mismatch, fallback, checkpoint hash mismatch, or state RMS above 2x control.
- No second seed or hyperparameter sweep. A positive result advances to one
  generic retrieval/language test; a negative result closes carrier gating.

## 9. Publication/Tag Record

No tag unless the mechanism passes its gate and the primary score is genuinely
strong. No model, checkpoint, or dataset is committed to Git.
