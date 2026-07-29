# Gain-Budgeted GDN2

## 1. Metainfo

- Plan: `P-GAIN-001`
- Status: scalar-certificate systems retry pending
- Date: `2026-07-29`
- Machine: AIStation `GPU1` only
- Branch: `codex/gain-budget-gdn2-20260729`
- FLA source: `9c8e42e762fce087c27b673af4922795d9edb85e`
- Parent checkpoint:
  `/huyang2/double-loop/models/gdn2-futureseed-d192l10-s12000-20260726T131301Z-42102bd/checkpoints/train_state_step009000.pt`
- Parent checkpoint SHA256:
  `606caf5229590f157d7a0f952423c719f4c17688003309a7bd0664e579588dd7`
- Parent source SHA:
  `42102bd65a28d60bde6b09ef93343692740582a1`

## 2. Mechanism Hypothesis

GDN2's channel-wise erase gate can preserve useful erase strength while still
creating a large non-normal one-step gain through anisotropy. This can make
memory updates sensitive without adding useful capacity. Projecting only that
anisotropic component under a decay-funded gain budget should improve stable
late-loop correction while preserving the original rank-one recurrence,
parameters, data, loss, and FutureSeed path.

The intervention is generic. It uses only the current normalized key, erase
gate, and actual log decay. It does not inspect Sudoku labels or rules.

## 3. Prediction And Decision

Only one matched comparison is authorized:

- official GDN2 plus FutureSeed through the audited external FP32 path with
  an identity erase projection, `mode=external_identity`;
- the same checkpoint and training contract, `mode=decay_funded`.

Both resume step9000 and stop at step9100. Their projection, normalization,
kernel, dtype, checkpoint, data, and RNG paths are identical except that the
candidate replaces `b` with `b_eff`. Untouched `mode=none` remains a bit-exact
upstream regression gate, not the causal comparison arm.

The parent checkpoint was trained with the older pinned FLA tree `fe8fce9`.
This probe intentionally transitions both arms to official FLA
`9c8e42e`. It is an exact model/optimizer/RNG continuation and a fair
within-probe comparison, but it is not an exact continuation of the parent's
software environment. Historical step9000 metrics are therefore context, not
the matched control score.

Success requires one of:

- mean loop5 exact over official 51-55, 56-60, and 61-64 improves by
  `>= +0.015`, with no individual range regressing by more than `0.01`, and
  mean loop1-to-loop5 exact gain does not regress;
- 56-60 or 61-64 improves by `>= +0.02`, 51-55 regresses by at most `0.01`,
  and mean loop1-to-loop5 exact gain improves over control by `>= +0.005`.

Systems overhead must be `<=20%`. The mechanism must be active:
projection clipping fraction `>=1%` and no infeasible budget. The strict FP32
official fused-recurrent lane must satisfy erase-strength error `<=5e-6` and
step-gain bound `<=1.0001`. The actual official BF16 chunk training lane must
satisfy post-cast erase-strength error `<=3e-3` and post-cast step-gain bound
`<=1.001`; it is a declared tolerance path, not a hard certificate.

Kill immediately on a provenance/kernel/gradient/checkpoint/no-fallback
failure, NaN/OOM/wrong GPU, or certificate violation. Stop the formal
intervention if clipping is `<1%`. Stop by step50 if CE is more than `0.15`
worse than matched control and has not recovered. Do not sweep cap, seed, LR,
loss, model size, or continuation length after a negative result.

## 4. Fixed Configuration

- Official FLA `GatedDeltaNet2`, D192/L10/H6/K32/V32, short conv on.
- Native FutureSeed1, five loops, equal CE on every loop.
- BF16 model projections, external FP32 q/k normalization, and FP32 erase
  projection. Normalized q/k and projected erase gate are then cast to BF16
  before the official chunk kernel. The actual post-cast key/gate/decay are
  audited against the low-precision tolerance. A separate official FP32
  fused-recurrent forward lane carries the hard mathematical certificate.
- Mixed FP32 q/k/g/b plus BF16 v/w is rejected by official WY Triton. Full
  FP32 official chunk is also rejected after a production H6/T81 GPU1 test
  triggered illegal memory access during autotuning. Neither failed path is
  used or hidden behind a fallback.
- Effective batch 128, seed 52, exact same curriculum and official fixed
  evaluation cases.
- No noise, repair, search, selector, oracle inference, or task-specific rule.

## 5. Commands And Environment

- Persistent root: `/huyang2/double-loop`
- All cache, wheel, model, run, and artifact paths stay below that root.
- Complete preflight (pure-math CPU properties, CUDA contract,
  strict FP32 fused-forward certification, official BF16 chunk parity and
  backward at the production head shape, state carry, benchmark, dataset manifest, and
  both two-step exact-resume smokes):

```bash
GPU1_UUID=<GPU1_UUID_FROM_AISTATION_HELPER> \
CUDA_VISIBLE_DEVICES=0 \
./scripts/run_gain_budget_gdn2_preflight.sh
```

- Formal matched run. This is the only authorized formal launcher because it
  enforces control-first execution, exact-PID termination, CE/certificate
  kill criteria, comparison, and decision generation:

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/run_gain_budget_gdn2_matched.sh
```

The detached source SHA and final run names will be filled after the
implementation commit. Both matched arms refuse to start without the same
passing formal-ready manifest bound to that exact source SHA, pinned FLA tree,
pre-registered GPU1 UUID, CUDA contract hash, and two exact-resume smoke
artifacts.

## 6. Required Validation

1. Pure Torch projection properties and closed-form singular value vs SVD.
2. CUDA official chunk output/state/backward parity at sequence boundaries,
   including Sudoku length 81, BF16-model-to-FP32-recurrence backward, and
   split state-carry equivalence.
3. Official fused recurrent forward parity for lengths at most 64; upstream
   fused backward is unsupported and must be reported as such.
4. `mode=none` bitwise equality to the untouched official layer and unchanged
   checkpoint parameter keys.
5. Full gain-budget layer finite, nonzero gradients for q/k/v/decay/erase/write.
6. Exact parent checkpoint hash, parent source SHA, optimizer/RNG state, and
   semantic resume contract.
7. Full checkpoint load, two-step GPU1 smoke, wall-time and VRAM benchmark.
8. SHA256 manifest over all official train/test input and label arrays,
   revalidated before each formal arm.
9. Official 512-case-per-range metrics remain the only primary score. A
   separate fixed 256-case-per-range pool is content-hashed and paired only
   for loop1..5 diagnostics and side-by-side visualization.
10. The comparator requires clean identical source SHA/config semantics,
    recomputes every paired-case error count, evaluates the preregistered
    score/systems/mechanism gates, and emits the sole leaderboard row.

## 7. Results

### 7.1 Strict preflight at `c4bcbbad`

The split-precision contract passed on GPU1:

- 23 pure-Torch mathematical/property tests passed.
- Official FLA was pinned exactly to
  `9c8e42e762fce087c27b673af4922795d9edb85e`.
- Official BF16 chunk forward matched the naive recurrence for
  `T=1,63,64,65,81,128`, including production `B2/H6/T81`.
- The strict FP32 official fused-recurrent forward certificate passed.
- Small-shape strict chunk backward VJP and production `H6` BF16 chunk
  backward VJP passed.
- State carry, untouched `mode=none` bitwise equality, external-identity VJP,
  full-layer gradients, checkpoint resume, data manifest, and both two-step
  exact-resume smokes passed.

The ABCCBA benchmark measured:

- official BF16 baseline: `0.01875085 s`;
- matched external identity: `0.02532083 s`;
- gain-budget candidate: `0.02536411 s`;
- projection time overhead versus matched identity: `+0.1709%`;
- projection peak-memory overhead versus matched identity: `+17.9985%`;
- clipped token/head fraction: `0.6337448`.

The larger `+35.27%` time difference from untouched official GDN2 to the
external path is not attributed to the projection: both causal arms pay the
same external-normalization and audit path.

### 7.2 First formal attempt at `c4bcbbad`

The matched control completed:

- mixed exact by loop 1/2/3/4/5:
  `0.0234/0.0449/0.1855/0.2344/0.2422`;
- official 51-55 loop5 exact/blank:
  `0.3516/0.7678`;
- official 56-60 loop5 exact/blank:
  `0.1426/0.6216`;
- official 61-64 loop5 exact/blank:
  `0.1895/0.8261`.

The candidate stopped before its first logged training step with:

```text
gain-budget FP32 numerical singular-value certificate failed
```

The wrapper captured exact PID `14566`, exit status `1`, and wrote
`abort.json`. No GPU process remained. Therefore this attempt gives no
candidate score and cannot accept or reject the mechanism.

The closed-form projection is mathematically valid. The failure occurs when a
very large real batch near `tau=1` rounds a few ulps outside the strict FP32
certificate.

### 7.3 Rejected double-certificate implementation at `755d5871`

The first numerical fix recomputed the full post-projection certificate after
moving residual violators to the isotropic endpoint. It passed 24/24 math
tests and all official chunk/fused/backward/state/layer correctness checks.
ABCCBA nevertheless measured candidate overhead `+23.3301%` versus matched
external identity, above the pre-registered `20%` maximum. The preflight
stopped before checkpoint smoke or formal training.

This is an implementation-efficiency failure, not a quality result. The next
exact SHA reserves half of the existing FP32 certificate tolerance before
computing the closed-form projection. It then computes the actual certificate
once. When this reserve consumes all feasible slack, `tau_projection` equals
the minimum feasible tau and the projection reaches `lambda=0`, the isotropic
endpoint. This still preserves `delta`, minimizes shear, and never relaxes the
requested budget. Its frequency remains explicit as
`gain_budget_numerical_endpoint_frac`.

### 7.4 One-pass performance at `ac566f96`

The one-pass formulation again passed 24/24 math tests and every official FLA
chunk/fused/backward/state/layer correctness gate. ABCCBA measured projection
overhead `+20.9806%`, narrowly above the unchanged `20%` gate, and stopped
before checkpoint smoke or formal training.

The remaining systems retry does not remove a condition or change the
projection. It combines the three device-wide fail-closed reductions
(feasibility, delta preservation, and spectral validity) into one reduction,
and removes an unnecessary finite-`tau` branch because validated
`fixed_sigma` and `decay_funded` modes always produce finite tau. Any failing
row still aborts before the recurrence.

### 7.5 Benchmark-instrument instability

After the GPU1 lease expired and was restarted, the reduction-fused
`ca24913b` implementation again passed 24/24 math tests and all official CUDA
correctness checks, but the same benchmark reported `+26.4590%`. Across
otherwise matched checks, the old timer has now reported `+0.17%`, `+20.98%`,
and `+26.46%`. It clears the CUDA allocator before every sample and times only
one forward/backward, so allocation and scheduling variance dominate a
threshold near `20%`.

The gate remains exactly `20%`; only the instrument is corrected. Each ABCCBA
timing point now averages five warmed forward/backward calls. Cold-cache peak
memory remains a separate measurement. Failed benchmark JSON retains all raw
time and memory samples instead of losing them when the assertion is raised.
This is one final stable systems measurement, not repeated sampling until a
favorable outcome.

### 7.6 Stable preflight and formal retry at `56030a13`

The corrected benchmark passed:

- projection time overhead versus matched identity: `+14.5554%`;
- projection memory overhead versus matched identity: `+18.4618%`;
- clipping fraction: `0.63497`;
- all 24 math tests, official CUDA checks, checkpoint gates, and both
  exact-resume smokes passed.

The formal control completed with mixed loop1-to-loop5 exact
`0.0234 -> 0.2324` and official 51-55/56-60/61-64 loop5 exact
`0.3750/0.1348/0.1543`.

The formal candidate again hit an FP32 post-certificate rounding violation
before its first logged step. The wrapper tracked exact PID `4282`, wrote
`abort.json`, and left no live GPU process. This still is not a candidate
quality score.

The final numerical retry does not increase the margin or rerun the same
formula. It checks the actual one-pass `effective_sigma`; rows that remain
outside the requested boundary select the analytic zero-shear endpoint
(`lambda=0`). No second dense certificate pass is required. Those endpoint
statistics use the exact rank-one result, while the BF16 gate actually passed
to the official chunk recurrence is independently recomputed and checked
against the declared `1.001` step-bound and `3e-3` delta tolerances.

### 7.7 Rejected dense endpoint selection at `4715a5a`

The analytic endpoint implementation passed all 24 math tests and every
official CUDA correctness gate. The stable benchmark nevertheless measured
projection time overhead `+28.1002%` and memory overhead `+18.4866%`.
Preflight stopped at the unchanged `20%` systems gate before checkpoint smoke
or formal training.

Profiling the code path isolated the regression to several additional
K-dimensional `where` operations after the scalar certificate had already
decided which token/head rows needed the endpoint. The next and last systems
retry makes all scale and endpoint decisions in scalar token/head space,
computes the analytic post-projection singular value there, and constructs the
K-dimensional gate once. It still recomputes the actual FP32 erase strength
from that gate, and the caller still audits the BF16 gate that enters the
official recurrence. No certificate, budget, or decision threshold changes.

## 8. Conclusion

The implementation and official-kernel contract are valid, but the formal
quality comparison is not yet complete. One scalar-certificate performance
retry is authorized because it removes redundant tensor materialization
without changing the mechanism. A systems failure or negative quality result
after that retry closes this projection without a cap, seed, loss, or
training-length sweep.

## 9. Submission

Not applicable. No tag unless the hard score is strong and the mechanism claim
passes its pre-registered gate.
