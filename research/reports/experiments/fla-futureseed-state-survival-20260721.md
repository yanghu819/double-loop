# Official FLA FutureSeed State-Survival Diagnostic

## 1. Metainfo

- Plan ID: `P-LA-003`
- Status: in progress
- Started: 2026-07-21 07:36 CST / 2026-07-20T23:36:10Z
- Machine: AIStation `GPU1`, NVIDIA A800-SXM4-80GB; GPU2 forbidden
- Source branch: `codex/fla-gdn2-kda`
- Diagnostic source SHA: pending prelaunch commit
- Checkpoint-training source SHA: `c3342b8326a6e00e012cd6972d8c0e295d82c0b3`

## 2. Hypothesis

The matched 500-step comparison does not prove that GDN has a higher ceiling:
the pre-fixed holes53 exact score is tied, all three methods have zero exact in
the 51-64-blank ranges, and KDA/GDN2 are still closing their CE gap at step 500.

The highest-information unresolved mechanism is whether a state with the same
injected RMS has the same effect in each recurrence. GDN applies a scalar decay
and erase/write rate per head. KDA applies channel-wise decay. GDN2 additionally
separates channel-wise erase and write. Those extra transformations may remove
the coherent FutureSeed direction faster even though the adapter reports the
same injected RMS.

Prediction: if KDA/GDN2 retain at least 30% less state or output influence by
token 81 than GDN, the old comparison is not semantically matched at the
FutureSeed interface. If all three are within 20%, seed erasure is rejected and
the remaining explanation is finite-budget optimization/sample efficiency.

## 3. Configuration

- Exact step500 checkpoints from all three P-LA-002 arms.
- Critical checkpoint config must match before any CUDA probe.
- Four deterministic official 51-55-blank test inputs; labels are not loaded.
- Every real layer-to-layer FutureSeed transition in D192/L10/H6/D32 models.
- Prefixes `1,8,32,81`.
- Actual injected fixed unit-RMS FutureSeed versus an all-zero state while block
  input and model parameters remain identical.
- Metrics: terminal-state difference divided by seed RMS, state/seed cosine,
  absolute and relative output difference, FutureSeed gate, and recurrent
  decay/erase/write statistics.
- No training, task loss, score selection, repair, search, selector, noise,
  seed sweep, LR sweep, or architecture-specific tuning.

Prelaunch audit found one raw-metadata difference that the original comparison
summary described too loosely. The GDN checkpoint retained its original planned
schedule `46-50:100,51-55:1400`, while the KDA/GDN2 relaunches planned
`46-50:100,51-55:400`. All checkpoints are exact step500 states. Source audit
shows there is no LR scheduler and the planned total is used only as the loop
endpoint/evaluation range. Therefore the executed curriculum through step500 is
identical for all arms: `46-50:100,51-55:400`. The diagnostic gate compares this
executed prefix and preserves the different unused tails in its metadata.

## 4. Environment

- GPU row: `GPU1` only, exactly one visible CUDA device.
- Required GPU: NVIDIA A800-SXM4-80GB.
- Python: `/opt/conda/bin/python`.
- PyTorch: `2.7.0+cu126`.
- FLA: pinned `flash-linear-attention==0.5.2` wheel with expected SHA256
  `65f57bf2aa937991fc497bd63f42883d263ead8646f21f2492735ccddd82d0eb`.
- Persistent cache root: `/huyang2/double-loop/.cache`.
- Strict environment: `FLA_DISABLE_BACKEND_DISPATCH=1`,
  `FLA_CONV_BACKEND=triton`, `FLA_STRICT_OFFICIAL=1`.

## 5. Commands

Strict provenance/reference/backward gate and state-survival command will be
recorded verbatim after the prelaunch commit is detached on GPU1.

## 6. Artifacts

The first launch correctly stopped before accepting results because it compared
the raw planned schedule rather than the executed step500 prefix. Its failure
record remains at
`/huyang2/double-loop/runs/fla-futureseed-state-survival-20260720T234811Z-3059fca/`.
The corrected run will archive `diagnostic.json`, self-contained `index.html`,
strict-gate JSON/log, launch environment, exact PID, source SHA, and completion
metadata below `/huyang2/double-loop` before pulling a lean copy locally.

## 7. Results

Pending.

## 8. Conclusions

Pending. The diagnostic determines one next decision only:

- unequal seed survival: calibrate a generic matched-influence FutureSeed
  interface, without using Sudoku scores;
- comparable seed survival: reject the erasure explanation and run at most one
  matched longer GDN2 crossover gate.

## 9. Submission Record

None. This is a mechanism diagnostic.
