# Official FLA FutureSeed State-Survival Diagnostic

## 1. Metainfo

- Plan ID: `P-LA-003`
- Status: completed; seed-erasure hypothesis rejected
- Started: 2026-07-21 07:36 CST / 2026-07-20T23:36:10Z
- Machine: AIStation `GPU1`, NVIDIA A800-SXM4-80GB; GPU2 forbidden
- Source branch: `codex/fla-gdn2-kda`
- Completed: 2026-07-21 08:25 CST / 2026-07-21T00:25:00Z
- Diagnostic source SHA: `d9b496378ff2748d86a59e5c7c2930b836d33cd4`
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

The detached launch script was
`/huyang2/double-loop/artifacts/launch/fla_futureseed_state_survival_d9b4963.sh`.
It first reran `check_fla_delta_backbones.py --backbone all` against the pinned
wheel, then ran `diagnose_fla_futureseed_survival.py` over prefixes
`1,8,32,81` and the three exact step500 checkpoints. The run used
`CUDA_VISIBLE_DEVICES=0`; all cache and temporary paths were below
`/huyang2/double-loop`.

## 6. Artifacts

The first launch correctly stopped before accepting results because it compared
the raw planned schedule rather than the executed step500 prefix. Its failure
record remains at
`/huyang2/double-loop/runs/fla-futureseed-state-survival-20260720T234811Z-3059fca/`.
The accepted run is
`runs/fla-futureseed-state-survival-20260721T001116Z-d9b4963/`; it contains
`diagnostic.json`, self-contained `index.html`, `strict_gate.json`, launch/GPU
metadata, source SHA, and completion metadata. No checkpoint is copied into Git.

## 7. Results

The strict gate passed again. Installed FLA source hashes match the pinned wheel,
the three observed backward nodes are the official chunk CUDA functions, and all
short convolutions use Triton. The label-free diagnostic itself took `86.4s`
and allocated `299.4 MiB` at peak.

| Mean over nine FutureSeed transitions | GDN | KDA | GDN2 |
|---|---:|---:|---:|
| injected seed RMS | 0.4887 | 0.4992 | 0.5036 |
| token1 state retention | 0.6680 | 0.7911 | 0.8488 |
| token32 state retention | 0.3783 | 0.5014 | 0.6144 |
| token81 state retention | 0.2945 | 0.3925 | 0.5298 |
| token81 state/seed cosine | 0.4288 | 0.5283 | 0.6280 |
| token81 output delta / zero-state output RMS | 0.8575 | 1.6283 | 1.4008 |
| token81 absolute output delta RMS | 0.4813 | 0.4957 | 0.3590 |
| learned FutureSeed gate | 0.4884 | 0.4986 | 0.5033 |

Relative to GDN at token81, KDA retains `1.33x` as much recurrent-state
influence and GDN2 retains `1.80x`. Their relative output effects are `1.90x`
and `1.63x`. GDN2's absolute output delta is `25.4%` lower than GDN, below the
predeclared `30%` erasure threshold, while its state and scale-normalized output
effects point strongly in the opposite direction.

The checkpoint gate also verifies identical executed curricula through step500:
`46-50:100,51-55:400`. It separately records the unused planned tails
(`1400` hard steps for GDN versus `400` for KDA/GDN2) instead of pretending the
raw metadata strings were identical.

## 8. Conclusions

The seed-erasure explanation is rejected. KDA and especially GDN2 preserve the
injected state direction longer than GDN; equal unit-RMS transfer did not
silently disadvantage them by making FutureSeed disappear. More persistence is
also not sufficient for better task quality: GDN's simpler recurrence learns the
shared 500-step recipe faster even though it forgets more of the injected state.

This narrows the plausible explanation to finite-budget optimization and recipe
compatibility. The official classes default to different state geometries:
KDA/GDN2 default to `expand_v=1, head_dim=128`, whereas this matched-interface
test forces all arms to `expand_v=2, head_dim=32`. Therefore P-LA-002 supports
only "GDN opens faster under this common small-state recipe", not "GDN has a
higher architecture ceiling". The next and only justified architecture check is
a matched GDN versus GDN2 continuation from step500 to step1000; no seed, LR,
loss, gate, or temperature sweep is warranted.

## 9. Submission Record

None. This is a mechanism diagnostic.
