# P-FS2-017 Train-Full, Serve-Momentum FutureSeed

## 1. Research Question

Can the proven P059 model retain native quality while transporting only its
Momentum plane at deployment, even though P-FS2-016 shows that removing the
base state during training destroys optimization?

## 2. Evidence And Hypothesis

P059 reaches balanced/future/past/joint
`.94425/.95150/.93700/.82400`. Masking its already-trained FutureSeed to
`[0,M]` changes those metrics by only `-.00050/-.00150/+.00050/-.00200`. In
contrast, P-FS2-016 trained the same recurrence from scratch with only `M`
crossing the layer boundary and collapsed to `.01325/.01350/.01300/0`.

The falsifiable interpretation is that `S` is an optimization scaffold used
to form the second-order representation, while the converged model's useful
future evidence is almost entirely in `M`. A phase-asymmetric contract may be
Pareto-better: train native `[S,M]`, then deploy the frozen model with the
production M-only edge.

## 3. Fixed Mechanism

Load the exact frozen P059 checkpoint. The control uses native normalized and
gated `[S,M]`; the candidate uses the production P-FS2-016 implementation,
packs only terminal `M`, applies the unchanged receiver normalization/gate,
and reconstructs `[0,M]`. Parameters, recurrent state, scans, Q/K/V,
convolutions and output correction are identical. No training, finetuning,
selector, cache, task logic or parameter change is allowed.

## 4. Gates

The clean pushed A800 run must pass the existing strict CUDA contract and then
replay the P059 test bank. R1/R2 established a cross-device calibration before
the production candidate ran: the new A800 changed exactly `1/4000` native
predictions, balanced accuracy by `.00025`, joint exact by `.001`, and CE by at
most `4.8e-6`, while all artifact, parameter and data hashes remained exact.
R3 therefore fixes, before candidate evaluation, a native replay bound of at
most `2/4000` changed predictions, `.001` accuracy/joint drift and `1e-4` CE
drift. It also freezes the prior exact same-weight M-only diagnostic SHA256
`75f7ee388aee78139c9a261238e102b4cb2670df1dbdc8f0acdf46d921091f07`.
A deployment Pareto pass requires:

- exact frozen artifact, data and parameter hashes, plus the fixed native
  cross-A800 replay calibration above;
- exactly 4,096 transported values instead of 8,192, with zero parameter,
  persistent-state and scan delta;
- one active M-only route, exact zero receiver S and finite nonzero M;
- balanced, future and past each within `.005` of P059, joint within `.01`,
  total errors at most `243`, and wrong-key swaps at most `171`;
- production M-only metrics within `.001` accuracy/joint and `1e-4` CE of the
  previously frozen exact same-weight M-only counterfactual;
- median interleaved warmed inference time below `1.05x` and allocation below
  `1.02x` the native control.

The claim is deployment-only. Failure closes phase-asymmetric compression;
success does not authorize M-only training or claim repair of the residual 151
owner swaps.

## 5. Result

R1 passed the strict CUDA contract but stopped before candidate evaluation when
the native replay differed from the frozen cases. R2 isolated native replay
before candidate construction and recorded the single changed prediction above,
proving the mismatch is a bounded cross-A800 replay effect rather than candidate
lifecycle contamination. R3 stopped before candidate construction because its
new audit zipped two hardest-first case lists; the one changed prediction moved
that case in the sort order and triggered a false identity error. R4 aligns by
the immutable `case_index`; model, data, mechanism and registered gates remained
unchanged.

R4 completed successfully on the clean pushed source
`d943ab6c34fabcae2af56711e592af2784b9005e`. Every registered integrity,
activation, quality and cost check passed:

| Metric | Native `[S,M]` | Production `[0,M]` | Delta |
|---|---:|---:|---:|
| balanced accuracy | `.94400` | `.94375` | `-.00025` |
| future accuracy | `.95150` | `.95000` | `-.00150` |
| past accuracy | `.93650` | `.93750` | `+.00100` |
| joint exact | `.82300` | `.82200` | `-.00100` |
| future CE | `.154794` | `.159012` | `+.004218` |
| past CE | `.193228` | `.193219` | `-.000009` |
| total errors | `224` | `225` | `+1` |
| wrong-key valid-value swaps | `152` | `153` | `+1` |

The candidate exactly reproduced the prior frozen M-only diagnostic within the
registered cross-device bounds. It transports `4,096` values per route instead
of `8,192`, for an exact `.5` ratio. Receiver S is exactly zero, receiver M RMS
is `.503104`, both layers are active, and parameter, persistent-state and scan
deltas are all exactly zero.

## 6. Mechanistic Interpretation

P-FS2-016 and P-FS2-017 together separate learnability from deployment content.
Training from scratch with only M crossing the layer boundary collapses, but
after native `[S,M]` training the same frozen model needs almost only M at that
edge. S is therefore an optimization scaffold for forming the Momentum
representation, not necessary production payload for this converged model.
This is a phase-asymmetric FS2 contract, not evidence that M-only transport can
support training.

The result also does not repair P059's residual ownership problem. The M-only
candidate has `153` wrong-key swaps versus `152` in the contemporaneous native
replay. Its contribution is a cheaper FutureSeed carrier around the better
Momentum GDN3 core.

## 7. Cost And GPU

The fixed interleaved benchmark order was control/candidate/candidate/control,
with eight warmup and forty measured batches per round. Median candidate versus
control elapsed time is `.280027s` versus `.297847s`, ratio `.940171`; median
throughput is `4570.99` versus `4331.14` examples/s. Peak allocation is exactly
`263,808,000` bytes for both arms. This proves no deployment cost regression;
the apparent median speedup is not claimed because the two control rounds have
visible timing spread.

The two-second GPU sampler recorded 105 rows. Only eight intersected the short
CUDA bursts: nonzero utilization averaged `28.25%`, peaked at `65%`, and maximum
observed device memory was `1,192 MiB`. All-row mean utilization was `2.15%`
because imports, NFS reads and compilation dominate this tiny D128/L2 frozen
evaluation; the interleaved throughput measurement is the relevant cost signal.

## 8. Provenance

- run: `p-fs2-017-r4-train-full-serve-momentum-20260816T154557Z-d943ab6`
- remote run directory: `/huyang2/double-loop/runs/p-fs2-017-r4-train-full-serve-momentum-20260816T154557Z-d943ab6`
- GPU: CUDA index 0, NVIDIA A800-SXM4-80GB,
  `GPU-7db97dd1-77e8-df7f-570e-09e7059ba5a3`
- frozen checkpoint SHA256:
  `13410aaad3be26fbdf07c7a12e1afa9fb38ced41a1939a519c02973be2b8508f`
- score SHA256:
  `318962aa8012723cbae98a9453453e1580047fd091e75fa99e7cd5ba6ce0e654`
- contract SHA256:
  `9124b5fb9bfd271b7423ca035ba72208547f426ecfc72c90d4f9debbd26cc51a`
- formal log SHA256:
  `87436020e31352e0deb79c0dbfae3ec064fcd3a6b951cf24efab1f0cad06a975`
- candidate/control cases SHA256:
  `44d2c774215f66333e03b45df3cfd2c2d424679b05d10b5bcfa2433ca8f8b1ef` /
  `ddf8698befe0ddcd37fc37f6d47e8992023680e48861d25a16940a8670c849a0`
- external Momentum DeltaNet SHA: `c6e77fa261fb0c002fae1a14b6209a5b28d2edc9`
- pinned host FLA SHA: `9c8e42e762fce087c27b673af4922795d9edb85e`

Endpoint, tee and combined wrapper statuses are all zero. The source worktree
was clean and its branch SHA was pushed and read back before launch.

## 9. Decision

Promote as a Pareto-better FS2 deployment contract: train P059 with native
`[S,M]`, but transport only M and reconstruct `[0,M]` at inference. This halves
logical cross-layer FutureSeed bandwidth at matched quality with no new
parameters, recurrent state, scans, allocation or measured latency regression.
Close M-only-from-scratch training and any claim that this solves the remaining
owner swaps. This result is on directional MQAR L1024; Sudoku transfer remains
a separate mainline endpoint rather than an implied result.
