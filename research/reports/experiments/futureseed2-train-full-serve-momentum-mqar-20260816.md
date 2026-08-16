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
the immutable `case_index`. No candidate quality result exists yet; the model,
data, mechanism and registered quality/cost limits remain unchanged.

## 6. Decision

Pending R4.
