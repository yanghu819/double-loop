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
replay the exact P059 test bank. A deployment Pareto pass requires:

- exact frozen checkpoint, data, cases and parameter hashes;
- exactly 4,096 transported values instead of 8,192, with zero parameter,
  persistent-state and scan delta;
- one active M-only route, exact zero receiver S and finite nonzero M;
- balanced, future and past each within `.005` of P059, joint within `.01`,
  total errors at most `243`, and wrong-key swaps at most `171`;
- median interleaved warmed inference time below `1.05x` and allocation below
  `1.02x` the native control.

The claim is deployment-only. Failure closes phase-asymmetric compression;
success does not authorize M-only training or claim repair of the residual 151
owner swaps.

## 5. Result

Pending the strict A800 endpoint.

## 6. Decision

Pending.
