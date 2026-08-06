# P-GDN3-004: Full-Scale Position-Address GDN3

## Status

- Status: step500 gate passed; formal trajectory continues unchanged
- Date: 2026-08-06
- Benchmark: official/full-diversity hard 9x9 Sudoku
- Compute: AIStation task-mode GPU1 only
- Seed: 52 only

## Decision Context

P-GDN3-003 fails its step500 carrier gate decisively. Exact cross-layer Q/K/V
initialization leaves the D256/L12 model near chance: holes50 loop5 exact/blank
is `0/0.1321` and train CE is `2.0089`, versus P-GDN3-002
`0.7778/0.9853` and CE `0.0143`. Initial equality is therefore not a usable
substitute for a stable address mechanism; the symmetric stack does not break
symmetry quickly enough to learn even the easy curriculum.

The strongest unscaled architectural evidence is P-ADDR-002. At matched D192
step9100, separating canonical position Q/K from content V and state-edit gates
improved mean official 51-64 blank accuracy by `+0.2296` and train CE from
`1.9370` to `1.2196`. Its candidate-only continuation reached mean hard blank
`0.5437` at step9300 with real loop correction, but it was never trained from
scratch at D256/L12 over the full 12k diversity budget.

## Mechanism

Each layer remains fully private and independently initialized. Q and K are
computed from the canonical token-plus-position anchor in canonical order and
then gathered into the sampled traversal. V, decay, erase, write, output gate,
FFN, and residual content remain functions of hidden content. The unchanged
official FLA GDN2 chunk recurrence receives those tensors and native
FutureSeed continues to initialize each deeper layer from the preceding
terminal state using unit per-example/head RMS normalization and its learned
per-head gate.

This is not a P-GDN3-003 initialization-strength or partial-sharing rescue. It
removes cross-layer copying entirely and tests whether stable address/payload
factorization is the scalable GDN3 abstraction.

## Fixed Contract

- D256/L12/H8/K32/V32, channel multiplier4;
- twelve strict pinned official-FLA GDN2 chunk/Triton layers;
- native FutureSeed scale1, unit RMS normalization, per-layer head gates;
- full-diversity 12k curriculum `46-50:500,51-55:3500,51-60:4000,51-64:4000`;
- random traversal, loop5, equal CE at every loop;
- microbatch32, gradient accumulation4, effective batch128;
- AdamW/LR/weight decay/data/order/seed copied from P-GDN3-002/003;
- one seed52 trajectory and no nearby mechanism, width, LR, loss, or duration
  table.

Before formal launch, one two-step production-runner fit must verify the exact
GPU UUID, clean pushed source, pinned FLA/Triton path, active position-Q/K
diagnostics, finite forward/backward, checkpoint write, and no fallback/OOM.

## Production Fit And Formal Launch

The sole two-step fit passed from clean detached source
`9f2ee8d1738032bc5f09b55db0b81d507780b376` on visible CUDA index0 / UUID
`GPU-53e9f3b4-2966-65d3-6614-09c540921519`. It instantiated `11,485,760`
trainable parameters and twelve exact `fla.layers.gdn2.GatedDeltaNet2` layers
from pinned FLA source `9c8e42e762fce087c27b673af4922795d9edb85e`, with Triton short convolutions
and backend dispatch disabled. Forward, backward, optimizer update, checkpoint
write, and evaluation were finite; peak CUDA allocation was `13,045.7 MiB`.

The address path is not a silent fallback. The fit JSON reports
`gdn2_address_enabled=1.0`, Q/K diagonal/off-diagonal cosine
`0.01234/0.01840`, and nonzero contrast `-0.00606`. The compact training line's
zero `addr_scale/qchg/kchg` fields belong to other address variants; position-Q/K
replaces the Q/K source directly and is measured by the mode-specific cosine
diagnostics. Fit log, config, metadata, score, checkpoint evaluation, JSON, and
HTML are archived under
`research/reports/visualizations/gdn3-position-qk-scale-20260806/`. The fit log,
output JSON, and step2 checkpoint SHA256 values are respectively
`281382610158e80036db7d235ecfb4791a6ce2d1a45b6c9d6780699c08807397`,
`1fa7febe18d08ed190c6be551c402e97e1df6b83138557176ea8c952f3ca6aac`, and
`58bc90c0413c4a39a1bc41f122aa94c239283f5c1b0af1d213a80d15eaa32133`.

The formal 12k trajectory started without changing the registered contract:

- run: `gdn3-position-qk-d256l12-s12000-20260806T131527Z-9f2ee8d`;
- exact PGID / Python child at launch: `128112` / `128187`;
- worktree: `/huyang2/double-loop/worktrees/p-gdn3-004-9f2ee8d`;
- launch log: `/huyang2/double-loop/artifacts/launch/p-gdn3-004/formal-9f2ee8d.log`.

The process remains healthy on the sole registered GPU. The two-step fit is
engineering evidence only; the first claim-bearing result is the independently
evaluated step500 gate below.

## Step500 Science Gate

The frozen step500 evaluator passes the registered easy-carrier gate by a wide
margin. On the 99-board holes50 fixed probe, exact accuracy across loops1-5 is
`0.3838/0.8384/0.8687/0.8788/0.8990`, and blank accuracy is
`0.9669/0.9901/0.9923/0.9927/0.9931`. Mean wrong cells therefore fall
`1.66 -> 0.49 -> 0.38 -> 0.36 -> 0.34`. Loop5 exact exceeds the `0.60`
floor by `+0.2990` and exceeds P-GDN3-002 at the same gate (`0.7778`) by
`+0.1212`. P-GDN3-003 remained at exact zero under the same gate.

This pass is not yet a hard-Sudoku result. The 512-board holes53/58/64 fixed
probes all remain at exact zero. Loop behavior is also heterogeneous:
holes53 wrong cells change `25.82 -> 25.30`, holes58 slightly regresses
`35.35 -> 35.40`, while holes64 improves materially `36.16 -> 34.04`.
Stable position addressing has solved the deep-stack optimization failure and
creates genuine late correction on the hardest probe, but it has not yet
opened full-board closure beyond the easy curriculum.

Train CE/total/loop1 loss at the gate is
`0.002967/0.008744/0.029285`, all finite. The address path remains active with
Q/K diagonal/off-diagonal cosine `0.04836/0.03296` and contrast `0.01540`.
No NaN, OOM, fallback, source drift, data drift, or GPU drift is present.
Training exposure is 64,000 effective boards, 5.184M board input cell tokens,
and 25.92M loop-cell evaluations.

The independent stable step100-to-step400 checkpoint intervals are
`716/723/725` seconds per 100 steps: `7.2133` seconds/step, `17.745` effective
boards/second, and `0.535%` interval CV. Exact full-shape fit peak allocation
is `13,045.7 MiB`; formal live NVML sampled `17,491 MiB`. Formal endpoint peak
allocation has not yet been emitted and is not inferred from NVML.

The evaluator JSON SHA256 is
`58356e226d798e6a761bc945ad90d50b4eea3f903c233b9b551419e18d369369`;
the step500 checkpoint SHA256 is
`522a0a96950b7d9ef30bd65c7b93cae98265fbd3a7fef7e21a3663d92d4fd602`.
Config, source HEAD, log snapshot, evaluator, summary, provenance, and an
aggregate loop visualization are archived under
`research/reports/visualizations/gdn3-position-qk-scale-20260806/`.
These fixed probes are not full official ranges and contain no same-board
trajectories.

Decision: pass and continue the exact trajectory without intervention. Step1000
is a scheduled diagnostic; step3000 is the next decision gate. No concurrent
FutureSeed experiment or rescue variant is authorized while this run owns GPU1.

## Predictions And Gates

- **Step500:** holes50 loop5 exact `>=0.60`, finite CE, and positive loop
  correction. Failure closes the full-scale position-address candidate.
- **Step3000:** holes53 loop5 exact `>=0.02` or blank `>=0.60`, with positive
  step1000-to-step3000 slope and loop-wise wrong-cell reduction.
- **Step6000:** holes53 exact `>=0.05` or holes58 exact `>=0.01`, train CE
  `<0.8711`, and genuine loop correction.
- **Endpoint:** hard-range macro exact `>=0.3191` or mixed exact `>=0.40`, with
  full official 51-55/56-60/61-64 evaluation, same-board loop1-5 evidence,
  and no systems regression that invalidates the mechanism claim.

Kill on any gate miss, source/data/GPU/metric mismatch, fallback, OOM, NaN, or
non-finite gradients. Stop by exact PID/PGID and write `abort.json`. Do not
rescue with another seed, LR, loss, batch, width, address scale, or nearby
position mechanism.

## Allowed Claim

The step500 pass establishes position-address/payload factorization as the first
full-size GDN3 candidate to combine private layer dynamics with faster easy-stage
optimization and genuine loop correction. It does not establish hard scaling.
That claim still requires the registered step3000/6000 gates and frozen endpoint
official ranges. A later miss closes this mechanism at scale and moves the next
test to the preregistered FutureSeed innovation representation rather than
another address variant.
