# P-GDN3-004: Full-Scale Position-Address GDN3

## Status

- Status: formal trajectory active; production-runner fit passed
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

The process is healthy and compiling/training on the sole registered GPU. The
first scientific decision remains the frozen step500 gate below; no result is
claimed from the two-step fit.

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

A pass would establish address/payload factorization as the first scalable
GDN3 candidate: stable locations for recurrent writes and reads, private
content dynamics in every layer, and native FutureSeed under a full hard-Sudoku
scaling budget. A miss closes this mechanism at scale and moves the next test
to a genuinely new FutureSeed state representation rather than another address
variant.
