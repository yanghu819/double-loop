# P-GDN3-059: Momentum-State FutureSeed

## 1. Research Question

Can a second-order live recurrent transition preserve key/value ownership at
L1024 better than GDN2 while retaining native cross-layer FutureSeed?

## 2. Mechanism Hypothesis

The local failure is not lack of value capacity. Native replay retains many
valid values but assigns them to adjacent wrong keys. P058 further shows that
adding an ownership reread changes the whole learned retrieval path, so an
inference-time edge-off cannot recover it. The candidate therefore changes the
primary token-scan update itself. Alongside matrix state `S`, it maintains a
momentum state `M`; committed residuals accumulate in `M` before updating `S`.
The hypothesis is that consistent ownership corrections reinforce while
single-token interference is damped. Native FutureSeed transports the stacked
`[S, M]` terminal state to the next layer.

This is not covered by the failed families. P025/P056 add auxiliary memory to
the GDN2 transition, P037 scalarizes its edit, and P058 adds a reverse sweep and
reread. P059 instead trains a distinct second-order primary recurrence end to
end. It adds no selector, cache admission rule, search, repair, reverse scan or
task-specific logic.

## 3. Fixed Implementation And Provenance

- External Momentum DeltaNet source:
  `HuuYuLong/MomentumDeltaNet@c6e77fa261fb0c002fae1a14b6209a5b28d2edc9`.
- The external repository has no declared license in this checkout. It is used
  read-only as a pinned experimental dependency and is not redistributed or
  copied into this MIT repository.
- External source hashes are fixed in the checker for the layer, chunk kernel
  and fused recurrent kernel.
- Current pinned FLA remains the host package for shared modules and cache;
  only the external momentum layer and operator namespace is added.
- The pinned host predates the external operator's uppercase
  `USE_CUDA_GRAPH` scheduling export. A contract-audited compatibility bridge
  supplies its upstream default value `False`; every other imported utility
  must already exist in the pinned host. This does not alter recurrence math,
  gates, kernel source, data, initialization, or any registered threshold.
- Candidate: D128/L2/H4/K32/V32, native state-and-momentum FutureSeed, chunk
  training, directional MQAR L1024, 10 epochs, batch32, seed123.
- The frozen P-REPRO initialization is loaded exactly into every same-name,
  same-shape shared parameter. Momentum-only gates use their deterministic
  architecture initialization. This is one from-scratch architecture run,
  not a short zero-init continuation.
- Candidate has 8,192 recurrent values per layer, exactly twice the GDN2
  matrix state, and 61,912 fewer total parameters than the frozen carrier.

## 4. Falsifiable Prediction

If second-order state dynamics address binding interference, balanced accuracy
must rise by at least `.10`, both future and past retrieval must improve rather
than trade off, and valid-value/wrong-key swaps must fall as a share of all
errors. Merely activating `M`, reducing one direction's errors, or improving
loss is not sufficient.

## 5. Registered Gates

Integrity and activation:

- exactly one visible registered GPU at CUDA index 0;
- exact pushed clean detached source and frozen artifact hashes;
- exact external Git SHA and three external source hashes;
- exact shared-parent tensor mapping, parameter count and 2x state geometry;
- two chunk momentum backward paths, gradients through Q/K/V and all four
  momentum gates, one active FutureSeed route;
- chunk/recurrent output and final-state relative RMS difference at most `.05`;
- finite nonzero `S` and `M`, with `M/S` RMS ratio in `[.01, 20]`.

Science:

- balanced accuracy `>=.65` and gain `>=.10` over frozen `.494`;
- future and past accuracy each `>=.62`;
- joint exact `>=.15` and gain `>=.10`;
- total query errors reduced at least 20%;
- wrong-key swap fraction of errors reduced by at least `.10`.

Cost:

- elapsed, post-warm wall and warmed-step ratios each `<2.0x`;
- peak allocation ratio `<1.5x`.

Any miss closes this mechanism. There is no momentum coefficient, gate,
state-size, width, seed, LR, loss, batch, depth or duration rescue. A complete
pass opens one hard-Sudoku transfer; otherwise the next decision returns to a
different scalable ownership transition.

## 6. Result

Exact pushed/read-back source
`96227cb425daf672794fb51265f2d5900f073347` ran on CUDA index 0, A800 UUID
`GPU-c1d7c624-a393-befa-3807-7e00602d65ca`. The strict contract passes:

- the external layer/chunk/recurrent source hashes and Git SHA are exact;
- both layers execute `Chunkmode_ruleFunctionBackward`;
- chunk/recurrent output and state relative RMS differences are
  `.006242/.004626`, inside the `.05` limits;
- contract `S/M` RMS is `.041696/.078351` (`M/S=1.8791`), and the receiver
  obtains finite stacked `[S,M]` state with exactly one FutureSeed route;
- Q/K/V, alpha, beta, momentum, eta and FutureSeed gradients are finite and
  nonzero;
- the candidate has exactly `599,672` parameters (`-61,912` versus control)
  and `8,192` recurrent values per layer.

The fixed endpoint is a large absolute quality improvement:

| arm | balanced | future | past | joint | errors | wrong-key swaps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen native control | .49400 | .45400 | .53400 | .04100 | 2,024 | 1,546 |
| momentum + native FutureSeed | .94425 | .95150 | .93700 | .82400 | 223 | 151 |

Future/past CE falls from `1.25939/1.22668` to `.15479/.19323`. Of the
control's wrong-key errors, `1,474` become correct; another `452` other errors
become correct. Only `125` control-correct queries regress. The candidate
opens earlier in training: validation accuracy is `.4590` at epoch 3 and
`.8455` at epoch 4, while the control is still near chance through epoch 4 and
ends at `.4940`.

The one failed gate is deliberately retained. Wrong-key swaps fall by
`90.23%` in absolute count, and total errors fall by `88.98%`, but wrong-key
swaps remain `.677130` of the 223 residual errors versus `.763834` of control
errors. The share improvement is `.086704`, short of the registered `.10`
floor by `.013296`. All other quality, integrity and activation checks pass.

Elapsed/post-warm/warmed-step/peak-allocation ratios are
`1.80087/1.79728/1.82637/.94984x`, all inside the registered ceilings. The
candidate trains 102.4 million tokens in `125.41 s`, reaches `976.03`
examples/s in the independent warmed benchmark, and allocates `998,742,528`
bytes at training peak. Across 59 formal five-second GPU samples, utilization
averages `35.15%` including import and compilation; the 37 active samples
average `56.05%`, peak at `85%`, and observed memory peaks at `3,844 MiB`.
Mean/peak sampled power is `124.44/244.95 W`.

## 7. Decision

Close P-GDN3-059 under its precommitted all-gates rule. Exit status `2` is a
science-gate miss, not an integrity failure. Do not transfer this checkpoint
to Sudoku and do not tune momentum coefficients, gates, state size, model
shape, seed, optimizer, loss, batch, depth or duration.

Retain the result as architecture-positive evidence: unlike every recent side
memory, router, reverse reread and key wrapper, a fully trained second-order
primary recurrence preserves both retrieval directions and removes nearly
nine tenths of all errors. The remaining bottleneck is now narrow: among the
small residual error set, wrong-owner retrieval is still dominant. A successor
must introduce a distinct scalable ownership representation rather than
rescue this exact Momentum Delta configuration.

## 8. Artifacts

- run:
  `/huyang2/double-loop/runs/p-gdn3-059-momentum-futureseed-20260815T230000Z-96227cb`
- source SHA: `96227cb425daf672794fb51265f2d5900f073347`
- external SHA: `c6e77fa261fb0c002fae1a14b6209a5b28d2edc9`
- contract JSON SHA256:
  `6ec0e25b4ce3704dea1d7e9c8079cb9c6a0723a0d6d5e85b22bccdd476b9b137`
- score SHA256:
  `7201d32834d3f9af4d1057750ba4eb62df492aa495eec770f9dad85f3d4f6e8a`
- checkpoint SHA256:
  `13410aaad3be26fbdf07c7a12e1afa9fb38ced41a1939a519c02973be2b8508f`
- cases SHA256:
  `f64b0ae0e45a65a8da2934ece8cfe208fedd835a373d79230320bdf945826f20`
- GPU samples SHA256:
  `fff38f556bc6c90aa9e96954daef4f5722f22cdaa6d55b1e124a087234f54c27`
- source snapshot SHA256:
  `57ce3b12aef627096f7fc9286746982c97ae5dd55e3755d89ef370f0a22e9b99`
- pulled core archive SHA256:
  `4fed7c14f748618f27f8a4a15f3657f4aa0ab6719f24c6b2f9e6e5d7e9c35cec`

## 9. Lessons

First-order live-state dynamics, not value width or an omitted read-time
wrapper, were the dominant L1024 optimization bottleneck. Momentum changes the
learning transition itself and makes both directional associations learnable
without extra model parameters. Native FutureSeed remains useful in this new
recurrence: its gate is `.50304`, and it transports a noncollapsed momentum
component rather than a lossy scalar extrapolation.

The strict miss also matters. High overall accuracy does not mean address
ownership is solved: 151 of 223 remaining query errors are still valid values
assigned to the wrong key. The next mechanism should target residual address
separability while preserving this kind of end-to-end scalable transition;
changing P059's hyperparameters would only measure a nearby operating point.
