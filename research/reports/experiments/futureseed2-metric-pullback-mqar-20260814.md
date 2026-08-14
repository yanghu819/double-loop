# P-FS2-009: Metric-Pullback FutureSeed

## 1. Metainfo

- Status: approved; not yet launched
- First decision field: directional MQAR L1024 wrong-key binding regime
- Fixed model: D128/L2/H4/K32/V32 pinned-official GDN2 plus native FutureSeed
- Fixed data/training: 10,000 train and 1,000 validation examples, ten epochs,
  batch32, seed123
- Fixed run order: private-Log-SPD native FutureSeed control, then one
  private-Log-SPD metric-pullback FutureSeed candidate
- Parameter, persistent-state and recurrent-scan delta versus control: zero

## 2. Evidence And Hypothesis

P-GDN3-020 is the only completed address intervention with a large positive
same-runtime signal: independent bounded Log-SPD Q/K metrics raised balanced
L1024 accuracy from `0.1735` to `0.48225`. P-GDN3-033 independently reproduced
the weak native control, but tying one metric across layers reduced balanced
accuracy to `0.125`. Thus useful layer-private address adaptation exists, while
hard namespace sharing is destructive.

Native FutureSeed nevertheless injects the producer KxV terminal state directly
into a receiver whose learned private metric is different. The falsifiable
hypothesis is that P020's gain is capped by this specific metric mismatch. A
receiver-read pullback derived from the two learned factors should preserve
private geometry while making inherited state easier for the receiver to read.

The claim is deliberately narrow. Because the official kernel normalizes Q/K
per token and different layers own different projections, no fixed matrix is
an exact covariance of every key and query. This experiment tests only whether
removing the known linear metric component of the mismatch is causal.

## 3. Exact Intervention

Each layer keeps P020's private bounded factor `C_l`, applied to both Q and K
before the unchanged official GDN2 recurrence. For the sole L2 FutureSeed edge,
form in FP32

```text
B_0_to_1 = solve(C_receiver, C_producer)
S_pullback = S_producer + (B_0_to_1 - I) S_producer.
```

The receiver then applies the existing native per-head RMS normalization,
learned scalar FutureSeed gate and fixed scale to `S_pullback`. Its normal main
scan, Q/K/V projections, decay, erase, write and output path are unchanged.
The residual form makes identity factors exactly the parent function while
retaining direct gradients through both private metrics.

`B=C_r^-1 C_p` is fixed before execution. There is no transpose/inverse
alternative arm, learned transport, scale, clipping, cache, canonical replay,
second state, extra scan or task-specific operation.

## 4. Why Existing Failures Do Not Cover It

- P033 hard-shared the full metric parameter and suppressed private adaptation;
  P-FS2-009 keeps both private metrics.
- P-FS3-004 learned a free static orthogonal K/V map on a mature Sudoku graft;
  this candidate has no transport parameters, changes only K coordinates, is
  determined by the actual learned metrics and trains from scratch on MQAR.
- P-DIAG-ADDR-001 tested receiver reprojection and cache admission on a frozen
  surprise-replay checkpoint. It did not test the interaction between P020's
  two active private metrics and native whole-state FutureSeed.
- P-FS2-007/008 rebuilt a receiver-native complement from token evidence. This
  candidate retains the exact terminal state and adds no tape, replay or solve
  over sequence tokens.

## 5. Strict CUDA Contract

From one clean detached pushed SHA on the registered single A100, prove:

1. exact GPU UUID, pinned FLA and Zoology SHAs, Triton short convolution, two
   official `ChunkGDN2FunctionBackward` paths and no fallback;
2. control and candidate have identical names, tensors, buffers, initialization
   hashes, exact `665,800` parameter count, state size and scan count;
3. identity metrics give bit-exact full output, terminal states and nonzero
   incoming-state behavior versus private-Log-SPD native FutureSeed;
4. non-metric parent gradients are exact at identity, while the candidate's
   producer and receiver metric gradients contain a finite nonzero transport
   contribution;
5. opened private metrics change the pullback and full output, all four heads
   are active, head permutation is equivariant, and `B` equals the direct FP32
   solve within `1e-6` relative error;
6. each private metric stays inside P020's bounds, pullback condition is below
   `4.60`, transported state is finite and board-varying, and raw transported
   RMS is at most `4.60x` producer RMS; and
7. fixed directional-MQAR data hashes, optimizer/RNG/batch order and first
   warmup batch are identical between arms, with no concurrent GPU model.

Any miss closes this implementation before science. No CPU model smoke is
allowed.

## 6. Fixed Matched Protocol

Run exactly two arms in one process and fixed order:

1. `future_seed_gdn2_log_spd` control;
2. `future_seed_gdn2_metric_pullback` candidate.

Both use L1024, four KV pairs, D128/L2/H4/K32/V32, ten epochs, batch32,
seed123 and the same 10,000/1,000 examples. RNG is reset before each arm. The
control is the only denominator; historical P020 is mechanism context, not an
absolute reproducibility requirement.

## 7. Registered Decision Gate

Activation and stability all pass:

- both private metrics have actual `||M-I||_F >= 1e-4`;
- pullback delta relative RMS, head variation and board variation are finite
  and at least `1e-4` where applicable;
- pullback condition `<4.60`, all states finite, and transported raw state RMS
  `<=4.60x` producer RMS;
- exactly one active FutureSeed route and two official main scans.

Quality all pass:

- balanced accuracy `>=max(0.55, control + 0.10)`;
- future and past accuracy each improve by at least `0.07`;
- joint exact `>=max(0.08, control + 0.04)`;
- total query errors strictly decrease; and
- wrong-key valid-value swap fraction among errors decreases by at least
  `0.05`.

Cost all pass relative to control:

- fit elapsed, post-warm wall and independent warmed step each `<1.30x`;
- peak CUDA allocation `<1.10x`.

## 8. Kill Rule And Next Decision

Any integrity, activation, stability, quality or cost miss discards
metric-pullback FutureSeed. There is no transport direction, transpose,
inverse, interpolation, scale, clipping, metric cap/rank, seed, LR, loss,
batch, width, depth, epoch or duration rescue.

A full pass authorizes one fixed hard-Sudoku transfer using the already frozen
P034 private-metric parent. A miss closes metric-derived cross-layer transport
and returns the next decision to a genuinely new live recurrent transition.

## 9. Commands And Artifacts

The exact launcher, pushed SHA, GPU identity, contract, run directory, hashes,
metrics and decision will be filled before launch and after completion.
