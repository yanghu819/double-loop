# P-GDN3-049: Shared Canonical-Address Companion State

## 1. Metainfo

- Status: completed; discarded at the single registered endpoint
- Task: directional MQAR, sequence length 1024, four future and four past queries
- Carrier: D128/L2/H4/K32/V32 pinned-official GDN2 plus native FutureSeed
- Fixed endpoint: 10 epochs, batch32, seed123, contemporaneous control then candidate
- Decision target: preserve value ownership before any hard-Sudoku transfer

## 2. Mechanism Hypothesis

P-GDN3-047 is the strongest recent mechanistic signal. Its semantic certificate
state raises balanced accuracy `.17475->.48300` and reduces errors `3301->2068`,
but 1,973 of 2,068 remaining errors are valid values assigned to another key.
The model therefore knows which values exist but lacks a stable ownership index.
P-GDN3-048 then shows that encoding ownership directly into the native payload
damages the parent retrieval path.

The falsifiable hypothesis is that ownership needs an independent address domain,
not another payload code or modification of the native state. A smaller companion
state can preserve the complete native value while learning a shared canonical
address. If this is the missing organization, it must improve both directions,
joint exact, total errors and wrong-key fraction. Merely recovering more legal
values while leaving the swap fraction high is a failure.

This is not P025's compressed committed-edit bank, P038's content router, P046's
residual added to the main address, or P047's stable-token payload under the same
native address. The primary state remains unchanged and the companion changes
only the organization of address ownership.

## 3. Exact Intervention

For each layer, compute the unchanged native tensors and parent scan:

```text
(o_main, S_main) = GDN2(q, k, v, g, b, w, S_in)
```

One learned matrix `P in R^(16x32)` is shared across layers and heads:

```text
a = normalize(P(normalize(q + k)))
g_aux = mean_K(g), broadcast to K16
b_aux = mean_K(b), broadcast to K16
(o_aux, C) = GDN2(a, a, v, g_aux, b_aux, w, 0)
o = o_main + tanh(r_layer,head) * normalize_V(o_aux)
```

The original output norm/projection consumes `o`. The projection contributes
512 parameters and two four-head zero-initialized read gates contribute eight,
for exactly 520 new parameters. Each layer adds one transient H4xK16xV32 state,
2,048 values, and one official scan. The companion state is rebuilt per layer;
native FutureSeed transports only the unchanged main state.

No token labels, lag rules, cache selector, task rule, search, repair, oracle,
extra target supervision or recurrence fallback is introduced.

## 4. Independent Data Contract

- frozen directional-MQAR L1024 train/test generator and hashes;
- train/test examples `10000/1000`, four key/value pairs, seed123;
- one serialized native initialization loaded into both arms;
- identical warmup batch, data order, optimizer and scheduler;
- run order: native FutureSeed control, then P049 candidate.

## 5. Integrity Contract

Before formal training require the sole visible registered GPU, clean detached
pushed/read-back source, pinned FLA and Zoology hashes, Triton short convolution
and no fallback. Candidate parameters must be `662,104` versus `661,584` native.
Main state remains 4,096 values/layer; companion state is exactly 2,048 values;
there are two official scans/layer and four `ChunkGDN2FunctionBackward` nodes.

At zero read gates require bit-exact full logits and, with finite nonzero incoming
main state, bit-exact layer output and main terminal state. Parent tensors and
parent-gradient topology must match native, calibrated against a byte-identical
native replay. All eight read gates must receive finite nonzero gradient. After
opening the gates, the shared projection must receive finite nonzero gradient,
rank must be 16, companion output/state must vary across tokens and boards, token
order must affect state, and simultaneous head permutation must commute within
`3e-3` BF16 tolerance.

## 6. Fixed Falsifiers And Gate

Activation requires both layers, all eight read gates, the shared rank-16
projection, finite nonzero canonical-address/output/state RMS, token and board
variation, nonzero decay/erase, and native FutureSeed.

All quality checks are conjunctive:

- balanced accuracy `>=0.55` and `>=control+0.10`;
- future and past accuracy each `>=0.50` and `>=control+0.08`;
- joint exact `>=0.06` and `>=control+0.04`;
- fewer total query errors;
- wrong-key valid-value swap fraction reduced by at least `0.10`.

Any miss closes projection source/map, K16 width, gate scale, mean gate reuse,
per-layer/shared projection, companion FutureSeed/transport, seed, data, LR,
loss, batch, model width/depth and duration rescue.

## 7. Cost And Scaling

The extra K16xV32 scan is the intended cost. Fixed ceilings are `<2.00x` for
elapsed, post-warm wall time and independent warmed-step time, and `<1.45x` for
peak allocated memory. The companion is linear in sequence length and has half
the key rows of the native state.

## 8. Results

Exact pushed/read-back source `3088b0da29e50315a94219bdcbb884ffea63728a`
ran from a clean detached worktree on A100-SXM4-40GB CUDA index0, UUID
`GPU-31166d8c-9fe5-d953-dc44-d0d549969ada`. The strict CUDA contract passed.
Zero-gated full output and finite nonzero incoming main state were bit exact,
four official `ChunkGDN2FunctionBackward` nodes were present, parameter/state
counts were exact, the shared projection had rank 16, all eight read gates and
the projection received finite gradients, and head permutation plus token
shuffle state dependency passed.

| metric | control | candidate | delta |
| --- | ---: | ---: | ---: |
| balanced accuracy | `.037000` | `.030750` | `-.006250` |
| future accuracy / CE | `.040000 / 4.06405` | `.039500 / 4.23220` | `-.000500 / +.16815` |
| past accuracy / CE | `.034000 / 4.08443` | `.022000 / 4.33548` | `-.012000 / +.25106` |
| joint exact | `0` | `0` | `0` |
| total query errors | `3,852` | `3,877` | `+25` |
| wrong-key valid-value swaps | `227` | `218` | `-9` |
| wrong-key fraction of errors | `.058930` | `.056229` | `-.002701` |

The mechanism is not dead. Both layers and all eight heads activate; companion
read-gate absolute means are `.013000/.010821`, the shared projection remains
rank 16 with RMS `.023395`, canonical-address token standard deviation is
`.036516/.055746`, and companion-state RMS is `.062019/.034721`. The two
companion reads are also not redundant with the main read: their cosine is
`.705997/-.378875`. None of this converts into better retrieval or ownership.

Elapsed, post-warm wall, independently warmed-step and peak-allocation ratios
are `1.1004/1.1087/1.3982/1.2777x`, so every registered cost ceiling passes.
The failure is scientific rather than a dead path, instability or runtime
problem.

One operator error occurred after the endpoint had already completed. The
score was written at `12:48:22.941 +08:00`, completion at `12:48:23.078`, and
an abort was mistakenly written at `12:48:24.283` on the concern that candidate
construction had advanced dropout RNG. The attempted exact-PGID TERM found no
process because the endpoint had naturally exited. Source inspection confirms
that `run_arm` calls `set_determinism(config.seed)` after warmup and immediately
before `Trainer`, so model-construction RNG cannot contaminate formal training.
The original `abort.json` is preserved and `integrity_resolution.json` records
the retraction; no rerun is authorized.

Artifacts:

- run: `/huyang2/double-loop/runs/p-gdn3-049-canonical-address-companion-l1024-20260815T043614Z-3088b0d`;
- score/comparison SHA256: `c31adfaa8d7ad2e76a31f18a59700eb599065ef67a5416b58a974bdf80b18f1c`;
- contract SHA256: `bdc195768f3747d424f2c85f839c36bbde8f2e3e7ff0038f135627f2d42a6550`;
- control/candidate checkpoint SHA256: `b9cdf2140e64ffae95d6f82cc098b39a2adcc2e665958ac5beefe0237316c9ba` / `eb0584fa5400e6148abbd8692d307e48f3ff32a1105d39d46c3061c912840720`;
- formal log/source snapshot SHA256: `f2c51b45f015d1c93db3fb0be5f6f61b40ac9ccc11f4718e92f97e877f9df780` / `a194879bd60e4a9f15d38787b676d6759e788a50d442d90ecfbec179a2663c05`;
- matched initialization SHA256: `7402e46c48cbd47070d65262d1a1b62a32ac55a4d644a6fb9644b96b0914850f`;
- integrity resolution SHA256: `7a90614884dbd5ce2e339127c162ab5434af3bc2bb1b9e26714a43ef9166984d`.

## 9. Decision

Discard P-GDN3-049. A smaller canonical address state can be fully active,
bounded and affordable while failing to improve either value retrieval or
key ownership. This closes projection source/map, K16, gate scale, mean-gate
reuse, per-layer/shared projection, companion transport and all training
rescues. There is no Sudoku transfer. A successor must change how individual
committed key-value edits occupy live memory, not add another dense address
bank around the same superposition problem.
