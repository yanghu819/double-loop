# P-GDN3-072 / P-FS2-019 Receiver-Live Boundary Recommit on Sudoku

## 1. Research Question

Can a receiving GDN2 layer retain useful FutureSeed evidence through its live
token scan by recommitting only the inherited component that the live state has
lost at the native 64-token chunk boundary?

## 2. Evidence And Hypothesis

The canonical official-FLA GDN2 plus native FutureSeed D192/L10 endpoint reaches
official 51-55/56-60/61-64 loop5 exact `.4492/.1543/.2637` and mixed exact
`.3379`. No-FutureSeed controls remain far below it, so cross-layer future state
is causal. Content codecs, basis transports, sparse replay, terminal readout,
pre-scan state feedback, parallel experts, terminal consolidation, and
first-order Momentum transfer have not improved hard Sudoku.

Native FutureSeed is consumed once, as the initial recurrent state before token
1. The receiver then applies 81 live edits. The untested hypothesis is that the
tail loses a useful direction from the inherited whole-board summary even when
that information was present at the layer entrance. Recommitting only the
missing inherited component at an existing chunk boundary should improve tail
closure without replacing the receiver's accumulated state.

## 3. Fixed Mechanism

For each receiving layer, let `F` be its exact normalized/gated initial
FutureSeed state and `S64` its live state after the first official 64-token
chunk. Per board and head:

```text
missing = F - proj_S64(F)
bounded = RMS(S64) * missing / max(RMS(missing), eps)
S64'    = S64 + tanh(alpha_layer,head) * bounded
```

`alpha` starts at zero. The unchanged pinned official `chunk_gdn2` then scans
tokens 65-81 from `S64'`. Layer 0 has no FutureSeed and uses the parent path.
The candidate adds exactly `10*6=60` scalars, no persistent state, no token,
no extra scanned token, and no task-specific operation. Projection, short
convolution, decay, erase, write, readout, channel mixing, loop feedback, loss,
and dataset remain unchanged. Boundary64 is fixed because it is the official
kernel's native chunk boundary; no boundary table is allowed.

This differs from the rejected orthogonal chunk-state controller: that method
rotated the receiver's live state from token outputs under position-QK. This
method uses the original inherited FutureSeed explicitly and only restores its
live-state-orthogonal component on the canonical content-addressed GDN2 parent.
It also differs from the P-FS2-006 hand-replayed diagnostic, which never reached
a quality verdict because its recurrence missed official-kernel parity.

## 4. Matched Sudoku Protocol

- Parent: `/huyang2/double-loop/models/gdn2-futureseed-clean-scale-s12000-final-20260802T163724Z-5889462/checkpoints/train_state_step012000.pt`.
- Parent source: `5889462cb9234ee8632a2dcb3c0690dee2f82e0d`.
- Model: official-FLA GDN2 D192/L10/H6/K32/V32, native terminal FutureSeed,
  loop5 equal CE, BF16, effective batch128, seed52.
- Data: unchanged official/full-diversity 9x9 Sudoku; append exactly 100
  `51-64` training steps after the complete parent curriculum.
- Arms, sequential on the same GPU: terminal control, then boundary-recommit
  candidate. Both exact-resume the same parent optimizer/RNG/data state.
- Evaluation: fixed mixed probes, official 51-55/56-60/61-64 loop1-5
  exact/blank/wrong-cell metrics, and same-board trajectories.

## 5. Integrity And Activation Gates

Before the matched run, a strict CUDA contract must prove the requested single
GPU UUID/index, pinned FLA SHA, official `GatedDeltaNet2` and
`ChunkGDN2FunctionBackward` provenance, exact parameter delta60, zero-gate
parent output/terminal parity including nonzero incoming state, finite nonzero
gradients in all receiving recommit scalars, boundary dependence, head
equivariance, and absence of fallback. A one-step exact-resume production probe
must preserve optimizer/RNG/data order and write complete metrics/checkpoint.

Formal activation requires exactly nine receiving paths, mean absolute gate and
state-residual relative RMS each at least `1e-4`, finite nonzero missing-state
fraction and board variation, boundary state norm ratio at most `2`, and
terminal-state RMS at most `4x` the matched control.

## 6. Quality And Cost Gates

Primary quality:

- hard51-64 macro loop5 exact at least `control + .015`;
- each official hard-range blank accuracy regresses by at most `.01`.

Alternate quality:

- mixed loop5 exact at least `control + .02`;
- 61-64 exact does not regress;
- same-board loop3-to-loop5 wrong-cell correction is stronger.

Independent post-warm elapsed overhead must be below `35%`; peak allocated CUDA
memory overhead must be below `10%`. Any integrity, activation, stability,
quality, or cost miss discards this exact mechanism. Do not rescue the boundary,
projection, gate scale/sharing, seed, LR, loss, batch, width, depth, or duration.

## 7. Next Decision

A complete pass authorizes a longer Sudoku trajectory and a fused single-call
kernel implementation. A miss closes coarse receiver-live recommit and returns
the next decision to a genuinely different scalable FutureSeed or GDN3 state
organization. No MQAR, Maze, language carrier, rule, search, repair, selector,
or low-information filler run is authorized.

## 8. Result

Overall decision: **closed: primary quality pass, allocation cost miss**. The
mechanism is retained as positive scientific evidence but is not promoted as a
formal endpoint or production winner.

The exact pushed source was
`c58ff80df1d79e2e6d1a9ef8660564885ee7b072`. The strict R2 CUDA contract passed
on A100 80GB index0, UUID
`GPU-0da20a4f-5e67-e47d-7aab-8c6efa2864ad`, with pinned FLA source
`9c8e42e762fce087c27b673af4922795d9edb85e`. It proved ten official
`GatedDeltaNet2` layers and `ChunkGDN2FunctionBackward`, exact +60 parameters,
zero-gate output and terminal-state identity including nonzero incoming state,
finite nonzero gradients for all ten scalar sets, exactly nine receiving
paths, head permutation error zero, inherited/live-state dependence,
orthogonality error `2.31e-9`, and opened boundary norm ratio `1.00495`.

The exact-resume step12001 probe passed without being used as a quality score.
After one step, exactly nine paths were enabled, gate/residual relative RMS was
`.001350`, missing inherited fraction `.806911`, boundary norm ratio
`1.000001`, and terminal RMS `3.335536`. The formal control and candidate then
ran sequentially from the same parent optimizer/RNG/data state to step12100.
Both exited status0 with complete metrics, checkpoint, config and source
snapshot hashes. No NaN, OOM, fallback, source drift, data drift or GPU drift
occurred.

### Official full-diversity result

Each cell is `full-board exact / blank accuracy / mean wrong blank cells` on
the same 512 boards. The wrong-cell denominator is reconstructed exactly from
the fixed board bank (`54.6152`, `56.8848`, and `63.9902` mean blanks).

| Range | Arm | loop1 | loop2 | loop3 | loop4 | loop5 |
|---|---|---|---|---|---|---|
| 51-55 | control | `.0000/.6185/20.84` | `.0469/.7296/14.77` | `.3301/.7878/11.59` | `.4199/.7979/11.04` | `.4316/.8002/10.91` |
| 51-55 | candidate | `.0000/.6205/20.73` | `.0527/.7359/14.42` | `.3672/.7915/11.38` | `.4512/.8015/10.84` | `.4629/.8035/10.73` |
| 56-60 | control | `.0000/.5290/26.79` | `.0137/.6052/22.46` | `.1289/.6389/20.54` | `.1855/.6485/20.00` | `.1973/.6500/19.91` |
| 56-60 | candidate | `.0000/.5310/26.68` | `.0215/.6084/22.28` | `.1387/.6434/20.29` | `.2031/.6523/19.78` | `.2109/.6540/19.68` |
| 61-64 | control | `.0000/.5736/27.29` | `.0000/.7327/17.11` | `.0820/.8280/11.00` | `.2500/.8584/9.06` | `.2891/.8645/8.67` |
| 61-64 | candidate | `.0000/.5709/27.46` | `.0000/.7305/17.25` | `.1016/.8290/10.94` | `.2637/.8598/8.97` | `.3047/.8650/8.64` |

At loop5 the official exact deltas are `+.031250/+.013672/+.015625` and the
blank deltas are `+.003290/+.003983/+.000488`. Hard51-64 macro exact is
`.305990` for control and `.326172` for candidate, delta `+.020182`; this
passes the registered `+.015` primary quality gate and every range stays
inside the blank regression bound. Loop3-to-loop5 mean wrong-cell correction
is slightly weaker in all three ranges (`.676/.633/2.336` control versus
`.652/.604/2.307` candidate), so the result is an earlier-closure improvement,
not evidence for stronger late-loop dynamics.

### Mixed result, optimization and activation

| Arm | loop1 exact/blank | loop2 | loop3 | loop4 | loop5 |
|---|---|---|---|---|---|
| control | `.0234/.5758` | `.0293/.6713` | `.2480/.7186` | `.3125/.7311` | `.3262/.7332` |
| candidate | `.0234/.5785` | `.0410/.6705` | `.2656/.7166` | `.3203/.7263` | `.3340/.7283` |

Mixed loop5 exact improves only `+.007812`, below the alternate `+.02` floor,
and mixed blank accuracy falls `.004895`. Control versus candidate training CE
is `.612266->.603699`; total loss is `.698357->.689683`. Parameter count is
`5,461,688->5,461,748`.

Candidate loop5 activation is exactly nine receiving paths, gate absolute mean
and residual relative RMS `.007730`, head standard deviation `.008033`,
FutureSeed/live-state cosine `.257475`, missing fraction `.799709` with board
standard deviation `.027927`, boundary norm ratio `1.000142`, terminal RMS
`3.069424`, terminal board standard deviation `.679353`, and scalar weight RMS
`.009228`. All activation and stability gates pass.

Control/candidate training time is `783.176/1006.752` seconds, or
`16.344/12.714` effective boards/s. Elapsed overhead is `+28.547%` and passes
the `<35%` gate. Peak allocated memory is `8095.40/8977.40 MiB`, overhead
`+10.895%`, which strictly fails the `<10%` cost gate. Reserved memory overhead
is `+9.408%` for reference. Five-second GPU samples during the resident phases
show mean utilization `19.98%/22.46%`, busy-sample mean `24.70%/24.34%`, and
observed device-memory peaks `11773/12583 MiB`; the small D192, length81
workload is launch- and evaluation-heavy on this A100.

### Same-board evidence

Both arms export identical 256-board case banks for each range. Their data
hashes are:

- 51-55: `8af125d2f365ef7b62b14724fe5cb885df2de855130a25c6a5adbc6ffb6507ac`;
- 56-60: `a362d2adddaf9297865b76aa064670ab87ad8a1a009f89a41c06d1612031d054`;
- 61-64: `bed22b6b6916569f62f521a5493150237b7c270e3b9ef95c1449cb54f5629cb3`.

The corresponding `index.html`, selected `cases.json`, and complete
`all_cases.json` live under each run's `output/case_bank/official_b*/`
directory. This is genuine same-board loop1-5 evidence, not an aggregate probe
or a hardest-board claim.

## 9. Gate Decision And Provenance

- **Integrity:** pass after an unchanged R2 contract rerun. R1 retained one
  BF16 output-quantum mismatch (`.0009765625`) at layer7 while terminal state
  stayed exact; no tolerance was relaxed. R2 contract log SHA256 is
  `8fbe522324840db278378fa03ed3f9737123f102d80e2c8479d01f1a368920e0`.
- **Activation/stability:** pass.
- **Primary quality:** pass (`+.020182` macro exact; all blank deltas positive).
- **Alternate quality:** fail (`+.007812` mixed exact and weaker late correction).
- **Elapsed cost:** pass (`+28.547% <35%`).
- **Allocation cost:** fail (`+10.895%`, not `<10%`).
- **Final:** close this exact split-chunk implementation. Do not reinterpret it
  as a production win and do not rescue it. Preserve the positive mechanism
  result: native FutureSeed is useful not only at receiver entry; a small,
  bounded recommit of the inherited component lost by the live scan improves
  exact hard-Sudoku closure across all three official ranges.

Formal artifact hashes:

- control metrics/checkpoint/config/source snapshot:
  `2111517d...785e`, `45514f1f...9b43`, `92c77484...b1fb`,
  `45c41c50...6ee3`;
- candidate metrics/checkpoint/config/source snapshot:
  `7ae2dd7d...69ff`, `f1a89f11...c15d`, `4e69e383...3b0f`,
  `45c41c50...6ee3`;
- sequence log/GPU CSV/manifest/complete:
  `9388380f...678d`, `d3f192a9...edfc`, `ec29aaec...b504`,
  `87f4296d...b338`.

The full hashes and structured gate result are stored in
`gdn3-boundary-recommit-sudoku-20260818-comparison.json`. Pre-science
orchestration aborts are retained separately: cached-FLA provenance check,
two unused-backend environment failures, sparse-worktree source snapshot, a
post-validator JSON-path error after the successful probe, and the R1 BF16
contract mismatch. None changed model settings or supplied a quality result.
