# FutureSeed2 Compatible Multihop Readout

## 1. Metainfo

- Plan: `P-FS2-005`
- Machine: AIStation GPU1, one A800 80GB
- Branch: `codex/futureseed2-multihop-readout-20260729`
- Status: completed, hypothesis rejected

## 2. Mechanism Hypothesis

FutureSeed1 passes an adjacent layer's terminal recurrent state directly as
the next layer's initial state. Prior experiments establish two boundaries:

- fixed or learned raw-state averaging over more layers is too blunt;
- rotating a state into another layer's coordinates damages late correction.

The remaining high-information hypothesis is that older terminal state still
contains useful future evidence, but must be read in the producer layer's own
coordinate system before it crosses more depth.

For destination layer `l`, the accepted FutureSeed1 path remains unchanged.
In parallel, layer `l-2` reads its saved terminal state with its own pretrained
GDN2 query, gated RMSNorm, and output projection:

```text
R(l) = Output(l-2, Query(l-2, x(l)) @ State(l-2))
x(l) = x(l) + alpha(l) * R(l)
```

`alpha` is one scalar per two-hop edge and starts at exactly zero. Therefore
the candidate is bit-exact to FutureSeed1 before optimization. This adds no
new recurrent state, query, value, output matrix, task rule, loss, or data.

## 3. Prediction

If FutureSeed's useful information extends beyond one adjacent layer, the
two-hop scale should move away from zero and improve hard exact or late-loop
correction. Unlike raw state carry, this path should not destroy the accepted
baseline because it converts the old state back to shared hidden space before
reuse.

If the scale learns but hard quality is neutral or worse, depth radius is not
the current bottleneck. If it stays at zero, the pretrained producer readout
does not expose a useful multihop residual at this continuation budget.

## 4. Budget And Kill Criteria

- Strict official FLA GDN2, D192/L10/H6/K32/V32.
- Frozen FutureSeed1 step9000 checkpoint, one seed, continuation to step9100.
- Official 512-board ranges: 51-55, 56-60, and 61-64 blanks.
- CE on every loop, loop5, identical data/RNG/optimizer/evaluator.
- GPU1 only; no CPU model smoke, GPU2, fallback, noise, repair, search,
  selector, or task-specific rule.

Integrity must pass:

1. zero-init full reasoner is bit-exact to FutureSeed1;
2. state readout is exact to its direct formula;
3. activated readout changes output;
4. the zero-init scale receives finite nonzero gradient;
5. official GDN2 class, chunk backward, Triton convolution, checkpoint
   migration, and clean detached source all pass.

Success:

- hard-range mean loop5 exact improves by at least `+0.01`, with 61-64 not
  regressing; or
- 56-60 or 61-64 improves by at least `+0.02`, with 51-55 regression no more
  than `0.01`;
- wall overhead remains below `20%`;
- later loops show net correction on matched hard cases.

Kill after this one 100-step continuation if the gates fail. Do not sweep hop
count, scale initialization, seed, LR, loss, or run length.

## 5. Publication Claim If Successful

A recurrent terminal state can supply cheap future context beyond adjacent
layers when it is queried in the producer's native memory coordinates and
returned to shared hidden space. This would extend FutureSeed from raw
one-boundary state transfer to compatible multihop memory readout without a
backward scan or quadratic attention.

## 6. Commands

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/run_futureseed2_multihop_readout_arm.sh contract
CUDA_VISIBLE_DEVICES=0 ./scripts/run_futureseed2_multihop_readout_arm.sh smoke
CUDA_VISIBLE_DEVICES=0 ./scripts/run_futureseed2_multihop_readout_arm.sh formal
```

## 7. Results

### 7.1 Integrity

The GPU1 contract passed before training:

- device: NVIDIA A800-SXM4-80GB;
- zero-init full-output maximum error: `0`;
- producer-native readout formula maximum error: `0`;
- activated readout output RMS change: `0.05837`;
- zero-init scale gradient norm: `0.12531`;
- official FLA source: `fe8fce9fc6984f22905f54cfa885dce1502baf26`;
- exact class: `fla.layers.gdn2.GatedDeltaNet2`;
- q/k/v convolution backend: Triton;
- backend dispatch/fallback: disabled.

The full D192/L10 model adds exactly eight parameters:
`5,461,688 -> 5,461,696`. A two-step GPU full-stack smoke resumed the
step9000 model, trained, checkpointed, and evaluated successfully. The new
scale moved from zero to about `0.0029` in that smoke.

### 7.2 Formal Run

- control:
  `futureseed2-identity-s9100-20260728T1450Z-57455e4`;
- candidate:
  `futureseed2-readout-s9100-20260729T051613Z-5da8fcd`;
- candidate source:
  `5da8fcda08edb6ec80ec041560dbb243d9308fc8`;
- both resume the same exact step9000 checkpoint and train to step9100.

Mixed exact by loop:

| Model | loop1 | loop2 | loop3 | loop4 | loop5 |
|---|---:|---:|---:|---:|---:|
| FutureSeed1 | 0.0234 | 0.0469 | 0.1953 | 0.2461 | 0.2520 |
| two-hop readout | 0.0234 | 0.0469 | 0.1797 | 0.2324 | 0.2363 |

Official loop5 exact:

| Blank range | FutureSeed1 | two-hop | delta |
|---|---:|---:|---:|
| 51-55 | 0.3672 | 0.3594 | -0.0078 |
| 56-60 | 0.1309 | 0.1367 | +0.0059 |
| 61-64 | 0.1973 | 0.1719 | -0.0254 |
| mean | 0.2318 | 0.2227 | -0.0091 |

The candidate learns a nonzero mean absolute scale of `0.01697`. Its readout
residual RMS by loop is `0.157/0.391/0.560/0.531/0.513`. Thus the mechanism
is active, and its largest intervention coincides with the point where the
loop curve first falls behind.

Final train CE is essentially identical (`0.6435` control versus `0.6437`
candidate), but full-board exact is worse. Candidate train time is `821.5s`
versus `659.9s`, an overhead of `24.5%`, which also fails the preregistered
`<20%` cost gate.

### 7.3 Matched Cases

- 51-55 blanks, batch 17: FutureSeed1 wrong cells
  `22 -> 9 -> 2 -> 0 -> 0`; candidate
  `22 -> 13 -> 8 -> 6 -> 5`.
- 56-60 blanks, batch 175: FutureSeed1
  `20 -> 13 -> 4 -> 1 -> 1`; candidate
  `19 -> 10 -> 0 -> 0 -> 0`.
- 61-64 blanks, batch 48: FutureSeed1
  `32 -> 12 -> 3 -> 0 -> 0`; candidate
  `32 -> 15 -> 8 -> 1 -> 0`.

This is a board-dependent operating change, not a uniformly better future
message. One medium-hard board improves, one easier hard board stops closing,
and one hardest board needs an extra loop. The aggregate hardest range
regresses.

Artifacts:

- dashboard:
  `research/reports/visualizations/futureseed2-multihop-readout-20260729/index.html`;
- machine-readable comparison:
  `research/reports/visualizations/futureseed2-multihop-readout-20260729/comparison.json`;
- remote run contract, score, logs, output, case banks, and lean source
  snapshot are under
  `/huyang2/double-loop/runs/futureseed2-readout-s9100-20260729T051613Z-5da8fcd`.

## 8. Decision

Reject compatible two-hop readout and stop this branch after one run. It
misses both quality gates, regresses 61-64 blanks, and exceeds the cost limit.
Do not sweep hop count, scale initialization, seed, LR, loss, or continuation
length.

The useful conclusion is narrower and stronger than "old state is useless":
the old terminal state can be queried exactly and can improve individual
boards, but a fixed extra cross-depth residual is not a reliable correction
direction. FutureSeed1's adjacent state transfer is already a specific useful
interface; sending state farther or retaining it across a macro step does not
automatically improve that interface.

Together with `P-FS2-004`, this closes the two user-proposed radius axes:

- raw same-layer block carry destroys iterative refinement;
- producer-compatible two-hop readout preserves the baseline much better but
  is slower and still weakens late correction.

A later FutureSeed2 should change how generic future evidence is formed or
compressed before injection. It should not merely transmit the existing
terminal state farther in depth or longer in time.
