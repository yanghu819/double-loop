# P-GDN3-011: Orthogonal Head-Write Routing

## 1. Metainfo

- Status: discarded at exact step3001 production stability gate
- Date: 2026-08-07
- Planned branch: `codex/gdn3-orthogonal-head-write-20260807`
- Benchmark: official/full-diversity hard 9x9 Sudoku 51-64 blanks
- Compute: AIStation task-mode GPU1 only
- GPU UUID: `GPU-53e9f3b4-2966-65d3-6614-09c540921519`
- Seed: 52 only
- Parent: D256/L12/H8/K32/V32 position-QK GDN3 plus native terminal FutureSeed
- Parent checkpoint:
  `/huyang2/double-loop/models/gdn3-position-qk-d256l12-s12000-20260806T131527Z-9f2ee8d/checkpoints/train_state_step003000.pt`
- Parent checkpoint SHA256:
  `6339c3cb2b5fc5230a581d6633716483e35ff8e4522f06a9d7aaf26512f023da`
- Parent source SHA: `9f2ee8d1738032bc5f09b55db0b81d507780b376`
- Frozen control:
  `p-fs3-001-terminal-s3100-20260806T200859Z-3e167b6`

## 2. Evidence Boundary

P-GDN3-005 through P-GDN3-008 activate within-head gate coupling, an
independent recurrent expert, inherited-state-conditioned write residuals, and
a second post-scan transition. None improves the registered exact score.
P-GDN3-009 and P-GDN3-010 then close the two direct composition routes around
the pinned operator: externally split official calls preserve forward identity
but do not expose the required state-gradient chain, while a single expanded
physical sequence retains one official graph but changes the parent
BF16/chunk computation even when every inserted write is zero.

The original 81-token, one-call geometry must therefore remain fixed unless a
separate CUDA recurrence is justified and fully audited. One structural degree
of freedom inside that fixed geometry is still untested. Position-QK creates
eight stable per-head address namespaces, but official GDN2 updates eight
independent K32xV32 states. The block output mixes heads only after those live
states have already committed the token. No existing P005-P010 candidate lets
one token route its payload among the existing recurrent heads before that
commit.

P011 tests this missing write topology. It is not another inherited-state
residual, transfer-content map, gate scalar, extra state, extra transition,
address rotation, or Sudoku-specific rule.

## 3. Mechanism Hypothesis

Hard global closure may require evidence represented in one content head to be
stored under another head's stable position-address namespace during the same
token update. Independent heads can otherwise learn useful but siloed partial
constraints that the later output projection combines too late to affect the
current recurrent memory.

A token-dependent orthogonal routing of V payloads across heads should improve
closure if this isolation is causal. The transform must vary across boards and
tokens, preserve aggregate payload norm, leave the parent exactly unchanged at
migration, and strengthen hardest-range late-loop correction. If it activates
without an exact gain, head isolation is not the local bottleneck and the
mechanism closes without changing its plane, angle, target, or duration.

## 4. Candidate

For each layer, after the parent's position-QK projections and short
convolution, reshape V as `[B,T,H8,V32]`. One bias-free, head-shared
`Linear(V32,3)` produces three scalar descriptors per head:

1. the first two rows are initialized nonzero and define two head-axis vectors;
2. Gram-Schmidt produces an orthonormal plane `u,v` over the eight heads;
3. the third row is initialized exactly zero; its head mean gives the
   permutation-invariant angle `theta = pi*tanh(mean_h(raw_theta_h))`;
4. rotate every V-coordinate in the same head plane;
5. pass the unchanged Q/K/g/b/w and rotated V to the original single
   pinned-official `chunk_gdn2` invocation.

For each payload coordinate `x in R^H`, the rotation is

```text
x' = x + (cos(theta)-1) * (u * <u,x> + v * <v,x>)
       + sin(theta) * (v * <u,x> - u * <v,x>).
```

At migration the angle row is zero, so `theta=0`, `sin(theta)=0`, and
`cos(theta)-1=0`; V, model output, and every recurrent state are exactly the
parent values regardless of the nonzero plane descriptors. Once the angle
opens, the transform is orthogonal for every signed angle and preserves the
aggregate cross-head V Frobenius norm. The shared descriptor, head reductions,
Gram-Schmidt, and rotation commute with arbitrary head permutations.

The angle row has a first-stage gradient at exact identity. The two plane rows
must receive gradients after a synthetic angle opening. The fixed candidate
adds exactly `12 * 32 * 3 = 1,152` parameters and no recurrent-state values,
tokens, scan, second core, reverse traversal, output residual, search, or task
logic.

## 5. Frozen Configuration

- D256/L12/H8/K32/V32, channel multiplier4
- position-QK addressing and registered random traversal
- native terminal FutureSeed, loop5, equal CE at every loop
- 12 pinned official-FLA GDN2/Triton layers
- one bias-free V32-to3 orthogonal head router per layer
- one head-plane V rotation per token before the unchanged official call
- official full-diversity curriculum and exact step3000 parent
- BF16, effective batch128, optimizer/RNG/data order exact resume
- LR0.0015, weight decay0.001, seed52
- candidate-only step3000-to3100 continuation

The frozen control is not rerun. There is no plane count/source, angle map or
cap, routing target, descriptor sharing, seed, optimizer, loss, batch,
width/depth, or duration follow-up.

## 6. Launch Gates

Before formal execution, one strict GPU1 CUDA contract must prove:

1. exactly CUDA index0 and the registered UUID;
2. pinned FLA source SHA
   `9c8e42e762fce087c27b673af4922795d9edb85e`;
3. exactly 12 official GDN2 modules and 12
   `ChunkGDN2FunctionBackward` graphs, each retaining the original 81-token
   call with no fallback;
4. exact 1,152-parameter migration and no state, token, scan, or core delta;
5. bit-exact full output and all 12 terminal states at zero angle, including
   synthetic nonzero incoming states;
6. finite nonzero first-stage gradients in all 12 angle rows and, after a
   synthetic angle opening, finite nonzero gradients in all 12 plane-row pairs;
7. exact head-permutation equivariance and changed routed payload/output/state
   when payloads or learned angle sources are shuffled across boards;
8. plane unit-norm and orthogonality errors each at most `1e-5` in FP32;
9. FP32 routed/base V Frobenius-norm ratio in `[0.9999,1.0001]` and BF16 ratio
   in `[0.995,1.005]` for an opened route;
10. exact-resume optimizer/RNG/data-order migration and complete step3001
    checkpoint plus metrics JSON;
11. no NaN, OOM, source/data drift, hidden CPU model path, concurrent GPU
    process, or Sudoku/traversal-specific dependency.

Any contract or production-probe miss closes the implementation. It does not
authorize a different plane, angle reduction/cap, payload target, tolerance,
descriptor sharing, or custom fallback.

## 7. Science, Stability, And Cost Gates

At step3100, activation and stability require all of:

- exactly 12 routing paths active;
- mean absolute angle and routed-V relative RMS each finite and `>=1e-4`;
- finite nonzero board and token variation;
- FP32 and BF16 norm ratios inside the registered ranges;
- finite terminal-state RMS at every loop and no more than `4x` the frozen
  control at the matching loop;
- no fallback.

Quality passes by exactly one route:

1. hard51-64 macro loop5 exact improves by at least `+0.02`, with every hard
   range losing no more than `0.01` blank accuracy; or
2. mixed loop5 exact improves by at least `+0.03`, official61-64 does not
   regress, and same-board loop3-to5 wrong-cell correction is stronger.

The original recurrent call and state size remain unchanged. Before any result,
independently warmed elapsed and peak allocated-memory overhead are each capped
below `25%` versus the frozen control. Timing instability, OOM, or either
ceiling miss kills the run regardless of quality.

Any activation, stability, quality, or cost miss discards this mechanism. Do
not rescue plane count/source, angle map/cap, routing target, descriptor
sharing, seed, LR, loss, batch, model width/depth, or duration.

## 8. Required Readout

Report and archive mixed and official51-55/56-60/61-64 loop1-5
exact/blank/wrong cells; train CE and same-board correction; angle, route,
plane and norm diagnostics; terminal-state geometry; independently warmed
throughput, peak allocation/reservation and timing; config/source/parent/
checkpoint/metrics/log hashes; and same-board loop1-5 visualization.

## 9. Decision

Discard before formal continuation. Exact source
`d093d21a3947e4ae891f63f9469ce7a841ba37d0` was pushed/read back and used from
a clean detached worktree. The strict R1 GPU1 CUDA contract passed: all 12
zero-angle outputs and terminal states were parent-exact, including nonzero
incoming states; all 12 official-GDN2 backward paths and two-stage angle/plane
gradients were present; parameter delta was exactly 1,152; and the opened
synthetic route passed head permutation, plane, FP32 and BF16 norm checks.

The exact-resume step3001 production probe completed status0 and activated all
12 routes. At loop5, angle abs/batch std/token std was
`0.059469/0.001724/0.006410`, routed-V relative RMS was `0.037769`, plane dot
and norm errors were only `1.40e-6/2.38e-7`, and terminal RMS was `6.545335`.
The production path nevertheless violated the preregistered FP32 norm gate:
loop5 routed/base max error was `5.1444e-4`, versus the fixed maximum `1e-4`
(worst loop `5.1445e-4`). Inspection shows that `.float()` tensors still pass
through `F.linear` and rotation operations inside an active CUDA autocast
region. The synthetic direct-route checker ran outside that region and did not
exercise this precision boundary.

The probe has no NaN, OOM or fallback, its process exited naturally, and GPU1
returned to 0 MiB. Contract/probe/metrics/checkpoint/abort SHA256 are
`da071473...d6a`, `598ba3be...3e6`, `290b4ca4...ef8`,
`5998f1b...fbd`, and `5c470fea...de1`. Because the preregistration states that
any production-probe miss closes the implementation, changing autocast scope,
precision, plane, angle, target, tolerance or training settings is not
authorized. No step3000-to3100 formal continuation, matched score or
visualization exists.
