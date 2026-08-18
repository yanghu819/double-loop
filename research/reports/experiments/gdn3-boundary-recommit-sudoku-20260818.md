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

Pending strict implementation contract and the single matched Sudoku decision.
