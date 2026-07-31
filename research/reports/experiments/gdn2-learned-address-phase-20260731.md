# GDN2 Learned Address-Specific Phase Field

## 1. Metainfo

- Plan: `P-ABIND-002`
- Status: discarded at the preregistered step9100 gate
- Approved by user: 2026-07-31
- Machine: AIStation GPU1, one A800 80GB
- Source branch: `codex/gdn2-address-operator-20260731`
- Parent checkpoint source: `42102bd65a28d60bde6b09ef93343692740582a1`

## 2. Hypothesis

`P-ABIND-001` showed that one scalar per head multiplying a fixed pairing of
anchor channels is active but harmful. The fixed map imposes an arbitrary
memory coordinate system. In contrast, pure position-Q/K is strong because
each layer's learned projections can assign every stable address a rich key.

The minimal generic repair is to learn the address-to-coordinate map while
retaining semantic content Q/K. Each stable input anchor should generate a
different rotation phase for every head and paired state plane. This directly
tests whether GDN2 needs a learned algebraic binding between token identity and
the key axes of its recurrent state.

## 3. Mechanism And Configuration

For every layer, token, head, and paired key plane:

```text
anchor = fixed token embedding + position embedding
theta(anchor) = pi * tanh(W_theta * LN(anchor))
q_bound = rotate(q_content, theta(anchor))
k_bound = rotate(k_content, theta(anchor))
```

`W_theta: D -> H*(K/2)` has no bias and initializes to exact zero. Thus the
candidate starts as normal official GDN2, while its first backward pass can
learn a nonzero address field. The same phase rotates Q and K, rotations are
orthogonal, and V/decay/erase/write/output, recurrent-state size, rank-1
update, FutureSeed, and the official FLA GDN2 kernel are unchanged.

At D192/H6/K32 this adds 18,432 parameters per layer, 184,320 total (about
3.4%). It adds no recurrent state. The mechanism is not Sudoku-specific: in a
language model the stable anchor is still the ordinary token-plus-position
input representation.

Formal configuration is the exact `P-ADDR-002` step9000 parent, D192/L10/H6,
five loops, random traversal, all-loop CE, official Sudoku data, batch32 x
grad-accum4, and strict official FLA SHA `9c8e42e7`. Only
`GDN2_ADDRESS_MODE=anchor_phase` changes.

## 4. Environment Contract

- GPU1 only: `CUDA_VISIBLE_DEVICES=0`, current exact GPU UUID required.
- No GPU2 and no CPU model smoke.
- `FLA_DISABLE_BACKEND_DISPATCH=1`, `FLA_CONV_BACKEND=triton`.
- Fail closed on source SHA, official class/kernel, short-conv backend,
  checkpoint hash, source cleanliness, or GPU UUID mismatch.
- Cache, environment, runs, logs, models, checkpoints, and artifacts remain
  below `/huyang2/double-loop`.

## 5. Prediction, Budget, And Kill Criteria

CUDA validation must first prove exact zero-init output/state/base-gradient
identity to normal official GDN2, nonzero first-step `W_theta` gradient,
finite active gradients through anchor/state/core, address-order consistency,
and `ChunkGDN2FunctionBackward` in the graph.

Then run one smoke and one 100-step continuation from step9000 to step9100.

- Primary signal: mean official 51-64 loop5 blank accuracy beats matched normal
  GDN2 by at least `+0.10`, train CE is at least `0.20` lower, later loops
  improve rather than regress, and runtime overhead is at most 20%.
- Strong signal: mean hard blank accuracy at least `0.40`, approaching the
  pure position-Q/K step9100 diagnostic `0.4333` without discarding content Q/K.
- Mechanism activity: phase-projection RMS, tokenwise phase variation,
  planewise phase variation, and Q/K relative changes are finite/nonzero.
- Kill at step9100 if mean delta versus normal is below `+0.05`, CE remains
  above `1.70`, the phase field collapses, or loops provide no correction.
- Only a primary pass authorizes an unchanged continuation to step9300.

One candidate only. No rank, angle, scale, seed, LR, loss, hidden-size, or
phase-parameterization table. Budget: roughly 30 minutes on GPU1.

Success supports a generic claim: learned address-specific orthogonal binding
can give matrix-state recurrent models stable memory coordinates while
preserving semantic keys and the original efficient recurrence. Sudoku remains
only a hard proxy; language transfer would require a separate positive test.

## 6. Artifacts

- Formal run: `gdn2-address-anchor_phase-s9100-20260731T121300Z-1107ecd`
- Remote run directory:
  `/huyang2/double-loop/runs/gdn2-address-anchor_phase-s9100-20260731T121300Z-1107ecd`
- Local mirror:
  `.codex-transfer/localrepo/runs/gdn2-address-anchor_phase-s9100-20260731T121300Z-1107ecd`
- CUDA contract log:
  `/huyang2/double-loop/artifacts/launch/gdn2-anchor-phase-cuda-check-1107ecd.log`
- Exact clean source: `1107ecd418173460d9cfe2cc0e36b6da2f8addb0`.
- The run archive contains config, score, metadata, full logs, source SHA/patch/
  snapshot, checkpoint evaluation, all official case banks, JSON/Markdown,
  summary HTML, and per-hard-case loop visualizations.

## 7. Results

All fail-closed CUDA checks passed on GPU1 against pinned official FLA
`9c8e42e7`:

- zero-init output, recurrent terminal state, and compared base gradients had
  max absolute error `0.0` versus normal official GDN2, both with and without
  an initial state;
- the zero-init phase weight already had a nonzero first-step gradient
  (`86.0`/`98.5` max absolute in the two checks);
- the active candidate retained `ChunkGDN2FunctionBackward`, propagated finite
  nonzero gradients to input, stable anchor, initial state, and phase weight;
- address reorder error was `0.0`; orthogonal rotation norm error was
  `4.77e-7`.

| Step9100 metric | Normal GDN2 | Learned phase | Delta |
|---|---:|---:|---:|
| train CE | 1.9370 | 1.9560 | +0.0190 worse |
| 51-55 loop5 blank acc | 0.2120 | 0.1797 | -0.0323 |
| 56-60 loop5 blank acc | 0.2119 | 0.1734 | -0.0386 |
| 61-64 loop5 blank acc | 0.1871 | 0.1556 | -0.0315 |
| mean hard blank acc | 0.2037 | 0.1696 | -0.0341 |
| train elapsed | 649.9 s | 770.2 s | +18.5% |
| parameters | 5,461,688 | 5,646,008 | +184,320 |

The field learned strongly rather than collapsing. By step9100, phase-weight
RMS was `0.01173`, mean absolute phase `1.0197` radians, tokenwise phase std
`0.7925`, and planewise phase std `1.1630`. Q/K relative changes reached
`1.0026/1.0931`: roughly one entire vector norm. Yet official loop1 to loop5
blank accuracy changed only `0.1801->0.1797`, `0.1738->0.1734`, and
`0.1571->0.1556`; every range moved slightly backward and exact stayed zero.

All four arm case banks have the same data hash. On mechanically selected
64-blank batch 11, wrong cells changed normal `52->56`, pure position-Q/K
`38->39`, fixed scalar rotation `56->54`, and learned phase `57->59` from
loop1 to loop5. The archived case HTML shows broad, unstable errors rather
than address-specific correction.

## 8. Decision And Lessons

Discard and do not continue to step9300. It misses both quality gates and loop
correction despite an active, diverse phase field. Runtime stays within the
20% systems ceiling, so systems cost is not the primary failure.

The important lesson is not to tune the `pi` multiplier. Rotation forces every
address edit to preserve the content Q/K norm and changes cross-token geometry
through angles. The optimizer immediately uses that freedom to rotate roughly
one full vector norm, but the resulting coordinates are not useful. Fixed and
learned rotations have now both failed, so scale/angle/rank/seed/LR/loss
sweeps are closed.

Pure position-Q/K remains much stronger because learned projections can create
direct Euclidean address directions. The next minimal falsifier therefore
adds one shared zero-init address residual directly to both content Q and K
before official in-kernel normalization:

```text
address_vector = W_address * LN(stable_input_anchor)
q_bound = q_content + address_vector
k_bound = k_content + address_vector
```

This gives read and write the same stable coordinate, lets official Q/K
normalization control magnitude, starts as exact GDN2, and remains generic to
language. It is a new parameterization test, not a phase hyperparameter sweep.

## 9. Submission

Not applicable.
