# GDN2 Learned Address-Specific Phase Field

## 1. Metainfo

- Plan: `P-ABIND-002`
- Status: in progress
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

Pending.

## 7. Results

Pending.

## 8. Decision And Lessons

Pending.

## 9. Submission

Not applicable.
