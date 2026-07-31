# GDN2 Generic Address Operator Probe

## 1. Metainfo

- Plan: `P-ABIND-001`
- Status: discarded at the preregistered step9100 gate
- Approved by user: 2026-07-31
- Machine: AIStation GPU1, one A800 80GB
- Source branch: `codex/gdn2-address-operator-20260731`
- Parent: `52edac2e13778777505736713e4845bf58748623`

## 2. Hypothesis

GDN2 compresses writes into a matrix state through key-value outer products,
but an additive input position embedding is not algebraically bound to the key
that selects the state direction. Similar content keys can therefore collide,
and the same token identity can drift to different state directions as hidden
content changes across layers and loops.

Pure position-Q/K proved that stable addresses matter on randomized hard
Sudoku, but it discards semantic addressing and is not a satisfactory language
algorithm. A generic alternative is to retain content-derived Q/K and bind a
stable input anchor into their geometry with a norm-preserving rotation before
the unchanged GDN2 recurrence.

## 3. Mechanism And Configuration

For each token and head:

```text
anchor = fixed input identity embedding
q_content, k_content = official GDN2 Q/K branches on evolving hidden state
phase = tanh(per_head_scale) * phase_from_paired_anchor_channels(anchor)
q_bound = rotate(q_content, phase)
k_bound = rotate(k_content, phase)
```

The same `k_bound` is consumed by the official erase/write recurrence and
`q_bound` reads the resulting state. Rotation is implemented as independent
2D orthogonal transforms, so it preserves Q/K norms and does not change state
size or rank-1 updates. Per-head scales initialize to exactly zero, making the
initial candidate numerically identical to normal GDN2. The stable anchor is
the ordinary input token embedding plus positional embedding, fixed across
loops; no row, column, box, Sudoku rule, or explicit memory slot is added.

Formal candidate:

- Backbone: strict official FLA GDN2 SHA `9c8e42e7`.
- Parent checkpoint: exact clean step9000 GDN2+native-FutureSeed checkpoint.
- D192/L10/H6/D32, five loops, all-loop CE, batch32 x grad-accum4.
- Official Sudoku train/test arrays and random cell traversal.
- Curriculum, optimizer, LR, seed, FutureSeed, loop update, and eval banks
  match `P-ADDR-002`.
- Candidate mode only: `GDN2_ADDRESS_MODE=anchor_rotary`.

## 4. Environment Contract

- GPU1 only: `CUDA_VISIBLE_DEVICES=0`, exact current GPU UUID required.
- No GPU2 and no CPU model smoke.
- `FLA_DISABLE_BACKEND_DISPATCH=1`, `FLA_CONV_BACKEND=triton`.
- Fail closed if the official class, source SHA, short-conv backend, chunk
  kernel, checkpoint hash, source cleanliness, or GPU UUID does not match.
- All caches, checkpoints, logs, models, runs, and artifacts stay under
  `/huyang2/double-loop`.

## 5. Prediction, Budget, And Kill Criteria

The first gate continues the exact step9000 parent for 100 optimizer steps to
step9100. Existing matched evidence supplies the normal-GDN2 step9100 control
and pure position-Q/K diagnostic ceiling on identical official case banks.

- Primary signal: candidate mean official 51-64 loop5 blank accuracy exceeds
  normal GDN2 by at least `+0.10`, train CE improves by at least `0.20`, and
  later loops remove more errors than loop1.
- Strong signal: candidate mean hard blank is at least `0.40`, approaching the
  pure position-Q/K `0.4333` result without discarding content Q/K.
- Mechanism activity: learned rotation scale and Q/K relative change must be
  nonzero and finite while the official recurrence path remains unchanged.
- Kill at step9100: mean hard blank delta below `+0.05`, or CE remains above
  `1.70` with no useful loop correction, or the learned operator remains
  effectively inactive.
- Only if the primary signal passes may the same candidate continue unchanged
  to step9300. No scale/rank/angle/LR/loss/seed/address-mode table.
- Budget: about 25-35 minutes for implementation CUDA validation plus the
  100-step formal gate; one additional 200-step continuation only on signal.

Success supports the claim that stable address binding can improve a GDN2
carrier without replacing semantic keys or modifying the recurrent kernel. It
does not yet support language improvement; a positive result authorizes one
language-style associative-retrieval falsifier before any large LM run.

## 6. Artifacts

- Formal run: `gdn2-address-anchor_rotary-s9100-20260731T113037Z-f6028dc`
- Remote run directory:
  `/huyang2/double-loop/runs/gdn2-address-anchor_rotary-s9100-20260731T113037Z-f6028dc`
- CUDA contract log:
  `/huyang2/double-loop/artifacts/launch/gdn2-anchor-rotary-cuda-check-f6028dc.log`
- Run archive includes `config.json`, `score.json`, `metadata.json`, full log,
  source SHA/patch/snapshot, checkpoint evaluation, case banks, JSON/Markdown,
  and HTML visualization.
- Exact source: `f6028dcd6d05d81111539e9d199ae5794249ae6c`, clean.

## 7. Results

The CUDA contract passed before training. At zero initialization the candidate
matched official GDN2 exactly with and without an initial recurrent state:
output, terminal state, and compared gradients all had max absolute error
`0.0`. With a nonzero operator, the autograd graph retained
`ChunkGDN2FunctionBackward`, the rotation-scale and anchor gradients were
finite/nonzero, and pairwise rotation preserved vector norms to float
tolerance.

| Step9100 metric | Normal GDN2 control | Anchor rotary | Delta |
|---|---:|---:|---:|
| train CE | 1.9370 | 2.0218 | +0.0848 worse |
| 51-55 loop5 blank acc | 0.2120 | 0.1849 | -0.0270 |
| 56-60 loop5 blank acc | 0.2119 | 0.1837 | -0.0283 |
| 61-64 loop5 blank acc | 0.1871 | 0.1586 | -0.0285 |
| mean hard blank acc | 0.2037 | 0.1757 | -0.0279 |
| train time | 668.8 s | 827.4 s | +23.7% |
| parameters | 5,461,688 | 5,461,748 | +60 |

The operator was active at step9100: mean absolute learned scale `0.00984`,
mean absolute phase `0.01277`, and relative Q/K changes
`0.01768/0.01639`. Activity did not become correction. Candidate loop1 to
loop5 blank accuracy fell `0.1930->0.1849`, `0.1957->0.1837`, and
`0.1644->0.1586` across the three official ranges. Exact remained zero.

The pure position-Q/K diagnostic on the same step9100 budget reached mean hard
blank accuracy `0.4333` and CE `1.2196`. Therefore the positive address signal
cannot be explained by any stable rotation; the learned projections that map
each address into a rich key space are material.

## 8. Decision And Lessons

Discard `anchor_rotary` and stop at step9100. It misses every preregistered
quality gate, exceeds the 20% runtime-overhead preference, and makes later
loops worse. Do not sweep scalar scale, angle, channel pairing, seed, LR, or
loss.

The mechanism lesson is narrower and useful: binding content Q/K to a stable
address remains promising, but a fixed anchor-channel-to-rotation map plus one
scalar per head is too rigid. It injects an arbitrary coordinate system rather
than learning which address features should identify each memory plane.

The next one-shot falsifier is a learned address-specific phase field:

```text
theta(address) = pi * tanh(W_theta * LN(stable_input_anchor))
q_bound = rotate(q_content, theta(address))
k_bound = rotate(k_content, theta(address))
```

`W_theta` is zero initialized, so the model still starts as exact official
GDN2. Unlike the failed scalar, it can learn a different binding for every
token/head/state plane. This remains generic: an address can be a token plus
position embedding in language, and the recurrent state/kernel are unchanged.

## 9. Submission

Not applicable.
