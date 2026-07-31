# GDN2 Decoupled Read/Write Address Residual

## 1. Metainfo

- Plan: `P-ABIND-004`
- Status: discarded; shared stable address remains the retained mechanism
- Approved by user: 2026-07-31
- Machine: AIStation GPU1, one A800 80GB
- Source branch: `codex/gdn2-address-operator-20260731`
- Parent checkpoint source: `42102bd65a28d60bde6b09ef93343692740582a1`

## 2. Hypothesis

The shared Euclidean address residual is the first generic mechanism here to
improve CE, all hard ranges, and later-loop correction. It still trails pure
position-Q/K. The sharp causal difference is symmetry: shared residual forces
read Q and write K to use the same address map, whereas normal attention and
the successful position-Q/K diagnostic use independent maps.

Independent read/write address maps can learn an asymmetric bilinear relation
between query identity and state-write coordinates. If that missing freedom is
the bottleneck, decoupling Q/K address projections should improve over the
shared residual without changing recurrent state or GDN2's update.

## 3. Mechanism And Configuration

```text
q_address = W_address_q * LN(stable token-plus-position anchor)
k_address = W_address_k * LN(stable token-plus-position anchor)
q_bound = q_content + q_address
k_bound = k_content + k_address
```

Both bias-free projections initialize to exact zero. Official in-kernel Q/K
normalization controls magnitude. V, decay, erase, write, output, rank-1
recurrence, recurrent state, FutureSeed, loops, and FLA kernel remain unchanged.
At D192/L10 this adds 737,280 parameters (13.5%), but no state.

This remains language-generic and contains no Sudoku structure. Formal setup
matches the exact step9000 parent, official data, random order, D192/L10/H6,
five loops, all-loop CE, batch32 x grad-accum4, seed52, and strict official FLA
SHA `9c8e42e7`. Only `GDN2_ADDRESS_MODE=anchor_qk_residual` changes.

## 4. Environment Contract

- GPU1 only; exact UUID and `CUDA_VISIBLE_DEVICES=0` required.
- No GPU2 and no CPU model smoke.
- Fail closed on official source/class/kernel/backend, checkpoint hash, source
  cleanliness, and GPU identity.
- All remote files remain below `/huyang2/double-loop`.

## 5. Prediction, Budget, And Kill Criteria

CUDA checks must prove exact zero-init official output/state/base-gradient
identity, nonzero first-step gradients for both maps, active input/anchor/state/
Q-map/K-map gradients, reorder consistency, and official chunk backward.

Run one smoke and one step9000->9100 formal gate.

- Primary: mean hard blank at least `+0.10` over normal, CE at least `0.20`
  lower, genuine loop correction, runtime overhead at most 25%.
- Mechanism improvement: beat shared residual mean `0.2844` by at least `+0.03`.
- Strong: mean hard blank at least `0.40`, approaching pure position-Q/K.
- Kill if mean is below `0.3144`, CE above `1.60`, loops fail to correct, either
  address map stays inactive, or address residuals explosively dominate content.
- Continue unchanged to step9300 only on primary plus mechanism-improvement pass.

No shared-vs-split table, rank, scalar, width, seed, LR, loss, or duration sweep.

## 6. Artifacts

- Formal run:
  `gdn2-address-anchor_qk_residual-s9100-20260731T134404Z-6469aa8`
- Remote run directory:
  `/huyang2/double-loop/runs/gdn2-address-anchor_qk_residual-s9100-20260731T134404Z-6469aa8`
- Local mirror:
  `.codex-transfer/localrepo/runs/gdn2-address-anchor_qk_residual-s9100-20260731T134404Z-6469aa8`
- GPU smoke:
  `gdn2-address-anchor_qk_residual-smoke-20260731T134032Z-6469aa8`
- Final CUDA contract log:
  `/huyang2/double-loop/artifacts/launch/gdn2-anchor-qk-cuda-final-6469aa8.log`
- Cross-process numerical diagnostic log:
  `/huyang2/double-loop/artifacts/launch/gdn2-address-cuda-final-4ba2f38.log`
- Remote visualization:
  `/huyang2/double-loop/runs/gdn2-address-anchor_qk_residual-s9100-20260731T134404Z-6469aa8/visualizations/address_binding_comparison/index.html`
- Local visualization:
  `.codex-transfer/localrepo/runs/gdn2-address-anchor_qk_residual-s9100-20260731T134404Z-6469aa8/visualizations/address_binding_comparison/index.html`
- Exact clean source: `6469aa8520f20d221946320e4f193caf7206abd6`.
- Archive includes `launch.env`, config, score, run/CUDA logs, source
  snapshot/SHA, checkpoint eval, complete case banks, comparison JSON, and
  HTML visualizations.

## 7. Results

The final CUDA contract used pinned official FLA `9c8e42e7`, GPU1 UUID
`GPU-430f57d9-8347-fb3b-4397-d7d51fcc218c`, and the official
`ChunkGDN2FunctionBackward`. Zero initialization produced exact `0.0` output
and terminal-state error with and without initial state. Both address maps had
large nonzero first-step gradients; active input, anchor, initial-state, Q-map,
and K-map gradients were finite/nonzero; token reorder error was `0.0`; and
the two maps produced different addresses.

Repeated clean processes exposed backend numerical behavior that the original
bit-exact gradient assertion hid: the same official BF16/Triton backward gave
base-gradient max differences of `0.0` and `0.0625` across launches while
output/state stayed exact. The final checker therefore reports every base
tensor and requires `atol=1e-4, rtol=1e-3`; it does not claim impossible
cross-process bit identity for atomic GPU accumulation.

| Step9100 metric | Normal GDN2 | Shared address | Split read/write | Split vs normal | Split vs shared |
|---|---:|---:|---:|---:|---:|
| train CE | 1.9370 | 1.6845 | 1.7730 | -0.1639 | +0.0886 worse |
| 51-55 loop5 blank | 0.2120 | 0.3106 | 0.2470 | +0.0351 | -0.0635 |
| 56-60 loop5 blank | 0.2119 | 0.2860 | 0.2367 | +0.0248 | -0.0493 |
| 61-64 loop5 blank | 0.1871 | 0.2565 | 0.2100 | +0.0229 | -0.0465 |
| mean hard loop5 blank | 0.2037 | 0.2844 | 0.2313 | +0.0276 | -0.0531 |
| mean loop1 -> loop5 | 0.2171 -> 0.2037 | 0.2754 -> 0.2844 | 0.2286 -> 0.2313 | - | - |
| train time | 668.8 s | 796.5 s | 1034.4 s | +54.7% | +29.9% |
| parameters | 5.462M | 5.830M | 6.199M | +13.5% | +6.3% |
| peak allocated VRAM | 7.91 GiB | 8.23 GiB | 8.37 GiB | +5.9% | +1.7% |

All hard-range exact scores remain zero. The split maps are active rather than
collapsed: final Q/K weight RMS is `0.0120/0.0116`, address RMS is
`0.5338/0.5690`, and content-relative Q/K change is `0.249/0.339`. They simply
learn a worse coordinate system than the tied map.

All four visualized arms use identical case hashes. The mechanically selected
51-55 blank batch 194 shows the sharp failure: normal wrong blanks are
`45->45->44->45->45`, shared address is `34->30->30->30->30`, split address
is `41->44->44->44->44`, and the position-only diagnostic is
`24->21->21->21->20`. On 61-64 blank batch 122, shared improves `43->39`,
while split regresses `49->52`. More split-map capacity does not turn into
better recurrent correction.

## 8. Decision And Lessons

Discard fully independent stable-address Q/K projections. It fails every
preregistered quality/system gate: mean is below `0.3144`, CE is above `1.60`,
it trails shared by `-0.0531`, loop gain is only `+0.0027`, runtime overhead is
`+54.7%`, and exact remains zero. Do not continue to step9300 and do not sweep
rank, scale, seed, LR, loss, or duration.

The useful conclusion is stronger than "position helps." In a recurrent
linear-attention memory, K defines the coordinate where an update is stored
and Q later reads that coordinate. A shared stable address gives both
operations a common namespace. Fully independent maps are legal but make the
model learn two coordinate systems and their alignment from only task loss;
the extra freedom and parameters hurt both optimization and late-loop repair.

Retain the shared Euclidean address residual from `P-ABIND-003` as the clean
generic mechanism. It is token-plus-position based, contains no Sudoku rule,
keeps the official rank-1 recurrence, and is directly transferable to sequence
tasks. The next proof should move tasks: matched associative retrieval or
language modeling should test whether the same stable namespace helps repeated
identity and long-range memory. Another Sudoku address parameterization would
be low ROI.

## 9. Submission

Not applicable.
