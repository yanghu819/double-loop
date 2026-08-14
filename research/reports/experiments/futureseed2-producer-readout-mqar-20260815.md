# P-FS2-010: Producer-Native Gated Readout FutureSeed

## 1. Metainfo

- Status: discarded; strict matched endpoint complete
- First decision field: directional MQAR L1024 wrong-key binding regime
- Fixed model: D128/L2/H4/K32/V32 pinned-official GDN2 with native FutureSeed
- Fixed data: four associations, 10,000 train and 1,000 validation examples
- Fixed optimization: 10 epochs, batch32, seed123, AdamW `1e-3`, weight
  decay `0.1`, cosine schedule, query-only cross entropy
- Arms: contemporaneous native FutureSeed control followed by the sole candidate
  in one process on one GPU
- Sweep: none

## 2. Evidence And Hypothesis

The L1024 failure regime is dominated by correct-value/wrong-key binding
errors. Direct decoupled erase keys, bounded query feedback, analytic product
addresses, parallel Raven context and whole-state routing all activated but
destroyed or weakened retrieval. Those failures argue against another change
to the live GDN2 erase/write recurrence without stronger evidence.

Native FutureSeed transfers a producer terminal KxV state into the next
layer's scan, but the receiver must use that state through its independently
learned Q/output coordinates. The hypothesis is that the transferred state
already contains useful evidence that is partially hidden by this coordinate
mismatch. Reading it with the producer's own query and output interface, then
allowing the receiver to fuse that hidden-space evidence, should improve
binding without changing the native recurrence.

The falsifiable prediction is a contemporaneous gain in both future and past
retrieval, fewer total and wrong-key errors, and a material same-checkpoint
drop when the trained readout edge is disabled. Activation without those
quality changes rejects the mechanism.

## 3. Exact Intervention

The native FutureSeed initial state remains unchanged. On the sole L1-to-L2
edge, the receiving layer's normalized and gated seed `F` is decoded using the
producer layer's existing Q convolution, GDN2 state read, output norm and
output projection:

```text
q_p = normalize(producer_q_conv(producer_q_proj(h_receiver)))
r_p = producer_o_proj(producer_o_norm(q_p^T F, producer_g_proj(h_receiver)))
```

The receiver fuses this evidence with one rank-32 multiplicative feature:

```text
z = silu(W_h h_receiver) * W_r r_p
raw = W_o z
delta = 0.5 * rms(h_receiver) * tanh(raw / rms(h_receiver))
h_receiver' = h_receiver + delta
```

`W_o` is exactly zero initialized. `W_h`, `W_r` and `W_o` have no bias, so
the increment is exactly 12,288 parameters. A zero producer state produces
zero evidence and zero residual; the candidate cannot reduce to a free extra
MLP. There is no new recurrent state and no new official GDN2 scan.

## 4. Distinction From Closed Families

- P-FS2-005 used a scalar two-hop readout on a mature Sudoku graft. P-FS2-010
  is a token-conditioned low-rank fusion trained from scratch in the validated
  binding-error regime.
- P-FS3-004 transported orthogonally modified K/V state. P-FS2-010 never
  rotates or rewrites K/V state.
- P-FS2-007/008 replayed or reconstructed selected events. P-FS2-010 reads the
  entire completed producer state through its native learned interface.
- P-GDN3-041 changed live erase content inside the token recurrence.
  P-FS2-010 leaves all decay, erase, write and scan equations untouched.

## 5. Matched Protocol

1. Build one native control initialization and save its exact parameter hash.
2. Load every parent tensor into the candidate; retain only the candidate's
   new fusion tensors.
3. Require exact zero-init control/candidate logits on CUDA with nonzero native
   FutureSeed state.
4. Train control then candidate in one process with identical data, order,
   optimizer, schedule, batch and seed.
5. Evaluate the trained candidate both enabled and edge-off on the same 1,000
   validation examples. Edge-off is diagnostic only and is excluded from the
   registered model-cost wall time.

## 6. Activation And Integrity Gates

- exactly one adjacent readout edge, rank32 and +12,288 parameters;
- exactly two pinned-official `ChunkGDN2FunctionBackward` paths;
- zero-init full-model and nonzero-incoming-state identity is exact;
- staged gradients: output projection first, then hidden/read projections;
- state shuffle changes decoded evidence and residual; zero state gives exact
  zero evidence; batch permutation is equivariant;
- read RMS, feature RMS and residual relative RMS are each at least `1e-3`;
- read board/token variation is finite and nonzero;
- trained output projection RMS is at least `1e-5`;
- elementwise residual is bounded by `0.5001 * rms(hidden)`;
- native FutureSeed remains active and terminal state is finite and variable.

## 7. Quality And Cost Gates

All quality checks must pass:

- balanced accuracy `>= max(0.55, control + 0.10)`;
- future and past accuracy each `>= max(0.53, control + 0.08)`;
- joint exact `>= max(0.08, control + 0.04)`;
- total errors decrease;
- wrong-key valid-value swap fraction decreases by at least `0.05`;
- enabled balanced accuracy exceeds trained edge-off by at least `0.03`, with
  fewer enabled errors.

Registered cost ceilings are `<1.50x` for elapsed, post-warm wall and warmed
step, and `<1.35x` for peak allocation. These ceilings include the extra
producer Q convolution and full-state read but exclude the one-off edge-off
science evaluation.

## 8. Kill Criteria And Risks

Any integrity, activation, quality or cost miss closes producer-native readout.
There is no rank, cap, injection point, decoder, gate, seed, learning-rate,
loss, width, depth, duration or Sudoku rescue. The main risk is that applying
producer coordinates to receiver hidden tokens is itself a mismatched query;
the edge-off attribution gate and balanced directional scores expose that
failure directly.

## 9. Next Decision

Discard P-FS2-010. The strict contract passes at exact pushed and read-back
source SHA `eb983102de0dbd49289008af7a2eb05f1780362f`: zero-init parent error is
zero, both layers use pinned-official `ChunkGDN2FunctionBackward`, the
candidate adds exactly 12,288 parameters and no state or scan, and all staged
gradients and state-dependency checks pass.

The mechanism trains and saturates its registered residual bound, but quality
collapses. Control versus candidate balanced/future/past/joint accuracy is
`.17475/.1610/.1885/0` versus `.0280/.0365/.0195/0`; total errors rise
`3301 -> 3888`. The candidate lowers the wrong-key valid-value fraction among
errors from `.233263` to `.074331`, but this is not binding closure because it
replaces wrong-key swaps with broader invalid retrieval. The trained edge is
also not causally useful at endpoint: edge-off balanced accuracy remains
`.0280`, and edge-off has the same 3,888 total errors.

Activation explains why the path is destructive rather than dead. Producer
read RMS is `.360067`, feature RMS is `2.708361`, output-projection RMS is
`.035193`, residual relative RMS is `.385321`, and maximum residual reaches
`.499865 * rms(hidden)` against the fixed `.5` cap. The transferred-state RMS
board standard deviation is exactly zero, but this is structurally forced by
native FutureSeed's per-board RMS normalization and cannot be interpreted as
identical state content. The preregistered state-RMS variation check is
therefore invalid and should not be reused. Quality and edge-off attribution
still reject producer-native state decode plus hidden residual fusion
independently of that activation check.

Elapsed and allocation ratios pass at `.9963x` and `1.1639x`; post-warm wall
passes narrowly at `1.4811x`. Independently warmed-step time is `1.5657x` and
misses the `<1.50x` ceiling. Quality, edge attribution and warmed cost
independently reject the experiment. Close rank, residual cap,
injection point, decoder, gate and training rescues; do not transfer to
Sudoku.

## 10. Commands And Artifacts

- Formal R2 run:
  `/huyang2/double-loop/runs/p-fs2-010-producer-readout-r2-l1024-20260814T193756Z-eb98310`
- Clean detached source SHA:
  `eb983102de0dbd49289008af7a2eb05f1780362f`
- Contract JSON/log SHA256:
  `450947e797f3a3227836da19ba6e2e0bbda5e4e30a5f6b8f28a30cd0f7bcbc93` /
  `c8ebb38a540e29bcac338500abb0aad0cee0324a6ac35ebb2ef4b8e549d40f03`
- Comparison SHA256:
  `32d143471f101462027f8d856771ed5844491e0b5526c607bab5746a592b79c9`
- Control/candidate checkpoint SHA256:
  `4d96b91f9278d38bd92e6f6cf35559ad067a8d5f3ea96d5cd519c558b8a346aa` /
  `4a48612f436af4e4a2fbedbe9964821197ee0d0abd0fa22adfa5d5ca66fac2c7`
- Formal log/source snapshot SHA256:
  `5b9a9dceaa74ba5dafabc5e22d6cad06595f4629333f50c71a5ddecd4c04baa3` /
  `c0d50d352d9c12b41e5d9dbb99bcbbd23a967992f27a273d4a81fa69abf1e1e9`

R1 passed the same strict CUDA contract but stopped before model construction
because its sparse checkout omitted a transitive diagnostics module. It has a
non-science infrastructure abort. R2 expanded only the checkout path and kept
the exact source SHA, mechanism, data and gates unchanged.
