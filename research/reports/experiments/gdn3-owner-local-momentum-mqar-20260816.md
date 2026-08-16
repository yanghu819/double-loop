# P-GDN3-068 Owner-Local Momentum Commit

## 1. Metainfo

- Plan: `P-GDN3-068`
- State: complete; discarded; owner-local commit family closed
- First decision field: fixed directional MQAR L1024/K4
- Resource: one AIStation A800 80GB, CUDA index 0 only
- Frozen reference: `P-GDN3-059` Momentum DeltaNet + native `[S,M]`
  FutureSeed

## 2. Evidence And Hypothesis

P059 is the only generic recurrent carrier in this regime that closes most of
the L1024 directional binding task: balanced/future/past/joint are
`.94425/.95150/.93700/.82400`, with 223 errors and 151 valid-value/wrong-key
swaps. All 151 swaps are between adjacent write ranks and 150 share one
direction. The values and traversal direction are therefore learned; the
remaining failure is local owner interference.

The negative experiments bound several tempting fixes. A separate prediction
key, global S/M phase rotation, global least squares, independent erase, and a
second arbitrary write all destroy the working owner geometry. Removing old
key-local Momentum is also non-selective: it lowers the immediate residual for
both failed and correct events. The useful Momentum must be retained, but its
effect on the committed value state should not be applied globally at every
token.

The hypothesis is that P059's velocity is a superposition of owner-specific
edits. At token `t`, only the component owned by the current normalized key
should be committed to `S`; orthogonal components stay in `M` for their own
future owners. This changes the live recurrent state organization without a
new address, cache, selector, pass, state tensor, parameter, or task rule.

## 3. Exact Mechanism

For normalized key `k_t`, define the exact rank-one owner projector

```text
P_k(M) = k (k^T M) / (k^T k).
```

The candidate recurrence is

```text
A_t = alpha_t S_(t-1)
r_t = v_t - k_t^T A_t
M_t = mu_t M_(t-1) - eta_t k_t r_t^T
S_t = A_t - beta_t P_k(M_t)
o_t = q_t^T S_t.
```

P059 uses `S_t = A_t - beta_t M_t`, which commits all owners' velocity at
every token. P068 changes only that final live commit. It preserves P059's
Q/K/V projections, alpha/mu/beta/eta parameterization, short convolutions,
output path, `[S,M]` state, native FutureSeed and deterministic training
protocol. With zero old Momentum and one token, `P_k(M_t)=M_t`, so the update
is exactly the parent delta edit. With arbitrary old Momentum it differs only
through owner-local projection.

A clean-room fused Triton scan implements the recurrence. It stores one FP32
state anchor every eight tokens; the backward recomputes and reverses only
within each short block, avoiding P062's known 1,024-step inverse-decay
amplification while leaving the forward recurrence unchanged. Within a block
it reconstructs

```text
A_t = S_t + beta_t P_k(M_t)
S_(t-1) = A_t / alpha_t
r_t = v_t - k_t^T A_t
M_(t-1) = (M_t + eta_t k_t r_t^T) / mu_t.
```

No source from the read-only external Momentum repository is redistributed.

## 4. Why Closed Families Do Not Cover It

- P-DIAG-MOMVEL-001 tested deleting current-key old Momentum. P068 deletes
  nothing from `M`; it delays orthogonal owner components while committing
  their exact component when their owner key appears.
- P063 learned an unconstrained second prediction address and overflowed.
  P068 uses the existing normalized key and an exact projector.
- P014 globally rotated `[S,M]` with virtually no board variation. P068 is
  token-, board-, head-, and owner-dependent without learned routing.
- P067 adds an independent erase address. P068 preserves the working P059
  residual and update address exactly.
- P062 changed where the residual is evaluated and failed production
  backward. P068 keeps the parent residual and changes state organization.

## 5. Fixed Protocol

- Model: D128/L2/H4/K32/V32, output gate enabled
- State: exactly `[S,M]`, 8,192 values/layer
- Parameters: exactly 599,672, zero delta versus P059
- FutureSeed: native adjacent full `[S,M]` transport, scale 1
- Data: fixed mixed-direction MQAR L1024/K4; 10,000 train, 1,000 validation
- Train: 10 epochs, batch32, seed123, inherited AdamW/LR/WD/schedule
- Initialization: exact frozen matched initialization used by P059
- Decision: one candidate-only run against frozen P059; no control rerun
- Sweep: none

## 6. Strict CUDA Contract

The exact pushed clean detached worktree on the sole A800 must prove:

1. exact GPU UUID, external Momentum and host FLA source hashes, one visible
   CUDA device and no fallback;
2. fused Triton versus explicit Torch output/state relative RMS `<=2e-4` and
   every input/gate/state gradient relative RMS `<=5e-3`;
3. `beta=0` parent parity and one-token zero-old-Momentum parent parity
   `<=.02` for output and complete `[S,M]` state;
4. arbitrary nonzero Momentum changes parent output/state by `>=.01`, while a
   component orthogonal to the first owner changes that first output by at
   most `.002`;
5. exact head equivariance, causality, two custom backward paths, finite
   nonzero full-stack projection/gate/FutureSeed gradients;
6. exact 599,672 parameters, 8,192 recurrent values/layer, matched parent and
   data hashes;
7. production L1024 forward/backward finite, both layers active, local commit
   relative RMS `>=.01` with board variation, projected fraction in
   `[.01,.95]`, discarded fraction `>=.10`, and Momentum/state RMS ratio in
   `[.01,20]`.

Any miss closes this exact recurrence before formal training. No numerical,
coefficient, state, kernel, seed, LR, loss, batch, width, depth, or duration
rescue is authorized.

## 7. Registered Decision Gates

Quality versus frozen P059, all required:

- balanced accuracy `>=.955` and absolute gain `>=.01`;
- future and past accuracy each regress by at most `.005`;
- joint exact `>=.84` and absolute gain `>=.01`;
- total errors `<=180`;
- wrong-key valid-value swaps `<=105`, conditional share `<=.60713`;
- adjacent-owner swaps reduced by at least 20% from 151.

Cost versus P059, all required:

- elapsed, post-warm wall, and independent warmed step `<2.5x`;
- peak training CUDA allocation `<1.25x`.

An integrity, activation, stability, quality, or cost miss discards P068 and
closes owner-local projection of the unchanged P059 Momentum. There is no
projector softening, rank, carry, gate, scale, normalization, seed, LR, loss,
batch, width, depth, or duration rescue.

## 8. Artifacts And Decision

Exact pushed/read-back source
`911ffaad9b06706dca80de38e5d4ed7a3e15f86d` passed the strict A800
contract. Fused Triton versus explicit Torch output/state relative RMS is
`1.2523e-7/6.1237e-8`; all recurrent gradients are below `3.3e-7` relative
RMS. The beta-zero and zero-old-Momentum limiting identities, arbitrary
Momentum dependence, owner orthogonality, causality, head equivariance, two
custom backward paths, complete full-stack gradients, exact parameter/state
counts and production L1024 activation checks all pass.

The fixed endpoint completed normally and failed the registered quality gate:

| Metric | P059 | P068 |
|---|---:|---:|
| balanced accuracy | `.94425` | `.01250` |
| future accuracy | `.95150` | `.01050` |
| past accuracy | `.93700` | `.01450` |
| joint exact | `.82400` | `0` |
| total errors | `223` | `3,950` |
| wrong-key valid-value swaps | `151` | `154` |
| adjacent-owner swaps | `151` | `77` |

The lower adjacent-owner count is not a repair: paired transitions include
`3,591` parent-correct queries becoming unrelated wrong values and `138`
becoming wrong-key values, while only two parent wrong-key events become
correct. Validation stays near chance for all ten epochs (`.01025` at epoch0,
`.01250` at epoch9). Layer-0 local-commit relative RMS reaches `129,049`, and
its Momentum/state RMS ratio reaches `5.52`; the exact projection is finite but
removes the globally coherent integration path needed to learn retrieval.

All cost gates pass. Elapsed/post-warm/warmed-step/peak-allocation ratios are
`1.7399/1.7385/1.2240/1.2320x`. The run sampled 112 GPU points with mean,
active-mean and peak utilization `19.59/51.02/73%`, peak observed memory
`2,230 MiB`, and mean/peak power `97.06/213.03 W`.

Decision: discard P068 and close exact current-key projection of the P059
Momentum commit. Do not soften the projector, add a carry gate, change rank or
rescue training. A successful successor must preserve P059's dense global
second-order integration while representing owner identity as an additional
learned state organization, not replace the parent commit with a rank-one
slice.

Remote run:
`/huyang2/double-loop/runs/p-gdn3-068-owner-local-momentum-20260816T110809Z-911ffaa`.
Key SHA256 values are comparison/score
`efc8ce5ca2e6aef8b8db5be03cbe4d0c46815796c37fa4c3e08f4c28e9528967`,
contract `73b52031349a4e364256c8a5b14418aff8097dcbc24ca62d0512deecf9942b83`,
formal log `41145525bed553ca71acb5314aad30f6c26cdd9227975e7d6cb7761bc14a426d`,
and candidate checkpoint
`3467bd3ec5481f4585d93ed10eaa073ff78af17233bbcca352f437cc40598e26`.
