# GDN Function-Preserving State Expansion

## 1. Metainfo

- Plan ID: `P-SCALE-035`
- Status: in progress
- Planned: 2026-07-22 19:14 CST / 2026-07-22T11:14:33Z
- Machine: AIStation `GPU1` A800 only; GPU2 forbidden
- Parent: exact P-SCALE-034 step30000 train-state checkpoint
- Parent source SHA: `f8e009bedcb0a6db88a49a79fc77b260cb3f3f30`

## 2. Mechanism Hypothesis

P-SCALE-034 raised mixed loop5 exact to `0.4805` and formal official 56-64
exact to `0.3848`, but the final 6000-step increment added only `+0.0156` on
mixed exact and remained non-monotonic across difficulty buckets. Continuing the
identical expand-v4 model to step36000 would therefore have low information
gain.

The remaining generic capacity hypothesis is narrower: the D224 token path may
be adequate, while each GDN head's `64 x 16` value-key state is too small to
retain all constraints needed by later loops. If that is true, increasing only
the value/state dimension should let loop4/5 keep improving. If a larger state
starts from the same learned function, receives gradient, and still does not
improve hard exact, then ordinary state width is not the missing mechanism.

This follows the bitter lesson: add general learned memory and compute, not a
Sudoku rule, repair pass, selector, search procedure, or hand-written loss.

## 3. Function-Preserving Transform

The checkpoint transform changes `GDN_EXPAND_V=4` to `8`, so each head's value
and recurrent-state dimension changes from64 to128.

- `v_proj` and `g_proj`: duplicate the learned 64-channel block inside every
  head.
- `o_norm_weight`: duplicate the learned 64 weights.
- `o_proj`: retain the original 64 columns for every head and initialize the
  additional 64 columns to exactly zero.
- AdamW first and second moments: duplicate moments for copied value/gate/norm
  channels and zero the new output-column moments.
- Preserve all other weights, optimizer state, Python/Torch RNG, and global
  step exactly.

At initialization, the expanded recurrent state contains two identical copies.
Its RMS normalization is therefore unchanged, while the zero new readout means
the logits should match the old network. The new readout columns see nonzero
features, so they must receive gradient and can break symmetry after training.

## 4. Gates, Budget, And Prediction

CUDA-only preflight on GPU1:

- same official 56-64 batch, loops1-5, BF16 Triton recurrent path;
- old versus expanded prediction mismatches must be zero;
- maximum logit difference `<=0.04`, RMS difference `<=0.004`;
- expanded backward gradients must be finite;
- the new half of every `o_proj` must have nonzero aggregate gradient;
- stop before training on wrong GPU, provenance mismatch, failed equivalence,
  zero new-branch gradient, NaN, or OOM.

Training keeps every other axis fixed: D224/L12/H14/D16, native FutureSeed
scale1, fixed update, loop5 with equal CE at every loop, full-diversity official
data, broad 51-64 curriculum, AdamW state, BF16 Triton recurrent execution, and
effective batch128. Microbatch32/accum4 is preferred; microbatch16/accum8 is an
allowed execution-only fallback if the 80GB fit gate requires it.

- step30500 is diagnostic only;
- step31500 is the decision gate;
- continue to step33000 only if mixed, holes60/64, or formal official56-64
  improves by at least `+0.02` over step30000;
- stop at step30500 if CE worsens by more than `0.20` or hard exact falls by
  more than `0.05` despite the equivalent start.

Success is formal official56-64 exact `>=0.415` while official51-55 remains
`>=0.60`, or mixed exact `>=0.50`. Strong conditional success at step33000 is
official56-64 `>=0.45` or mixed `>=0.52`.

## 5. Paper Decision

A positive result supports progressive recurrent-memory scaling: FutureSeed
supplies global direction, loops spend recurrent compute, and state capacity can
be grown without forgetting the learned solver. A negative result is equally
decisive: do not run expand-v6/v10, LR, seed, loss, or noise tables; move to a
different generic state formulation or higher independent-data coverage.

## 6. Results

Pending CUDA equivalence and training gates.
