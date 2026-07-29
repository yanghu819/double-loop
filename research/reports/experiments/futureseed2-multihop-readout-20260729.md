# FutureSeed2 Compatible Multihop Readout

## 1. Metainfo

- Plan: `P-FS2-005`
- Machine: AIStation GPU1, one A800 80GB
- Branch: `codex/futureseed2-multihop-readout-20260729`
- Status: approved, implementation in progress

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

Pending.

## 8. Decision

Pending.
