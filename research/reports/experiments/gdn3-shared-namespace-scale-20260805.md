# P-GDN3-002: Clean Shared-Namespace Scale

## Metainfo

- Status: in progress; GPU fit gate passed, formal launch pending
- Preregistered: 2026-08-05 12:17 CST
- Resource: AIStation task-mode GPU1 A100 80GB only
- Seed: 52 only
- Benchmark: official hard 9x9 Sudoku

## Mechanism Question

P-GDN3-001 showed that one address projection shared across recurrent layers
greatly improves a mature GDN2+FutureSeed checkpoint and produces real loop
correction, but all hard full-board exact rates remain zero after only 100
adaptation steps.

The decisive question is:

> When the shared address namespace, layer-local GDN2 content maps, and
> FutureSeed route co-adapt from initialization under more model, state, data,
> and optimization scale, does the local gain turn into globally correct hard
> Sudoku boards?

## Fixed Configuration

- strict official FLA GatedDeltaNet2 chunk recurrence and Triton short conv;
- D256/L12/H8/K32/V32, increasing width, depth, heads, and total recurrent
  state over the D192/L10 quick probe;
- one zero-initialized D256 shared address projection reused in all 12 layers;
- native terminal-state FutureSeed, scale 1, per-head learned gate;
- five loops with equal CE supervision at every loop;
- official 3.832M-board full-diversity split and random cell traversal;
- 12,000 optimizer steps, effective batch 128, BF16, seed 52;
- curriculum `46-50:500,51-55:3500,51-60:4000,51-64:4000`;
- no noise, special margin loss, search, repair, selector, oracle, or Sudoku
  rule.

This is one model, one seed, and one training trajectory. It is not a
width/depth/address/LR/loss table.

## Prediction And Gates

The GPU fit probe must show exactly one visible GPU1, pinned official FLA,
finite forward/backward, active shared-namespace gradients, no fallback, and
no OOM/NaN at full D256/L12 shape.

Training gates:

- step500: 50-blank loop5 exact at least 0.60 and finite CE, proving the scaled
  carrier opens before entering hard training;
- step3000: 51-55 loop5 exact at least 0.02 or blank accuracy at least 0.58,
  with a positive checkpoint slope;
- step6000: 51-55 exact at least 0.10, or nonzero 56-60 exact with positive
  loop1-to-loop5 correction;
- endpoint: beat the canonical D192/L10 FutureSeed mean official 51-64 exact
  0.2891 by at least 0.03, or reach mixed exact at least 0.40 while preserving
  genuine loop correction.

Kill on provenance/data/kernel mismatch, fallback, OOM/NaN, no easy carrier at
step500, or a flat step3000/6000 hard curve under the registered gates. Stop by
exact PID/PGID and write `abort.json`. Do not rescue with another seed, LR,
address scale, rank, loss, batch, or nearby width.

## Claim On Success

Success supports:

> A cross-layer shared address namespace is a scalable GDN3 design for making
> FutureSeed-transferred linear-attention state usable by deeper layers and
> repeated computation.

The scale run alone does not isolate the mechanism at D256. A strong endpoint
would authorize one later matched D256 normal-GDN2 control; a weak endpoint
closes this GDN3 candidate without a sweep.

## Execution Log

- The first fit invocation stopped before model construction because the new
  wrapper inherited the detached worktree as `PERSIST_ROOT` and therefore did
  not see the persistent official-FLA Python path. CUDA allocation stayed at
  zero. This is a launcher preflight failure, not model evidence.
- The wrapper now fixes persistent storage to `/huyang2/double-loop`, reads the
  installed Python path marker, and checks both one-visible-GPU and the exact
  registered GPU1 UUID before invoking the runner. The scientific config and
  all decision gates are unchanged.
- The corrected full-shape run
  `gdn3-shared-namespace-d256l12-fit-20260805T043507Z-185d4f8` passed. It used
  11,551,296 parameters, all 12 pinned official-FLA GDN2 layers, Triton short
  convolutions, 1024 recurrent-state values per head, and finite two-step
  forward/backward on GPU1. Sampled allocation reached about 17.1 GiB with no
  OOM or fallback. The reported 643 seconds includes first-shape Triton compile
  and checkpoint evaluation and is not a steady-state speed measurement.
- At step2 the shared projection has weight RMS `0.00150`, residual RMS
  `0.1269`, and Q/K relative changes `0.7115/0.7185`. This proves the path is
  active but also exposes a real optimization risk. The formal run keeps the
  registered optimizer unchanged; the step500 easy-carrier gate decides
  whether co-adaptation stabilizes naturally. There is no LR or scale rescue.
