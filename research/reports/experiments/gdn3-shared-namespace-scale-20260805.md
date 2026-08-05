# P-GDN3-002: Clean Shared-Namespace Scale

## Metainfo

- Status: in progress; GPU fit, step500, and step3000 gates passed
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
- The formal trajectory
  `gdn3-shared-namespace-d256l12-s12000-20260805T044918Z-a68c683` launched from
  clean detached SHA `a68c683d0a7eee9b9b67da16ebdbd355ff4aed94` on the
  registered GPU1. PID/PGID is `101373`; only CUDA index 0 with UUID
  `GPU-53e9f3b4-2966-65d3-6614-09c540921519` is visible.

## Step500 Science Gate

The easy carrier opens cleanly and passes the registered gate:

| Metric | loop1 | loop2 | loop3 | loop4 | loop5 |
|---|---:|---:|---:|---:|---:|
| 50-blank full-board exact | 0.2121 | 0.7475 | 0.7677 | 0.7778 | **0.7778** |
| 50-blank blank accuracy | 0.9436 | 0.9824 | 0.9844 | 0.9848 | **0.9853** |

- train CE is finite and falls from `2.0175` at step100 to `0.0143` at
  step500;
- the registered step500 exact threshold is `0.60`; observed loop5 exact is
  `0.7778`;
- loop computation is already useful on the carrier: exact gains `+0.5657`
  from loop1 to loop5, rather than merely copying loop1;
- harder fixed probes are intentionally not used as a stop at this curriculum
  boundary: loop5 blank accuracy is `0.5090/0.3878/0.4408` at 53/58/64 blanks
  and exact remains zero, before any registered hard-stage optimization;
- shared-address weight RMS is `0.02351`; the path remains active, finite, and
  on the pinned official-FLA/Triton implementation.

The run therefore continues unchanged into the 51-55 curriculum. The next
decision gate remains step3000. The archived step500 JSON and static loop/range
dashboard are under
`research/reports/visualizations/gdn3-shared-namespace-scale-20260805/`.
A board-level hardest-case export is deferred until it can reuse a frozen
checkpoint without contending with the only formal training process; this
does not alter or pause the registered trajectory.

## Step3000 Science Gate

The first hard-stage gate passes through its registered blank-accuracy route.
The h53 fixed probe is the representative checkpoint readout for the 51-55
range:

| Fixed probe | Metric | loop1 | loop2 | loop3 | loop4 | loop5 |
|---|---|---:|---:|---:|---:|---:|
| h50 | exact | 0.7778 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| h53 | exact | 0.0000 | 0.0000 | 0.0117 | 0.0137 | **0.0156** |
| h53 | blank accuracy | 0.5583 | 0.5841 | 0.5904 | 0.5909 | **0.5907** |
| h58 | blank accuracy | 0.4448 | 0.4569 | 0.4580 | 0.4572 | **0.4576** |
| h64 | blank accuracy | 0.5267 | 0.5859 | 0.6046 | 0.6092 | **0.6108** |

- h53 loop5 blank accuracy rises `0.5090 -> 0.5625 -> 0.5907` across the
  step500/1000/3000 checkpoints. It clears the registered `0.58` route with a
  positive slope. Its exact rate also opens from zero to `0.0156`, although it
  remains below the alternative `0.02` exact route.
- Later loops now create complete solutions on h53: exact is zero through
  loop2, then `0.0117/0.0137/0.0156` at loops3/4/5. This is real recurrent
  correction rather than five copies of one readout.
- The hardest fixed probe still exposes the unresolved boundary. On h64,
  loop1-to-loop5 blank accuracy improves by `+0.0841`, equivalent to about
  `30.29 -> 24.91` wrong blank cells per board, but full-board exact remains
  zero. More correct cells are not yet enough for global closure.
- Step3000 train CE is `0.8762`; loop1/loop5 losses are `0.9568/0.8762`.
  Shared-address weight RMS is `0.04981`, residual RMS is `2.4129`, and all
  diagnostics remain finite on the pinned official-FLA/Triton path.
- From step1000 to step3000, elapsed time is `14,688.3 s`, or about
  `7.344 s/optimizer-step`, `17.43 effective boards/s`, and `1.41k cells/s`.
  The exact same-shape fit recorded a CUDA allocator peak of `13,014.7 MiB`;
  the live process currently occupies about `17,687 MiB` by NVML. The runner
  does not emit a formal-run allocator peak, so no stronger peak claim is made.

The checkpoint evaluator emits representative h50/h53/h58/h64 aggregates, not
the full official blank ranges or per-board predictions. The archived
dashboard therefore provides a loop1-to-loop5 visualization of the hardest
h64 fixed condition without inventing a board-level case. Full official-range
metrics and same-board hardest-case exports remain mandatory at the next
frozen evaluation window. The formal process continues unchanged to step6000;
there is no LR, address, loss, seed, batch, or width rescue.

- aggregate dashboard: `../visualizations/gdn3-shared-namespace-scale-20260805/visualizations/index.html`;
- hardest h64 fixed-condition page: `../visualizations/gdn3-shared-namespace-scale-20260805/hardest-case-bucket-h64-step3000.html`;
- rendered h64 image: `../visualizations/gdn3-shared-namespace-scale-20260805/hardest-case-bucket-h64-step3000.png`.

Artifact hashes:

- step1000 checkpoint-eval JSON:
  `45e3be544c94d7f2ef8f91ef2f757d2024bb5e13a496f31413d2b59d82bacb5a`;
- step3000 checkpoint-eval JSON:
  `5e679626d9a160d01c3defa27b8e50b39345d6d843b8694c766f823b63556dec`;
- exact step3000 train-state checkpoint:
  `f1870347deabaef0e45a8464edc475bcc0df9892ba7dfd34c9beaa20743c1958`.
