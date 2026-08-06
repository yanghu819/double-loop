# P-GDN3-002: Clean Shared-Namespace Scale

## Metainfo

- Status: complete; endpoint gate failed, shared-namespace scale candidate closed
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
  registered GPU1. PID/PGID was `101373`; only CUDA index 0 with UUID
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

## Step6000 Science Gate

The second hard-stage gate passes through its registered 56-60 route. The
checkpoint evaluator still emits representative fixed probes, so h58 is the
registered readout for that range:

| Fixed probe | Metric | loop1 | loop2 | loop3 | loop4 | loop5 |
|---|---|---:|---:|---:|---:|---:|
| h50 | exact | 0.9495 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| h53 | exact | 0.0000 | 0.0039 | 0.0234 | 0.0293 | **0.0293** |
| h53 | blank accuracy | 0.5792 | 0.6655 | 0.6893 | 0.6935 | **0.6946** |
| h58 | exact | 0.0000 | 0.0000 | 0.0059 | 0.0059 | **0.0059** |
| h58 | blank accuracy | 0.4537 | 0.4696 | 0.4736 | 0.4744 | **0.4747** |
| h64 | blank accuracy | 0.5301 | 0.5828 | 0.6015 | 0.6078 | **0.6093** |

- The alternative h53 `0.10` exact route is not met. The registered gate still
  passes because h58 full-board exact is nonzero and appears only after recurrent
  computation: `0/0/0.0059/0.0059/0.0059` across loops1-5. At the same time,
  h58 removes `1.22` wrong cells per board (`31.68 -> 30.46`) from loop1 to
  loop5. This is positive loop correction on the representative 56-60 probe,
  not a loop1 operating point copied five times.
- The easier hard probe strengthens but remains far from solved: h53 loop5
  exact is `0.0293`, blank accuracy is `0.6946`, and mean wrong cells fall
  `22.30 -> 16.19` across loops. Compared with step3000, h53 loop5 exact rises
  `0.0156 -> 0.0293` and blank accuracy rises `0.5907 -> 0.6946`.
- The unresolved boundary remains global closure at the hardest condition. h64
  removes `5.06` wrong cells per board (`30.07 -> 25.01`) and gains `+0.0791`
  blank accuracy across loops, but full-board exact remains zero at every loop.
- Step6000 train CE is finite at `0.8711`; loop1/loop5 losses are
  `0.9832/0.8711`. Shared-address weight RMS is `0.07567`, residual RMS is
  `4.0939`, and Q/K relative changes are `0.9437/0.9737`. The official-FLA
  GDN2/Triton path, native FutureSeed route, source SHA, and single-GPU
  identity remain intact with no NaN, OOM, or fallback.
- From step3000 to step6000, elapsed time is `22,206.4 s`, or about
  `7.402 s/optimizer-step`, `17.29 effective boards/s`, and `1.40k cells/s`.
  Live NVML occupancy is about `17,687 MiB`; the runner still does not emit a
  formal allocator peak, so no stronger peak-memory claim is made.

The evaluator does not expose full official-range aggregates or per-board
predictions at this live checkpoint. The archived step6000 visualization is
therefore explicitly an aggregate fixed-probe audit: it shows h58 exact opening
at loop3 and the unsolved h64 tail without pretending to be a selected-board
case. Full official 51-55/56-60/61-64 metrics and same-board loop exports remain
mandatory at a frozen evaluation window or the endpoint.

The sole registered trajectory continues unchanged to the scheduled step9000
readout and step12000 endpoint. No LR, address scale, seed, loss, batch, width,
or mechanism rescue is authorized.

- aggregate dashboard:
  `../visualizations/gdn3-shared-namespace-scale-20260805/visualizations/index.html`;
- step6000 fixed-probe loop audit:
  `../visualizations/gdn3-shared-namespace-scale-20260805/hardest-case-buckets-step6000.html`;
- rendered step6000 audit:
  `../visualizations/gdn3-shared-namespace-scale-20260805/hardest-case-buckets-step6000.png`.

Artifact hashes:

- step6000 checkpoint-eval JSON:
  `71c6bd997679162379f30f780b477fada2ae072a090d19dd1b326cb0dd4db3a4`;
- exact step6000 train-state checkpoint:
  `8d36c7c37d6eb394b6a02bf0ce1979153aa979e759cb9868386a78d835114d1e`.

## Step9000 Science Readout

The registered trajectory remains healthy and every hard fixed probe improves
from step6000. The live evaluator still reports aggregate h50/h53/h58/h64
conditions rather than full official ranges or per-board predictions.

| Fixed probe | Metric | loop1 | loop2 | loop3 | loop4 | loop5 |
|---|---|---:|---:|---:|---:|---:|
| h50 | exact | 0.9798 | 1.0000 | 1.0000 | 1.0000 | **1.0000** |
| h50 | blank accuracy | 0.9996 | 1.0000 | 1.0000 | 1.0000 | **1.0000** |
| h53 | exact | 0.0000 | 0.0059 | 0.1250 | 0.2012 | **0.2148** |
| h53 | blank accuracy | 0.6078 | 0.7264 | 0.7787 | 0.7875 | **0.7896** |
| h58 | exact | 0.0000 | 0.0000 | 0.0098 | 0.0117 | **0.0117** |
| h58 | blank accuracy | 0.4579 | 0.4882 | 0.5001 | 0.5007 | **0.5005** |
| h64 | exact | 0.0000 | 0.0000 | 0.0137 | 0.0254 | **0.0293** |
| h64 | blank accuracy | 0.5504 | 0.6474 | 0.7171 | 0.7295 | **0.7313** |

- Step6000-to9000 loop5 slopes are positive across all hard probes. h53 exact
  and blank accuracy gain `+0.1855/+0.0950`; h58 gains `+0.0059/+0.0257`; h64
  gains `+0.0293/+0.1221`.
- The hardest fixed condition now reaches global closure. h64 exact remains
  zero through loop2 and opens at loop3, then rises
  `0.0137 -> 0.0254 -> 0.0293`. Mean wrong cells fall
  `28.78 -> 22.56 -> 18.11 -> 17.31 -> 17.20` across loops1-5. This is genuine
  recurrent correction, not a copied one-pass prediction.
- h53 is the strongest opening: exact rises `0 -> 0.2148` across loops and mean
  wrong cells fall `20.79 -> 11.15`. h58 remains the bottleneck, but it still
  removes `2.47` wrong cells per board and preserves nonzero loop3-5 exact.
- Train CE is finite at `0.7786`, down from `0.8711` at step6000. The
  step6000-to9000 interval takes `22,189.1 s`, or `7.396 s/optimizer-step`,
  `17.31` effective boards/s, and `1.402k` cells/s. Live NVML occupancy remains
  about `17,687 MiB`; the runner does not emit a formal allocator peak.
- The exact PID/PGID, source SHA, one-visible-GPU UUID, official-FLA GDN2/Triton
  implementation, and native FutureSeed route remain unchanged. No NaN, OOM,
  fallback, source drift, or GPU drift is present.

The curve is healthy, so the sole registered trajectory continues unchanged to
step12000. The endpoint still requires full official 51-55/56-60/61-64 metrics
and same-board hardest-case loop exports. No LR, address scale, rank, seed,
loss, batch, width, or mechanism rescue is authorized.

- aggregate dashboard:
  `../visualizations/gdn3-shared-namespace-scale-20260805/visualizations/index.html`;
- step9000 fixed-probe loop audit:
  `../visualizations/gdn3-shared-namespace-scale-20260805/hardest-case-buckets-step9000.html`;
- rendered step9000 audit:
  `../visualizations/gdn3-shared-namespace-scale-20260805/hardest-case-buckets-step9000.png`.

Artifact hashes:

- step9000 checkpoint-eval JSON:
  `80d62a33057b892590694b3cbdf5c356d619b18d1caf1949a324137d8b1bd3df`;
- exact step9000 train-state checkpoint:
  `21e3840973e4c0cc8d4c45b32106cec76424e57ff1dc5f6ac03a3a51d10e4e2b`;
- archived run-through-step9000 log:
  `46479b4d87b83c0a8120d7fb3aeb075eec975782967af1c9fd29019ebb0306db`;
- immutable source snapshot retained remotely:
  `f1cc2c269d12b109f99b8dea7e064aa4a3a50db62e5791012f5d629c8223d004`.

## Step12000 Endpoint

The sole registered trajectory reached step12000 and exited cleanly. The
checkpoint, full official-range evaluation, per-board case bank, and source
provenance were frozen before making the endpoint decision.

### Fixed-probe readout

Each entry is `full-board exact / blank accuracy` on the same aggregate fixed
probe used at earlier gates.

| Probe | loop1 | loop2 | loop3 | loop4 | loop5 |
|---|---:|---:|---:|---:|---:|
| h50 | 0.9798 / 0.9994 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 1.0000 / 1.0000 | **1.0000 / 1.0000** |
| h53 | 0 / 0.6213 | 0.0059 / 0.7475 | 0.2539 / 0.8083 | 0.3301 / 0.8166 | **0.3418 / 0.8177** |
| h58 | 0 / 0.4636 | 0.0020 / 0.5058 | 0.0117 / 0.5236 | 0.0156 / 0.5261 | **0.0156 / 0.5261** |
| h64 | 0 / 0.5630 | 0 / 0.6749 | 0.0293 / 0.7451 | 0.0625 / 0.7614 | **0.0840 / 0.7652** |

The step9000-to12000 loop5 slopes remain positive: h53 exact/blank gains
`+0.1270/+0.0282`, h58 gains `+0.0039/+0.0257`, and h64 gains
`+0.0547/+0.0338`. This confirms that training did not collapse after the
step9000 opening. It does not by itself satisfy the full official endpoint.

### Full official ranges

The frozen full evaluation uses 512 boards per range. Each table entry is
again `full-board exact / blank accuracy`.

| Official range | loop1 | loop2 | loop3 | loop4 | loop5 |
|---|---:|---:|---:|---:|---:|
| 51-55 blanks | 0 / 0.5834 | 0.0156 / 0.6843 | 0.2207 / 0.7327 | 0.2852 / 0.7421 | **0.3008 / 0.7432** |
| 56-60 blanks | 0 / 0.4987 | 0.0117 / 0.5521 | 0.0801 / 0.5766 | 0.0957 / 0.5815 | **0.0996 / 0.5831** |
| 61-64 blanks | 0 / 0.5587 | 0 / 0.6679 | 0.0332 / 0.7349 | 0.0586 / 0.7524 | **0.0762 / 0.7562** |

The hard-range macro loop5 exact is `0.1589`, and macro blank accuracy is
`0.6942`. Mixed exact rises `0.0234 -> 0.0352 -> 0.1562 -> 0.2051 ->
0.2168` over loops1-5, while mixed blank accuracy rises `0.5482 -> 0.6210 ->
0.6569 -> 0.6643 -> 0.6658`. Therefore genuine recurrent correction passes,
but neither registered quality route passes:

- hard-range macro exact `0.1589 < 0.3191`;
- mixed loop5 exact `0.2168 < 0.40`.

The scaled candidate also trails the sealed D192/L10 canonical model in every
hard range: `0.3008/0.0996/0.0762` versus
`0.4492/0.1543/0.2637`, and mixed exact is `0.2168` versus `0.3379`.

### Loop behavior on the same boards

The archived 61-64 hard-failure case-bank export for batch 6 reduces wrong
cells `22 -> 16 -> 7 -> 6 -> 6` and conflict units
`23 -> 17 -> 13 -> 8 -> 8`, but remains unsolved. The prediction freezes from
loop4 to loop5, exposing a residual global-consistency stall rather than lack
of early correction. A contrasting batch-15 board reduces wrong cells
`29 -> 13 -> 1 -> 0 -> 0` and becomes exact at loop4. Both are direct
same-board trajectories, not aggregate bucket proxies.

- hard-failure HTML:
  `../visualizations/gdn3-shared-namespace-scale-20260805/output/case_bank/official_b61_64/official_b61_64_hard_failure_08_b0006.html`;
- rendered hard-failure image:
  `../visualizations/gdn3-shared-namespace-scale-20260805/hardest-same-board-b61-64-step12000.png`;
- solved-by-loop contrast:
  `../visualizations/gdn3-shared-namespace-scale-20260805/output/case_bank/official_b61_64/official_b61_64_solved_by_loop_01_b0015.html`;
- endpoint dashboard:
  `../visualizations/gdn3-shared-namespace-scale-20260805/endpoint-dashboard-step12000.png`.

### Cost and integrity

- train CE/total loss is `0.7102/0.7778`; loop1/loop5 loss is
  `0.9450/0.7102`;
- 12,000 optimizer steps at effective batch128 take `88,991.9 s`, or
  `17.26 effective boards/s`; the step9000-to12000 interval is independently
  consistent at `17.31 boards/s` after earlier compilation;
- the model has `11,551,296` trainable parameters;
- CUDA peak allocation/reservation is `13,016.8/14,428.0 MiB`; live sampled
  NVML occupancy during training was about `17,687 MiB`;
- all 12 layers use pinned official-FLA GDN2 source
  `9c8e42e762fce087c27b673af4922795d9edb85e` with Triton short convolutions;
- the process exited normally, GPU1 is idle, source SHA is unchanged, and the
  complete log has no NaN, OOM, fallback, CUDA error, or traceback.

### Decision

`P-GDN3-002` fails its preregistered endpoint and is closed without a matched
D256 normal-GDN2 control, second seed, or tuning rescue. The result separates
two claims:

1. Cross-layer address compatibility remains a real local mechanism signal:
   P-GDN3-001 and the loop trajectories here both show substantial correction.
2. Forcing one shared Q/K namespace across all 12 layers is not a
   quality-preserving scalable GDN3 design under the registered full-diversity
   budget.

The next high-information hypothesis is not a shared-namespace strength/rank
variant. It is an identity-initialized cross-layer state-coordinate transport
that maps the terminal KxV state into each receiving layer's private K/V basis,
preserving layer-private address dynamics while making transported state
readable. This is analysis only: no compute is authorized until a read-only
mechanism audit and a fresh preregistration fix identity behavior, prediction,
budget, and kill criteria.

Endpoint hashes:

- endpoint manifest:
  `../visualizations/gdn3-shared-namespace-scale-20260805/endpoint_manifest.json`;
- step12000 checkpoint-eval JSON:
  `369fecf8f8af8f11b8e9f5e28e7fc71d2582d395ee7dc0f76cb906c63f4e6b6a`;
- full final JSON:
  `6a844540af049802263ab267d6076548e60e5eb62f50e4a541ef59301f330b4e`;
- exact step12000 train-state checkpoint:
  `e44887754d23aa38174c34f7fda51645f0497125caff4ee89356877936f32f52`;
- complete launch log:
  `648cd6cc3f89f64cd924ca8a379318d99c574abc1fbf45ea48c07d4fd32eb5c7`;
- config:
  `4cb98ffb912c8079adc9290eee0190f6108428913b7280de011d886d65d3ad2e`;
- immutable source snapshot:
  `f1cc2c269d12b109f99b8dea7e064aa4a3a50db62e5791012f5d629c8223d004`.
