# FutureSeed GDN2 Compound Scale

## Metainfo

- Plan: `P-SCALE-037`
- Status: formal D192/L10 training in progress
- Preregistered: 2026-07-26 19:43 CST / 2026-07-26 11:43 UTC
- Machine: AIStation GPU1, NVIDIA A800-SXM4-80GB
- GPU2: forbidden
- CPU model smoke: forbidden
- Seed: 52 only

## Question

The strict step-500 carrier gate makes official FLA GDN2 the strongest
finite-step FutureSeed carrier: 46-50-blank loop-5 exact is `0.7969`, ahead of
RWKV7 `0.7285`, KDA `0.6465`, and GDN `0.3633`.

Does that advantage survive useful scaling, and can a compute-efficient GDN2
trained on substantially more independent official boards turn the
51-64-blank per-cell improvement into nonzero full-board exactness?

## Mechanism Hypothesis

GDN2's stronger finite-budget result is not enough to establish a higher
frontier. At 51-64 blanks it is still zero full-board exact, and its per-blank
accuracy is only slightly above RWKV7.

If the remaining failure is data and compute under-training, then an official
GDN2 with native FutureSeed should show a delayed but clear hard-range
exact-learning curve as training coverage increases. If loss and blank
accuracy improve while hard full-board exact stays flat, the bottleneck is
global closure rather than insufficient ordinary scaling.

This is one compound scaling experiment, not a width/depth/LR/seed table.

## Configuration

- official FLA `GatedDeltaNet2`, pinned wheel and exact source;
- backend dispatch disabled, official chunk backward, Triton short conv;
- D192, 10 layers, 6 heads, head dimension 32, expand-v 1;
- native terminal-state FutureSeed, scale 1, unit normalization;
- five recurrent loops with equal CE supervision on every loop;
- BF16, AdamW grouped contract, LR `0.0015`, weight decay `0.001`;
- microbatch 32, accumulation 4, effective batch 128;
- official 3.832M-board full-diversity training split;
- 12,000 optimizer steps, up to 1.536M sampled boards;
- curriculum `46-50:500,51-55:3500,51-60:4000,51-64:4000`;
- checkpoints every 100 steps;
- formal eval at steps `500,1000,3000,6000,9000,12000`;
- eval ranges `46-50`, `51-55`, `56-60`, and `61-64`;
- no noise, scratch state, feedback, special margin loss, search, repair,
  selector, rollout oracle, or Sudoku rule.

Canonical files:

- `configs/sudoku/gdn2_scale.env`
- `scripts/run_canonical_gdn2_scale.sh`
- `CUDA_VISIBLE_DEVICES=0 ./run.sh baseline`

## Predictions

1. The long-data run may trail initially, but by step 3000 it should show
   either nonzero 51-55 exact or a strong CE/blank-accuracy slope.
2. By step 6000 it should approach or exceed the completed D224 GDN reference:
   51-55 exact `0.3926` and 56-64 exact `0.1270`.
3. More loops must increase full-board exact, not only blank accuracy.
4. If GDN2 is the correct quality carrier, its hard-range slope should justify
   continuing to step 12000 and then running one long no-FutureSeed control.

## Decision And Kill Criteria

Preflight rejects the run if any of these fail:

- GPU1 identity or `CUDA_VISIBLE_DEVICES=0`;
- pinned official FLA source/class;
- official GDN2 chunk backward and Torch output/state/gradient reference;
- initial-state/FutureSeed gradients;
- Triton convolution or no-fallback contract;
- clean detached source, official data identity, or finite-value checks.

The two-step GPU capacity probe uses the canonical model and
microbatch geometry.
OOM, nonfinite values, wrong GPU, or low utilization with excessive memory
stops the launch. There is no automatic batch/kernel/model fallback; a resource
change requires an explicit recorded config commit.

Training decisions:

- At step 3000, stop only if 51-55 exact is below `0.02`, 51-55 blank accuracy
  remains below `0.58`, and CE improves by less than `0.03` from step 1000.
- At step 6000, stop if 51-55 exact remains below `0.10`, 56-60 exact remains
  zero, and the checkpoint curves are flat.
- Continue to step 12000 if either 51-55 exact reaches `0.30`, 56-60 exact is
  nonzero, or hard-range exact has a clear positive slope.
- Primary success: 51-55 exact `>=0.50` and 56-64 combined exact `>=0.20`.
- Strong success: mixed exact `>=0.45` or 56-64 exact `>=0.30`.

If step 6000 is flat under these rules, ordinary GDN2 capacity/data scaling is
not the missing mechanism. Do not respond with D288/D320/L14/L16 or LR tables.

## Claim Enabled On Success

Success would promote official GDN2 to the main quality carrier and support the
claim that FutureSeed transfers across recurrent state formulations and
continues to benefit from generic model/data/compute scaling. It would not by
itself prove FutureSeed's causal contribution at long scale; that requires the
conditional matched GDN2 no-FutureSeed control.

## Results

### Strict preflight and rejected execution geometry

At 2026-07-26 19:49-20:00 CST, exact SHA
`973a082212f5bfbac1e07c09ba944a5cf39eae6e` passed the GPU1-only strict
preflight. The pinned FLA wheel and installed files match, backend dispatch is
disabled, all 12 GDN2 layers use `fla.layers.gdn2.GatedDeltaNet2`, short
convolution is Triton, and the official chunk recurrence has maximum
output/state/gradient reference errors `5.99e-5/2.76e-4/2.34e-4`. The shared
shell is identical across all four carriers for all `77/77` common tensors.

The first full-size D256/L12 microbatch-64 probe used the real official GDN2
forward and backward on GPU1, allocated about `30 GiB`, and completed two
optimizer steps without OOM or nonfinite values. Its checkpoint reports
`530.06s` elapsed. A warm-cache repeat reports `99.34s`.

These two numbers were initially interpreted as pure training throughput. That
interpretation was wrong: the two-step probe also requested checkpoint
evaluation at step 2, and `train_sec` includes that evaluation and artifact
work. They establish functional capacity but do not establish `49.67s/step`
steady-state training. The record is retained with this correction instead of
silently rewriting the failed reasoning.

### Explicit amendment

At 2026-07-26 20:23 CST, the execution geometry changes from microbatch
`64 x accumulation 2` to `32 x accumulation 4`. Effective batch `128`, sampled
boards per optimizer step, model, data, loop count, loss, optimizer, seed, and
all scientific decision rules remain unchanged. This is not an automatic
fallback: the failed geometry and both probe runs remain archived, and the
change is committed before a new detached probe.

The amended D256/L12 microbatch-32 probes also included step-2 checkpoint
evaluation, so their `206.94s` and `110.45s` totals cannot be used to rank
microbatch throughput either.

At 2026-07-26 20:41 CST, the formal run was conservatively re-scoped to the
already validated compute-efficient D192/L10/H6 geometry while retaining
official GDN2, native FutureSeed, effective batch 128, the full-diversity
corpus, the 12k-step curriculum, and every quality decision rule. The decision
uses the project's prior evidence that data and optimizer-step scaling moved
the hard boundary while modest width scaling did not, not the invalid
two-step timing comparison above.

The next gate is one exact-SHA D192/L10 GPU1 probe. Launch the formal 12k-step
experiment only if measured steady-state throughput is near the accepted GDN2
range and no implementation or memory gate regresses.

The exact-SHA 10-step training-only probe reports `137.24s` total and
`8.02GiB` peak allocation. The average `13.72s/step` includes the process's
first large-shape compile; it passes the `15s/step` launch boundary and is
consistent with approaching the historical hot rate after amortization.

The first formal launch then failed closed before step 1 because the diagnostic
checkpoint list requested exact blank count `62`, while the official test split
contains no 62-blank rows. Its observed hard-tail counts are `61:25`, `62:0`,
`63:20`, and `64:4865`. At 2026-07-26 21:14 CST, only the diagnostic point
changes from `62` to `64`. The primary official interval remains `61-64`; no
training, model, loss, data, seed, or decision rule changes.

### Formal launch and execution-shape check

The accepted formal run is bound to clean detached SHA
`42102bd65a28d60bde6b09ef93343692740582a1`:

- run: `gdn2-futureseed-d192l10-s12000-20260726T131301Z-42102bd`;
- GPU: GPU1 only;
- step 100 CE: `1.6522`;
- step 200 CE: `0.2834`;
- checkpoint cadence: every 100 optimizer steps.

Repeated GPU sampling showed that microbatch `32 x accumulation 4` uses only
about `11.8 GiB` and does not saturate the A800. Because a 12k-step run makes
throughput consequential, the process was stopped by its exact PID immediately
after the complete step-200 checkpoint for one execution-only gate.

With model, data, loss, seed, and effective batch unchanged, microbatch
`128 x accumulation 1` took `187.59s` for a checkpoint-eval-free 10-step run,
versus `137.24s` for `32 x accumulation 4`: `36.7%` slower despite using about
`36.6 GiB` resident memory. The larger microbatch is rejected. No intermediate
batch sweep follows.

The formal run resumed from the exact step-200 model, optimizer, scheduler,
data RNG, and training RNG checkpoint with microbatch `32 x accumulation 4`.
This pause changes execution time only; it does not create a second quality
arm.

### Step-500 gate

Step 500 completes the `46-50` warm-up stage with train CE `0.00927`. The
fixed-count 512-board checkpoint evaluator reports:

| Exact blanks | loop1 exact / blank | loop2 exact / blank | loop3 exact / blank | loop5 exact / blank |
|---:|---:|---:|---:|---:|
| 50 | `0.3535 / 0.9610` | `0.7172 / 0.9834` | `0.7374 / 0.9861` | `0.7374 / 0.9861` |
| 53 | `0.0000 / 0.4986` | `0.0000 / 0.5150` | `0.0000 / 0.5136` | `0.0000 / 0.5133` |
| 58 | `0.0000 / 0.3807` | `0.0000 / 0.3870` | `0.0000 / 0.3839` | `0.0000 / 0.3830` |
| 64 | `0.0000 / 0.4146` | `0.0000 / 0.4348` | `0.0000 / 0.4358` | `0.0000 / 0.4356` |

This is a clean easy-range opening, not a hard-range success. Loop 2 provides
large real compute gain at 50 blanks, while loops 3-5 saturate. Exact remains
zero at 53/58/64 blanks, as expected before the model has received any
`51-55` curriculum updates.

The run continues into `51-55`. Step 1000, after 500 hard-stage updates
(`64,000` sampled hard boards), is the first decision-relevant comparison with
the short-budget carrier baseline.

### Step-1000 gate

Hard-stage train CE moves monotonically from `0.9832` at step 600 to `0.9675`,
`0.9493`, `0.9355`, and `0.9145` at steps 700-1000. The fixed checkpoint
profile is:

| Exact blanks | loop1 exact / blank | loop2 exact / blank | loop3 exact / blank | loop5 exact / blank |
|---:|---:|---:|---:|---:|
| 50 | `0.7273 / 0.9889` | `0.9495 / 0.9990` | `0.9596 / 0.9992` | `0.9798 / 0.9996` |
| 53 | `0.0000 / 0.5398` | `0.0000 / 0.5603` | `0.0000 / 0.5605` | `0.0000 / 0.5599` |
| 58 | `0.0000 / 0.4335` | `0.0000 / 0.4479` | `0.0000 / 0.4458` | `0.0000 / 0.4448` |
| 64 | `0.0000 / 0.5036` | `0.0000 / 0.5278` | `0.0000 / 0.5305` | `0.0000 / 0.5315` |

Relative to step 500, loop-5 blank accuracy improves by `+0.0466`, `+0.0618`,
and `+0.0959` at 53, 58, and 64 blanks, respectively. This is real transfer
from the 51-55 training stage, but it has not crossed the full-board closure
threshold.

A zero-training-step read-only evaluation from the exact step-1000 checkpoint
reports:

| Evaluation | loop5 exact | loop5 blank accuracy |
|---|---:|---:|
| mixed official test | `0.0234` | `0.5262` |
| official 46-50 | `0.9961` | `0.9998` |
| official 51-55 | `0.0000` | `0.5541` |
| official 56-60 | `0.0000` | `0.4787` |
| official 61-64 | `0.0000` | `0.5290` |

Mixed exact rises only from `0.0195` at loop 1 to `0.0234` at loop 2 and then
stays flat. Most aggregate recurrent gain still arrives in the second loop.

The case bank nevertheless shows genuine incomplete self-correction. A
54-hidden-cell 51-55 failure reduces wrong hidden cells
`15 -> 12 -> 10 -> 9` and duplicate conflicts `20 -> 17 -> 17 -> 16` across
loops `1/2/3/5`. Loop computation is doing useful work, but it does not yet
finish the board.

Across all selected cases, the mean trajectories are:

| Official range | selected cases | wrong cells loop1/2/3/5 | conflict units loop1/2/3/5 |
|---|---:|---|---|
| 46-50 | 5 | `2.80 / 0.40 / 0.20 / 0.20` | `4.80 / 1.00 / 0.60 / 0.60` |
| 51-55 | 4 | `16.00 / 13.00 / 11.25 / 10.50` | `21.00 / 19.50 / 18.25 / 17.50` |
| 56-60 | 4 | `16.25 / 12.75 / 11.25 / 11.25` | `22.50 / 18.75 / 19.25 / 18.75` |
| 61-64 | 4 | `24.50 / 21.00 / 19.00 / 19.25` | `24.75 / 24.00 / 23.25 / 23.25` |

This locates the mechanism boundary more precisely than aggregate exact alone.
The loop strongly cleans easy boards and provides diminishing local correction
on hard boards. At 61-64 blanks, later loops barely reduce conflict units and
slightly regress mean wrong cells from loop 3 to loop 5. The unresolved
question is whether ordinary training scale changes this trajectory, not
whether the current loop is numerically active.

The archived browser-ready case bank is
`runs/gdn2-futureseed-d192l10-step1000-eval-20260726T1514Z-42102bd/output/case_bank/`.

The step-3000 stop rule is not triggered early: hard blank accuracy and train
CE have clear positive slopes. Training therefore resumes from the exact
step-1000 model/optimizer/RNG checkpoint after a recorded GPU1 lease rollover.
