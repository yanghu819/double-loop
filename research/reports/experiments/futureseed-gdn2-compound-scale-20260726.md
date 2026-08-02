# FutureSeed GDN2 Compound Scale

## Metainfo

- Plan: `P-SCALE-037`
- Status: step-3000 gate passed; exact checkpoint resumed toward step 6000
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

### Step-3000 gate

The 51-55 stage remains noisy, but train CE reaches `0.8270` at step 3000,
down `0.0875` from step 1000. The fixed-count checkpoint profile is:

| Exact blanks | loop1 exact / blank | loop2 exact / blank | loop3 exact / blank | loop5 exact / blank |
|---:|---:|---:|---:|---:|
| 50 | `0.8384 / 0.9960` | `1.0000 / 1.0000` | `1.0000 / 1.0000` | `1.0000 / 1.0000` |
| 53 | `0.0000 / 0.5627` | `0.0039 / 0.6102` | `0.0176 / 0.6256` | `0.0215 / 0.6271` |
| 58 | `0.0000 / 0.4475` | `0.0000 / 0.4592` | `0.0000 / 0.4598` | `0.0000 / 0.4598` |
| 64 | `0.0000 / 0.5177` | `0.0000 / 0.5772` | `0.0000 / 0.5880` | `0.0000 / 0.5916` |

The read-only 512-board official evaluation reports:

| Evaluation | loop5 exact | loop5 blank accuracy |
|---|---:|---:|
| mixed official test | `0.0352` | `0.5639` |
| official 46-50 | `1.0000` | `1.0000` |
| official 51-55 | `0.0098` | `0.6072` |
| official 56-60 | `0.0098` | `0.5132` |
| official 61-64 | `0.0000` | `0.5898` |

The case-bank log also prints exact rates from its separate 256-board sample:
`0.0117/0.0195/0.0000` on 51-55/56-60/61-64. Those are visualization-sample
rates, not the primary 512-board scores above.

Mixed exact across loops 1-5 is
`0.0234 / 0.0234 / 0.0313 / 0.0352 / 0.0352`; mixed blank accuracy is
`0.5202 / 0.5548 / 0.5622 / 0.5638 / 0.5639`. The model now uses loops 3-4
for additional full-board correction, although loop 5 is nearly saturated.

Selected-case trajectories make the delayed change easier to see:

| Official range | selected cases | wrong cells loop1/2/3/5 | conflict units loop1/2/3/5 |
|---|---:|---|---|
| 46-50 | 4 | `2.50 / 0.00 / 0.00 / 0.00` | `5.50 / 0.00 / 0.00 / 0.00` |
| 51-55 | 10 | `16.20 / 6.10 / 3.70 / 3.20` | `20.60 / 11.00 / 5.70 / 5.00` |
| 56-60 | 12 | `16.08 / 7.00 / 3.25 / 2.58` | `21.42 / 11.83 / 5.92 / 4.50` |
| 61-64 | 4 | `27.25 / 15.75 / 10.75 / 10.00` | `26.25 / 21.50 / 17.00 / 16.00` |

This is genuine recurrent correction, not a broad-mask artifact: later loops
remove wrong digits and duplicate row/column/box values. It remains incomplete
at 61-64 blanks.

The historical D224/L12 GDN step-3000 reference is not a matched architecture
comparison, but it is a useful frontier check. Its mixed loop-5 exact is
`0.0313` versus GDN2 `0.0352`; its official 51-55 exact/blank is
`0.0117/0.6560` versus GDN2 `0.0098/0.6072`. GDN2 therefore has not yet
separated from the long-run GDN frontier despite winning the strict 500-step
carrier gate.

The preregistered step-3000 stop conjunction is false: 53-blank loop-5 exact
is above `0.02`, blank accuracy is above `0.58`, CE improved by more than
`0.03`, and official 56-60 exact is nonzero. Continue to step 6000. The exact
checkpoint is
`train_state_step003000.pt`, SHA256
`1c715ad3cdd5e17f4ce6e3fc3be93cdbd2434636242acd7c8bb464afb8d5653f`.
The GPU1 lease rollover and read-only evaluation are recorded separately.
After restart, strict launch attempt 1 failed before model load because
`PERSIST_ROOT` was omitted and project-local FLA was intentionally unavailable;
attempt 2 restored `/huyang2/double-loop`, loaded the exact checkpoint, and
resumed on clean SHA `42102bd65a28d60bde6b09ef93343692740582a1`.

### Step-6000 gate

Train CE reaches `0.7974`. The fixed-count 512-board checkpoint profile is:

| Exact blanks | loop1 exact / blank | loop2 exact / blank | loop3 exact / blank | loop4 exact / blank | loop5 exact / blank |
|---:|---:|---:|---:|---:|---:|
| 50 | `0.9192 / 0.9972` | `1.0000 / 1.0000` | `1.0000 / 1.0000` | `1.0000 / 1.0000` | `1.0000 / 1.0000` |
| 53 | `0.0000 / 0.6090` | `0.0059 / 0.7245` | `0.1055 / 0.7748` | `0.1641 / 0.7855` | `0.1777 / 0.7871` |
| 58 | `0.0000 / 0.4596` | `0.0000 / 0.4808` | `0.0059 / 0.4858` | `0.0117 / 0.4858` | `0.0117 / 0.4853` |
| 64 | `0.0000 / 0.5272` | `0.0000 / 0.6163` | `0.0020 / 0.6528` | `0.0020 / 0.6612` | `0.0059 / 0.6628` |

A zero-training-step read-only evaluation of the exact checkpoint reports:

| Evaluation | loop5 exact | loop5 blank accuracy |
|---|---:|---:|
| mixed official test | `0.1426` | `0.6390` |
| official 46-50 | `1.0000` | `1.0000` |
| official 51-55 | `0.1914` | `0.7212` |
| official 56-60 | `0.0605` | `0.5505` |
| official 61-64 | `0.0078` | `0.6503` |

Mixed exact across loops 1-5 is
`0.0234 / 0.0254 / 0.0996 / 0.1387 / 0.1426`; mixed blank accuracy is
`0.5439 / 0.6045 / 0.6323 / 0.6375 / 0.6390`. This is not a wider-mask
artifact: Sudoku emits one digit per blank, and later loops remove wrong cells
and duplicate constraint violations.

The separately sampled 256-board case bank gives the following diagnostic
trajectories; its exact rates are not substituted for the primary scores:

| Official range | selected cases | wrong cells loop1/2/3/5 | conflict units loop1/2/3/5 |
|---|---:|---|---|
| 46-50 | 4 | `2.00 / 0.00 / 0.00 / 0.00` | `3.50 / 0.00 / 0.00 / 0.00` |
| 51-55 | 12 | `21.00 / 11.75 / 3.00 / 2.00` | `23.75 / 16.42 / 5.83 / 2.58` |
| 56-60 | 12 | `19.08 / 10.67 / 4.33 / 2.83` | `20.42 / 13.08 / 5.58 / 2.92` |
| 61-64 | 7 | `26.43 / 14.29 / 6.57 / 4.14` | `26.00 / 18.86 / 7.86 / 5.57` |

One 64-blank board moves from `26` wrong cells and `26` conflict units at
loop 1 to `15/21`, `1/3`, and finally `0/0` at loops 2, 3, and 5. A hard
failure still moves in the correct direction, from `30/25` to `16/19`,
`9/14`, and `5/12`, but does not close. These two visualizations establish
both the mechanism's current capability and its remaining boundary.

The delayed scaling slope is now unambiguous: from step 3000 to 6000,
official 51-55 exact rises `0.0098 -> 0.1914`, 56-60 rises
`0.0098 -> 0.0605`, 61-64 rises `0.0000 -> 0.0078`, and mixed rises
`0.0352 -> 0.1426`. However, the historical D224/L12 GDN step-6000 reference
is still ahead at official 51-55 exact `0.3926` and combined 56-64 exact
`0.1270`. This is not a matched asymptotic architecture comparison, but it
prevents a premature claim that GDN2 is already the best carrier.

The preregistered step-6000 stop rule does not fire: official 51-55 is above
`0.10`, 56-60 is nonzero, and every hard exact curve has positive slope. The
run therefore continues unchanged to step 9000; no width, depth, learning-rate,
loss, or seed branch is opened. The exact checkpoint is
`train_state_step006000.pt`, SHA256
`23eaa434720f931965b3057a725b4f1f5a3aa8030da76be2c18c214d6ed9fc11`.

### Step-7100 lease rollover

The unchanged stage-3 continuation reaches step 7100 with train CE `0.7770`.
Before the GPU1 lease expires, the complete model, AdamW, scheduler, data RNG,
and Python/NumPy/Torch RNG checkpoint is hashed and the exact launcher process
group `2102` is stopped. The checkpoint is
`train_state_step007100.pt`, size `72,233,438` bytes, SHA256
`952f3377642748d3f378be010a4443b61823edb6b984f635ded962c82f4860f8`.
GPU1 workload `b1fdabeb-aafd-48ac-86db-1e86d7cac72a` is pending allocation.
Resume must use this exact checkpoint and keep every experiment variable
unchanged.

### Step-9000 gate and step-10000 resume

The unchanged run reached step 9000 in the final `51-64` curriculum stage.
Train CE is `0.6524`. The exact checkpoint is
`train_state_step009000.pt`, size `72,248,862` bytes, SHA256
`606caf5229590f157d7a0f952423c719f4c17688003309a7bd0664e579588dd7`.

The original read-only evaluation from source SHA `42102bd65a28...` reports:

| Evaluation | loop1 exact | loop5 exact | loop5 blank accuracy |
|---|---:|---:|---:|
| mixed official test | `0.0234` | `0.2500` | `0.6949` |
| official 46-50 | `0.9902` | `1.0000` | `1.0000` |
| official 51-55 | `0.0000` | `0.3926` | `0.7803` |
| official 56-60 | `0.0000` | `0.1211` | `0.6084` |
| official 61-64 | `0.0000` | `0.1094` | `0.8058` |

Compared with step 6000, mixed exact rises `0.1426 -> 0.2500` and official
51-55 rises `0.1914 -> 0.3926`. The two harder ranges are nonzero and later
loops still account for almost all solved boards. This passes the original
continuation criterion; ordinary scaling has not yet been falsified.

A subsequent identity-only FutureSeed1 continuation to step 9100 is exactly
the same scientific arm and preserves model, optimizer, scheduler, data RNG,
training RNG, data, loop loss, and kernel. Under the current evaluator it gives
mixed loop5 exact `0.2520`, official 51-55/56-60/61-64 exact
`0.3672/0.1309/0.1973`, and train CE `0.6435`. Its exact checkpoint SHA256 is
`5f80e850c1407a3c3225adb1e335a72026abc66aa9486f609005ce65bad91429`.
It is therefore the nonduplicative resume point for the step-10000 gate.

The step-10000 intervention changes only the amount of final-stage optimizer
compute: `51-64` exposure grows from 1100 to 2000 steps. It does not add a
module, loss term, noise source, traversal change, seed, width, state, or loop.
Prediction: if the remaining failure is ordinary under-training, mixed exact
or mean official 51-64 exact should improve by at least `0.02` while the large
loop1-to-loop5 exact gain remains. If CE/blank improves but both full-board
signals are flat or regress, stop and do not spend the final 2000 steps.

Exact launch config:

- `configs/sudoku/gdn2_scale_resume_10000.env`
- `CUDA_VISIBLE_DEVICES=0 BASELINE_CONFIG=configs/sudoku/gdn2_scale_resume_10000.env scripts/run_canonical_gdn2_scale.sh`

### Step-10000 gate and final continuation decision

The exact step-10000 run is
`gdn2-futureseed-clean-scale-s10000-20260802T104025Z-cfdab41`, bound to clean
source SHA `cfdab41acf5ce5a161307b60c906b94de9f28896`. The train-state checkpoint
is `train_state_step010000.pt`, SHA256
`2b720c4a4a38e5eb37df2c0c30a193f55c14d36167ccb578274c4aaa9806ceca`.
Strict GPU1 preflight verified official FLA source `9c8e42e...`, the official
`GatedDeltaNet2` class, `ChunkGDN2FunctionBackward`, Triton Q/K/V convolutions,
nonzero initial-state gradients, and Torch/CUDA output, state, and gradient
agreement. No fallback or CPU model execution occurred.

| Evaluation | loop1 exact | loop5 exact | loop5 blank accuracy |
|---|---:|---:|---:|
| mixed official test | `0.0234` | `0.2852` | `0.7115` |
| official 46-50 | `0.9961` | `1.0000` | `1.0000` |
| official 51-55 | `0.0000` | `0.4062` | `0.7977` |
| official 56-60 | `0.0000` | `0.1387` | `0.6188` |
| official 61-64 | `0.0000` | `0.1797` | `0.8312` |

Mixed exact across loops 1-5 is
`0.0234 / 0.0312 / 0.1953 / 0.2734 / 0.2852`. Relative to the matched
step9100 readout, mixed loop5 exact improves `0.2520 -> 0.2852`, a `+0.0332`
gain that passes the predeclared `+0.02` gate. Its loop1-to-loop5 gain also
widens `+0.2285 -> +0.2617`. Mean official 51-64 exact rises only
`0.2318 -> 0.2415`; 51-55 improves, while the two harder buckets fluctuate.
The proper conclusion is therefore that ordinary scaling still improves
aggregate full-board closure and recurrent correction, not that every hard
bucket is monotone.

The case bank makes the loop mechanism concrete. One 64-blank board changes
from `31` wrong cells at loop1 to `18`, `3`, `1`, and `0` at loops 2-5; its
constraint-conflict count reaches zero. Other boards stall on a small set of
high-confidence wrong digits. This is genuine iterative correction rather
than a token-accuracy-only gain, but it also identifies the remaining upper
tail.

Decision: continue the unchanged trajectory to the original step12000
endpoint. The only intervention is 2000 additional `51-64` optimizer steps.
The exact launch config is `configs/sudoku/gdn2_scale_resume_12000.env`. No
architecture, loss, noise, order, seed, model size, state size, loop count,
batch, optimizer, scheduler, or evaluator changes are allowed. Because one
GPU1 lease cannot hold all 2000 steps, stop only on a complete 100-step
checkpoint, record the infrastructure rollover, and resume exactly.

### Step-11000 lease rollover

The first final-stage leg reached step11000 before the GPU1 lease boundary.
Train CE over steps10100-11000 is
`0.6347/0.5732/0.6188/0.6889/0.5547/0.6591/0.5702/0.6979/0.7585/0.6527`.
This is stochastic hard-batch variation around the established regime, with no
NaN, OOM, or persistent divergence.

The guard waited for both `train_state_step011000.pt` and the matching
checkpoint-complete log line. It then stopped exact process group `1851` and
wrote `abort.json` with `scientific_failure=false`. The checkpoint is
72,313,438 bytes, SHA256
`3879e6c0af3958fca4b64c2466e6d66867297f487d3287b7bd88148c899e7ed7`.
GPU memory returned to zero before GPU1 restart. The sole valid resume config
is `configs/sudoku/gdn2_scale_resume_12000_from11000.env`; the next leg must
restore model, optimizer, scheduler, data RNG, and training RNG unchanged and
run the remaining 1000 steps on GPU1.
