# FutureSeed + Loop Paper Plan

## Working Title

Future Seeds for Cheap Bidirectional Computation in Recurrent Reasoners

## Core Claim

FutureSeed is a small, generic state-conditioning mechanism that lets a causal or
recurrent backbone receive a learned summary of future context without paying for
full bidirectional mixing at every layer. Looping then turns that initial
direction into extra computation. The mechanism should be judged by whether it
improves reasoning under the same code path, compute budget, and path-aware task
metric, not by solver-specific repair or selector tricks.

## Draft Abstract

Recurrent neural networks are attractive for reasoning because they can trade
test-time compute for accuracy, but causal recurrence is a poor fit for tasks
whose constraints are defined by both past and future context. We introduce
FutureSeed, a lightweight state-conditioning module that seeds recurrent
computation with a learned summary of future information. FutureSeed does not add
task-specific search, repair, or symbolic rules; it changes only the initial
direction of the recurrent state, leaving looped computation to refine the
answer. On synthetic constraint tasks, FutureSeed removes the position bias of a
causal RWKV backbone and opens hard Sudoku regimes where the same model without
FutureSeed collapses. We further use maze path finding as a diagnostic proxy and
show that standard token accuracy can be misleading when the desired path tokens
are sparse, motivating path-aware evaluation. Our experiments are designed to
separate three questions: whether FutureSeed supplies cheap bidirectional
information, whether loops perform real correction rather than copying a broad
mask, and whether the same mechanism improves official EqR-style recurrent
reasoning under matched compute. The resulting evidence supports FutureSeed as a
general mechanism for cheap bidirectional context in recurrent reasoners, while
also identifying objective alignment and late-loop correction as the main limits
for scaling to harder spatial reasoning tasks.

## Current Evidence

- **Important retained milestone:**
  [`Address/Payload-Factorized GDN2`](MILESTONE_GDN2_ADDRESS_PAYLOAD.md) is the
  canonical algorithm record for the position-Q/K carrier, its matched-compute
  evidence, its scaling boundary, and the claims that are not yet allowed.
- Strong cross-carrier causal gate: under byte-identical paired initialization,
  the same data/order/objective/optimizer/state/loop budget, and official
  kernels, b46-50 loop5 full-board exact with/without FutureSeed is
  `0.7285/0`, `0.3633/0`, `0.7969/0`, and `0.6465/0` for
  RWKV7/GDN/GDN2/KDA. FutureSeed adds no parameters, costs
  `7.2-17.2%` wall time and about `0.29-0.34 GiB`, and passes a separate
  same-trained-weights functional contract. This supports a generic
  short-budget causal opening claim, not hard closure: every arm is still zero
  exact at 51-64 blanks.
- New carrier-side mechanism evidence: under randomized Sudoku cell traversal,
  separating GDN2 memory address from payload raises equal-compute step9100
  mean official 51-64 blank accuracy from `0.2037` to `0.4333`. At matched
  step9300 it raises the same metric from `0.2753` to `0.5437` and lowers CE
  from `1.5707` to `0.9661`. Canonical position drives only Q/K; content still
  drives V and all memory-edit/output gates, with no new parameters or
  recurrent-kernel change. On matched 64-blank batch69, normal GDN2 changes
  `53->50->50->51->51` wrong cells while the split carrier changes
  `32->21->18->15->15`. This is persistent optimization and loop-correction
  evidence. A clean step9300->9600 hard-stage continuation first opens exact
  to `0.0059` in 51-55 and 56-60 through later loops, but mean hard blank only
  rises `+0.0364` and 61-64 exact remains zero. The paper can claim improved
  address learning and conditional recurrent correction, not robust global
  closure. The next evidence gate needs matched capacity/data scaling.
- Strong positive: RWKV9 Sudoku with and without FutureSeed, same CUDA
  state-passing backbone and same budget. Without FutureSeed, h12 loop5 exact is
  `0.0156` and early blank cells are much worse than late blank cells. With
  FutureSeed, h12 loop5 exact is `0.9492` and the early/late gap nearly
  disappears.
- Correction: one official EqR mixer-replacement probe was misnamed as
  FutureSeed. It actually replaced the token mixer with a forward+reverse
  recurrent scan. That is not FutureSeed and must not enter the main evidence
  chain. The side probe failed the e1024 gate: official mixer-base reached
  `accuracy=0.6644`, `exact=0.0249`, `lm_loss=0.7664`, while the scan
  replacement stayed at `accuracy=0.4231`, `exact=0`, `lm_loss=1.4029`.
- Strong boundary: official EqR Maze e256/e512 token accuracy is high but path
  F1 is near zero under plain token CE. A generic path-token-weighted objective
  opens path recovery to loop16 path F1 about `0.468`, but the model predicts a
  broad high-recall mask rather than a clean path.
- Official EqR + FutureSeed is neutral under the path-aware objective: loop16
  path F1 is `0.4684` versus clean EqR `0.4682`. This suggests EqR's mixer
  already supplies the noncausal interaction that FutureSeed is meant to cheaply
  add to causal/recurrent backbones.
- Causal RWKV on the same official path-weighted Maze objective is also neutral:
  no-FutureSeed loop8 path F1 is `0.4690`; FutureSeed loop8 path F1 is `0.4666`.
  Both arms converge to broad high-recall masks, so this objective does not
  isolate the future-context mechanism.
- A follow-up generic PATH/non-PATH boundary objective makes broad masks costly
  but exposes a different failure. no-FutureSeed loop8 path F1 falls to
  `0.4089`; FutureSeed improves to `0.4278` by preserving recall, but loop gain
  is still near zero. Treat this as weak evidence that FutureSeed can protect
  useful future-context signal under pruning pressure, not as a Maze win.
- A learned path-budget decoder separates mass calibration from ranking. Both
  no-FutureSeed and FutureSeed learn the total PATH fraction accurately, but
  budgeted decoding misses many true-path cells: no-FutureSeed budget loop8 F1
  is `0.3387`, FutureSeed is `0.3289`, and both produce about `78-80` false
  negatives per case. This shows the Maze bottleneck is not simply predicted
  mass; it is true-vs-false PATH ordering and late-loop correction.
- The official EqR causalized-mixer gate is negative as a FutureSeed result.
  With the same official code path, path-weighted objective, and e256 budget,
  causal base loop16 path F1 is `0.468067`; causal+FutureSeed is `0.468157`
  (`+0.000090`). Both arms still predict a broad high-recall mask with
  precision about `0.306`, recall about `1.0`, predicted PATH fraction about
  `0.433`, and roughly `270` false positives per case. FutureSeed has an early
  step500 optimization edge, but no final path-aware win.
- Important caveat: the current official EqR FutureSeed patch is a cross-level
  H/L latent injection. It is useful as an add-on boundary test, but it is not
  a clean "cheap bidirectional" implementation under causal attention because
  it does not introduce an independent future-token source. The neutral Maze
  causal/compression results should therefore not be over-read as disproving the
  broader FutureSeed mechanism.
- The official EqR H96 mixer-compression gate is also negative. H96 base loop16
  path F1 is `0.467337`; H96+FutureSeed is `0.467160` (`-0.000178`). Both arms
  keep recall at `1.0`, predicted PATH fraction around `0.434`, and roughly
  `271` false positives per case. Compression did not reveal hidden FutureSeed
  value on this Maze objective.
- The generic RWKV denoising-attractor feedback probe is negative. no-FutureSeed
  DAT at step300 has loop1/loop16 path F1 `0.4683/0.4680` and FP
  `268.3->269.0`; FutureSeed DAT has `0.4691/0.4688` and FP `265.9->266.4`.
  DAT loss decreases, but the feedback loop learns the same broad-mask fixed
  point. FutureSeed changes the operating point only slightly (`+0.0008`
  loop16 F1 over no-FS), not the loop correction behavior.
- Loop evidence is mixed: Sudoku scale-up shows loop matters; Maze visualizations
  often show later loops copying the same operating point. A paper claim about
  loops must report precision, recall, predicted mass, and false positives, not
  only final token accuracy.

## Experimental Roadmap

### E0. Quality-Matched Causal Efficiency

Hypothesis: if FutureSeed is a cheap future-context mechanism rather than only
an early curriculum shortcut, an efficient official causal carrier with
FutureSeed should reach the same full-board quality in materially fewer
optimizer steps or open 51-55 exact under the same clean scaling recipe.

Method: select one carrier from the strict four-way gate, then train one FS and
one noFS arm on the same independent-data schedule. Compare time/tokens to a
preregistered quality target and evaluate 46-50/51-55/56-64 exact at fixed
checkpoints. Do not tune the two arms separately and do not add noise, repair,
selector, search, or task-specific losses.

Decision:
- A publishable efficiency result requires at least `20%` less wall time or
  tokens to the same exact target, or a nonzero 51-55 exact frontier that noFS
  does not reach at matched compute.
- If both arms eventually converge at similar cost, narrow the claim to
  finite-budget optimization.
- If neither opens 51-55, stop carrier tables and treat global closure as a
  separate state-capacity/data-scaling problem.

### E1. Official EqR Maze Objective Alignment

Hypothesis: the official Maze failure is mostly an objective/metric mismatch.
Plain token CE is dominated by non-PATH cells, so the model can score about
`0.87` accuracy while predicting almost no path. A generic class-balanced loss
should make official Maze a meaningful proxy before we compare FutureSeed.

Method: patch the official EqR code path with an optional class-balanced token
loss or path-token weight. Train only the clean official EqR baseline first. This
is not a solver and does not encode maze rules; it aligns the supervised loss
with the path-recovery metric.

Prediction: if the proxy is viable, loop16 path F1 and predicted PATH fraction
should move away from zero by step500-1000. If it still predicts no PATH, Maze is
not ready for FutureSeed claims under this setup.

Decision:
- Result: passed the viability gate. Clean EqR reaches loop16 path F1 `0.4682`
  with recall `0.9998`, precision `0.3064`, and pred PATH frac `0.4325`.
- Interpretation: objective alignment works, but the task remains a broad-mask
  diagnostic rather than solved path finding.

### E2. Official EqR Maze FutureSeed Under Matched Objective

Hypothesis: once the Maze objective actually rewards path recovery, FutureSeed
should improve sample efficiency or final path F1 under the same official code
path and compute.

Method: matched base vs FutureSeed run under the E1 objective. Same official EqR
SHA, same data, same batch, same steps, same visualization runner.

Prediction: a real FutureSeed win should show better path F1, better recall at
similar predicted mass, or earlier opening at step500/1000.

Decision:
- Result: neutral. FutureSeed loop16 path F1 `0.4684`, clean EqR `0.4682`,
  delta `+0.0001`; hard-case false positives remain about `270` per case.
- Decision: do not claim FutureSeed beats official EqR. Use EqR as the strong
  noncausal baseline and boundary result.

### E3. Causal Maze Backbone With FutureSeed On/Off

Hypothesis: EqR's mixer already has noncausal mixing, so FutureSeed may be
masked. A causal/RWKV Maze backbone should expose the same future-context
failure mode seen in Sudoku.

Method: use the existing CUDA recurrent backbone, train Maze path recovery with
and without FutureSeed, and visualize loop trajectories. No selector, repair, or
maze-specific search.

Prediction: no-FutureSeed should show directional or position bias; FutureSeed
should improve path recall/F1 and reduce that bias.

Decision:
- Result on official path-weighted Maze: negative. RWKV no-FutureSeed reaches
  loop8 path F1 `0.4690`; RWKV + FutureSeed reaches `0.4666`; loop gains are
  near zero or negative.
- Decision: do not use official path-weighted Maze as a positive FutureSeed
  benchmark. It mostly measures broad PATH coverage and leaves hundreds of false
  positives per case.

### E4. Loop Correction Diagnostics

Hypothesis: FutureSeed gives a good initial direction, but top-conference-level
evidence needs to show whether loops actually correct errors.

Method: for the best Sudoku and Maze runs, report loop1/4/8/16 precision, recall,
predicted mass, false positives, false negatives, and hard-case visualizations.

Prediction: a strong loop story requires later loops to reduce false positives
without destroying recall. If later loops only copy loop1, the paper should not
overclaim loop correction.

Decision:
- Keep loop as a variable-compute mechanism if late-loop correction is visible.
- Otherwise position loop as compute reuse and future work for state dynamics.

### E5. Causal Maze Boundary Objective

Hypothesis: the path-weighted Maze objective permits broad-mask shortcuts. A
generic PATH/non-PATH boundary objective should make false positives expensive
without adding maze rules or postprocessing. If FutureSeed provides cheap future
context to a causal backbone, it should help preserve true path cells when the
model is pushed away from high-recall coverage.

Method: causal RWKV7 state-passing Maze runner, D128/L8, train loops 4, eval
loops 8, same official `maze-30x30-unique-1k` data. Compare
`FUTURE_SEED_SCALE=0` and `1` under matched compute. Objective is token CE with
PATH weight plus a generic PATH-vs-non-PATH margin BCE and per-sample PATH-mass
calibration.

Decision:
- Result: weak positive for FutureSeed, negative as a final benchmark.
  no-FutureSeed loop8 path F1 is `0.4089`; FutureSeed is `0.4278`.
  The gain is mainly recall (`0.5828 -> 0.6280`), while precision is almost
  unchanged (`0.3256 -> 0.3270`). Loop gain remains effectively zero.
- Interpretation: the objective can reduce broad-mask coverage, but it overprunes
  true path cells. Do not sweep boundary weights. Maze remains useful as a
  failure analysis tool, not yet as a positive FutureSeed benchmark.

### E6. Causal Maze Learned Budget Decoder

Hypothesis: the path-weighted Maze failure may be a calibration problem: the
causal RWKV logits may rank true PATH cells above false positives, while raw
argmax uses the wrong boundary.

Method: add a generic learned path-count head and decode PATH as the top
model-ranked cells under the model's own predicted budget. Compare no-FutureSeed
and FutureSeed under matched compute. No oracle true count, selector, search,
repair, or maze rules.

Decision:
- Result: negative but informative. The budget head learns path fraction well
  (`abs_err≈0.0095-0.0098`), but budget decoding trades false positives for false
  negatives. no-FutureSeed budget loop8 F1 is `0.3387`; FutureSeed is `0.3289`.
- Interpretation: the model has mass calibration but lacks ranking/correction.
  FutureSeed does not repair this Maze bottleneck. Do not sweep count/budget
  weights; the next useful Maze experiment must target generic ranking or
  self-correction directly.

### E7. Official EqR Causalized-Mixer Gate

Hypothesis: if FutureSeed cheaply supplies future-side information, removing
full noncausal attention from the official EqR mixer should create a condition
where FutureSeed helps a causalized EqR path recover Maze structure.

Method: official EqR upstream SHA `aba94e9`, same official
`maze-30x30-unique-1k` data, same path-weighted objective, same e256 budget,
same path-aware visualization. Compare causalized EqR no-FutureSeed against the
same causalized code path patched with FutureSeed. No selector, search, repair,
or maze-specific postprocessing.

Decision:
- Result: negative. causal base loop16 path F1 `0.468067`, causal+FutureSeed
  `0.468157`, delta `+0.000090`.
- Loop behavior remains broad-mask fixed point. Loop1 to loop16 gains are only
  about `+0.0018`, and false positives remain around `270` per case.
- Interpretation: causalizing EqR does not make this Maze proxy expose the cheap
  bidirectional value. The path-weighted objective still mostly rewards high
  recall broad masks. Do not sweep FutureSeed gate bias, seed, path weight, or
  causal-attention details.

### E8. Official EqR H96 Mixer-Compression Gate

Hypothesis: if FutureSeed cheaply substitutes for part of the noncausal mixer,
then a compressed official EqR model should benefit more from FutureSeed than
the full D128 model did.

Method: official EqR upstream SHA `aba94e9`, official `maze-30x30-unique-1k`,
path-token weight `8`, e256 budget, `hidden_size=96,num_heads=8`, matched base
vs FutureSeed, no selector/search/repair/postprocessing.

Decision:
- Result: negative. H96 base loop16 path F1 `0.467337`; H96+FutureSeed
  `0.467160`, delta `-0.000178`.
- Both arms are broad masks: precision about `0.306`, recall `1.0`,
  predicted PATH fraction about `0.434`, false positives about `271` per case.
- Interpretation: official EqR Maze path-weighted objective is not a positive
  FutureSeed benchmark under full, causalized, or compressed mixer settings.
  Stop hidden-size/gate/seed sweeps.

## Bitter-Lesson Boundaries

Allowed:
- Larger clean training budgets.
- Generic loss alignment to the task metric.
- Generic state dynamics such as learned FutureSeed gates or normalization.
- Causal/backbone comparisons under matched compute.

Not allowed:
- Maze repair, Sudoku repair, symbolic search, rollout selector, or best-of-K
  oracle at inference.
- Maze-specific hand rules.
- Broad seed/weight/temperature tables without a decision.
- Claiming success from token accuracy when the target metric is path recovery.

## Immediate Next Experiment

E1, E2, and the first E3 official-Maze variants are complete. They show that
token accuracy is misleading, broad-mask path recovery is easy to learn, and
extra decision heads or mass objectives do not yet make loops reliably repair
false positives.

The official-EqR mixer-compression gate is now complete and still broad-mask
neutral. Maze should not be used as positive FutureSeed evidence yet. The only
remaining Maze work worth running is a generic recurrent state/training
dynamics experiment that directly tests whether loops can leave the broad-mask
attractor without search, repair, selector, or maze-specific rules.

The denoising-attractor probe is now complete and negative. It did not reduce
false positives while preserving recall; both no-FutureSeed and FutureSeed arms
hit the step300 broad-mask kill rule. Maze should therefore remain failure
analysis unless the next mechanism changes the recurrent decision/state dynamics
more substantially while still staying generic.

P-MAZE-007 is complete. Abortable probes now dump hard-case
input/target/loop1/4/8/16 visuals before termination. The verification run is
not positive model evidence: at step100, loop1 and loop16 were the same broad
mask (`F1 0.4614 -> 0.4614`, precision `0.3006 -> 0.3006`, recall `1.0`,
FP `272.1 -> 272.1`). Its value is that future Maze failures are now auditable.

The next Maze mechanism is only worth running if it creates a genuinely
different state variable for true-vs-false PATH competition; another
loss-weight, corruption-mix, temperature, or seed sweep is explicitly low ROI.
For small runner probes, prefer a minimal Git archive of required source paths
when full worktree/source snapshot checkout stalls on historical tracked
artifacts.

The paper claim is allowed to proceed only if FutureSeed either improves hard
path F1 by `>= +0.03` without broad-mask inflation, improves official Sudoku
sample-efficiency/exact under a matched cheap backbone, or reaches the same
accuracy with `>=20%` lower train/inference compute. If the official EqR
baseline cannot be reproduced, stop there. If FutureSeed only increases token
accuracy or predicted-path coverage without improving exact/FP/FN, treat it as
a negative result and visualize the failure.

Do not run another weight/seed/temperature table for Maze.

The official-codebase bidirectional scan mixer replacement probe is complete and
is off-mainline. It should not be described as FutureSeed. The next paper
experiment should not be another old-patch gate or a longer run of that scan
replacement.

The highest-ROI next experiment is a simple generic mechanism change that keeps
the FutureSeed idea but avoids the stable-wrong attractor:

- Use official EqR as the strong noncausal ceiling and baseline.
- Keep FutureSeed as terminal-state seeding: a recurrent layer's final state
  seeds a later layer or loop. Do not add a hidden right-to-left scan under the
  FutureSeed name.
- Test either a gated/normalized FutureSeed state update or a minimal setting
  where FutureSeed supplies a future-conditioned initialization while learned
  capacity handles final decisions.
- Prefer Sudoku sample efficiency/exact for the first official-codebase gate
  because Maze path-weight repeatedly collapses to broad masks.
- Keep the mechanism generic: no Sudoku rules, no solver, no selector, no
  best-of-K oracle, no repair.
- Success means FutureSeed improves matched official sample efficiency without
  losing long-budget exact, or reaches the same quality with less recurrent
  compute. Failure means the current formulation is only an early optimization
  aid, not a standalone mixer replacement.

## 2026-08-04 Cross-Task Directionality Gate

`P-CAUSAL-007` supplies the first clean non-Sudoku causal mechanism result. In
the validated upstream Zoology shell with pinned official FLA GDN2, the matched
no-FutureSeed model reaches 93.05% on write-before-query retrieval but only
1.10% on query-before-write retrieval. Native cross-layer terminal-state
seeding reaches 99.55% and 99.30%, respectively, with identical parameters,
initialization, data, optimizer, kernel, and epoch budget. Future-query exact
changes from 0% to 98.60%.

This supports the narrow paper claim that FutureSeed is a cheap future-context
route for causal recurrent stacks. It does not yet support language quality,
asymptotic scaling, or wall-time efficiency. The next paper gate should test
one meaningful transfer axis rather than repeat seeds: OOD sequence/association
load or an established masked/retrieval language task with a warmed full
noncausal reference.

## 2026-08-04 Established-Text Gate Boundary

`P-CAUSAL-008` does not extend the paper claim to natural language. The matched
WikiText byte-MLM run produced masked accuracy `0.4247` for causal GDN2 and
`0.4218` for FutureSeed, but the proposed full-bidirectional attention ceiling
reached only `0.1879`. This fires the preregistered carrier-validity kill rule:
the attention reference did not learn the task, so the run cannot say whether
FutureSeed closes a meaningful future-context language gap.

The mechanism implementation itself passed the strict checks: scale 0 was
identical to causal GDN2, causal future dependency was zero, FutureSeed future
dependency and gate gradient were nonzero, both recurrent arms had identical
parameters and initialization, and official FLA/Triton executed without a
fallback. FutureSeed was nevertheless slightly worse in this exact invalid
carrier. Report that boundary honestly; do not present it as either positive
or negative language evidence and do not tune the failed attention shell.

The next paper gate must start by reproducing a validated, established
bidirectional masked-language implementation and recipe. It must visibly beat
a strict causal reference on masked recovery before any recurrent mixer is
substituted. Once that carrier is valid, compare matched official-FLA GDN2
scale 0 and native FutureSeed scale 1 with the same data, initialization,
training tokens, optimizer, width, depth, and metric. Until then, P-CAUSAL-007
remains the clean cross-task directionality evidence and language transfer is
an open question.

### P-CAUSAL-009 Established MLM Boundary

The exact Transformers BERT-mini carrier did learn on the fixed WordPiece
WikiText stream, but it reached only `0.0716` masked accuracy at 20.48M input
tokens and missed the preregistered `0.10` endpoint. No causal, GDN2, or
FutureSeed arm was run. This is not a language result for or against
FutureSeed. The paper should now prioritize the validated directional-MQAR
scaling and cost frontier; natural-language transfer remains a later gate that
requires a genuinely opened published recipe.

### P-CAUSAL-010 Scaling Carrier Boundary

The L64 matched GDN2 pair replicated the mechanism strongly: no-FutureSeed
past/future accuracy was `0.9965/0.0100`, while native FutureSeed reached
`0.9985/0.9920` and `0.981` joint exact. However, the registered
parameter-matched full SDPA reference reached only about `0.50` on both
directions in ten epochs, despite a verified nonzero future dependency. The
run therefore stopped before L1024. This is not a Transformer comparison.
Before drawing a scaling curve, reproduce the exact upstream Zoology MHA with
only its causal mask removed and allow the pre-existing official opening
budget; only an opened attention carrier can define the quality/cost ceiling.

### P-CAUSAL-011 Official-MHA Mask-Removal Boundary

The stricter carrier check also failed. The exact upstream MHA, with identical
parameters and initialization and only its causal mask removed, had verified
future dependency but finished 30 epochs at past/future accuracy
`0.4850/0.4845` and joint exact `0.048`. This closes the mask-removal carrier,
not bidirectional Transformers in general. The paper must not claim an
attention quality win from P-CAUSAL-010/011. The next core figure should first
establish causal GDN2 versus native FutureSeed length scaling at L64/L1024;
an attention quality/cost frontier remains conditional on a separate task and
published bidirectional recipe that independently opens.

### P-CAUSAL-012 Long-Context FutureSeed Boundary

The direct GDN2 endpoint gives strong but bounded evidence. At length64,
causal/FutureSeed past accuracy is `0.9965/0.9985` and future accuracy is
`0.0100/0.9920`. At length1024 under the same ten-epoch, four-association
protocol, causal GDN2 does not open either direction (`0.0110/0.0085`), while
FutureSeed reaches past/future accuracy `0.7535/0.7415`, directional exact
`0.590/0.572`, and joint exact `0.339`. The future delta is `+0.733`, but the
registered `0.80` future and `0.90` past thresholds and 80% L64-retention gate
are missed. The paper may claim a large long-context optimization and
information-routing benefit, not length-invariant quality.

The error structure gives the next mechanism question. At L1024, `82.4%` of
wrong future predictions and `78.9%` of wrong past predictions are valid
values belonging to another key in the same sample. FutureSeed transports the
value set but increasingly confuses bindings. Test one generic recurrent
address/state-capacity increase at L1024 before filling middle lengths. If it
reduces these swaps and opens the registered endpoint, then measure the full
length curve; otherwise treat terminal-state compression/update quality as the
limit. Do not rescue P-CAUSAL-012 with epochs, seed, LR, loss, or task rules.

Do not use the archived sequential throughput difference as a cost claim. The
two arms shared one Triton process and likely inherited different autotuning
cache state. The final cost-quality figure requires independent fresh-process
warmup and alternating execution order, plus a separately validated
bidirectional attention or bidirectional recurrent baseline.

### P-CAUSAL-013 Whole-Model Capacity Boundary

The first capacity intervention failed cleanly. At the exact L1024/K4 endpoint,
official-FLA GDN2 D256/L2/H8/D32 doubled recurrent-state values per layer but
increased total parameters `3.39x`. Both matched D256 arms stayed near chance:
causal past/future accuracy was `0.0145/0.0120`, and FutureSeed was
`0.0090/0.0115`, versus frozen D128 FutureSeed `0.7535/0.7415`. Validation CE
rose after epoch5 in both larger arms, while D128 FutureSeed had opened sharply
from epoch3. The FutureSeed path itself remained active and all official
FLA/Triton, data, initialization and dependency checks passed.

This result rejects the claim that undifferentiated width scaling is sufficient
under matched optimization compute. It does not show that larger recurrent
state is intrinsically harmful: width, parameter count and state capacity moved
together, and the larger model failed before binding diagnostics became
meaningful. The next high-information test is state-only scaling at the proven
D128 geometry through official GDN2 value expansion. Do not run middle lengths
or a D256 epoch/LR/seed rescue before that isolation test.

### P-CAUSAL-014 Value-State Axis Boundary

The isolation test also failed, but it makes the next question more precise.
With D128/L2/H4 and key dimension32 fixed, official GDN2 `expand_v=2` doubled
value-state scalars per layer from4,096 to8,192. It added22.49% parameters,
reduced warmed diagnostic throughput15.48%, and increased peak training memory
10.72%. Native FutureSeed balanced accuracy fell from the frozen `0.7475` to
`0.27675`; past/future accuracy was `0.2555/0.2980`, and joint exact was zero.
The matched causal arm stayed directionally valid with future accuracy0.0125.

This is not evidence that FutureSeed stops working: within the expand-v2
carrier it still adds `+0.25025` balanced accuracy over scale0. It is evidence
that widening the V/payload axis is a poor way to repair long-context binding.
The recurrent matrix remained K32 x V64, so the number of independent key
address directions did not increase. The main remaining paper mechanism
question is whether generic address separability or FutureSeed compression can
scale without changing the already-open D128 residual backbone. Do not present
state-element count as address capacity, and do not continue expand-v, epochs,
LR, or seeds.

### P-CAUSAL-015 Relative-Address Attention Boundary

Adding parameter-free standard RoPE to the frozen near-parameter-matched full
SDPA carrier did not open it. Strict preflight proved identical 539,136
parameters and initialized tensors, bit-exact `rope_scale=0`, exact data,
nonzero future dependency and finite CUDA backward. After 30 epochs, RoPE SDPA
reached past/future accuracy `0.4955/0.4820` and joint exact `0.046`, versus
plain SDPA `0.4980/0.4995` and FutureSeed `0.9985/0.9920` with joint exact
`0.981`. Its best aggregate validation accuracy was only `0.5030` at epoch 2.

All wrong predictions from all three bidirectional attention controls are
another valid value from the same sample. They learn the candidate value set
but not the key/value binding. Relative position alone does not repair that
algorithmic ambiguity in this two-layer shell. Close this MQAR proxy for an
attention quality/cost claim and do not rescue it with RoPE, optimizer, model,
epoch or seed tuning. The paper can retain the matched causal-versus-FutureSeed
directionality and long-context results, but any Transformer ceiling must be
established on a separate carrier that independently opens.

### P-CAUSAL-016 FutureSeed Context-Length Curve

The matched five-point curve is now strong paper evidence. Causal/FutureSeed
future-query accuracy at lengths `64/128/256/512/1024` is respectively
`0.0100/0.9920`, `0.0085/0.9810`, `0.0090/0.9780`, `0.0105/0.9850`, and
`0.0085/0.7415`. FutureSeed joint exact is
`0.981/0.955/0.932/0.942/0.339`; causal joint exact is zero throughout. All
arms use the same official-FLA GDN2 D128/L2/H4/D32 architecture, parameters,
data recipe, optimizer, training-token budget and CUDA kernels at each length.
All preregistered absolute and matched-delta gates pass.

The result changes the scaling diagnosis. There is no gradual quality dilution
through 512 tokens: FutureSeed future accuracy stays in `0.978--0.992`. The
single large drop is `0.2435` from L512 to L1024. At L1024, `82.4%` of wrong
future predictions select another key's valid value from the same sequence,
so the remaining boundary is address binding/compression rather than absence
of future content. Wider whole models and wider V state already failed; do not
reopen those axes or fill additional middle lengths.

The paper may now claim that native terminal-state seeding provides a compact
future-context route that scales cleanly through 512 tokens and remains highly
consequential at 1024. It still may not claim superiority to bidirectional
Transformers: all attempted attention carriers on this proxy failed their own
binding gate. The final cost-quality figure requires an independently opened
published bidirectional carrier. P-CAUSAL-016 peak-memory differences are
consistently about `1.50 MiB`, but its 20-step timing samples are too short and
variable for a precise throughput claim; rerun only a robust repeated timing
protocol when the valid comparison carrier exists.

### P-CAUSAL-017 Explicit Bidirectional Recurrent Boundary

A conventional two-pass recurrent ceiling was tested at L512: every layer ran
independent official-FLA GDN2 streams in forward and reversed order, flipped
the reverse output back, concatenated both streams and learned a linear fusion.
This baseline had 894,608 parameters versus 596,048 for causal and FutureSeed
GDN2. Strict preflight proved active future dependency, four official
chunk/Triton streams and nonzero reverse/fusion gradients.

The carrier nevertheless failed. Its past/future accuracy was
`0.0345/0.0120`, joint exact was zero and best aggregate validation accuracy
was only `0.02525`; frozen FutureSeed reached `0.9860/0.9850` and `0.942`
joint exact on the same L512 data. Robust cost measurement was stopped because
comparing speed against a chance-level ceiling would not answer the paper
question.

This cannot support a claim that FutureSeed beats bidirectional recurrent
models in general. It does support a more precise mechanism interpretation:
reverse visibility alone is insufficient when each causal stream still must
retain random associations over long distances. FutureSeed supplies a
trainable cross-layer memory route that helps both formally future and already
causally available long-range bindings. The paper still needs an independently
opened published bidirectional carrier before making a quality-cost claim.

### P-CAUSAL-018 Established Real-Text Carrier

The fixed WikiText masked-token carrier is now independently validated with the
official pretrained Google BERT-Tiny checkpoint. The same 4,416,698 checkpoint
parameters were evaluated with native bidirectional attention and with only the
attention semantics changed to strict causal. On 4,742 deterministic masked
targets, accuracy was `0.355546` versus `0.215732`, a `+0.139814` bidirectional
gain; CE was `4.033269` versus `5.453952`, a `1.420684` improvement. Strict
causal future dependency was exactly zero while native bidirectional dependency
was `3.567824`. Every preregistered carrier check passed.

This is carrier evidence, not a FutureSeed result and not a fair trained-model
quality comparison. It proves that the fixed text task exposes useful right
context under an established checkpoint, removing the invalid-carrier problem
that killed P-CAUSAL-008/009. The next paper experiment is one separately
preregistered, matched official-FLA GDN2 scale-0/scale-1 replacement with common
lexical initialization. Do not add a model-size, tokenizer, mask, seed, or
training-budget table before that causal gate.

### P-CAUSAL-019 Matched Real-Text FutureSeed Gate

The first fair real-text replacement is valid but weak. Two official-FLA GDN2
arms used identical D128/L2/H4/D32 parameters, frozen tied BERT-Tiny word
embeddings, fresh contextual modules, 160,000 identical corruption/window
pairs, 20.48M input tokens and the same optimizer. The only difference was
native terminal-state FutureSeed scale 0 versus 1. Strict preflight proved
scale-0 identity, exact causal non-leakage, active FutureSeed dependency,
official chunk/Triton execution and byte-identical provenance.

Causal/FutureSeed masked accuracy is `0.276466/0.278996`; CE is
`5.350222/5.262873`. The `0.087349` CE gain has paired-window 95% interval
`[0.06723,0.10778]` and the same sign at steps 1000 and 1250, but it misses the
registered `0.20` strong threshold. Accuracy gains only 12/4,742 targets and
its interval crosses zero. The correct paper wording is therefore not that
FutureSeed improves language-model accuracy.

The mechanism itself transfers: removing the suffix changes causal CE by
exactly zero but worsens FutureSeed CE by `0.439155`, with 95% interval
`[0.32460,0.56772]`. FutureSeed repairs 119 top-1 predictions while regressing
107. It has learned to use right context, but the one available L1-to-L2 state
transfer mostly improves probabilities rather than decisions. Retain this as
real-text directionality evidence and a scaling motivation. Do not tune or
extend the exact L2 endpoint; a future language test must change one meaningful
scaling axis, preferably depth/multiple transfer opportunities, under a new
preregistration.

### P-CAUSAL-020 Real-Text Depth Boundary

The registered depth test scaled only the official-FLA GDN2 stack from L2 to
L4 while keeping D128 state width, WordPiece/WikiText tensors, frozen lexical
table, optimizer, seed and 20.48M training tokens fixed. The scale-0 and
scale-1 arms each had 4,965,722 parameters. Strict preflight verified three
active terminal-state routes with nonzero gate gradients and zero causal future
dependency.

L4 causal/FutureSeed accuracy is `0.276677/0.284268`; its `+0.007592` paired
interval is `[0.00084,0.01426]`. CE is `5.323694/5.224210`, a `0.099484`
improvement with interval `[0.07965,0.11964]`. Suffix removal costs FutureSeed
`0.586523` CE, confirming stronger right-context use than L2. Yet CE advantage
increases by only `0.01213` over L2 and misses the registered `0.05` depth gain,
the `0.20` strong CE gate and the `+0.03` accuracy gate.

The paper may state that the real-text route persists across depth and becomes
statistically visible in top-1 accuracy. It may not state that depth scaling
solves masked language modeling or yields competitive quality. Close small
depth sweeps; a future language headline requires a materially larger
pretraining regime or another independently validated carrier.

### P-CAUSAL-021 Real-Text Data-Diversity Scaling

The previous fixed-token runs repeatedly remasked only 10,000 grouped text
windows. P-CAUSAL-021 replaced that schedule with 160,000 independent windows
from the full, hash-pinned WikiText-103 raw training split while holding the
official-FLA GDN2 D128/L4 architecture, initialization, optimizer, batch size,
1,250 steps and 20.48M input tokens fixed. This isolates data diversity from
model and compute scaling.

Causal/FutureSeed masked accuracy is `0.275833/0.292493`, a `+0.016660`
difference with paired 95% interval `[0.00973,0.02364]`. CE is
`5.300604/5.131930`, a `0.168674` FutureSeed advantage with interval
`[0.14567,0.19203]`. Relative to P-CAUSAL-020, the CE advantage grows by
`0.069190`, passing the preregistered data-diversity mechanism gate; the
accuracy advantage grows by `0.009068`, narrowly missing its separate `0.01`
gate. FutureSeed makes 182 repairs and 103 regressions, for 79 net repairs
versus P-CAUSAL-020's 36. Suffix removal costs FutureSeed `0.689943` CE and
causal GDN2 exactly zero, so the gain still comes from usable right context.

This is the clearest real-text scaling result so far, but it remains below the
strong paper gate of `+0.03` accuracy or `0.20` CE. The paper may claim that
independent language-data diversity amplifies the FutureSeed advantage at
fixed compute. It may not yet claim competitive masked-language quality. The
positive endpoint slope and passed mechanism gate authorize one clean joint
data-and-compute continuation at the same model size; they do not authorize a
model-size, seed, learning-rate or loss sweep.

### P-CAUSAL-022 Joint Data-and-Compute Language Scale

The preregistered continuation kept P-CAUSAL-021's D128/L4 official-FLA GDN2,
4,965,722 parameters, initialization, optimizer, seed, tokenizer, validation
and kernel fixed. It consumed 640,000 independent WikiText windows once over
5,000 steps, or 81.92M input tokens per arm. The only arm difference remained
native terminal-state FutureSeed scale 0 versus 1.

Causal/FutureSeed masked accuracy is `0.309363/0.373893`, a `+0.064530`
difference with paired 95% interval `[0.05492,0.07438]`. CE is
`4.618276/3.905131`, a `0.713145` advantage with interval
`[0.66992,0.75717]`. Both strong paper routes pass. The advantage also scales
smoothly from step1000 to step5000: accuracy delta grows
`0.00569->0.06453` and CE advantage `0.15974->0.71315`. Suffix removal costs
FutureSeed `2.12152` CE and causal exactly zero, directly tying the gain to
right-context use.

The aggregate visual audit contains 440 repairs and 134 regressions, for 306
net corrected targets across both halves of the text windows. Warmed diagnostic
throughput is 499k/475k tokens per second for causal/FutureSeed and peak
training allocation differs by about 19 MB. These support small implementation
overhead but are not yet a full bidirectional systems frontier.

P-CAUSAL-022 upgrades the real-text result from weak mechanism evidence to a
strong matched quality result: FutureSeed's benefit grows sharply under
ordinary independent-data and training-compute scaling. The paper should use
P019/P020/P021/P022 as a decision sequence, not as an ablation grid. Do not add
intermediate token budgets, multiple seeds or width/depth tables. The next
headline experiment must either move to a materially larger established
language regime or compare quality and robust cost against a valid
bidirectional carrier.
