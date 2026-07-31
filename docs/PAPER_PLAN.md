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
