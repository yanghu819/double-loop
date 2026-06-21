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

- Strong positive: RWKV9 Sudoku with and without FutureSeed, same CUDA
  state-passing backbone and same budget. Without FutureSeed, h12 loop5 exact is
  `0.0156` and early blank cells are much worse than late blank cells. With
  FutureSeed, h12 loop5 exact is `0.9492` and the early/late gap nearly
  disappears.
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
- Loop evidence is mixed: Sudoku scale-up shows loop matters; Maze visualizations
  often show later loops copying the same operating point. A paper claim about
  loops must report precision, recall, predicted mass, and false positives, not
  only final token accuracy.

## Experimental Roadmap

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

E1, E2, and the first E3 official-Maze variant are complete. Official Maze is
useful as a failure analysis tool, but under the current path-weight objective it
is not a positive FutureSeed benchmark. The next experiment should change the
proxy or objective so broad masks are not a cheap answer, while still avoiding
maze rules, search, repair, or selectors.
