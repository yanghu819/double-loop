# P-DIAG-MOMVEL-001: Momentum Key-Locality Diagnostic

## 1. Question

P059 leaves only 223 errors after improving directional MQAR L1024 balanced
accuracy from `.494` to `.94425`. Of those errors, 151 return another valid
value, usually from an adjacent write in the same temporal direction. Does the
remaining owner bleed come from stale Momentum that survives along the key of
the current committed write?

This question is narrower than P061. P061 added a second residual microstep
and failed its parent-read identity contract. This diagnostic adds no
microstep, changes no state or logit, and cannot rescue P061.

## 2. Frozen Intervention

Use the exact P059 checkpoint, score, cases and test data. Select every test
case containing at least one frozen wrong-key valid-value swap and an equal
number of deterministically chosen all-correct cases. Run the ordinary frozen
model unchanged. Around each of its two external Momentum chunk calls, replay
the same transition one token at a time with the external fused recurrent op.

At each true write token, compute the native committed residual and the
decayed old Momentum component along the normalized current key. Also compute
one algebraic counterfactual that removes only that old key-local Momentum
component immediately before the unchanged native write. The counterfactual
is diagnostic only; its state and output are never returned to the model.

For future-owned events, layer 0 is the causal commit path later transported
by native `[S,M]` FutureSeed. For past-owned events, layer 1 is the causal
within-sequence commit path. Both layers and both directions remain archived.

## 3. Integrity Gates

- exact frozen checkpoint, score, cases, trained parameter and test-data hashes;
- sole CUDA index 0 with the registered A800 UUID;
- ordinary model predictions exactly match every selected frozen event;
- original chunk output and terminal state versus sequential external replay
  each have relative RMS at most `.05` in every batch and layer;
- the manual native state equation matches the replay state at all measured
  writes and every recorded scalar is finite;
- logits, weights, optimizer, data and formal scores are not modified.

Any integrity miss invalidates the diagnostic. It does not open a tolerance,
precision, batching or source rescue.

## 4. Registered Decision

Open exactly one clean-room key-local Momentum erase successor only if all
five causal-layer conditions hold:

1. swap median old-velocity/write-update RMS ratio is at least `.25`;
2. that swap median is at least `1.25x` the matched-correct median;
3. the fixed erase counterfactual reduces swap post-commit residual to at most
   `.85x` native;
4. matched-correct post-commit residual is at most `1.05x` native;
5. correct-minus-swap erase ratio is at least `.10`.

The successor, if opened, must modify the live token transition, preserve one
linear scan and the native `[S,M]` FutureSeed interface, and receive its own
strict identity, gradient, stability, quality and cost gates before training.

If any selection gate misses, close key-local Momentum erase. Use the measured
lifetime evidence to choose a distinct Momentum state organization; do not
sweep erase strength, decay, rank, width, seed, learning rate, loss, batch,
depth or duration.

## 5. Cost

No training and no quality evaluation arm. The diagnostic replays at most all
swap-bearing cases plus an equal control sample in batches of 128. Persistent
parameters and state are unchanged. A 40-minute hard timeout and 2-second GPU
telemetry sampling are fixed before launch.

## 6. Status

Registered. No GPU result has been observed.

## 7. Next Decision

Only the preregistered five-way gate may select the next recurrent mechanism.
This diagnostic cannot itself justify Sudoku transfer or a quality claim.
