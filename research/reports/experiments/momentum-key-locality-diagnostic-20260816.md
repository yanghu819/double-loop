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

Complete; key-local Momentum erase is closed.

The exact pushed source `4e855671e6cfdeabafc0a73446f7a99670622d16`
ran once in the clean detached worktree on the sole A800 CUDA index 0. The
diagnostic selected all 135 swap-bearing cases and 135 deterministic
all-correct controls. Frozen predictions matched exactly, every chunk and
sequential replay gate passed, and the maximum manual native-equation versus
external recurrent-state relative RMS was `.000949852`, below the registered
`.05` limit.

At the causal layer, the matched-correct and swap old-velocity/update medians
were `12.38849` and `12.53231`. Their ratio was only `1.01161`, far below the
registered `1.25`. Removing old Momentum along the current key reduced the
immediate residual strongly in both groups: the matched-correct and swap
erase/native medians were `.115448` and `.131297`. The resulting
correct-minus-swap selectivity was `-.015849`, not the required `+.10`.

The five registered gates therefore resolved as:

1. swap old-velocity/update at least `.25`: pass (`12.53231`);
2. swap/correct old-velocity ratio at least `1.25`: fail (`1.01161`);
3. swap erase/native residual at most `.85`: pass (`.131297`);
4. correct erase/native residual at most `1.05`: pass (`.115448`);
5. correct-minus-swap erase selectivity at least `.10`: fail (`-.015849`).

This is a useful negative result. Old key-local Momentum dominates the write
update for essentially every event, but it is not selectively elevated on the
remaining owner swaps. A direct key-local erase would suppress a universal
part of the trained integrator rather than target the error mechanism.

The bounded run emitted 81 telemetry samples, including 56 active-memory
samples. Active GPU utilization averaged `17.625%`, peaked at `86%`, used at
most `3,688 MiB`, and peaked at `286.17 W`. The low mean reflects checkpoint
and cache I/O plus many one-token recurrent launches; the run completed
naturally with status 0 and left no compute process.

Archived artifact SHA256 values:

- diagnostic JSON: `f2367fcd1f4e04b6a4ca17ffbe4b722a26e024f415ff6e8e9e86cafc0d93d2c0`;
- GPU samples: `0e29e2fa5f5f0e10a3b3e5537e83c9198a6930e629ec88bab335af25b31d0ba9`;
- source snapshot: `b639f1e9a8909271acb1ec77d923b4f44f7b8e674f4536d5da47512fbbf13af3`;
- diagnostic log: `1c279a0889a4c547d29f26d4e1d210f7225baae597b97b9f0862ceff86969b6c`.

## 7. Next Decision

Do not launch key-local erase, decay, strength, coefficient, rank, width or
seed variants. The remaining branch is a distinct Momentum lifetime or state
organization that changes the live recurrent transition and is first judged
from scratch on the directional MQAR L1024 binding regime. It must preserve a
single scalable scan and receive its own identity, gradient, stability,
quality and cost gates before consuming formal training compute.
