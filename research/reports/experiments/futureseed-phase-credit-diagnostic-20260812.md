# P-LOOP-003: Receiver-Edge FutureSeed Phase-Credit Diagnostic

## 1. Metainfo

- Status: approved; not yet run
- Parent: position-QK GDN2 plus native terminal FutureSeed, exact step3000
- Task: official/full-diversity hard 9x9 Sudoku, 51-64 blanks
- Intervention: none; this diagnostic does not alter logits, parameters or optimizer state

## 2. Hypothesis

P-LOOP-001 found aggregate loop1-versus-continuation conflict in the 88 native
FutureSeed gate scalars, but P-LOOP-002 showed that projecting the shared gate
gradient does not improve closure. That leaves one narrow explanation: one
shared gate value may be asked to serve incompatible opening and refinement
roles. Aggregate parameter gradients cannot prove this because each later loss
also backpropagates through every earlier FutureSeed injection.

This audit decomposes the exact equal-five-loop loss at the receiver boundary.
It asks whether the actual cotangent of each injected K32xV32 state gives a
coherent, cross-board gradient to an opening-versus-continuation coordinate.
It is not a model candidate and cannot establish quality.

## 3. Configuration

Use the exact D256/L12/H8/K32/V32 parent and the same three fixed official B8
ranges as P-LOOP-001. Register a read-only pre-hook on each `FLADeltaBlock` and
capture all 165 non-null receiver states: five macro loops times two L cycles
plus one H update times 11 receiving edges. The final graph must contain
exactly 180 `ChunkGDN2FunctionBackward` nodes.

For target loop loss `j`, injection `r`, board `b` and head `h`, compute the
exact radial logit credit

```text
q[j,r,b,h] = (1 - sigmoid(a)) * <d L_j / d I_r, I_r>.
```

Under terminal/unit/head/layer FutureSeed this exactly reconstructs the direct
`future_seed_logit` gradient. Average the five target losses, then sum records
from macro loop 1 into `O` and records from loops 2-5 into `C`. The virtual
zero-mean phase coordinate is

Let `O` be the loop-1 usage sum and `C` be the mean usage sum for each of
loops 2-5. Then the exact virtual coordinate is

```text
G_phase = 4/5 (O - C),
G_shared = O + 4 C.
```

Also report the continuation cotangent component orthogonal to the injected
state. This is descriptive only: P-FS3-004 already closed simple orthogonal
basis transport, so it cannot reopen that family.

## 4. Environment And Integrity

Run on one visible AIStation GPU as CUDA index 0 from a clean detached pushed
SHA. Require parent checkpoint SHA256
`6339c3cb2b5fc5230a581d6633716483e35ff8e4522f06a9d7aaf26512f023da`,
parent source `9f2ee8d1738032bc5f09b55db0b81d507780b376`, pinned FLA source
`9c8e42e762fce087c27b673af4922795d9edb85e`, 12 official GDN2 layers and no
fallback. Parameter SHA256 before and after must match, `.grad` must stay empty,
and reconstructed versus direct gate-gradient relative error must be at most
`5e-3` with cosine at least `0.9999`. CPU model smoke and concurrent GPU
model/eval are forbidden.

## 5. Commands

```bash
EXPECTED_UUID=<admitted-uuid> scripts/run_futureseed_phase_credit_diagnostic.sh
```

## 6. Artifacts

The launcher writes `futureseed_phase_credit.json`, `run.log`, `gpu.txt`, source
provenance and hashes below one immutable `p-loop-003-phase-credit-*` run
directory. It creates no checkpoint.

## 7. Frozen Admission Gate

A range qualifies only when all conditions hold:

- opening/continuation aggregate cosine `<= -0.25`;
- magnitude-weighted local board/edge/head sign conflict `>=0.60`, with
  opening/continuation overlap ratio `>=0.20`;
- `||G_phase|| / ||G_shared|| >=0.25` and absolute phase-energy ratio
  `>=0.15`;
- cross-board aggregation `||sum_b G_phase,b|| / sum_b ||G_phase,b|| >=0.50`,
  maximum single-board phase-energy share `<=0.30`, and minimum leave-one-board
  direction cosine `>=0.75`;
- at least 8 of 11 edges independently contribute at least 2% phase energy,
  have weighted conflict `>=0.50`, and phase/shared leverage `>=0.15`;
- loop5-minus-loop1 blank accuracy `>=0.02` with fewer wrong blank cells.

Exactly one P-LOOP-004 opening/continuation phase-gate candidate is admitted
only if at least two of three ranges qualify, share at least six active edges,
and have aggregate phase-gradient cosine `>=0.25`. Otherwise close scalar
FutureSeed loop-credit coordination and move to a genuinely new recurrent
transition for GDN3. Thresholds cannot be changed after observing results.

The 8 boards in each range are the independent units; the board-edge-head
coordinates are not treated as independent samples. These boards were already
used by P-LOOP-001, so this is a mechanism replication and not an independent
quality estimate. A passing cotangent gate only authorizes one 100-step causal
test; it does not predict endpoint exact by itself.

## 8. Conclusions

Pending.

## 9. Submission Record

Not applicable.
