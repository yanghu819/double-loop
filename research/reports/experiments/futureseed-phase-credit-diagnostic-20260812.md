# P-LOOP-003: Receiver-Edge FutureSeed Phase-Credit Diagnostic

## 1. Metainfo

- Status: completed; diagnostic rejected; P-LOOP-004 not admitted
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

- Run: `p-loop-003-phase-credit-20260812T185345Z-82c4847`
- Source: clean detached pushed SHA
  `82c48479f11e95ef9f337453f8cf8d5d377b6fb0`
- GPU: A100-SXM4-80GB, CUDA index 0, UUID
  `GPU-05f3e3f9-3f6f-82e9-1107-6e86672e77b5`
- JSON SHA256:
  `e42f505a468ecf1c1a9f9fd02fe5954d26a21de740b535182b2a032062c00ee4`
- Log SHA256:
  `39c1734ce4dc3a7af650160a46afbf745b5ba35081f713848fa7f25c3d19ce2a`
- Elapsed/peak allocated: `601.739s` / `3517.831 MiB`

The immutable evidence is mirrored under
`research/reports/visualizations/futureseed-phase-credit-20260812/`. The
diagnostic creates no checkpoint.

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

The integrity gate passes. Each range captures exactly 165 non-null receiver
states and 180 official `ChunkGDN2FunctionBackward` nodes. Direct gate-gradient
reconstruction has maximum relative error `3.71e-5` and minimum cosine
`0.99999987`; parameter SHA256 is identical before and after.

No hard range qualifies:

| Range | loop1->5 blank | wrong cells | O/C cosine | conflict | phase/shared | board aggregation | max board share | leave-one-out | active edges |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 51-55 | `0.5413->0.5872` | `25.00->22.50` | `-0.9711` | `0.7909` | `1.4811` | `0.4698` | `0.7911` | `0.4060` | `7,8` |
| 56-60 | `0.5210->0.5386` | `27.13->26.13` | `+0.9580` | `0.1619` | `0.2365` | `0.4081` | `0.3506` | `0.9012` | none |
| 61-64 | `0.5195->0.5859` | `30.75->26.50` | `+0.8898` | `0.4590` | `0.3035` | `0.1707` | `0.2757` | `0.2989` | `8` |

The strongest-looking 51-55 opposition is concentrated in receiver edge 8
and one board rather than a stable cross-board, cross-depth mechanism. The two
harder ranges are aggregate-aligned, and 56-60 misses the frozen `+0.02`
correction floor. Although the continuation cotangent is almost entirely
orthogonal to the injected state (`0.99997` mean), P-FS3-004 already rejected
simple orthogonal basis transport, so this descriptive statistic does not
reopen that family.

Decision: `0/3` ranges qualify, so P-LOOP-004 is forbidden. Close scalar
FutureSeed phase-credit coordination, including phase gates, projection
strength, edge selection and loss-weight rescues. The next experiment must
alter a genuinely live recurrent address/state transition and earn a fresh
from-scratch directional binding result before any Sudoku scale claim.

## 9. Submission Record

Not applicable.
