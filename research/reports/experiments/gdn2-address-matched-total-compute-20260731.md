# GDN2 Address-Payload Matched-Total-Compute Falsifier

## 1. Metainfo

- Plan: `P-ADDR-003`
- Status: preregistered; control continuation pending
- Machine: AIStation GPU1, one A800 80GB
- Source branch: `codex/gdn2-address-binding-20260730`
- Preregistered: 2026-07-31 13:10 CST

## 2. Question

Does canonical-position Q/K remain better than content-entangled GDN2 when both
arms receive the same total optimizer steps, or did the final step9300 gap only
come from continuing the positive candidate for 200 extra steps?

This is the missing fairness gate from `P-ADDR-002`, not a new address ablation.

## 3. Hypothesis

Ordinary GDN2 asks one hidden stream to specify both memory address and payload.
Position-Q/K separates those jobs without adding parameters or changing the
official recurrent kernel. If this factorization genuinely improves learning,
its advantage should remain after the normal-GDN2 control is also trained from
its exact step9100 state to step9300.

## 4. Matched Contract

- Control resume:
  `gdn2-address-control-s9100-20260731T034101Z-4b94220`.
- Candidate result:
  `gdn2-address-position_qk-s9300-20260731T041227Z-eec018c`.
- Both originate from the same strict-official-FLA step9000 parent.
- Same model size, parameter count, official boards, random cell-order policy,
  curriculum, optimizer, seed, batch, FutureSeed, all-loop CE, and official
  FLA GDN2 chunk/Triton path.
- The only model difference remains GDN2 address mode:
  `none` versus `position_qk`.
- One control continuation only. No seed, loss, learning-rate, address-mode,
  width, depth, or duration sweep.
- No Sudoku rule, repair, search, selector, oracle rollout, fallback, CPU model
  smoke, or GPU2.

## 5. Prediction And Decision

- Primary success: at step9300, position-Q/K exceeds control by at least
  `+0.10` mean loop5 blank accuracy across official 51-55, 56-60, and 61-64
  ranges.
- Mechanism support also requires lower train CE and stronger loop1-to-loop5
  correction on matched hard boards.
- Strong success: candidate opens hard exact while control remains closed.
- Inconclusive: final mean blank gap is `[+0.05,+0.10)`.
- Reject the scale claim: final mean blank gap is below `+0.05`; classify the
  earlier result as transient optimization speed rather than a persistent
  carrier improvement.

## 6. Budget And Kill Criteria

- Budget: exactly 200 additional control optimizer steps plus the existing
  official 512-board-per-range evaluation and case bank.
- Expected wall time: 25-35 minutes on GPU1.
- Stop immediately on source/checkpoint/hash/GPU UUID mismatch, non-official
  FLA runtime, kernel fallback, NaN, or OOM.
- At step9200, stop if train CE is above `2.2` and has not improved from the
  step9100 value `1.9370`.

## 7. Artifacts

Pending.

## 8. Results And Decision

Pending.

## 9. Submission

Not applicable.
