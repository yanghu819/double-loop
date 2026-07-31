# GDN2 Address-Payload Matched-Total-Compute Falsifier

## 1. Metainfo

- Plan: `P-ADDR-003`
- Status: completed; hypothesis strongly supported on partial-board accuracy,
  not on exact solve
- Machine: AIStation GPU1, one A800 80GB
- Source branch: `codex/gdn2-address-binding-20260730`
- Preregistered: 2026-07-31 13:10 CST
- Completed: 2026-07-31 14:16 CST

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

- Control run:
  `runs/gdn2-address-control-s9300-20260731T053946Z-3a54897/`
- Candidate run:
  `runs/gdn2-address-position_qk-s9300-20260731T041227Z-eec018c/`
- Matched comparison JSON, HTML, and screenshots:
  `runs/gdn2-address-matched-s9300-20260731/`
- Control source SHA:
  `3a54897c48708baf66f3cc6c7c4358a486f4d5b0`, clean.
- Control resume checkpoint SHA256:
  `451d943e02aaeea79eb476c6dfcab04ca1434d2c818d3ff6ff0f38ccd4cc29ae`.
- Source snapshots remain outside Git and are pinned by
  `source_snapshot.ref`; model checkpoints are not committed.

## 8. Results And Decision

The comparison contract passed:

- Both arms finish at optimizer step `9300`.
- Both have `5,461,688` parameters.
- Both use official FLA SHA `9c8e42e7`, strict official layer forward,
  chunk/Triton execution, the same FutureSeed and five-loop training contract.
- All three paired official case-bank hashes match exactly.
- The control step9200 CE was `1.7398`, below the `2.2` kill threshold and
  improving, so the run continued as preregistered.

| official blank range | control loop5 blank | position-Q/K loop5 blank | delta | both exact |
|---|---:|---:|---:|---:|
| 51-55 | 0.3035 | 0.5660 | +0.2625 | 0.0000 |
| 56-60 | 0.2781 | 0.5057 | +0.2275 | 0.0000 |
| 61-64 | 0.2444 | 0.5594 | +0.3150 | 0.0000 |
| mean | 0.2753 | 0.5437 | **+0.2683** | 0.0000 |

The primary `+0.10` gate passes by a wide margin. Train CE is
`1.5707 -> 0.9661` from control to candidate. The 200-step continuation wall
time is `1311.3s -> 1272.8s`; peak allocated CUDA memory is
`8094.9 -> 8239.4 MiB`. This is not a parameter or extra-step win.

The matched visualization is also directionally causal. On mechanically
selected 64-blank batch `69`, the control changes wrong blanks
`53 -> 50 -> 50 -> 51 -> 51` across loops 1-5. Position-Q/K changes
`32 -> 21 -> 18 -> 15 -> 15`. Thus the candidate starts from a better
representation and later loops remove another 17 errors; the control only
removes two and then regresses.

**Decision:** retain generic address/payload factorization as the strongest
GDN2 carrier modification found so far. The persistent matched-compute claim
survives. Do not claim that Sudoku is solved: every separately sampled
51-64-blank exact rate is still zero. The next expensive experiment must test
clean from-scratch/model-data scaling of this carrier against matched normal
GDN2, not another address, loss, or seed table.

## 9. Submission

Not applicable.
