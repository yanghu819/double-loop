# GDN2 Address-Payload Clean Hard-Stage Scaling Gate

## 1. Metainfo

- Plan: `P-ADDR-004`
- Status: preregistered; one GPU1 continuation pending
- Machine: AIStation GPU1, one A800 80GB
- Source branch: `codex/gdn2-address-binding-20260730`
- Preregistered: 2026-07-31 14:20 CST

## 2. Question

After matched-total-compute evidence shows that canonical position-Q/K removes
a large random-order optimization bottleneck, can more clean hard-stage data
and compute convert partial blank accuracy into full-board exact solves?

This is a scaling-law gate, not another address mechanism comparison.

## 3. Hypothesis

At step9300, position-Q/K has a strong positive slope: train CE is `0.9661`,
mean official 51-64 blank accuracy is `0.5437`, and later loops remove errors,
but each separately sampled hard-range exact rate is zero. If the remaining
problem is primarily insufficient hard-distribution compute, 300 more optimizer
steps on the unchanged 51-64 curriculum should open exact and preserve useful
loop correction.

## 4. Contract

- Resume exact position-Q/K step9300 checkpoint:
  `gdn2-address-position_qk-s9300-20260731T041227Z-eec018c`.
- Checkpoint SHA256:
  `8efc1b09e68830ddff44b2b9ef65db1b497af7d0322c24c3fdb99dcbaff57cb4`.
- Continue exactly to step9600. The extra 300 steps extend only the existing
  final `51-64` stage.
- Keep D192/L10/H6/D32, `5,461,688` parameters, batch128, seed52,
  random cell order, native FutureSeed, five-loop all-loop CE, optimizer,
  official data, and strict official FLA GDN2 chunk/Triton path unchanged.
- One run only. No matched control is needed because this run tests closure
  slope, not a new causal architecture comparison.
- No noise, loss change, task rule, repair, search, selector, fallback, CPU
  model smoke, GPU2, or seed/hyperparameter sweep.

## 5. Prediction And Decision

- Primary success: loop5 exact becomes at least `0.02` in any official hard
  range, mean hard blank accuracy reaches at least `0.60`, and loop5 improves
  over loop1 rather than merely changing the first-pass operating point.
- Strong success: nonzero exact appears in 61-64 blanks.
- Partial signal: blank mean improves by at least `+0.04` but exact remains
  zero. This supports more capacity/data scaling but not closure.
- Negative: mean blank improves by less than `+0.02` and exact remains zero.
  Do not add more short continuations; the next experiment must increase
  generic model/state capacity or independent data coverage.

## 6. Budget And Kill Criteria

- Budget: 300 additional optimizer steps plus the standard official
  512-board-per-range evaluation and case bank.
- Expected wall time: 32-40 minutes on GPU1.
- At step9450, stop by exact PID if train CE is still above `0.95` and has
  improved by less than `0.03` from step9300.
- Stop immediately on source/checkpoint/hash/GPU UUID mismatch, non-official
  FLA runtime, kernel fallback, NaN, OOM, or AIStation lease risk.
- If stopped, write `abort.json` with the exact step and reason.

## 7. Claim If Successful

Generic address/payload factorization plus clean data/compute scaling can turn
FutureSeed-GDN2's partial random-order correction into full-board recurrent
closure. This claim is allowed only if official exact opens; blank accuracy
alone is insufficient.

## 8. Artifacts

Pending.

## 9. Results And Decision

Pending.
