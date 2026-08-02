# GDN2 FutureSeed-Tied Preconditioner

## Status

- Plan: `P-PCOND-001`
- Status: implementation complete; GPU1 queued for CUDA contract and one probe
- Date: 2026-08-02 CST

## Mechanism Question

Position-Q/K already gives GDN2 a stable address, but it does not change the
conditioning of repeated online writes. Does a causal estimate of how often
each key direction has been updated reduce interference, especially when
FutureSeed provides a nonzero initial memory?

## Intervention

For normalized key `k`, GDN2 key-axis log-decay `g`, and erase gate `b`, keep a
parameter-free diagonal statistic:

```text
A_t = exp(g_t) * A_(t-1) + b_t * k_t^2
```

Convert `A_t` to the bounded multiplier used by the open FLA PGDN work. The
current GDN2 gates are reused instead of adding another projection. FutureSeed
state row energy initializes `A_0`, so the seed supplies both memory content
and an estimate of which key directions already contain evidence.

The unchanged official GDN2 kernel receives:

```text
k_kernel = multiplier * k
b_kernel = b / multiplier
```

Therefore `b_kernel * k_kernel = b * k`: erase semantics remain unchanged,
while the rank-one write direction is preconditioned. No Sudoku rule, repair,
search, selector, extra loss, extra parameter, or recurrent-state expansion is
introduced.

## Prediction And Decision

The single candidate resumes the exact strict-FLA GDN2+FutureSeed step9000
checkpoint and keeps the successful position-Q/K random-order setting. The
frozen matched position-Q/K step9100 run is the control.

- Success: mean official 51-64 loop5 blank accuracy improves by at least
  `+0.03`, no individual range regresses by more than `0.03`, and overhead is
  at most `20%`.
- Integrity: parallel ATK matches token recurrence, the folded erase address
  differs by less than `8e-3` after BF16 quantization, official chunk forward,
  terminal state, and backward match a direct CUDA Torch recurrence, and the
  autograd graph contains `ChunkGDN2FunctionBackward`.
- Kill: any non-finite gradient, wrong GPU/source/kernel, excessive numerical
  mismatch, overhead above `20%` without a clear quality signal, or no primary
  gain.
- No follow-up table: do not sweep squash range, center, gate, seed, LR, loss,
  width, or duration after a negative result.

## Results

Pending GPU1 allocation.
