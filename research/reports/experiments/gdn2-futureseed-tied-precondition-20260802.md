# GDN2 FutureSeed-Tied Preconditioner

## Status

- Plan: `P-PCOND-001`
- Status: discarded after one preregistered GPU1 probe; weak quality signal did
  not clear the quality or systems gate
- Date: 2026-08-02 CST
- Experiment source SHA: `d985b6a8b05184578581925d9a1e2484503553db`
- Official FLA source SHA: `9c8e42e762fce087c27b673af4922795d9edb85e`

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

### CUDA contract

The contract ran on GPU1 (`NVIDIA A800-SXM4-80GB`, UUID
`GPU-e7f175ea-1d38-93c4-2d07-fc1a938dc9d2`) with PyTorch `2.7.0+cu126` and
the pinned official FLA source. There was no fallback.

- Parallel ATK versus sequential recurrence max absolute error:
  `7.15e-7`.
- BF16 folded erase-address max absolute error: `0.00247`.
- Official `chunk_gdn2` output/state max absolute error:
  `0.00328/0.00873`.
- Backward relative L2 errors across Q/K/V/gates/initial state:
  `0.31%-0.78%`; all tested gradients were finite and nonzero.
- The real autograd graph contained `ChunkGDN2FunctionBackward`.
- Kernel microbenchmark overhead was `+24.89%` time and `+12.13%` peak
  allocated memory.

Two preflight failures were fixture bugs, not model results: the first tensor
fixture omitted `requires_grad`; the second used a K-sized write gate for a
non-square K/V test. Both tests were corrected before training, and the final
contract uses a non-square state to detect K/V axis swaps.

### Matched experiment

Both arms start from the exact step9000 checkpoint and run 100 optimizer steps
with the same official Sudoku data, random cell order, optimizer, seed 52,
effective batch 128, five loops with all-loop CE, 5,461,688 parameters, and
official GDN2 kernel. The control is the frozen position-Q/K step9100 run.

| Metric | Position-Q/K control | Preconditioned candidate | Delta |
|---|---:|---:|---:|
| Step9100 train CE | 1.2196 | 1.1434 | -0.0762 |
| 51-55 blank loop5 | 0.4704 | 0.4919 | +0.0214 |
| 56-60 blank loop5 | 0.4293 | 0.4459 | +0.0166 |
| 61-64 blank loop5 | 0.4001 | 0.4312 | +0.0311 |
| Mean hard blank loop5 | 0.4333 | 0.4563 | +0.0230 |
| Official hard exact | 0 in every range | 0 in every range | 0 |
| Train time | 690.45 s | 880.10 s | +27.47% |
| Peak allocated VRAM | 8,239 MiB | 13,254 MiB | +60.86% |

Candidate loop1-to-loop5 blank gains are `+0.0290/+0.0293/+0.0356` across
the three ranges. The control gains are `+0.0266/+0.0241/+0.0342`, so most of
the candidate advantage is already present at loop1; the additional recurrent
correction over control is only about `+0.0023/+0.0053/+0.0014`.

The mechanism is active rather than collapsed: final multiplier mean/std is
`1.3021/0.1920`, its range is `0.7317-1.4590`, write direction changes by
`29.34%`, and the folded erase error remains `0.00529` max.

### Case-level evidence

All visualized arms use identical case banks; all three SHA256 hashes match.
On 61-64 blanks:

- Largest final improvement, batch 73: control wrong cells
  `43->46->47->47->47`; candidate `39->37->35->34->34`.
- Strongest candidate loop correction, batch 194: candidate
  `43->31->32->31->31`, but it starts worse than the control and ends only
  three cells better.
- Hardest candidate failure, batch 186: candidate stays
  `46->46->46->46->46`, while control ends at 40 wrong cells.
- Largest regression, batch 46: control ends at 31 wrong cells; candidate ends
  at 42.

The mixed 512-board evaluation has candidate exact `0 -> 0.00391` from loop1
to loop5, so some loops do close boards. The preregistered separately sampled
hard ranges remain exact zero and are the decision metric.

## Decision

Discard this exact implementation as a new baseline. It produces a real but
insufficient quality gain (`+0.0230 < +0.03`) and exceeds the systems budget
(`+27.47% > +20%`). It also does not establish a stronger loop mechanism:
most gain appears at loop1, and official hard exact remains zero.

Do not sweep multiplier bounds, center, strength, gate, seed, LR, loss, width,
or duration. Keep position-Q/K as the strong GDN2+FutureSeed baseline. The
reusable insight is narrower: causal write geometry can improve optimization,
but a generic prefix `logcumsumexp` implementation is too expensive and does
not yet convert that gain into reliable recurrent closure. Revisit only with a
fused kernel or on a direct online-memory interference/retrieval benchmark.

## Artifacts

- Run: `runs/gdn2-fs-precondition-positionqk-s9100-20260802T070402Z-d985b6a/`
- Contract: `preflight/preconditioned-gdn2-d985b6a.json`
- Matched metrics: `visualizations/comparison.json`
- Interactive visualization: `visualizations/index.html`
