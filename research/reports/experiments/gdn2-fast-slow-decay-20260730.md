# GDN2 Fast-Slow Decay

## 1. Metainfo

- Plan: `P-FSMC-001`
- Status: approved, awaiting GPU1
- Date: `2026-07-30`
- Machine: AIStation `GPU1` only
- Branch: `codex/gdn2-fast-slow-decay-20260730`
- FLA source: `9c8e42e762fce087c27b673af4922795d9edb85e`
- Parent checkpoint:
  `/huyang2/double-loop/models/gdn2-futureseed-d192l10-s12000-20260726T131301Z-42102bd/checkpoints/train_state_step009000.pt`
- Parent checkpoint SHA256:
  `606caf5229590f157d7a0f952423c719f4c17688003309a7bd0664e579588dd7`
- Parent source SHA:
  `42102bd65a28d60bde6b09ef93343692740582a1`

## 2. Mechanism Hypothesis

GDN2 currently predicts a separate forgetting rate for every token, head, and
key channel. A useful memory can therefore be weakened by short-lived local
fluctuations even when the erase and write decisions should remain fast.

The candidate creates two timescales without adding recurrent state:

```text
raw hazard = -log_decay
slow hazard = positive normalized causal FIR(raw hazard)
effective hazard = raw + rho * (slow - raw)
```

Only forgetting is smoothed. GDN2's current-token erase and write gates remain
untouched, as do its official rank-one recurrence and Triton chunk kernel.
This is a generic memory-control bias. It does not inspect Sudoku rules,
labels, board units, or solved states.

## 3. Prediction And Decision

Run exactly one matched comparison from the same frozen step9000 checkpoint:

- control: `external_identity`, which traverses the same wrapper, creates the
  same K=4/rho parameters, and computes the same FIR diagnostics, but returns
  raw decay unchanged;
- candidate: `positive_causal`, initialized with current-token weight `0.85`
  and per-head slow mixing `rho=0.10`.

Prediction: the candidate measurably reduces forgetting-hazard temporal
variation (`TV ratio <=0.995`). If abrupt forgetting is a real hard-Sudoku
bottleneck, that should improve 51-64 blank exact rather than only blank
accuracy.

Accept only if systems overhead is at most 20%, the mechanism is active, and:

- hard three-range mean loop5 exact improves by at least `+0.01`, no range
  regresses by more than `0.01`, and mixed loop5 does not regress; or
- 56-60 or 61-64 improves by at least `+0.02`, 51-55 regresses by at most
  `0.01`, and mixed loop5 does not regress.

Kill on official FLA provenance/kernel/backward/checkpoint/no-fallback failure,
NaN/OOM/wrong GPU, systems overhead above 20%, or candidate step9050 CE more
than `0.10` above control. A negative result closes decay smoothing. Do not
sweep kernel length, rho, seed, LR, loss, width, or continuation length.

## 4. Fixed Configuration

- Official FLA GDN2, D192/L10/H6/K32/V32, short conv on.
- Native FutureSeed1, five loops, equal CE at every loop.
- Effective batch 128, seed 52, BF16, exact optimizer/RNG continuation.
- Official Sudoku train/test arrays and 512 fixed evaluations per blank range.
- Parent step9000 to step9100, exactly 100 optimizer steps per arm.
- No noise, repair, search, selector, rollout oracle, or task-specific rule.

## 5. Commands And Environment

```bash
GPU1_UUID=<UUID_FROM_AISTATION_GPU1_PROBE> \
CUDA_VISIBLE_DEVICES=0 ./scripts/run_gdn2_fast_slow_decay_preflight.sh
CUDA_VISIBLE_DEVICES=0 ./scripts/run_gdn2_fast_slow_decay_matched.sh
```

All caches, wheels, models, runs, and artifacts remain under
`/huyang2/double-loop`.

## 6. Required Validation

1. Constant preservation, strict causality, positivity/range, identity, total
   variation reduction, and parameter/input gradients in pure Torch.
2. Exact pinned official FLA GDN2 source and wheel file tree, with backend
   dispatch disabled and the physical CUDA UUID bound to AIStation GPU1.
3. Official output/state and input/state/core-gradient parity for the external
   identity wrapper.
4. Candidate backward traverses `ChunkGDN2FunctionBackward`; controller,
   input, and initial-state gradients are finite and nonzero.
5. Production-shape warmed time and peak-memory overhead at most 20% versus
   the matched external-identity path.
6. Exact parent checkpoint hash and full-stack two-step GPU1 smokes for both
   arms.
7. Same official evaluation and paired case hashes, with loop1/3/5 hard-case
   visualization for all three blank ranges.

## 7. Results

Pending GPU1 allocation.

## 8. Interpretation

Pending.

## 9. Decision And Reuse

Pending.
