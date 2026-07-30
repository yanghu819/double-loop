# GDN2 Fast-Slow Decay

## 1. Metainfo

- Plan: `P-FSMC-001`
- Status: discarded after the single preregistered probe
- Date: `2026-07-30`
- Machine: AIStation `GPU1` only
- Physical GPU UUID: `GPU-c1d7c624-a393-befa-3807-7e00602d65ca`
- Branch: `codex/gdn2-fast-slow-decay-20260730`
- Implementation/result source SHA:
  `2867c8892b97e1c2d6111241b538c499543cd639`
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
   identity wrapper, both without an initial state and with a FutureSeed-like
   initial state.
4. Candidate backward traverses `ChunkGDN2FunctionBackward`; controller,
   input, and initial-state gradients are finite and nonzero.
5. Production-shape warmed time and peak-memory overhead at most 20% versus
   the matched external-identity path.
6. Exact parent checkpoint hash and full-stack two-step GPU1 smokes for both
   arms.
7. Same official evaluation and paired case hashes, with loop1/3/5 hard-case
   visualization for all three blank ranges.

## 7. Results

### Integrity and mechanism

- Six pure-Torch FIR tests pass: constant preservation, strict causality,
  positivity/range, exact external identity, TV reduction, and finite nonzero
  gradients.
- Pinned FLA `9c8e42e` reference errors are small:
  output `5.99e-05`, terminal state `2.76e-04`, and maximum gradient
  `2.34e-04`.
- `external_identity` is exactly equal to untouched official GDN2 both with
  and without a FutureSeed-like initial state: output, terminal state, and all
  checked gradients have maximum absolute error `0`.
- The candidate backward graph contains `ChunkGDN2FunctionBackward`.
  Controller, input, and initial-state gradients are finite and nonzero.
- The layer contract measures `+6.97%` time and `+8.18%` peak memory.
- The full-model run measures `763.52s -> 818.41s` (`+7.19%`) and
  `8278.24MiB -> 9987.85MiB` (`+20.65%`) peak memory.
- At step9100 the learned mechanism remains active:
  hazard TV ratio `0.9839`, `rho=0.0990`, lag mass `0.1475`, and relative
  hazard change `0.0111`.

### Primary quality

| Evaluation | Control | Fast-Slow | Delta |
|---|---:|---:|---:|
| Mixed loop1 exact | 0.0234 | 0.0234 | +0.0000 |
| Mixed loop3 exact | 0.1777 | 0.1855 | +0.0078 |
| Mixed loop5 exact | 0.2402 | 0.2500 | +0.0098 |
| Official 51-55 loop5 exact | 0.3594 | 0.3535 | -0.0059 |
| Official 56-60 loop5 exact | 0.1387 | 0.1465 | +0.0078 |
| Official 61-64 loop5 exact | 0.1641 | 0.1621 | -0.0020 |
| Official three-range mean | 0.2207 | 0.2207 | +0.0000 |

The score gate fails: mean hard exact does not improve, and neither hardest
range gains the required `+0.02`. The systems gate also fails because
full-model peak memory is `+20.65%`.

### Paired case distribution

The 256-board visualization bank is diagnostic and is not substituted for the
primary 512-board official metrics.

| Range | Candidate better / worse / tied at loop5 | Mean candidate-control wrong cells |
|---|---:|---:|
| 51-55 | 87 / 68 / 101 | -0.242 |
| 56-60 | 106 / 90 / 60 | -0.133 |
| 61-64 | 103 / 117 / 36 | +0.074 |

The extrema expose instability rather than a reliable gain:

- largest rescue: control loop5 has 19 wrong cells; Fast-Slow loop5 solves;
- largest regression: control goes `15 -> 6 -> 3` wrong across loops
  `1/3/5`, while Fast-Slow goes `18 -> 27 -> 26`;
- hardest shared case: control goes `32 -> 43 -> 43`; Fast-Slow goes
  `32 -> 42 -> 41`.

Archived evidence:

- `runs/gdn2-fast-slow-comparison-20260730T064321Z-2867c88/score.json`
- `runs/gdn2-fast-slow-comparison-20260730T064321Z-2867c88/comparison.json`
- `runs/gdn2-fast-slow-comparison-20260730T064321Z-2867c88/contracts/`
- `runs/gdn2-fast-slow-comparison-20260730T064321Z-2867c88/visualizations/index.html`

## 8. Interpretation

The attachment's mechanical claim is correct: a positive causal FIR creates a
slow forgetting timescale while keeping erase/write token-fast, and it can be
implemented without changing the official recurrent kernel. The optimization
path also remains healthy.

The task claim is not supported. A one-percent change in the forgetting
hazard is enough to move some boards into a better attractor and other boards
into a much worse one. The average official exact score is unchanged. Smooth
forgetting therefore changes the selected solution basin, but does not make
the recurrent dynamics reliably converge toward the globally consistent
board.

The mixed `+0.0098` is not a reason to tune K or rho. It is smaller than the
preregistered gate, does not replicate across the fixed official ranges, and
comes with both large paired wins and large paired regressions. More ordinary
data/compute on the clean GDN2+FutureSeed baseline remains better supported
than hand-tuning a memory timescale.

## 9. Decision And Reuse

- Discard decay-only Fast-Slow Memory Control for hard Sudoku.
- Do not continue to step9300.
- Do not sweep FIR length, rho, initialization, seed, LR, loss, model size, or
  training length.
- Reuse the positive-causal FIR and exact identity/CUDA contract only if a
  different task supplies independent evidence for an explicit slow-memory
  timescale.
- Keep the frozen strict-official GDN2+FutureSeed1 line as the baseline.
- The next high-ROI axis is clean data/model/compute scaling or a generic
  state update with an explicit convergence reason, not another gate filter.
