# FutureSeed Sudoku Four-Backbone Final Baseline

## 1. Status

- Plan ID: `P-BASELINE-001`
- Status: completed
- Training window: 2026-07-23 06:15-09:12 UTC
- Machine: AIStation `GPU1`, NVIDIA A800-SXM4-80GB
- GPU2: forbidden and unused
- Model source SHA:
  `8c7c7596987b365877603419ad4a7e0b91486119`
- Branch: `codex/final-baseline`
- Aggregate:
  `runs/sudoku-backbone-benchmark-20260723T061503Z-manual-8c7c759/`

This run turns the scattered architecture experiments into one reproducible
baseline. It compares four recurrent carriers for the same native FutureSeed
and depth-loop mechanism:

- local RWKV7 state-passing CUDA;
- official FLA GatedDeltaNet, published as `gdn`;
- official FLA GatedDeltaNet2, published as `gdn2`;
- official FLA KimiDeltaAttention, published as `kda`.

It is an architecture-carrier benchmark. It is not a with/without-FutureSeed
ablation and does not estimate any architecture's asymptotic ceiling.

## 2. Question And Decision Rule

Question:

> Under one data order, outer model, FutureSeed rule, loop budget, optimizer,
> and evaluator, which recurrent carrier is the best default for continued
> clean scaling?

The decision was not based on one token metric. A replacement for GDN needed a
real hard full-board advantage or comparable quality with at least 20% lower
training cost. If every arm stayed at zero exact beyond 50 blanks, the result
would select an engineering default only; it would not support a universal
architecture claim.

No seed, learning-rate, loss, temperature, or architecture-specific tuning
sweep was allowed.

## 3. Fairness Contract

All four arms used:

- official 9x9 full-diversity Sudoku train/test arrays;
- executed curriculum `46-50:100,51-55:400`;
- D192, 10 layers, 6 heads, head dimension 32, channel multiplier 4;
- five depth loops and equal CE supervision on every loop;
- microbatch 32, gradient accumulation 4, effective batch 128;
- AdamW, LR `0.0015`, weight decay `0.001`;
- BF16 forward and seed 52;
- fixed unit-normalized native FutureSeed with scale 1;
- no feature noise, hidden noise, feedback, scratch state, margin loss,
  selector, rollout oracle, repair, search, or Sudoku rule;
- 500 optimizer steps and the same 512-board evaluator;
- fixed official blank ranges `46-50`, `51-55`, and `56-64`.

Architecture-specific parameter count, speed, and memory were measured rather
than hidden. Parameters were not artificially padded to equality.

## 4. Kernel And Source Gate

The GPU1 preflight passed complete-model forward, backward, finite-gradient,
parameter-count, and runtime-provenance checks for all four arms.

- RWKV used `local_rwkv7_statepassing`; automatic fallback was disabled.
- GDN used official
  `fla.layers.gated_deltanet.GatedDeltaNet`.
- GDN2 used official `fla.layers.gdn2.GatedDeltaNet2`.
- KDA used official `fla.layers.kda.KimiDeltaAttention`.
- Every official FLA short convolution reported the Triton backend.
- The installed official FLA source matched source marker
  `fe8fce9fc6984f22905f54cfa885dce1502baf26`.
- Pinned wheel SHA256:
  `65f57bf2aa937991fc497bd63f42883d263ead8646f21f2492735ccddd82d0eb`.

The successful preflight was recorded at parent SHA `0ce0d93`. The benchmark
SHA `8c7c759` only prepends the persistent project-local toolchain directory in
the launcher; model, kernel, config, and evaluator semantics are unchanged.

Every training arm reports the same exact source SHA. Source patches are either
empty or touch only generated `leaderboard.csv` and
`runs/visualization_index.html`; the summarizer now rejects any model or config
source patch.

## 5. Results

### Matched Step-500 Result

| Metric | RWKV | GDN | GDN2 | KDA |
|---|---:|---:|---:|---:|
| Parameters | 5.580M | 5.980M | 6.946M | 5.852M |
| Train CE | 1.0432 | **1.0001** | 1.0114 | 1.0065 |
| Fixed h53 exact | 0.00586 | **0.01758** | **0.01758** | 0.01562 |
| Mixed loop1 exact | 0.00000 | **0.01367** | 0.00781 | 0.00781 |
| Mixed loop5 exact | 0.00781 | **0.02148** | 0.01953 | 0.01758 |
| Mixed loop5 blank accuracy | 0.48666 | **0.50544** | 0.50131 | 0.50184 |
| Loop1-to5 exact gain | 0.00781 | 0.00781 | **0.01172** | 0.00977 |
| Seconds per optimizer step | **3.87** | 4.93 | 5.45 | 5.55 |
| Peak allocated VRAM | 9.04 GiB | **7.04 GiB** | 9.08 GiB | 7.62 GiB |

RWKV is fastest per step, but its easy-range opening and mixed exact are much
weaker. GDN has the best CE, mixed exact, blank accuracy, easy-range closure,
and lowest measured allocation among the three official FLA arms. GDN2 spends
16% more parameters and about 29% more allocated memory than GDN without a
quality win. KDA nearly catches GDN2 by step 500 but is the slowest arm.

### Official Blank Ranges At Loop 5

| Blank range | RWKV exact / blank | GDN exact / blank | GDN2 exact / blank | KDA exact / blank |
|---|---:|---:|---:|---:|
| 46-50 | 0.3359 / 0.9659 | **0.8770 / 0.9952** | 0.7754 / 0.9916 | 0.7441 / 0.9905 |
| 51-55 | 0 / 0.5097 | 0 / **0.5325** | 0 / 0.5199 | 0 / 0.5234 |
| 56-64 | 0 / 0.4553 | 0 / **0.4738** | 0 / 0.4712 | 0 / 0.4687 |

No arm opens full-board exact at 51 or more blanks in 500 steps. Therefore this
run does not establish a hard-quality winner or an architecture ceiling. It
does establish that GDN gives the best finite-budget opening and quality-cost
tradeoff under the shared recipe.

## 6. Loop Behavior And Same-Puzzle Visualization

On the identical 53-blank visualization puzzle, wrong cells change as follows:

| Backbone | Loop1 | Loop2 | Loop3 | Loop4 | Loop5 |
|---|---:|---:|---:|---:|---:|
| RWKV | 24 | 24 | 23 | 23 | 22 |
| GDN | 21 | 21 | 22 | 20 | 20 |
| GDN2 | 24 | 22 | 22 | 22 | 22 |
| KDA | 22 | 20 | 20 | 20 | 20 |

All arms perform most useful refinement by loop 2 or 3 and then largely
freeze on this hard board. The loop is not neutral in aggregate: mixed exact
gains by 0.8-1.2 percentage points. But this short run does not show sustained
late-loop global correction beyond the 50-blank boundary.

The aggregate HTML embeds all four loop-by-loop boards and the train/loop
curves. It deliberately uses the same puzzle rather than selecting a separate
success for each architecture.

## 7. Lease Recovery And Memory Measurement

GPU1's first lease expired after KDA had written the exact step-500 model,
optimizer, RNG, checkpoint evaluation, and primary case, but before the full
final JSON was written.

Recovery did not retrain or select a checkpoint:

1. GPU1 was reopened.
2. The exact step-500 KDA checkpoint was loaded with total steps still set to
   500, so the training loop executed zero optimizer steps.
3. Only deterministic final evaluation was rerun.
4. The original training log and run config were restored and the resume log
   was appended.

CUDA peak counters do not survive a lease restart. KDA memory was therefore
recovered in a separate diagnostic run by loading the exact step-500 checkpoint
and executing one matched microbatch-32, accumulation-4 optimizer step. It
measured 7807.97 MiB allocated and 8082 MiB reserved, consistent with the prior
same-shape KDA run. This probe contributes only the memory number; all reported
quality and speed remain from the original step-500 run. Raw recovery logs and
the pre-correction JSON are under the aggregate `operations/` directory.

## 8. Final Baseline Decision

Retain two distinct canonical artifacts:

1. `configs/sudoku/gdn_scale.env` is the strongest scale baseline. Its clean
   D224/L12 local GDN-Triton run reaches mixed loop5 exact `0.4805` at step
   30000, with official exact `1.0000/0.6152/0.3848` on the three blank ranges.
2. `configs/sudoku/backbone_benchmark.env` is the fair architecture contract
   for RWKV, official FLA GDN, GDN2, and KDA.

GDN is the default scale carrier because it has the demonstrated long-run
slope and the best finite-budget quality-cost tradeoff. RWKV remains maintained
as the native CUDA mechanism reference. GDN2 and KDA remain supported benchmark
backbones, but no separate scaling line is justified.

This does not mean GDN2 or KDA are universally worse. Previous exact
continuations show the early ranking narrows by step 1000. The valid claim is
narrower: under the current shared recipe, GDN is the best place to spend the
next unit of compute.

## 9. Deprecated Boundary

The final baseline excludes and marks as deprecated or forbidden:

- feature-difference and hidden-aggregate noise;
- scratch state, feedback corruption, margin/ranking/mass/boundary losses;
- progressive split-bank state expansion as a default;
- selector, best-of-K, rollout oracle, repair, search, and task rules;
- reverse scanning mislabeled as FutureSeed;
- EqR mixer transplants and Maze-specific objective paths;
- seed, loss-weight, temperature, and architecture enumeration tables.

The old parsers and runs remain in Git for checkpoint readability and negative
evidence. They are not maintained launch paths.

## 10. Artifacts

- Aggregate JSON/CSV/HTML:
  `runs/sudoku-backbone-benchmark-20260723T061503Z-manual-8c7c759/`
- Browser-reviewed overview and loop screenshots:
  `runs/sudoku-backbone-benchmark-20260723T061503Z-manual-8c7c759/benchmark-*.png`
- GPU1 preflight:
  `runs/sudoku-backbone-preflight-20260723T053346Z-0ce0d93/`
- Four formal arms:
  `runs/sudoku-backbone-{rwkv,gdn,gdn2,kda}-s500-20260723T061503Z-manual-8c7c759/`
- KDA memory diagnostic:
  `runs/sudoku-backbone-kda-memory-probe-s501-20260723T0920Z-8c7c759/`
- Shared source snapshot SHA256:
  `2e4defe0907602894f138f24ad1002402691b63263651c963e16eb5bf7794598`

Checkpoints and datasets remain only under `/huyang2/double-loop` and are not
committed.
