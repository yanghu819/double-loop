# FutureSeed Sudoku Four-Backbone Clean Rerun

## Status

- Plan: `P-BASELINE-002`
- Result: completed
- Time: 2026-07-24 02:00:11-05:09:31 UTC
- Machine: AIStation GPU1, NVIDIA A800-SXM4-80GB
- GPU2: unused
- CPU model smoke: unused
- Source: `3bc5dec7677606cc3f12def6538891c199798ca5`
- Source mode: detached, tracked-clean worktree
- Aggregate:
  `runs/sudoku-backbone-benchmark-clean-20260724T020321Z-3bc5dec/`

This is a from-zero rerun. No old model, optimizer, or checkpoint was loaded.

## Question

Under one FutureSeed rule, data order, model shell, loop budget, optimizer, and
evaluator, which recurrent carrier is the best finite-budget default?

This run compares:

- RWKV7 with the local state-passing CUDA kernel;
- official FLA GatedDeltaNet;
- official FLA GatedDeltaNet2;
- official FLA KimiDeltaAttention.

It does not compare FutureSeed against no-FutureSeed and does not estimate an
architecture's asymptotic ceiling.

## Locked Contract

Every arm used:

- D192, 10 layers, 6 heads, head dimension 32;
- native unit-normalized FutureSeed with scale 1;
- five depth loops with equal CE supervision on every loop;
- microbatch 32, accumulation 4, effective batch 128;
- BF16, AdamW, LR 0.0015, weight decay 0.001, seed 52;
- official 3,831,994-board train split and 422,786-board test split;
- curriculum `46-50:100,51-55:400`;
- the same 512-board mixed and official blank-range evaluations;
- no noise, feedback, scratch state, selector, repair, search, or task rule.

Only the recurrent carrier changed. Architecture-specific parameter count,
runtime, and memory were measured rather than padded to equality.

## Post-Run Fairness Audit

This is a valid shared-recipe engineering comparison, but it is not a
publication-grade matched-capacity or matched-compute comparison.

What is genuinely matched:

- the official data paths, deterministic Python sampling stream, curriculum,
  number of sampled boards, and evaluation seeds;
- the outer D192/L10/H6/D32 shell, channel MLP, five loops, per-loop loss,
  optimizer hyperparameters, BF16 setting, and model initialization seed;
- the FutureSeed rule, gate initialization, normalization rule, and scale;
- the source SHA and strict CUDA/no-fallback implementation gates.

What is not matched:

- **Recurrent state capacity.** RWKV carries a `6 x 32 x 32` state, or 6,144
  scalars per sample. All three FLA arms were launched with `expand_v=2`, so
  they carry `6 x 64 x 32` states, or 12,288 scalars. With unit-RMS
  FutureSeed and a 0.5 initial gate, the corresponding preflight state norms
  are 16.0 for RWKV and 22.627 for the FLA arms. The FLA carriers therefore
  receive twice the state capacity and `sqrt(2)` times the total normalized
  seed energy.
- **Native defaults.** `expand_v=2` is the pinned FLA GDN default, but the
  pinned GDN2 and KDA defaults are `expand_v=1`. The run deliberately gave all
  FLA arms the same state size, so GDN2 and KDA are not native-default runs.
- **Parameters and compute.** Parameters range from 5.580M to 6.946M. Relative
  to RWKV, GDN/GDN2/KDA use 7.2%/24.5%/4.9% more parameters and take
  47.3%/58.3%/55.4% more wall time per optimizer step. The comparison cannot
  support a compute-efficiency claim.
- **Native optimizer conventions.** The shared AdamW applies weight decay to
  every parameter. The official FLA layers mark `A_log` and `dt_bias` as
  no-weight-decay parameters, but this runner does not honor those markers.
  This is a symmetric optimizer recipe, not each architecture's native recipe.
- **Cryptographic data provenance.** The code and arguments imply the same
  deterministic train/eval rows, but the artifact records dataset sizes,
  seeds, and one primary puzzle hash rather than hashes of every data file and
  sampled row stream.

State precision is not listed as a confirmed mismatch. The FLA preflight
records FP32 recurrent states. RWKV state-passing also computes its recurrent
state in FP32 and casts it to the block input dtype, but the benchmark did not
archive the resulting RWKV state dtype. A publication run must record state
shape, dtype, and normalized seed energy for every arm.

Fairness grade: **B- for selecting a provisional engineering carrier under one
shared recipe; insufficient for claiming that one recurrent equation is
intrinsically better.**

## Hard Gates

Before training, the suite required:

- GPU1 binding through `CUDA_VISIBLE_DEVICES=0`;
- CUDA forward/backward and finite gradients for all four full models;
- RWKV state-passing availability with silent fallback disabled;
- official FLA class and operation provenance;
- Triton short-convolution backend;
- installed FLA source hashes equal to the pinned wheel;
- a real gradient into the injected FutureSeed state.

Pinned FLA wheel SHA256:

`65f57bf2aa937991fc497bd63f42883d263ead8646f21f2492735ccddd82d0eb`

All gates passed. Every arm records the same source SHA, data identity, eval
seed, and primary puzzle hash. RWKV has an empty source patch. The other arms'
patches contain only generated `leaderboard.csv` and
`runs/visualization_index.html`; the strict summarizer rejects model or config
source changes.

## Result

| Metric | RWKV | GDN | GDN2 | KDA |
|---|---:|---:|---:|---:|
| Parameters | 5.580M | 5.980M | 6.946M | 5.852M |
| Train CE | 1.0432 | **0.9964** | 1.0078 | 1.0051 |
| Fixed h53 exact | 0.00586 | **0.01758** | 0.01562 | 0.01562 |
| Mixed loop1 exact | 0.00000 | 0.01172 | 0.01172 | 0.01172 |
| Mixed loop5 exact | 0.00781 | 0.02148 | 0.01953 | **0.02344** |
| Mixed loop5 blank accuracy | 0.48666 | **0.50589** | 0.50163 | 0.50351 |
| Loop1-to5 exact gain | 0.00781 | 0.00977 | 0.00781 | **0.01172** |
| Seconds per optimizer step | **3.43** | 5.06 | 5.44 | 5.34 |
| Peak allocated VRAM | 9.04 GiB | **7.04 GiB** | 9.08 GiB | 7.55 GiB |

KDA's mixed score is 12 solved boards out of 512; GDN solves 11. That
one-board edge is too small to select a carrier.

### Official Blank Ranges

| Blanks | RWKV exact / blank | GDN exact / blank | GDN2 exact / blank | KDA exact / blank |
|---|---:|---:|---:|---:|
| 46-50 | 0.3359 / 0.9659 | **0.8652 / 0.9955** | 0.7676 / 0.9905 | 0.7285 / 0.9899 |
| 51-55 | 0 / 0.5097 | 0 / **0.5279** | 0 / 0.5205 | 0 / 0.5257 |
| 56-64 | 0 / 0.4553 | 0 / **0.4727** | 0 / 0.4689 | 0 / 0.4674 |

GDN solves 443 of 512 official 46-50 boards; KDA solves 373. The 70-board
opening advantage is much larger than KDA's one-board mixed advantage.
No carrier solves a full board with 51 or more blanks at this budget.

## Loop Behavior

The aggregate evaluates the same puzzle for every carrier. Wrong cells change:

| Backbone | Loop1 | Loop2 | Loop3 | Loop4 | Loop5 |
|---|---:|---:|---:|---:|---:|
| RWKV | 24 | 24 | 23 | 23 | 22 |
| GDN | 24 | 22 | 22 | 22 | 22 |
| GDN2 | 27 | 25 | 24 | 23 | 23 |
| KDA | 22 | 21 | 21 | 21 | 20 |

All four get useful loop gains, but most corrections happen by loop 2 or 3.
Later loops mostly repeat the same answer on this hard board. This is evidence
for finite recurrent refinement, not sustained hard-board closure.

## Decision

Retain GDN as the provisional scaling carrier under this shared recipe.

- It has the best CE and official opening.
- It has the lowest memory among the official FLA arms.
- It already has the independent 30k-step hard-Sudoku scaling result.
- KDA's mixed edge is one board and does not survive the stronger official
  opening comparison.
- GDN2 uses 16% more parameters and more memory without a quality win.
- RWKV remains the fastest CUDA mechanism reference.

The correct claim is narrow: GDN is the best engineering default under this
shared finite-budget recipe. Because state capacity and compute are not
matched, the run does not show that GDN's recurrent equation is intrinsically
better. It also does not remove the 51-blank closure cliff.

## Reproducibility Notes

Relative to `P-BASELINE-001`, the broad result reproduced:

- RWKV metrics are effectively unchanged.
- GDN remains the best official opening carrier.
- GDN2 remains close but more expensive.
- KDA moved by three mixed-eval boards and now leads GDN by one board.

That small KDA movement is a warning against architecture claims from a single
512-board mixed count. The large official opening gap and the all-zero hard
range are the stable decisions.

## Artifacts

- Aggregate JSON/CSV/HTML and screenshots:
  `runs/sudoku-backbone-benchmark-clean-20260724T020321Z-3bc5dec/`
- Formal arms:
  `runs/sudoku-backbone-{rwkv,gdn,gdn2,kda}-s500-clean-20260724T020321Z-3bc5dec/`
- Raw preflight and suite log:
  `runs/sudoku-backbone-benchmark-clean-20260724T020321Z-3bc5dec/provenance/`
- Full source snapshots remain on GPU1 under the run directories; their hashes
  are recorded in `provenance/source_snapshots.sha256`.
- Model checkpoints remain under `/huyang2/double-loop/models` and are not in
  Git.

No experiment tag was created: every mixed score is below 0.50 and hard exact
is zero.
