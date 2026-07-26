# FutureSeed Causal Four-Backbone Comparison

## 1. Metainfo

- Experiment: `E-BASELINE-005`
- Plan: `P-BASELINE-005`
- Status: completed, causal gate supported
- Scheduled: 2026-07-26 10:27 CST / 2026-07-26 02:27 UTC
- Controls completed: 2026-07-26 13:18 CST / 2026-07-26 05:18 UTC
- Final causal contract completed: 2026-07-26 13:38 CST /
  2026-07-26 05:38 UTC
- Machine: AIStation GPU1, NVIDIA A800-SXM4-80GB
- GPU2: forbidden for this experiment and never addressed by the launcher;
  its status may be controlled by another task
- CPU model smoke: forbidden
- Formal source: `6e51f067a18d936bda7f3d588b7b3f32625b2b18`
- Suite: `nofs-causal-20260726T022718Z-6e51f06`

## 2. Hypothesis

The accepted strict table ranks four FutureSeed-enabled recurrent carriers but
cannot estimate FutureSeed's causal contribution.

If FutureSeed is a generic cheap-future-context mechanism, enabling it should
improve full-board exactness across at least two independently implemented
causal recurrent carriers under identical initialization, data, objective,
optimizer, recurrent-state geometry, loop budget, and evaluator. A blank
accuracy increase without full-board exact improvement does not pass.

This is the minimum causal comparison required by the paper claim. It is not a
seed, geometry, loss, LR, or budget sweep.

Here FutureSeed has its native meaning. Layer `l` performs one ordinary
left-to-right causal recurrent scan. Its terminal recurrent state, which has
seen the whole sequence, is unit-normalized, gated, and used as the initial
state of layer `l+1`. There is no right-to-left scan, bidirectional mixer,
selector, or oracle. Setting `future_seed_scale=0` skips only that initial-state
injection while preserving all parameters and the same causal carrier.

## 3. Configuration

Both FutureSeed states use:

- carriers: official RWKV7 TimeMix, official FLA GDN, GDN2, and KDA;
- D192, 10 layers, 6 heads, head dimension 32;
- matched six-head 32-by-32 recurrent state;
- five loops and equal cross-entropy supervision on every loop;
- microbatch 32, accumulation 4, effective batch 128;
- BF16, AdamW, LR 0.0015, weight decay 0.001;
- seed 52 and backbone-independent shared-shell initialization seed 52;
- official full-diversity Sudoku and curriculum `46-50:100,51-55:400`;
- identical fixed-53, mixed, and official 46-50/51-55/56-64 evaluation;
- no noise, feedback, scratch state, selector, repair, search, or task rule.

The sole substantive intervention is `FUTURE_SEED_SCALE=1` versus `0`.
Run/checkpoint output paths necessarily differ. Parameter counts must match
within each carrier.

The four carriers are state-geometry matched, not padded to identical parameter
counts. Their measured sizes range from 4.736M to 5.462M parameters. Therefore
the cross-carrier ordering is a finite-budget engineering comparison. The
causal claim is stricter: every FS/noFS pair has exactly the same parameter
count and byte-identical initialization.

## 4. Environment

- Persistent root: `/huyang2/double-loop`
- Formal detached worktree:
  `/huyang2/double-loop/artifacts/worktrees/fs-causal-6e51f06`
- Python: `/opt/conda/bin/python`
- Python extras: `/huyang2/double-loop/.cache/python-extra-pylib`
- Data: `/huyang2/double-loop/data/sudoku-extreme-full`
- Models: `/huyang2/double-loop/models`
- Runs: `/huyang2/double-loop/runs`
- Required device: `CUDA_VISIBLE_DEVICES=0`
- FLA: backend dispatch disabled, Triton convolution required

The official data identity was captured after both launch windows. The train
arrays were last modified on 2026-07-11 and the official test arrays on
2026-06-21, before the FutureSeed runs started on 2026-07-24. SHA256:

- train inputs:
  `979f0f27411dfde97b725082cd29da03bea7c3b1c6b2bd2b15bfe95b76549cbc`;
- train labels:
  `84e8a40dbb12d032eb9eba601c378d8388ea949430e07c00e80bc0947a59e48a`;
- test inputs:
  `cbc2ab9f1743d79da9c0c26273460140bf542a5fb3229cfdd80575615163bcd5`;
- test labels:
  `3d0bb8e73ecc5681675803c1cb518ebb4e771dac93a6741256a7a85ab8e29c2b`.

## 5. Commands

Preflight and formal launch are archived under:

`/huyang2/double-loop/artifacts/launch/nofs-causal-20260726T022718Z-6e51f06/`

Each no-FutureSeed arm runs:

```bash
CUDA_VISIBLE_DEVICES=0 \
BENCHMARK_CONFIG=<archived-nofs-config> \
BENCHMARK_STEPS=500 \
PERSIST_ROOT=/huyang2/double-loop \
RUNS_ROOT=/huyang2/double-loop/runs \
PYTHON_BIN=/opt/conda/bin/python \
./scripts/run_sudoku_backbone_benchmark.sh <rwkv|gdn|gdn2|kda> <run-name>
```

Kill conditions:

- any configuration difference beyond FutureSeed scale and output paths;
- source, formula, CUDA/Triton backward, state, shared initialization, data,
  evaluator, parameter-count, or no-fallback gate fails;
- step 100 takes more than 15 minutes;
- NaN, OOM, wrong GPU, or low utilization with high memory.

There is no automatic kernel, batch, device, or implementation fallback.

The fresh GPU1 preflight requires:

- pinned official FLA 0.5.2 source SHA `fe8fce9` and wheel SHA256
  `65f57bf2aa937991fc497bd63f42883d263ead8646f21f2492735ccddd82d0eb`;
- exact official `GatedDeltaNet`, `GatedDeltaNet2`, and
  `KimiDeltaAttention` classes;
- expected official chunk autograd nodes and Triton short convolutions;
- official RWKV7 source commit `9521024` and state-passing CUDA SHA256
  `59a90a0521b1851da17c008c685f959d586af1a7d28056b29a7478ab92c1c892`;
- finite full-stack backward and nonzero initial-state gradients;
- all 77 shared-shell tensor hashes identical across carriers.

After training, a separate GPU1 causal contract constructs FS/noFS models from
the same RNG stream and requires every initial parameter tensor to be
byte-identical, FS-on state injection and gate gradients to be nonzero, FS-off
state injection and gate gradients to be exactly zero, and both arms to use the
same official runtime identity.

## 6. Artifacts

- Strict aggregate:
  `runs/futureseed-causal-four-carrier-20260726T022718Z-6e51f06/`
- Browser report:
  `runs/futureseed-causal-four-carrier-20260726T022718Z-6e51f06/index.html`
- Machine-readable result:
  `runs/futureseed-causal-four-carrier-20260726T022718Z-6e51f06/comparison.json`
- Pair validator:
  `runs/futureseed-causal-four-carrier-20260726T022718Z-6e51f06/strict_pair_validation.json`
- CSV:
  `runs/futureseed-causal-four-carrier-20260726T022718Z-6e51f06/comparison.csv`
- GPU1 launch, contract, and fail-closed attempt logs:
  `runs/futureseed-causal-four-carrier-20260726T022718Z-6e51f06/provenance/`
- Browser screenshots:
  `browser-top.png`, `browser-rwkv-curves.png`, and
  `browser-gdn-case.png` in the strict aggregate directory.
- At 2026-07-26 14:04 CST / 06:04 UTC, the four no-FutureSeed HTML pages were
  mechanically relabeled from the generator's old FutureSeed-only title to
  `No-FutureSeed`. The result JSON, scores, configs, logs, and metrics were not
  changed; the strict aggregate was then regenerated from those same results.
- The four no-FutureSeed run directories are archived as siblings under
  `runs/`; model and optimizer checkpoints remain only under
  `/huyang2/double-loop/models/`.

## 7. Results

All four paired validators and the aggregate fairness gate passed.

| Carrier | Params | CE FS / noFS | b46-50 exact FS / noFS | Mixed exact FS / noFS | FS time overhead | FS VRAM delta |
|---|---:|---:|---:|---:|---:|---:|
| RWKV7 | 5.093M | 1.0190 / 1.6274 | 0.7285 / 0 | 0.0195 / 0 | +9.3% | +0.29 GiB |
| GDN | 4.867M | 1.0509 / 1.6258 | 0.3633 / 0 | 0.0059 / 0 | +7.2% | +0.34 GiB |
| GDN2 | 5.462M | 1.0062 / 1.6227 | 0.7969 / 0 | 0.0234 / 0 | +17.2% | +0.34 GiB |
| KDA | 4.736M | 1.0179 / 1.6266 | 0.6465 / 0 | 0.0176 / 0 | +8.2% | +0.34 GiB |

The preregistered b46-50 exact deltas are therefore
`+0.7285/+0.3633/+0.7969/+0.6465` for RWKV7/GDN/GDN2/KDA. The success rule
required two carriers above `+0.10`; all four pass.

This is not only a lower final CE. On b46-50, loop-5 blank accuracy is
`0.9903/0.9701/0.9924/0.9854` with FutureSeed and
`0.3826/0.3825/0.3938/0.3862` without it. Full-board exact rises from zero, so
the result passes the stronger global metric.

The learning curves localize the effect. GDN, GDN2, and KDA noFS are initially
as good as or better than FS at step 100, but after the curriculum enters
51-55 blanks they plateau near CE `1.62-1.63`; their FS pairs continue toward
`1.05/1.01/1.02`. This is inconsistent with explaining the result as merely a
lucky easier-stage initialization.

Loop behavior on b46-50 is:

- RWKV7 FS: `0.0762 -> 0.6758 -> 0.7090 -> 0.7246 -> 0.7285`;
- GDN FS: `0.1172 -> 0.3047 -> 0.3457 -> 0.3594 -> 0.3633`;
- GDN2 FS: `0.4160 -> 0.7305 -> 0.7715 -> 0.7910 -> 0.7969`;
- KDA FS: `0.1387 -> 0.5723 -> 0.6348 -> 0.6445 -> 0.6465`;
- every noFS sequence remains zero.

On the identical primary puzzle, wrong-cell counts from loop 1 to 5 are FS
versus noFS:

- RWKV7: `24 -> 23` versus `40 -> 41`;
- GDN: `23 -> 17` versus `42 -> 40`;
- GDN2: `21 -> 20` versus `39 -> 39`;
- KDA: `28 -> 24` versus `43 -> 40`.

The independent GPU1 functional contract loaded the same accepted FS step-500
checkpoint into FS-on and FS-off models. Output RMS deltas are
`0.992/1.367/1.604/1.050` for RWKV7/GDN/GDN2/KDA. FS-on gate gradient norms
are `0.0861/0.1003/0.0511/0.0432`; every FS-off gate gradient is exactly zero.
Constructor parameter hashes and loaded functional parameter hashes are
identical within every pair.

Two contract attempts failed closed and are archived. Attempt 1 omitted the
project-local `ninja` path and refused to load RWKV CUDA instead of falling
back. Attempt 2 incorrectly required nonzero gate gradients at the official
RWKV7 zero-output initialization. The final contract separates byte-identical
constructor initialization from a functional check on identical trained
weights, resolving that mathematical initialization issue without changing
the experiment.

Success requires at least two carriers with official 46-50 blank loop-5 exact
gain `FS - noFS >= 0.10`, plus loop/case evidence that the gain is not only an
isolated operating-point fluctuation. A stronger result is FutureSeed reaching
the no-FutureSeed loop-5 quality in fewer loops.

## 8. Conclusions

**Supported at this budget:** native FutureSeed is not an RWKV-specific trick.
The same terminal-state-to-next-layer-initial-state mechanism opens moderate
9x9 Sudoku in official RWKV7, GDN, GDN2, and KDA while adding no parameters.
It costs 7.2-17.2% wall time and about 0.29-0.34 GiB in this implementation.

**Not supported yet:** this experiment does not establish a hard-task or
asymptotic win. All eight conditions remain zero full-board exact at 51-55 and
56-64 blanks. FutureSeed roughly doubles hard-range per-blank accuracy, but
local correctness has not become global closure. The carrier ordering is also
only a state-matched finite-budget comparison, not a universal ranking.

The highest-ROI next paper experiment is not another seed or carrier table. It
is a quality-matched efficiency/scale gate: choose one efficient official
carrier, give FS and noFS enough clean data and compute to reach a common
quality target, and measure whether FutureSeed reaches that target materially
earlier or opens 51-55 exact. EqR remains the noncausal ceiling comparison; this
result isolates the causal-recurrent mechanism.

## 9. Submission Record

No tag or submission is allowed unless the primary score is at least 0.50 and
the mechanism conclusion is clean.
