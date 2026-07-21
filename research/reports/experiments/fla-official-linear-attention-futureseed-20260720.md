# Official FLA Linear Attention + FutureSeed Comparison

## 1. Metainfo

- Plan ID: `P-LA-002`
- Status: completed; architecture-enumeration hypothesis rejected at the matched
  500-step gate
- Started: 2026-07-20 12:36 CST / 2026-07-20T04:36:00Z
- Completed: 2026-07-21 01:10 CST / 2026-07-20T17:10:00Z
- Machine: AIStation `GPU1`, NVIDIA A800-SXM4-80GB; GPU2 forbidden and unused
- Branch: `codex/fla-gdn2-kda`
- Training source SHA: `c3342b8326a6e00e012cd6972d8c0e295d82c0b3`
- Official FLA source marker: `fe8fce9fc6984f22905f54cfa885dce1502baf26`
- Package: `flash-linear-attention==0.5.2`
- Pinned wheel SHA256:
  `65f57bf2aa937991fc497bd63f42883d263ead8646f21f2492735ccddd82d0eb`

## 2. Hypothesis

The earlier GDN/KDA/GDN2 comparison changed implementation and training recipe
at the same time. This run isolates the Linear Attention recurrence: under one
pinned official FLA package, one outer model, one native FutureSeed rule, one
dataset, and one compute budget, does a more expressive memory update improve
difficult global closure?

KDA uses key-channel-dependent decay. GDN2 separately controls key-channel
erase and value-channel write. The predeclared finite-budget prediction was
that at least one of training-loss slope, 51-55-blank exact accuracy, or
late-loop gain would order `GDN2 >= KDA >= GDN`. If all implementations trained
normally but failed that prediction, the decision was to stop enumerating
Linear Attention equations and return to the demonstrated data/compute scaling
axis.

## 3. Configuration

The only model variable was the official FLA layer class:

| Arm | Official class | FutureSeed cache layout |
|---|---|---|
| GDN | `fla.layers.gated_deltanet.GatedDeltaNet` | `[B,H,V,K]` |
| KDA | `fla.layers.kda.KimiDeltaAttention` | `[B,H,V,K]` |
| GDN2 | `fla.layers.gdn2.GatedDeltaNet2` | `[B,H,K,V]` |

Shared configuration:

- D192, L10, H6, head dimension 32, value expansion 2, short convolution 4,
  channel multiplier 4
- five outer loops and equal CE supervision at every loop
- BF16 forward, microbatch 32, gradient accumulation 4, effective batch 128
- AdamW, LR `0.0015`, weight decay `0.001`, seed 52
- native FutureSeed scale 1, unit-normalized fixed state transfer
- official full-diversity Sudoku arrays under
  `/huyang2/double-loop/data/sudoku-extreme-full`
- effective curriculum through the compared step500 checkpoint
  `46-50:100,51-55:400`; the GDN launch retained an unused planned hard tail to
  step1500, whereas the later KDA/GDN2 launches ended their plan at step500
- evaluation on holes 53 and official blank ranges `46-50`, `51-55`, `56-64`
- no feature noise, task rules, repair, search, selector, best-of-K, CPU smoke,
  GPU2, or architecture-specific hyperparameter changes

The original plan requested 1500 steps per arm. The first GDN arm required
2933 seconds to reach step 500, so three matched 1500-step arms could not fit in
the available GPU1 lease. GDN was stopped by its exact PID after writing the
step-500 checkpoint and `abort.json`; KDA and GDN2 then received the identical
500-step budget. This budget decision was made before their results were known.
The first GDN2 launch was also stopped before it produced a usable result because
automatic tracking output made that worktree dirty. Its exact-PID abort record is
preserved; the reported GDN2 arm is the subsequent clean-SHA relaunch. That run
finished before the GPU lease expired, and its checkpoint, log, and evaluation
artifacts remained intact under `/huyang2`.

## 4. Environment And No-Fallback Gate

- Python: `/opt/conda/bin/python`
- PyTorch: `2.7.0+cu126`
- GPU: `NVIDIA A800-SXM4-80GB`, visible device index 0 only
- Caches and temporary files: below `/huyang2/double-loop/.cache`
- `FLA_DISABLE_BACKEND_DISPATCH=1`
- `FLA_CONV_BACKEND=triton`
- `FLA_STRICT_OFFICIAL=1`

The strict gate passed before training:

- ten installed FLA source files are byte-identical to the pinned wheel;
- imported classes and unwrapped operators resolve into the installed FLA
  package, including functions wrapped by `torch.compiler`;
- each complete official layer `forward` is called, and FutureSeed only injects
  the prior layer's terminal state through official FLA `Cache`;
- the observed CUDA backward nodes are
  `ChunkGatedDeltaRuleFunctionBackward`, `ChunkKDAFunctionBackward`, and
  `ChunkGDN2FunctionBackward`;
- initial-state gradients are finite and nonzero: GDN `0.006503`, KDA
  `0.012747`, GDN2 `0.011757`;
- CUDA-vs-Torch maximum output/state errors are GDN
  `4.54e-5/1.60e-4`, KDA `6.56e-5/1.18e-4`, and GDN2
  `5.99e-5/2.76e-4`; all passed the BF16-aware gate;
- all Q/K/V short-convolution backends reported `triton`.

There is no bidirectional scan disguised as FutureSeed and no local Torch
recurrent fallback in these training arms. A provenance, backend, autograd-node,
state-layout, or CUDA-gradient mismatch makes the strict gate fail rather than
silently selecting another path.

## 5. Commands

Strict provenance/reference/backward gate:

```bash
CUDA_VISIBLE_DEVICES=0 \
FLA_DISABLE_BACKEND_DISPATCH=1 FLA_CONV_BACKEND=triton \
XDG_CACHE_HOME=/huyang2/double-loop/.cache \
TRITON_CACHE_DIR=/huyang2/double-loop/.cache/triton \
TORCHINDUCTOR_CACHE_DIR=/huyang2/double-loop/.cache/torchinductor \
TORCH_EXTENSIONS_DIR=/huyang2/double-loop/.cache/torch_extensions \
TMPDIR=/huyang2/double-loop/.cache/tmp \
PYTHONPATH=/huyang2/double-loop/.cache/python-extra-pylib \
/opt/conda/bin/python \
  experiments/rwkv_fs_sudoku/check_fla_delta_backbones.py \
  --backbone all \
  --wheel /huyang2/double-loop/wheelhouse/flash_linear_attention-0.5.2-py3-none-any.whl \
  --out artifacts/fla-official-futureseed-strict-gate-c3342b8/gate.json
```

Matched training command, once for each backbone:

```bash
CUDA_VISIBLE_DEVICES=0 FULL_STEPS=500 EVAL_CHECKPOINT_STEPS=500 \
  scripts/run_fla_official_futureseed_compare.sh \
  <fla_gdn|kda|gdn2> <run-name>
```

## 6. Artifacts

- Strict gate: `artifacts/fla-official-futureseed-strict-gate-c3342b8/`
- GDN train/checkpoint:
  `runs/fla-official-gdn-fs-d192l10-s1500-20260720T0825Z-c3342b8/`
- GDN unified evaluation:
  `runs/fla-official-gdn-fs-d192l10-step500-unifiedeval-20260720T1435Z-c3342b8/`
- KDA:
  `runs/fla-official-kda-fs-d192l10-s500-20260720T0908Z-c3342b8/`
- GDN2:
  `runs/fla-official-gdn2-fs-d192l10-s500-clean-20260720T1034Z-c3342b8/`
- GDN full-model backward-memory diagnostic:
  `runs/fla-official-gdn-trainmem-step501-20260720T1455Z-c3342b8/`
- Aggregate JSON/HTML/screenshots:
  `runs/fla-official-futureseed-comparison-20260720-c3342b8/`
- Launch and exact-stop records: selected `artifacts/launch/fla-official-*`
  files, including the budget truncation and dirty-worktree GDN2 abort
- Remote checkpoints and lean source snapshots remain under repository-local
  `/huyang2/double-loop/models/` and each remote run directory; model weights
  are not committed.

## 7. Results

### Matched 500-step gate

| Metric | GDN | KDA | GDN2 |
|---|---:|---:|---:|
| Parameters | 5.980M | 5.852M | 6.946M |
| Train CE at step 100 | 1.5918 | 1.6655 | 1.6054 |
| Train CE at step 300 | 1.0492 | 1.1097 | 1.0803 |
| Train CE at step 500 | **0.9981** | 1.0252 | 1.0101 |
| Pre-fixed checkpoint holes53 loop5 exact | **0.0176** | **0.0176** | **0.0176** |
| Final holes53 loop1 exact | **0.0117** | 0.0078 | 0.0078 |
| Final holes53 loop5 exact | **0.0215** | 0.0156 | 0.0195 |
| Final holes53 loop5 blank accuracy | **0.5049** | 0.4971 | 0.5012 |
| Loop1-to5 exact gain | 0.0098 | 0.0078 | **0.0117** |
| Train wall to step 500 | 48.9 min | 53.6 min | **47.9 min** |
| Peak full-model allocated VRAM | **7.11 GiB** | 7.55 GiB | 9.08 GiB |

The pre-fixed checkpoint batch is exactly tied. The separate final 512-example
batch differs by at most three solved boards, so it is not evidence for a hard
exact winner. The more stable finite-budget differences are CE, easy-to-medium
opening, parameters, wall time, and memory.

### Official blank ranges at loop 5

| Blank range | GDN exact / blank | KDA exact / blank | GDN2 exact / blank |
|---|---:|---:|---:|
| 46-50 | **0.8594 / 0.9947** | 0.6504 / 0.9855 | 0.7539 / 0.9903 |
| 51-55 | 0 / **0.5303** | 0 / 0.5220 | 0 / 0.5218 |
| 56-64 | 0 / **0.4733** | 0 / 0.4587 | 0 / 0.4685 |

All three cross the same full-board cliff at 51 blanks. GDN2 uses about 16%
more parameters and 28% more allocated VRAM than GDN, but does not open that
range. KDA uses slightly fewer parameters than GDN but is about 9.6% slower to
step 500 and has worse CE and opening accuracy.

### Same 53-blank case

The puzzle region has SHA256
`b65514a35fb363f26847cc165b76d08b41063210d7d91de00aaf852977128aa5`
in all three visualizations.

| Arm | Loop1 | Loop2 | Loop3 | Loop4 | Loop5 |
|---|---:|---:|---:|---:|---:|
| GDN wrong cells | 20 | 22 | 22 | 22 | 22 |
| KDA wrong cells | 23 | 21 | 20 | 20 | 20 |
| GDN2 wrong cells | 25 | 24 | 23 | 23 | 23 |

This board is deliberately not cherry-picked as a success. KDA and GDN2 make a
small early correction and then freeze; GDN makes this board worse after loop1.
The aggregate metrics show the same broad pattern: most benefit arrives by
loop2 or loop3, not as sustained late-loop global correction.

## 8. Conclusions

`GDN2 >= KDA >= GDN` is rejected at this matched finite-compute gate. The
newer equations are more expressive, but that extra freedom does not make them
easier to optimize on this task. Official GDN has the best CE slope, strongest
46-50-blank closure, and lowest measured full-model memory among the three;
therefore it remains the scaling backbone.

The result does not say GDN2 or KDA are universally worse. It says that swapping
the recurrence equation alone is not the missing mechanism for hard Sudoku at
this scale. All three fail to convert roughly 50% blank-cell accuracy into a
globally valid board beyond 50 blanks.

This comparison ranks three FutureSeed-enabled backbones. It does not estimate
the causal gain of FutureSeed itself. The paper-quality causal comparison should
remain one matched no-FutureSeed control at the selected GDN scaling point,
instead of tripling this into a low-information no-FS table.

Decision: stop Linear Attention architecture enumeration, keep official GDN,
and spend the next compute on the already positive clean data/longer-training
scaling path. No score tag is created because the primary score is below 0.50.

## 9. Submission Record

None. This is a mechanism/architecture gate, not a competition submission.
