# FLA GDN2/KDA FutureSeed Sudoku Gate

## 1. Metainfo

- Plan ID: `P-LA-001`
- Status: done; both matched probes completed and the scale gate failed
- Planned: 2026-07-19 11:00 CST / 2026-07-19T03:00:00Z
- Completed: 2026-07-19 13:15 CST / 2026-07-19T05:15:00Z
- Machine: AIStation `GPU1` A800 only
- Branch: `codex/fla-gdn2-kda`
- Training source SHA: `bf4b33820599f2f5782c128ac55b5a867d7e9d3d`
- Official FLA source SHA: `fe8fce9fc6984f22905f54cfa885dce1502baf26`
- Runs:
  - `fla-gdn2-futureseed-sudoku-gate-20260719T042329Z-bf4b338`
  - `fla-kda-futureseed-sudoku-gate-20260719T043622Z-bf4b338`

## 2. Hypothesis

FutureSeed already supplies each deeper recurrent layer with the terminal state of
the preceding layer. The unresolved question is whether the state update itself
is too coarse for difficult global closure.

The current GDN v1 state has one decay and one erase/write strength per head.
KDA uses a separate decay for each key channel. GDN2 additionally separates
channel-wise erase on the key axis from channel-wise write on the value axis.
These are generic learned memory operations, not Sudoku rules.

If state-edit granularity is the bottleneck, matched training should show a
mechanistic ordering: GDN2 opens or closes hard boards earlier than KDA, and KDA
earlier than GDN v1. If both new recurrences only improve cell accuracy without
improving full-board exact, the missing ingredient is not a richer delta-rule
state and this branch should stop.

## 3. Configuration

- Upstream implementation: official FLA commit
  `fe8fce9fc6984f22905f54cfa885dce1502baf26`.
- Wheel: locally built `flash_linear_attention-0.5.2-py3-none-any.whl`, SHA256
  `65f57bf2aa937991fc497bd63f42883d263ead8646f21f2492735ccddd82d0eb`.
- Task/data: official EqR Sudoku arrays,
  `sudoku-extreme-1k-aug-1000` train/test.
- Model: D128/L6/H8/head-dim16, channel multiplier4, expand-v1.
- Recurrent compute: loop4, every-loop CE, fixed native FutureSeed scale1.
- Training: 600 steps, batch128, BF16, LR `0.0015`, weight decay `0.001`.
- Arms: one `BACKBONE=gdn2`, one `BACKBONE=kda`.
- Matched reference: P-GDN-005 GDN v1 FutureSeed, exact `0.0107`, blank accuracy
  `0.4925`, CE `1.0667`, training `212.06s`, peak allocation `15380.5MB`.
- Disabled: noise, scratch, repair, search, selector, task rules, seed/LR/loss
  sweeps, and CPU smoke.

## 4. Environment

- GPU row: `GPU1`; `CUDA_VISIBLE_DEVICES=0` only.
- Expected GPU: NVIDIA A800-SXM4-80GB.
- Remote root: `/huyang2/double-loop`.
- Python: `/opt/conda/bin/python`, PyTorch `2.7.0+cu126`.
- All caches, wheel installs, checkpoints, logs, and runs remain below the
  repository root.

## 5. Commands

CUDA integration gate, including an actual three-layer FutureSeed stack at the
training head dimension, to run from a clean detached worktree:

```bash
CUDA_VISIBLE_DEVICES=0 \
XDG_CACHE_HOME=/huyang2/double-loop/.cache \
TRITON_CACHE_DIR=/huyang2/double-loop/.cache/triton \
TORCHINDUCTOR_CACHE_DIR=/huyang2/double-loop/.cache/torchinductor \
TORCH_EXTENSIONS_DIR=/huyang2/double-loop/.cache/torch_extensions \
TMPDIR=/huyang2/double-loop/.cache/tmp \
PYTHONPATH=/huyang2/double-loop/.cache/python-extra-pylib \
  /opt/conda/bin/python \
  experiments/rwkv_fs_sudoku/check_fla_delta_backbones.py \
  --backbone gdn2 \
  --out artifacts/fla-delta-kernel-check.json
```

The first cold-cache combined invocation was stopped by exact process group
`4157` at `20m28s`, as required by the integration-gate kill criterion. It had
not reported a numerical mismatch; it was still generating FLA autotune
kernels and had produced 3,848 persistent Triton cache files. The checker now
accepts `--backbone gdn2|kda`, logs each sub-check as it starts and passes, and
writes partial JSON after every passed sub-check. The split reruns reuse the
same repository-local cache and make a compile stall distinguishable from a
forward/backward/state mismatch.

The split GDN2 run passed its strict recurrence reference and real adapter
checks. Kernel versus Torch maximum errors were `5.99e-5` for output,
`2.76e-4` for terminal state, and `2.34e-4` over all gradients including the
initial state. The BF16 adapter had a nonzero initial-state gradient norm
`0.00735`; full versus split-sequence error was `0.00781` for output and
`0.00279` for terminal state; measured forward plus backward was `7.90ms` at
the test shape.

The initial three-layer check then exposed a checker bug rather than a kernel
failure: it treated every `grad=None` as non-finite, even though
`blocks.0.future_seed_logit` is structurally unused because layer zero has no
preceding terminal state to seed it. The corrected check permits only that
named missing gradient, reports any other missing gradient separately from
NaN/Inf, and supports `--check futureseed_stack` so the already-passed kernel
tests do not need to be rerun.

After both backbones passed, the first GDN2 training launch
`fla-gdn2-futureseed-sudoku-gate-20260719T041850Z-986e725` stopped before its
first optimizer step. P-GDN-005 was recorded before official-data blank-range
filtering existed, so its nominal default `2-4` was ignored and it actually
sampled uniformly from the full 46-64-blank train split. The current loader
correctly rejected the impossible range. The matched launcher now sets
`HOLES_MIN=46` and `HOLES_MAX=64`; because that is the complete train split,
the seeded index sampling is semantically identical to the historical
baseline. No curriculum or difficulty reweighting was introduced.

Training command template:

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 SOURCE_SNAPSHOT_MODE=lean \
PYTHON_BIN=/opt/conda/bin/python \
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
OFFICIAL_SUDOKU_TRAIN_SPLIT=train OFFICIAL_SUDOKU_EVAL_SPLIT=test \
SUDOKU_SIZE=9 HOLE_PATTERN=random EVAL_HOLES=16 EVAL_HOLES_LIST=16 \
BACKBONE=<gdn2-or-kda> GDN_MODE=chunk GDN_USE_SHORT_CONV=0 GDN_EXPAND_V=1.0 \
D_MODEL=128 LAYERS=6 HEADS=8 HEAD_DIM=16 CHANNEL_MULT=4 L_CYCLES=2 MAX_LOOPS=4 \
LOOP_LOSS=all FULL_STEPS=600 FULL_BATCH=128 FULL_EVAL_N=1024 \
FULL_ROLLOUT_KS= FULL_LOG_EVERY=100 FORWARD_DTYPE=bfloat16 \
LR=0.0015 WEIGHT_DECAY=0.001 BLANK_LOSS_WEIGHT=8 FUTURE_SEED_SCALE=1 \
RUN_NAME=<resolved-name> ./run.sh full
```

## 6. Artifacts

- CUDA checks: `runs/fla-delta-kernel-gate-20260719-986e725/`
- Pre-step abort: `runs/fla-gdn2-futureseed-sudoku-gate-20260719T041850Z-986e725/`
- GDN2 run: `runs/fla-gdn2-futureseed-sudoku-gate-20260719T042329Z-bf4b338/`
- KDA run: `runs/fla-kda-futureseed-sudoku-gate-20260719T043622Z-bf4b338/`
- Aggregate JSON/HTML and Playwright screenshots:
  `runs/fla-gdn2-kda-comparison-20260719-bf4b338/`

The generated source snapshot tarballs were intentionally not committed. Each
run still carries the exact source SHA and source patch; checkpoints remain
under repository-local `models/` and are not tracked by Git.

## 7. Results

### 7.1 CUDA correctness and state path

| Backbone | output max abs | state max abs | gradient max abs | initial-state grad norm | hot adapter fwd+bwd |
| --- | ---: | ---: | ---: | ---: | ---: |
| GDN2 | `5.99e-5` | `2.76e-4` | `2.34e-4` | `0.00735` | `7.896ms` |
| KDA | `6.56e-5` | `1.18e-4` | `2.58e-4` | `0.00783` | `12.522ms` |

Both kernels passed Torch-reference output, terminal-state, backward,
initial-state-gradient, sequence-split continuity, and three-layer native
FutureSeed stack checks. KDA is about `59%` slower than GDN2 in the hot adapter
microbenchmark. Its first cold adapter invocation spent `708s` compiling and
autotuning, so cold check time is not a throughput measurement.

### 7.2 Matched 600-step quality

Parameter counts were read from the exact step600 checkpoint state dictionaries.

| Backbone | parameters | CE@600 | loop1 exact | loop4 exact | exact gain | loop4 blank | peak alloc | observed train wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GDN v1 | `1.307M` | `1.0667` | `0.0010` | **`0.0107`** | **`+0.0098`** | **`0.4925`** | `15.02GiB` | `212.1s` |
| FLA GDN2 | `1.444M` | `1.0997` | `0.0000` | `0.0020` | `+0.0020` | `0.4813` | `10.45GiB` | `669.5s` |
| FLA KDA | `1.253M` | `1.1141` | `0.0010` | `0.0010` | `+0.0000` | `0.4697` | `8.70GiB` | `568.3s` |

Relative to GDN v1, GDN2 loses `0.00879` exact, `0.01115` blank accuracy,
and `0.03297` CE. KDA loses `0.00977` exact, `0.02282` blank accuracy, and
`0.04739` CE. Neither passes the predeclared quality gate of `+0.01` exact or
`+0.03` blank accuracy. Lower peak allocation is not an efficiency success
because quality is materially worse. End-to-end wall includes one-time Triton
compilation and is reported only as an observed number.

Training CE by step:

| Backbone | 100 | 200 | 300 | 400 | 500 | 600 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GDN v1 | `1.7918` | `1.3550` | `1.1813` | `1.1387` | `1.0861` | **`1.0667`** |
| FLA GDN2 | `1.8915` | `1.6666` | `1.3028` | `1.1694` | `1.1282` | `1.0997` |
| FLA KDA | `1.8847` | `1.6858` | `1.3464` | `1.1911` | `1.1519` | `1.1141` |

The simple GDN v1 recurrence learns faster from the first 100-step readout and
maintains the lead through step600. This is an optimization/sample-efficiency
boundary, not a claim that GDN2 or KDA can never catch up under unlimited
compute.

### 7.3 Difficulty transition and loop behavior

| Official blank range | GDN2 exact / blank | KDA exact / blank |
| --- | ---: | ---: |
| 46-50 | **`0.0732 / 0.9271`** | `0.0127 / 0.8970` |
| 51-55 | `0.0000 / 0.5010` | `0.0000 / 0.4922` |
| 56-64 | `0.0000 / 0.4513` | `0.0000 / 0.4410` |

GDN2 is clearly stronger than KDA on the easier 46-50-blank range, but both
collapse to zero exact at 51 blanks and above. FutureSeed is active in both:
GDN2 loop4 gate/state norm is `0.498 / 7.964`; KDA is `0.483 / 7.722`.
The failure is therefore not an accidentally disabled seed path.

On the same batch-index-0, 56-blank board, GDN2 wrong cells move
`23 -> 20 -> 20 -> 20`; KDA moves `23 -> 22 -> 21 -> 21`. Both make a small
early correction and then freeze. The aggregate loop curves agree: GDN2 exact
gain is only `+0.0020`, while KDA has no exact gain. More granular state edits
did not create stronger recurrent closure.

## 8. Conclusions

The mechanistic prediction was `GDN2 > KDA > GDN v1` if state-edit granularity
were the main hard-Sudoku bottleneck. The observed order is
`GDN v1 > GDN2 > KDA`. This rejects the proposed bottleneck at the current
matched sample/compute budget and makes a GDN2/KDA scale-up low ROI.

What remains positive is infrastructure and portability evidence: native
terminal-state FutureSeed works with both official FLA matrix-state kernels,
including real gradients through the seeded initial state. FutureSeed is not
tied to one RWKV/GDN equation. This experiment did not add new no-FS arms, so
it does not by itself provide a new causal estimate of FutureSeed benefit.
That estimate already exists for matched GDN v1 in P-GDN-005.

Decision:

- stop GDN2/KDA seed, LR, loss, width, and longer-training tables;
- do not promote either variant to D224/full-diversity;
- resume unchanged P-SCALE-034 GDN v1 clean full-diversity scaling from the
  exact step16000 checkpoint toward step20000, because that line still has a
  measured hard-exact positive slope;
- retain GDN2/KDA as tested backbone options, not as current mainline models.

## 9. Submission Record

None.
