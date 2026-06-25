# GDN/FLA FutureSeed Sudoku Gate

## 1. Metainfo

- run family: `gdn-fla-futureseed-sudoku-20260625`
- plan ID: `P-GDN-001`
- machine: AIStation GPU1 only
- remote work dir: `/huyang2/double-loop`
- local branch: `codex/gpu1-experiment-tracking`
- code commit: `5f8a20795c395751ad3c3c43eb4f4674502142e9`
- run time: 2026-06-25 Asia/Shanghai

## 2. Hypothesis

FutureSeed should be a generic recurrent-state mechanism, not an RWKV-only trick. If the mechanism is real, a Gated DeltaNet matrix-state backbone should support the same definition: normalize the previous layer terminal recurrent state and inject it as the next layer initial recurrent state.

The first decision is not whether GDN+FS beats everything. The first decision is whether GDN can replace RWKV in the standalone runner and train at all. If no-FS GDN cannot move optimization, modifying Triton for FutureSeed is premature.

## 3. Configuration

- dataset: generated 9x9 random-hole Sudoku for the fast viability gate. Official EqR Sudoku arrays were not present under `/huyang2/double-loop/repos/eqr`, so official-data GDN is deferred.
- runner: `experiments/rwkv_fs_sudoku/study_rwkv_futureseed_loop.py`
- target fast path: FLA `chunk_gated_delta_rule`, source SHA `9b20d26dc4922e67c1332ef77e30b71111406d96`
- viable training path used here: FLA `naive_recurrent_gated_delta_rule` on CUDA tensors
- native FutureSeed state layout preserved as `(B,H,V,K)`; naive recurrent internally transposes to FLA's `(B,H,K,V)` and transposes back.
- matched small gate: D64/L2/H4/head_dim16, loop2, generated holes `8-16:50,16-24:50`, batch32, 100 steps, bf16, rollout disabled.
- no selector, no repair, no oracle rollout, no Sudoku rule postprocessing.

## 4. Environment

- python: `/opt/conda/bin/python`, Python 3.10
- torch: `2.7.0+cu126`
- CUDA device: GPU1, NVIDIA A800-SXM4-80GB
- FLA runtime: minimal closure under `/huyang2/double-loop/.cache/fla-gdn-scp`
- remote execution note: the existing worktree reported base git SHA `a81a35d` plus dirty patch because a clean `git worktree add` for `5f8a207` stalled in checkout. The exact executed code is archived in each run's `source.patch` and `source_snapshot.tar.gz`; GitHub contains the clean commit.

## 5. Commands

No-FS generated gate:

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 SOURCE_SNAPSHOT_MODE=lean \
PYTHON_BIN=/opt/conda/bin/python PYTHON_EXTRA_PATH=/huyang2/double-loop/.cache/fla-gdn-scp \
BACKBONE=gdn GDN_MODE=naive_recurrent GDN_USE_SHORT_CONV=0 \
SUDOKU_SIZE=9 HOLE_STAGES=8-16:50,16-24:50 EVAL_HOLES=16 EVAL_HOLES_LIST=8,16,24 \
D_MODEL=64 LAYERS=2 HEADS=4 HEAD_DIM=16 CHANNEL_MULT=2 \
L_CYCLES=1 MAX_LOOPS=2 FUTURE_SEED_SCALE=0 FORWARD_DTYPE=bfloat16 \
FULL_STEPS=100 FULL_BATCH=32 FULL_EVAL_N=128 FULL_ROLLOUT_KS= FULL_LOG_EVERY=20 \
RUN_NAME=gdn-naive-nofs-small-20260625T1024-5f8a207 ./run.sh full
```

Matched FS generated gate:

```bash
... FUTURE_SEED_SCALE=1 RUN_NAME=gdn-naive-fs-small-20260625T1030-5f8a207 ./run.sh full
```

## 6. Artifacts

- no-FS run: `runs/gdn-naive-nofs-small-20260625T1024-5f8a207`
- FS run: `runs/gdn-naive-fs-small-20260625T1030-5f8a207`
- logs: `logs/run.log`
- metrics: `output/futureseed_loop_seed52.json`, `score.json`
- case HTML: `output/futureseed_loop_case_seed52.html`
- visual index: `visualizations/index.html`
- aborted large no-FS probe: `runs/gdn-naive-nofs-gate-20260625T1018-5f8a207/abort.json`, killed because D96/L4/B64 did not reach step50 after 4.5 minutes.

## 7. Results

Chunk-kernel viability:

- Minimal FLA closure import succeeded.
- `chunk_gated_delta_rule` forward succeeded on CUDA with output shape `(2,64,2,16)` and final state shape `(2,2,16,16)`.
- Chunk backward did not complete within the probe budget. Both `loss = o^2 + state^2` and output-only `loss = o^2` reached forward then stalled in backward for more than 2.5-3.5 minutes with low GPU utilization.
- Decision: do not run training through FLA chunk backward yet.

Naive-recurrent training gate:

| arm | step20 CE | step40 CE | step100 CE | loop1 blank_acc | loop2 blank_acc | loop2 exact | train sec |
|---|---:|---:|---:|---:|---:|---:|---:|
| no-FS | 1.7915 | 1.5985 | 1.5995 | 0.2319 | 0.2319 | 0.0000 | 182.47 |
| FS | 1.7942 | 1.6101 | 1.6113 | 0.2466 | 0.2383 | 0.0000 | 182.10 |

Hole transfer, loop2:

| arm | h8 blank_acc | h16 blank_acc | h24 blank_acc |
|---|---:|---:|---:|
| no-FS | 0.2588 | 0.2319 | 0.1934 |
| FS | 0.2520 | 0.2383 | 0.2061 |

Diagnostics:

- no-FS confirms GDN replacement can train on CUDA: CE drops clearly by step40/100 and blank accuracy opens above random.
- FS has active state injection: `fs_gate ~= 0.471`, `fs_state_norm ~= 7.537`.
- FS does not show a strong matched improvement. CE is slightly worse; loop1 blank accuracy is slightly better; loop2 and hole transfer are mixed.
- Loop is neutral: loop2 does not improve exact, and in FS loop2 blank accuracy is lower than loop1.

## 8. Conclusions

Mixed, mostly infrastructure-positive and mechanism-neutral.

What is solid:

- Native GDN state plumbing works in the standalone runner.
- Native FutureSeed definition was preserved; no reverse scan or task rule was introduced.
- GDN no-FS is trainable under a small CUDA budget, so the backbone replacement is not dead.
- FLA chunk forward/state works, so the intended accelerated path is plausible.

What is not yet solid:

- FLA chunk backward is not usable under current GPU1 setup without further autotune/kernel work.
- Naive recurrent GDN is too slow to scale meaningfully; D96/L4/B64 is already low ROI.
- The small matched FS-vs-noFS gate does not show convincing FutureSeed gain on GDN.

Decision:

- Stop naive-recurrent scaling; it is a debugging bridge, not the paper path.
- Next high-ROI work is kernel/modeling infrastructure: make a fast trainable GDN path, either by patching FLA chunk backward/autotune or implementing a FutureSeed-aware GDN Triton state path directly.
- Do not claim GDN+FutureSeed works yet. Current paper evidence still comes from RWKV FutureSeed on official Sudoku; GDN is an open extension.

## 9. Submission Record

None.
