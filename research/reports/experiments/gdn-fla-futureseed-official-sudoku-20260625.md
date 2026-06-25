# GDN/FLA FutureSeed Official Sudoku Gate

## 1. Metainfo

- run family: `gdn-fla-futureseed-official-sudoku-20260625`
- plan ID: `P-GDN-001`
- machine: AIStation GPU1 only
- work dir: `/huyang2/double-loop`
- local branch: `codex/gpu1-experiment-tracking`
- start time: 2026-06-25 Asia/Shanghai

## 2. Hypothesis

FutureSeed should be a generic recurrent-state mechanism, not an RWKV-only trick. If the mechanism is real, a Gated DeltaNet matrix-state backbone should support the same definition: normalize the previous layer terminal recurrent state and inject it as the next layer initial recurrent state.

The first decision is not whether GDN+FS beats everything. The first decision is whether FLA GDN can replace RWKV inside the standalone runner and train on official Sudoku at all. If no-FS GDN cannot open, modifying Triton for FutureSeed is premature.

## 3. Configuration

- dataset: official EqR Sudoku arrays, train/test split
- runner: `experiments/rwkv_fs_sudoku/study_rwkv_futureseed_loop.py`
- backbone options:
  - baseline: `BACKBONE=gdn FUTURE_SEED_SCALE=0`
  - mechanism arm, conditional: `BACKBONE=gdn FUTURE_SEED_SCALE=1`
- GDN implementation: FLA `chunk_gated_delta_rule`, pinned source SHA `9b20d26dc4922e67c1332ef77e30b71111406d96`
- no selector, no repair, no oracle rollout, no Sudoku rule postprocessing
- expected probe shape: D192/L10/H12/head_dim16, loop5, official train/test, bf16, rollout disabled

## 4. Environment

To fill after GPU1 probe:

- python:
- torch:
- CUDA device:
- FLA import/source:
- triton:

## 5. Commands

CUDA probe:

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 BACKBONE=gdn \
SUDOKU_SIZE=9 D_MODEL=48 LAYERS=2 HEADS=3 HEAD_DIM=16 CHANNEL_MULT=2 \
MAX_LOOPS=2 FULL_STEPS=1 FULL_BATCH=4 FULL_EVAL_N=8 FULL_ROLLOUT_KS= \
FORWARD_DTYPE=bfloat16 FUTURE_SEED_SCALE=0 RUN_NAME=gdn-fla-cuda-probe-<ts> ./run.sh full
```

No-FS gate:

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 SOURCE_SNAPSHOT_MODE=lean \
BACKBONE=gdn GDN_MODE=chunk GDN_EXPAND_V=1.0 GDN_USE_SHORT_CONV=1 \
SUDOKU_SIZE=9 D_MODEL=192 LAYERS=10 HEADS=12 HEAD_DIM=16 CHANNEL_MULT=4 \
L_CYCLES=2 MAX_LOOPS=5 FUTURE_SEED_SCALE=0 FORWARD_DTYPE=bfloat16 \
OFFICIAL_SUDOKU_DATA_DIR=<official-sudoku-dir> \
HOLE_STAGES=8-16:300,16-24:300 EVAL_HOLES=16 EVAL_HOLES_LIST=16,24,32 \
FULL_STEPS=600 FULL_BATCH=128 FULL_EVAL_N=512 FULL_ROLLOUT_KS= FULL_LOG_EVERY=100 \
RUN_NAME=gdn-official-sudoku-nofs-<ts>-<sha> ./run.sh full
```

Matched FS gate runs only if no-FS GDN is trainable:

```bash
... FUTURE_SEED_SCALE=1 RUN_NAME=gdn-official-sudoku-fs-<ts>-<sha> ./run.sh full
```

## 6. Artifacts

To fill:

- logs:
- score/config JSON:
- case HTML:
- source SHA:
- FLA wheel:

## 7. Results

Pending.

Primary readouts:

- train CE at steps 100/300/600
- loop1 vs loop5 exact and blank accuracy
- FS gate/state diagnostics
- wall time and peak memory

## 8. Conclusions

Pending.

Decision rules:

- Continue to native-FS GDN if no-FS GDN shows real optimization movement: CE decreases materially and blank accuracy opens above random.
- Treat as GDN baseline/scale failure if no-FS remains flat or FLA kernel is not viable under GPU1 budget.
- Claim FutureSeed generality only if matched FS improves sample efficiency or loop gain under the same GDN backbone.

## 9. Submission Record

None.
