# GDN Transition Soft-Group Aggregate Noise Gate

## 1. Metainfo

- plan_id: P-DIAG-024
- run_name: gdn-transition-softgroupnoise-d224l12-s1000-20260705T0517Z-75f9936
- machine: AIStation GPU1 only
- status: completed, discarded as main direction
- created_utc: 2026-07-05T05:17Z
- completed_utc: 2026-07-05T06:07Z

## 2. Hypothesis

P-DIAG-023 showed that one-hot Gumbel aggregation destroys optimization. That does not kill the broader noise idea. The more plausible version is bounded multi-token stochastic aggregation: sample a small group of hidden states, aggregate them softly, and cap the perturbation norm so noise regularizes global summary formation without turning into a single-cell shock.

Prediction: entropy should stay nonzero, perturbation should be clipped to a small norm, and step500 should stay above random blank accuracy. If it survives that gate and step1000 beats the clean D224/L12 same-step transition curve, this noise direction deserves scale. If not, stop aggregation noise for now.

## 3. Configuration

- backbone: GDN + native FutureSeed terminal-state seeding
- model: D224/L12/H14/head_dim16, `GDN_EXPAND_V=4.0`, `GDN_USE_SHORT_CONV=0`
- loops: `MAX_LOOPS=5`, `LOOP_LOSS=all`
- data: official EqR Sudoku extreme arrays
- curriculum: `46-50:100,51-55:900`
- batch: microbatch32, grad accumulation4, effective batch128
- new mechanism: `HIDDEN_AGG_NOISE_MODE=soft_group`, `HIDDEN_AGG_NOISE_SCALE=0.03`, `HIDDEN_AGG_NOISE_TEMP=4.0`, `HIDDEN_AGG_NOISE_TOPK=8`, `HIDDEN_AGG_NOISE_MAX_NORM=2.0`, `HIDDEN_AGG_NOISE_DETACH=1`
- disabled: feature-diff noise, scratch, repair, search, selector, seed sweep

## 4. Environment

- remote work_dir: `/huyang2/double-loop`
- GPU: GPU1 only, `CUDA_VISIBLE_DEVICES=0`
- Python: `/opt/conda/bin/python`
- repo branch: `codex/gpu1-experiment-tracking`
- launch SHA: `75f9936e2dd45eea1af19c331e9986a8065db0f4`

## 5. Commands

```bash
CUDA_VISIBLE_DEVICES=0 SMOKE_DONE=1 SKIP_SETUP=1 SOURCE_SNAPSHOT_MODE=lean \
PYTHON_BIN=/opt/conda/bin/python SUDOKU_SIZE=9 \
OFFICIAL_SUDOKU_DATA_DIR=/huyang2/double-loop/official_eqr_sudoku_repro_20260623/data/sudoku-extreme-1k-aug-1000 \
BACKBONE=gdn GDN_MODE=triton_recurrent GDN_USE_SHORT_CONV=0 GDN_EXPAND_V=4.0 \
D_MODEL=224 LAYERS=12 HEADS=14 HEAD_DIM=16 CHANNEL_MULT=4 L_CYCLES=2 \
FUTURE_SEED_SCALE=1 MAX_LOOPS=5 LOOP_LOSS=all SCRATCH_MODE=none \
HIDDEN_AGG_NOISE_MODE=soft_group HIDDEN_AGG_NOISE_SCALE=0.03 HIDDEN_AGG_NOISE_TEMP=4.0 \
HIDDEN_AGG_NOISE_TOPK=8 HIDDEN_AGG_NOISE_MAX_NORM=2.0 HIDDEN_AGG_NOISE_DETACH=1 \
HOLE_STAGES=46-50:100,51-55:900 EVAL_HOLES=60 EVAL_HOLES_LIST=53,60,64 \
FULL_BATCH=32 GRAD_ACCUM_STEPS=4 FULL_STEPS=1000 FULL_EVAL_N=512 \
FULL_LOG_EVERY=100 FULL_ROLLOUT_KS=1 SAVE_TRAIN_CHECKPOINT_EVERY=100 \
EVAL_CHECKPOINT_STEPS=500,1000 EVAL_CHECKPOINT_HOLES_LIST=53,60,64 \
OFFICIAL_EVAL_BLANK_RANGES=46-50,51-55,56-64 \
CASE_BANK_HOLES=53,60,64 CASE_BANK_N=4 CASE_BANK_EVAL_N=256 CASE_BANK_LOOP_VALUES=1,3,5 \
RUN_NAME=gdn-transition-softgroupnoise-d224l12-s1000-20260705T0517Z-75f9936 ./run.sh full
```

## 6. Artifacts

- Remote run dir: `/huyang2/double-loop/.worktrees/pdiag024-softgroup-75f9936/runs/gdn-transition-softgroupnoise-d224l12-s1000-20260705T0517Z-75f9936`
- Local archive: `runs/gdn-transition-softgroupnoise-d224l12-s1000-20260705T0517Z-75f9936`
- Included locally: `config.json`, `score.json`, `metadata.json`, `source_HEAD.txt`, `source.patch`, `logs/run.log`, `output/*.json`, `output/*.md`, `output/*.html`, `visualizations/index.html`, `visualizations/summary.json`
- Excluded from Git: train checkpoints and `source_snapshot.tar.gz`

## 7. Results

Primary result:

| Metric | loop1 | loop5 | Delta |
|---|---:|---:|---:|
| mixed official exact | 0.0195 | 0.0234 | +0.0039 |
| mixed official blank acc | 0.5003 | 0.5252 | +0.0250 |

Checkpoint evals:

| Step | CE | hidden agg entropy | holes53 loop5 exact/blank | holes60 loop5 exact/blank | holes64 loop5 exact/blank |
|---:|---:|---:|---:|---:|---:|
| 500 | 1.0450 | 0.9775 | 0.0078 / 0.4855 | 0.0137 / 0.4943 | 0.0176 / 0.4901 |
| 1000 | 0.9906 | 0.9797 | 0.0176 / 0.5169 | 0.0215 / 0.5265 | 0.0254 / 0.5196 |

Official blank-range eval at step1000:

| Range | loop1 exact/blank | loop5 exact/blank |
|---|---:|---:|
| 46-50 | 0.5879 / 0.9852 | 1.0000 / 1.0000 |
| 51-55 | 0.0000 / 0.5196 | 0.0000 / 0.5478 |
| 56-64 | 0.0000 / 0.4645 | 0.0000 / 0.4871 |

Noise diagnostics:

- The noise did not collapse: final `hidden_agg_noise_entropy=0.9797`, `hidden_agg_noise_max_weight=0.1998`.
- The cap was always active: final `hidden_agg_noise_norm=2.0`, raw norm `1199.08`, `clip_frac=1.0`.
- Runtime to step1000 was `2908.4s`, peak CUDA allocation about `47.3GB`.

Same-step comparison:

- Clean D224/L12 H14/D16 step1000 baseline from P-DIAG-011 had holes53 loop5 `0.0176/0.5284`.
- This run has holes53 loop5 `0.0176/0.5169`: exact ties, blank accuracy is lower.
- H7/D32 state-geometry step1000 had holes60 loop5 `0.0215/0.5390`; this run has holes60 loop5 `0.0215/0.5265`.

## 8. Conclusions

Decision: discard this as a main direction.

This is a useful boundary, not a useful improvement. P-DIAG-024 fixes the mechanical failure from P-DIAG-023: the aggregate noise is no longer one-hot, it is bounded, and training remains stable. But it does not improve the transition cliff. The model still solves 46-50 blanks and gets exact zero on 51-55 and 56-64 official ranges.

The core lesson is blunt: the user's noise intuition needs to be treated carefully. Smoother stochastic hidden aggregation is trainable, but it is not the missing board-level consistency mechanism at this budget. It also does not beat plain D224/L12 clean scaling at the same step count. Do not sweep `topk/temp/scale/max_norm`; that would be table filling. The stronger evidence remains clean compute/data scaling plus recurrent capacity, especially the D224/L12 6000-12000 step line.

Next high-ROI direction: resume or redesign around the proven D224/L12 long-run line, with a simple generic recurrent state/capacity improvement only if it can be trained at the same or better throughput. Noise should be paused unless it is tied to a clearer scaling mechanism.

## 9. Submission Record

No tag/submission planned unless the primary score is genuinely strong.
