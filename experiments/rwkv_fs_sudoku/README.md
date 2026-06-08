# RWKV FutureSeed Loop Mainline

This directory keeps one active mechanism:

```text
RWKV recurrent backbone
+ FutureSeed cross-layer recurrent state initialization
+ EqR-style depth loop
```

Sudoku is only a structured probe. The mechanism must still make sense for a
DiffusionLM, language model, or multimodal token model.
For the current Mac-only phase, active experiments stay on Sudoku because it
gives the fastest full-board signal under local compute limits.

## Versioning

There is one active checkpoint and one active run artifact:

- active checkpoint: `checkpoint/clean-feature-diff-main-2026-06-01`
- active runs:
  - `runs/final_feature_diff_main_6x6`

## Five-Step Discipline

1. Make the requirement less wrong: the goal is scalable latent reasoning, not
   a Sudoku solver.
2. Delete: remove mechanisms that did not survive evidence.
3. Simplify: keep one mechanism and only decision-relevant gates.
4. Accelerate: use small 6x6 gates before any 9x9 or language-scale work.
5. Automate: keep smoke/static checks and GitHub checkpoints clean.

## Kept

| Mechanism | Standard term | Why it stays |
| --- | --- | --- |
| RWKV recurrent scan | recurrent sequence model | general backbone for token streams |
| FutureSeed | cross-layer recurrent state initialization | global state propagation without copying input views |
| Depth loop | iterative refinement / inference-time compute | tests whether more computation improves latent state |
| Feature-difference stochastic rollout | activation-space latent transition | optional diagnostic, disabled by default |
| Rollout oracle diagnostics | best-of-N sampling diagnostic | measures whether optional stochastic proposals create useful candidates |

## Deleted From Mainline

The clean mainline intentionally excludes Sudoku-specific shortcuts:

- no row/column/box state pooling
- no persistent unit memory
- no unit-consistency loss
- no row/column/box/unit hole-pattern evaluation

Rows, columns, and boxes remain only in the data generator, valid-board metric,
and visualization because those are the proxy task definition rather than model
assistance.

## Run

Smoke:

```bash
uv run python study_rwkv_futureseed_loop.py \
  --size 6 --steps 2 --batch 8 --eval_n 8 --max_loops 3 \
  --d_model 32 --layers 4 --heads 4 --head_dim 8 --channel_mult 2 \
  --l_cycles 1 \
  --holes_min 2 --holes_max 4 --eval_holes 2 \
  --eval_holes_list 2,4,6 --hole_pattern random \
  --blank_loss_weight 20 \
  --out_dir runs/rwkv_fs_sudoku_smoke
```

Best-known feature-diff run:

```bash
uv run python study_rwkv_futureseed_loop.py \
  --size 6 --steps 300 --batch 32 --eval_n 256 --max_loops 3 \
  --d_model 32 --layers 4 --heads 4 --head_dim 8 --channel_mult 2 \
  --l_cycles 1 \
  --holes_min 2 --holes_max 4 --eval_holes 2 \
  --hole_pattern random \
  --blank_loss_weight 20 \
  --noise_scale 0.0 \
  --rollout_ks 1,4,8,16 \
  --rollout_noise_scale 0.0 --log_every 50 \
  --out_dir runs/final_feature_diff_main_6x6
```

## Current Evidence

Best feature-difference rollout result:

| run | K | oracle exact | confidence | consistency | residual | majority | token disagreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `runs/final_feature_diff_main_6x6` | 16 | 0.5039 | 0.3477 | 0.3477 | 0.3438 | 0.3477 | 0.1171 |

Interpretation:

- Without FutureSeed, flattened RWKV is causal and cannot represent the intended
  noncausal Sudoku mechanism.
- Feature-difference noise is the maintained stochastic basis.
- FutureSeed plus feature-diff can create useful parallel candidates: K16 oracle
  rises above K1. The current bottleneck is not candidate generation; it is
  selecting or learning to prefer the good candidate.

## Next Experiments

Do not add another Sudoku-specific mechanism. The next high-ROI path must build
on the simple maintained mainline:

| Experiment | Question | Decision rule |
| --- | --- | --- |
| self-evaluation from latent state | Can the same model estimate which sampled latent trajectory is better? | Continue only if selector closes the oracle gap without a separate Sudoku-specific evaluator |
| FutureSeed schedule | Can small FutureSeed scale help after selection works? | Continue only if K16 oracle improves from loop1 to loop3 |

Non-goals:

- no Sudoku rule assistance
- no more shallow selector heads
- no multi-seed sweeps
- no 9x9 until 6x6 shows loop-depth candidate improvement
