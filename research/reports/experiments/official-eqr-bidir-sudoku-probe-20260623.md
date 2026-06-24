# official-eqr-bidir-sudoku-probe-20260623

## 1. Metainfo

- Plan ID: `P-EQR-011`
- Status: completed
- Machine: AIStation `GPU1` only
- Local branch: `codex/gpu1-experiment-tracking`
- Remote work dir: `/huyang2/double-loop/official_eqr_compare`

## 2. Hypothesis

The old official EqR FutureSeed patch was only H/L latent cross-injection, not a
clean cheap-bidirectional source. If FutureSeed's real value is cheap future
context, then a causal cheap Sudoku backbone should benefit from an explicit
generic reverse-causal future-token seed, under matched data, train budget, and
eval path.

## 3. Configuration

- Official EqR upstream SHA: `aba94e9cde0f273ce644db5261cd6915ba6561f0`
- Dataset: official `sudoku-extreme-1k-aug-1000`
- Compare arms:
  - `causal-cheap`: official EqR with `attention_causal=true`
  - `causal-bidir-futureseed`: same causal backbone plus
    `future_seed_mode=reverse_causal`
- Critical guard: force `arch.mlp_t=false`; upstream Sudoku sets
  `arch.mlp_t=true`, which is noncausal and would make `attention_causal=true`
  meaningless.
- Probe budget defaults used: `hidden_size=192`, `num_heads=6`,
  `global_batch=128`, `epochs=64`, checkpoint/eval interval `250`.
- No path-token loss, no solver, no search, no repair, no selector, no
  Sudoku-specific rule.

## 4. Environment

Reuse GPU1 only and existing official EqR environment under
`/huyang2/double-loop/official_eqr_compare`. The quick baseline gate is no
longer a 5-seed table: official Maze paper-point reproduction plus Sudoku
quick seed0/1/2 in-range is sufficient to stop baseline work and test the
mechanism.

## 5. Commands

The launcher is intentionally guarded so it refuses to start unless the official
Sudoku quick baseline has at least two per-seed metric files and no active
`evaluate.py`/`pretrain.py`/`torchrun` process is present. This avoids
low-ROI seed-table completion while still checking that the official eval path
is alive.

```bash
cd /huyang2/double-loop/.worktrees/official-eqr-bidir-probe-current
bash runs/official-eqr-bidir-sudoku-probe-20260623/launch_probe.sh
```

## 6. Artifacts

Planned remote artifact root:
`/huyang2/double-loop/official_eqr_compare/artifacts/bidir_sudoku_probe_<timestamp>`.

## 7. Results

Launched at `2026-06-24T03:37:41Z` from detached source SHA
`45fe80a4b31426b81fae6dda99997871af6c6a4c` on GPU1. Remote launcher:

- Worktree:
  `/huyang2/double-loop/.worktrees/official-eqr-bidir-probe-45fe80a`
- Launch root:
  `/huyang2/double-loop/artifacts/launch/official-eqr-bidir-sudoku-20260624`
- Run stamp: `20260624T033741Z-45fe80a`
- Log root:
  `/huyang2/double-loop/official_eqr_compare/artifacts/bidir_sudoku_probe_20260624T033741Z-45fe80a`
- Launcher PID: `430`

This is a single mechanism probe, not a seed sweep:

- causal-cheap: causal attention, `arch.mlp_t=false`, no future-token source.
- causal-bidir-futureseed: same backbone plus reverse-causal FutureSeed on input
  embeddings.
- Default budget: `epochs=64`, `global_batch_size=128`.

Prediction: if FutureSeed is a cheap way to inject future context into a cheap
causal recurrent backbone, the FutureSeed arm should reach higher early eval
exact or lower eval loss under the same train budget. If both arms are bad or
indistinguishable, the claim weakens and the next move should not be a seed
sweep; it should be either a larger single-budget scale-up or a different
generic FutureSeed state mechanism.

Early health check: causal-cheap training started as PID `581` with official
EqR command `pretrain.py --config-name train/eqr_sudoku`, overrides
`arch.attention_causal=true arch.mlp_t=false arch.hidden_size=192
arch.num_heads=6 arch.halt_max_steps=16 arch.noise_scale=0.01`. GPU1 was active
on `NVIDIA A100-SXM4-80GB`, around `1717 MiB / 81920 MiB`, `94%` utilization
during the first eval pass. The launcher will run causal-cheap first, then
causal-bidir-futureseed under the same budget.

Completed comparison parsed at `2026-06-24T04:13:25Z`.

Remote comparison artifact:
`/huyang2/double-loop/official_eqr_compare/artifacts/bidir_sudoku_probe_20260624T033741Z-45fe80a/comparison.json`.

Local archived comparison:
`runs/official-eqr-bidir-sudoku-probe-20260623/comparison_20260624T033741Z-45fe80a.json`.

| metric | causal-cheap | causal-bidir-futureseed | FS - cheap |
|---|---:|---:|---:|
| `accuracy` | `0.0931727` | `0.0996616` | `+0.0064888` |
| `exact_accuracy` | `0.0` | `0.0` | `0.0` |
| `lm_loss` | `2.5434555` | `2.5613002` | `+0.0178447` |
| `total_loss` | `2.5475551` | `2.5653838` | `+0.0178288` |
| `residual_of_1_steps` | `235.8250` | `221.2877` | `-14.5374` |
| `residual_of_16_steps` | `167.9040` | `82.8528` | `-85.0512` |

Both arms finished training and wrote step448 checkpoints. The probe is not a
score win: exact remains zero for both arms, and the FutureSeed arm has slightly
worse eval loss at this tiny budget. The non-trivial signal is state dynamics:
the reverse-causal FutureSeed source almost halves `residual_of_16_steps`
under the same cheap causal backbone.

## 8. Conclusions

This is a useful mechanism probe but not yet a paper result.

The positive signal is precise: when official EqR's Sudoku mixer is forced into
a cheap causal form (`arch.mlp_t=false`, `attention_causal=true`), adding a
generic reverse-causal FutureSeed source changes the recurrent trajectory
substantially. The 16-step residual drops from `167.9040` to `82.8528`, and
token accuracy rises by `+0.00649`.

The negative signal is equally important: this budget does not open exact
solves, and the FutureSeed arm is not lower-loss. So the claim cannot be
"FutureSeed beats EqR" yet. The only justified next experiment is a single
larger-budget scale gate asking whether the residual contraction converts into
sample efficiency or exact accuracy. Running more seeds at this budget would be
table filling.

Next decision:

- Continue only with one longer matched-compute causal-cheap vs
  causal-bidir-futureseed run if GPU1 is free.
- Kill that direction if exact stays zero and the residual advantage no longer
  grows by the first extended eval.
- If the longer run opens FutureSeed earlier, move the same cheap-bidirectional
  test to a path-aware Maze/Sudoku visualization.

## 9. Submission Record

Not applicable.
