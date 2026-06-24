# official-eqr-bidir-sudoku-probe-20260623

## 1. Metainfo

- Plan ID: `P-EQR-011`
- Status: ready to launch after quick baseline gate
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
- Probe budget defaults: `hidden_size=192`, `num_heads=6`, `global_batch=128`,
  `epochs=128`, checkpoint/eval interval `250`.
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

## 8. Conclusions

Pending.

## 9. Submission Record

Not applicable.
