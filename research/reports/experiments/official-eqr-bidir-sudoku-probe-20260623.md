# official-eqr-bidir-sudoku-probe-20260623

## 1. Metainfo

- Plan ID: `P-EQR-011`
- Status: planned, blocked on `P-EQR-010` quick5 baseline completion
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

Pending launch. Must reuse GPU1 only and existing official EqR environment under
`/huyang2/double-loop/official_eqr_compare`.

## 5. Commands

The launcher is intentionally guarded so it refuses to start until the official
Sudoku quick5 baseline has written `summary.json` and no active
`evaluate.py`/`pretrain.py`/`torchrun` process is present.

```bash
cd /huyang2/double-loop/.worktrees/official-eqr-bidir-probe-current
bash runs/official-eqr-bidir-sudoku-probe-20260623/launch_probe.sh
```

## 6. Artifacts

Planned remote artifact root:
`/huyang2/double-loop/official_eqr_compare/artifacts/bidir_sudoku_probe_<timestamp>`.

## 7. Results

Pending. Do not interpret this experiment until the strong official EqR baseline
gate has completed and been archived.

## 8. Conclusions

Pending.

## 9. Submission Record

Not applicable.
