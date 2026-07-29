# FutureSeed2 Same-Layer Block Memory

## 1. Metainfo

- Plan: `P-FS2-004`
- Machine: AIStation GPU1, one A800 80GB
- Branch: `codex/futureseed2-block-memory-20260729`
- Status: approved, implementation in progress

## 2. Hypothesis

FutureSeed1 passes the terminal recurrent state from one layer into the next
layer. That cheaply exposes future information, but every later reasoning call
reconstructs its recurrent memory. On hard Sudoku, the remaining failures may
come from this reconstruction rather than insufficient seed strength.

FutureSeed2 stores every layer's terminal state after a complete backbone call.
At the next recurrent call, layer `l` starts from layer `l`'s own previous
terminal state. The state stays in the exact same learned coordinate system, so
the next loop can continue computation instead of translating memory between
unrelated layers.

## 3. Configuration

- Frozen control checkpoint: strict official GDN2 + FutureSeed1, step 9000.
- Model: D192/L10/H6/K32/V32, loop5, two low-level cycles, every-loop CE.
- Data: official full-diversity Sudoku; formal ranges 51-55, 56-60, 61-64.
- Intervention: `future_seed_scope=block`; no new parameters, gates, loss,
  noise, task rules, repair, search, or selector.
- First recurrent call remains exactly FutureSeed1. Later calls seed each layer
  from the same layer's previous terminal state.

## 4. Environment

- `CUDA_VISIBLE_DEVICES=0`; GPU2 forbidden.
- Official pinned FLA `GatedDeltaNet2`, chunk backward, Triton convolution.
- All caches, data, checkpoints, logs, and runs under `/huyang2/double-loop`.
- CPU model smoke and backend fallback are forbidden.

## 5. Commands

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/run_futureseed2_block_arm.sh contract
CUDA_VISIBLE_DEVICES=0 ./scripts/run_futureseed2_block_arm.sh smoke
CUDA_VISIBLE_DEVICES=0 ./scripts/run_futureseed2_block_arm.sh formal
```

## 6. Artifacts

Pending.

## 7. Results

Pending.

## 8. Decision

Success requires official 61-64 loop5 exact at least `0.2173` (`+0.02` over
the matched FutureSeed1 continuation), hard-range mean exact at least `0.2418`
(`+0.01`), and 51-55 exact no lower than `0.3572`. Loop visualizations must
show net correction rather than later-loop regression.

Kill after the single 100-step continuation if these gates fail. Do not sweep
blend coefficients, gate strength, seed, learning rate, loss, or run length.

## 9. Publication

No tag unless the mechanism passes the preregistered hard-range and loop
correction gates.
