# FutureSeed2 Same-Layer Block Memory

## 1. Metainfo

- Plan: `P-FS2-004`
- Machine: AIStation GPU1, one A800 80GB
- Branch: `codex/futureseed2-block-memory-20260729`
- Source SHA: `d9228e73fafa1e94ea22c089d72287d829eca7fc`
- Status: completed, hypothesis rejected

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

- Contract: `futureseed2-block-contract-20260729T040124Z-d9228e7`
- Full-stack smoke: `futureseed2-block-smoke2-20260729T040416Z-d9228e7`
- Formal run:
  `/huyang2/double-loop/runs/futureseed2-block-s9100-20260729T040936Z-d9228e7`
- Matched control:
  `/huyang2/double-loop/runs/futureseed2-identity-s9100-20260728T1450Z-57455e4`
- Formal checkpoint:
  `/huyang2/double-loop/models/futureseed2-block-s9100-20260729T040936Z-d9228e7/checkpoints/train_state_step009100.pt`
- Interactive comparison:
  `research/reports/visualizations/futureseed2-block-memory-20260729/index.html`
- Machine-readable comparison:
  `research/reports/visualizations/futureseed2-block-memory-20260729/comparison.json`

## 7. Results

The CUDA contract passed exactly:

- First complete reasoner call versus FutureSeed1: max absolute difference `0`.
- Same-layer state routing versus the explicit oracle: max absolute difference
  `0`.
- Second-call intervention RMS: `0.422361`.
- Initial-state gradient norm: `0.198962`.
- Block-seed gate gradient norm: `0.000921692`.
- Exact class: `fla.layers.gdn2.GatedDeltaNet2`.
- Official FLA source SHA:
  `fe8fce9fc6984f22905f54cfa885dce1502baf26`.
- Backend dispatch was disabled; q/k/v convolutions used Triton.

The formal run was a matched step9000 to step9100 continuation. Both arms have
`5,461,688` parameters. The control/block train times were
`659.925/666.384` seconds. Final train CE was `0.643511/0.959755`.

| Metric | FutureSeed1 control | Same-layer block memory |
|---|---:|---:|
| mixed loop1 exact | 0.0234 | 0.0215 |
| mixed loop2 exact | 0.0469 | 0.0215 |
| mixed loop3 exact | 0.1953 | 0.0195 |
| mixed loop4 exact | 0.2461 | 0.0195 |
| mixed loop5 exact | 0.2520 | 0.0195 |
| official 51-55 loop5 exact | 0.3672 | 0.0000 |
| official 56-60 loop5 exact | 0.1309 | 0.0000 |
| official 61-64 loop5 exact | 0.1973 | 0.0000 |
| official hard-range mean | 0.2318 | 0.0000 |

The first-loop score is nearly unchanged. The failure appears when recurrence
should refine the answer: the control gains `+0.2285` exact from loop1 to
loop5, while block memory loses `-0.0020`.

Matched visual cases show the same mechanism:

- 51-55 blanks, batch 120: control wrong cells
  `17 -> 4 -> 1 -> 1 -> 1`; block memory
  `18 -> 13 -> 13 -> 13 -> 14`.
- 56-60 blanks, batch 86: control
  `14 -> 11 -> 4 -> 1 -> 1`; block memory
  `15 -> 10 -> 11 -> 11 -> 11`.

Two preflight defects were found and fixed before accepting the run. The first
oracle cast expected recurrent states to the original BF16 input instead of
each layer's actual state dtype. The second strict provenance check inspected
the wrapper instead of its nested reasoner. Neither issue was hidden by a
tolerance change or backend fallback; the accepted contract is exact.

## 8. Decision

Reject direct same-layer terminal-state reuse. A terminal recurrent state is
not automatically a compatible FutureSeed for the next macro reasoning step.
FutureSeed1's cross-layer directional initialization and same-layer temporal
carry are not interchangeable.

This is not evidence that more training or a tuned blend would rescue the
idea. The intervention preserves first-loop ability but removes the baseline's
large later-loop gain. Per the preregistered kill rule, do not sweep blend
coefficients, gate strength, decay, seed, learning rate, loss, or run length.

The next FutureSeed2 candidate must transform or summarize future evidence
before reuse, and must be initialized as an exact identity relative to the
strong FutureSeed1 line. It should not carry raw terminal state across macro
steps.

## 9. Publication

No tag. The negative result is retained as a mechanism boundary and prevents a
low-information hyperparameter sweep.
