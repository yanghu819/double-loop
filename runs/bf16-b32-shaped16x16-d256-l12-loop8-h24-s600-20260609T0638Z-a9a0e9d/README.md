# BF16 Batch32 Clean 16x16 Scaling Probe

Run: `bf16-b32-shaped16x16-d256-l12-loop8-h24-s600-20260609T0638Z-a9a0e9d`

Source SHA: `a9a0e9d6ca7464a8700931af7035cf758f001cc5`

GPU: GPU1 only, NVIDIA A800-SXM4-80GB.

## Question

Can a purely generic scaling-efficiency change move clean FutureSeed/RWKV 16x16 upward?

The previous clean D256/L12/loop8 fp32 path could run batch16 at about 43GB, but batch32/batch48 OOMed near the 80GB limit. This run asks whether CUDA bf16 forward autocast can make batch32 fit and whether the larger batch plus shaped loop supervision improves local digit reliability.

## Configuration

- Board: 16x16 Sudoku, random holes, 4x4 boxes.
- Model: D256, 12 layers, 8 heads, head dim 32, channel mult 4.
- Loop: max loops 8, L-cycles 2.
- Curriculum: `8-16:300,16-24:300`.
- Batch/eval: batch 32, eval N 256.
- Forward dtype: `bfloat16`.
- FutureSeed: `FUTURE_SEED_SCALE=1`, `FUTURE_SEED_DECAY=0.5`, fixed update.
- Loop feedback: off.
- Loop supervision: shaped, biased toward later loops.
- Kernel: RWKV CUDA state-passing path.

## Result

| readout | value |
|---|---:|
| peak observed memory | about 67.8GB |
| train time | 969.6s |
| step100 CE | 1.2038 |
| step300 CE | 0.9221 |
| step600 CE | 0.9751 |
| h16 loop8 blank_acc | 0.4043 |
| h24 loop1 blank_acc | 0.3678 |
| h24 loop8 blank_acc | 0.3711 |
| h24 loop8 exact | 0.0000 |
| h32 loop8 blank_acc | 0.3571 |
| loop8 - loop1 blank_acc on h24 | +0.0033 |

## Decision

Keep bf16 forward as a scaling default for GPU1 clean experiments. It makes batch32 feasible and improves local blank accuracy versus the previous clean b16 runs:

- h24 blank_acc improves from about `0.335-0.338` to `0.3711`.
- h32 blank_acc improves from about `0.311-0.321` to `0.3571`.
- exact remains `0.0000`, so this is not yet a viable 16x16 solver.

Do not treat this as evidence that loop feedback/gate sweeps are useful. Loop gain remains tiny. The gain came from scaling efficiency and larger batch, not from deeper-loop refinement.

## Insight

The clean line is not dead, but it is far below the exact-board regime. BF16 is a real engineering unlock because it changes what fits on 80GB. The next high-ROI clean experiment should spend this memory headroom on a stronger generic model/credit mechanism, not on feedback-scale sweeps.

Two concrete directions are worth considering next:

1. Activation checkpointing to fit D320/D384 or longer loops under bf16.
2. A stronger generic intermediate-state objective that improves per-cell reliability without injecting Sudoku unit priors.

## Artifacts

- `config.json`
- `score.json`
- `metadata.json`
- `logs/run.log`
- `launcher.log`
- `output/futureseed_loop_seed52.json`
- `output/futureseed_loop_seed52.md`
- `output/futureseed_loop_case_seed52.html`
- `source_HEAD.txt`
- `source.patch`
