# GDN2 Cell-Order Counterfactual

## 1. Metainfo

- Plan: `P-ADDR-001`
- Status: done
- Machine: AIStation GPU1, one A800 80GB
- Source base: `e1d44ec5ffb0af0e487398da1b00ff374fcfe872`
- Date: 2026-07-30 CST

## 2. Hypothesis

The accepted GDN2+FutureSeed model already adds a learned absolute position
embedding to every cell token. What it does not guarantee is a canonical
state address: every layer still derives key, query, and value from the mixed
hidden representation. If this binding is the 51-64 blank bottleneck, the
same board should degrade when complete token-position pairs are traversed in
a different order.

## 3. Configuration

- Frozen strict-official-FLA GDN2+FutureSeed step9000 checkpoint.
- Official Sudoku test split, 51-64 blanks, `n=256`.
- Traversals: row-major, reverse, column-major, box-major, one fixed random
  permutation.
- Each content token keeps its original learned absolute position embedding.
- Logits are restored to canonical cell order before scoring.
- Five loops, BF16 forward, no noise, no training, no selector or repair.

## 4. Environment

- `CUDA_VISIBLE_DEVICES=0`; GPU1 only.
- CPU model execution is forbidden.
- Checkpoint SHA256 and strict official FLA runtime are checked before eval.

## 5. Decision Contract

- Mechanism prediction: at least one paired traversal loses more than `0.03`
  full-board exact or `0.02` blank accuracy relative to row-major.
- Kill criterion: if every paired traversal stays inside both tolerances,
  reject canonical address binding as the current bottleneck and do not build
  Address-Payload GDN2.
- Positive next decision: only a clear order-sensitivity result authorizes one
  generic position-addressed GDN2 training probe. It still must beat the same
  row-major baseline and retain official FLA kernels.

## 6. Artifacts

- Remote run:
  `/huyang2/double-loop/runs/gdn2-cell-order-counterfactual-20260730T0845Z-c430c9b`
- Local mirror:
  `runs/gdn2-cell-order-counterfactual-20260730T0845Z-c430c9b`
- Files: `score.json`, `cases.json`, `index.html`, `run.log`.
- Probe source SHA: `c430c9b41df807683335d8f230d1e33ffc412dbf`.

## 7. Results

| traversal | loop5 exact | loop5 blank accuracy |
|---|---:|---:|
| row-major | 0.2617 | 0.6970 |
| reverse | 0.0000 | 0.1339 |
| column-major | 0.0000 | 0.1277 |
| box-major | 0.0000 | 0.1839 |
| fixed-random | 0.0000 | 0.1228 |

- Worst non-row exact delta: `-0.2617`.
- Worst non-row blank-accuracy delta: `-0.5743`.
- Paired token-plus-position encoding roundtrip max error: exactly `0`.
- Runtime: exact official `fla.layers.gdn2.GatedDeltaNet2`, pinned FLA
  `9c8e42e`, Triton Q/K/V short convolutions, backend dispatch disabled.

## 8. Conclusion

The model has absolute position information, but it does not use that
information as an order-independent recurrent address. Traversal order is a
dominant part of the learned computation. This result authorizes one causal
mechanism probe that separates position-addressed Q/K from content-driven V
and gates.

It does **not** yet prove that address entanglement causes the 51-64 blank
cliff. That stronger claim requires `P-ADDR-002`: the new mechanism must beat
a matched random-order normal-GDN2 control without sacrificing row-major
quality.

## 9. Submission

Not applicable.
