# GDN2 Cell-Order Counterfactual

## 1. Metainfo

- Plan: `P-ADDR-001`
- Status: in progress
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

Pending.

## 7. Results

Pending.

## 8. Conclusion

Pending.

## 9. Submission

Not applicable.
