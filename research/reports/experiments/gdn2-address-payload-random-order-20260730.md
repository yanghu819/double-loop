# GDN2 Address-Payload Random-Order Probe

## 1. Metainfo

- Plan: `P-ADDR-002`
- Status: in progress
- Machine: AIStation GPU1, one A800 80GB
- Source branch: `codex/gdn2-address-binding-20260730`
- Date: 2026-07-30 CST

## 2. Hypothesis

The frozen-model counterfactual shows that learned absolute position metadata
is present but does not produce an order-independent recurrent address.
Ordinary GDN2 derives Q, K, V, and all state-edit gates from the same hidden
stream. The hypothesis is that this entangles where a memory is read or
written with what content is carried. A parameter-neutral split should make
the binding easier to learn:

- canonical learned position embeddings produce Q and K;
- token hidden states produce V, decay, erase, write, and output gates;
- the unchanged official GDN2 recurrence performs the state update.

## 3. Configuration

- Parent checkpoint: strict-official-FLA GDN2+native-FutureSeed step9000.
- Parent checkpoint SHA256:
  `606caf5229590f157d7a0f952423c719f4c17688003309a7bd0664e579588dd7`.
- Matched arms: normal GDN2 control and `position_qk` address-payload GDN2.
- Both arms continue the same optimizer/model state and use the same
  deterministic random cell permutation per microbatch.
- Labels and returned logits remain in canonical board order.
- First gate: 100 optimizer steps; maximum continuation: 300 steps.
- One seed only. No loss, temperature, bias, or address-size sweep.
- No row, column, box, Sudoku constraint, repair, search, selector, or oracle.

## 4. Environment

- `CUDA_VISIBLE_DEVICES=0`; GPU1 only.
- CPU model execution is forbidden.
- Exact official FLA class and pinned source SHA are mandatory.
- Q/K/V short convolutions must report Triton backends.
- Recurrent training remains `fla.ops.gdn2.chunk_gdn2`.

## 5. Decision Contract

- Prediction: the address-payload arm improves average non-row blank accuracy
  by at least `+0.10`, or non-row exact by at least `+0.03`, relative to the
  matched random-order normal-GDN2 control.
- Quality guard: row-major exact may not trail control by more than `0.03`.
- Step100 kill: stop an arm if CE does not decrease and fixed-random blank
  accuracy is below `0.30`.
- Positive decision: continue one address-payload scale run and test whether
  order robustness also shifts the 51-64 blank cliff.
- Negative decision: archive this as evidence that traversal dependence is a
  symptom rather than the causal capacity bottleneck; do not sweep address
  variants.

## 6. Commands

Pending exact clean-SHA launch commands.

## 7. Artifacts

Pending.

## 8. Results And Conclusion

Pending.

## 9. Submission

Not applicable.
