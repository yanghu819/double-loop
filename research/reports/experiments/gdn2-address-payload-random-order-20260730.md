# GDN2 Address-Payload Random-Order Probe

## 1. Metainfo

- Plan: `P-ADDR-002`
- Status: completed; positive address-learning result, hard exact still open
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

```bash
GPU1_UUID=<current-gpu1-uuid> CUDA_VISIBLE_DEVICES=0 \
  scripts/run_gdn2_address_arm.sh control formal

GPU1_UUID=<current-gpu1-uuid> CUDA_VISIBLE_DEVICES=0 \
  scripts/run_gdn2_address_arm.sh position_qk formal
```

Only after the candidate passed the 100-step gate, it was resumed from its
exact step9100 checkpoint toward step9300 using
`P_ADDR_TARGET_STEP=9300` plus the checkpoint path, SHA256, and source SHA.

## 7. Artifacts

- Control:
  `runs/gdn2-address-control-s9100-20260731T034101Z-4b94220`
- Position-Q/K:
  `runs/gdn2-address-position_qk-s9100-20260731T035618Z-4b94220`
- Candidate continuation:
  `runs/gdn2-address-position_qk-s9300-20260731T041227Z-eec018c`
- Equal-compute visualization:
  `runs/gdn2-address-comparison-s9100-20260731/index.html`
- Final candidate-continuation visualization:
  `runs/gdn2-address-comparison-s9300-20260731/index.html`

## 8. Results And Conclusion

The 100-step gate is a strong positive mechanism result:

| Official range | control blank acc | position-Q/K blank acc | delta | exact |
|---|---:|---:|---:|---:|
| 51-55 | 0.2120 | 0.4704 | +0.2585 | 0 / 0 |
| 56-60 | 0.2119 | 0.4293 | +0.2173 | 0 / 0 |
| 61-64 | 0.1871 | 0.4001 | +0.2130 | 0 / 0 |

Both arms use the same official boards, parent checkpoint, continuation
batches, optimizer budget, FutureSeed, loop supervision, parameters, and
official FLA GDN2 kernel. Their case-bank data hashes match exactly. The only
mechanism difference is whether Q/K come from the mixed content stream or the
canonical position stream.

The candidate also optimizes faster: final train CE is `1.2196` versus
`1.9370`. On mechanically selected matched 64-blank batch 226, normal GDN2
ends with 52 wrong blank cells. Position-Q/K moves from 38 wrong cells at
loop1 to 28 at loop2 and remains there through loop5. This is real loop
correction followed by a plateau, not solved Sudoku.

Conclusion at step9100: separating memory address from payload removes a large
optimization bottleneck under shuffled traversal. It does not yet solve global
consistency, because exact remains zero and later loops stop improving after
the early correction. The preregistered blank-accuracy gate passes, so only
the positive candidate is extended to 300 continuation steps; no address,
loss, seed, or control-length sweep is justified.

The candidate-only continuation to step9300 strengthens the mechanism result
without closing the task:

| Official range | step9100 blank acc | step9300 blank acc | loop1 -> loop5 at step9300 | step9300 exact |
|---|---:|---:|---:|---:|
| 51-55 | 0.4704 | 0.5660 | 0.5097 -> 0.5660 | 0 |
| 56-60 | 0.4293 | 0.5057 | 0.4674 -> 0.5057 | 0 |
| 61-64 | 0.4001 | 0.5594 | 0.4542 -> 0.5594 | 0 |

Mean official hard-range blank accuracy rises from `0.4333` at step9100 to
`0.5437` at step9300. Train CE falls to `0.9661`. The mixed 512-board
evaluation reaches exact `0.00195 -> 0.02344` from loop1 to loop5, but every
separately sampled official 51-64 range remains at zero exact.

The matched-data visualization makes the recurrent effect concrete. On the
mechanically selected 64-blank batch 110, the stopped control changes
`48 -> 48 -> 49` wrong blanks at loops 1/2/5. The step9300 position-Q/K
candidate changes `37 -> 23 -> 15`. Thus the later loops are doing useful
work after enough candidate training; the remaining failure is not merely
"loops copy loop1." They still leave too many mutually inconsistent cells for
full-board closure.

Final decision: retain address/payload separation as a promising generic GDN2
design direction. Do not claim solved Sudoku and do not use the step9300 versus
step9100 control delta as a same-compute quality estimate. The causal
same-compute claim comes only from the step9100 comparison; the later run
establishes a positive scaling slope and real loop correction. The next
high-information experiment should preserve this address split and scale the
clean backbone/data budget, then compare against a control trained for the
same total compute. It should not be an address-mode, loss-weight, or seed
sweep.

## 9. Submission

Not applicable.
