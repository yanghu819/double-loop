# P-DIAG-MOMFS-001: Momentum FutureSeed Component Attribution

## 1. Question

Are P-GDN3-059's 223 residual retrieval errors caused by transporting the
wrong second-order component across layers, or do they remain inside the live
token recurrence even when FutureSeed is removed?

## 2. Evidence And Hypothesis

P-GDN3-059 raises L1024 balanced accuracy from `.494` to `.94425`, but 151 of
its 223 remaining errors are valid values assigned to a wrong key. A frozen
case audit sharpens the pattern: all `151/151` swaps select the adjacent write
rank, and `150/151` stay within the same future/past direction. The value and
coarse direction are therefore present; only local owner identity remains.

The momentum carrier transports a stacked `[S,M]` terminal state through one
native FutureSeed edge. `S` and `M` are normalized independently but share the
same learned head gate. Before designing another FS transform or another live
recurrence, the same trained weights can attribute which boundary contains the
tail.

## 3. Fixed Counterfactuals

Load the exact P-GDN3-059 checkpoint and exact frozen L1024 test bank. Evaluate
four deterministic modes without training or parameter changes:

1. native normalized and gated `[S,M]` transport;
2. `S` only, zeroing transported `M` after native normalization;
3. `M` only, zeroing transported `S` after native normalization; and
4. no FutureSeed, zeroing both transported components.

The native replay must reproduce every formal prediction exactly. All modes
retain the same model, recurrence, data order and kernel; only the one
inter-layer seed tensor is masked.

## 4. Registered Decision

A component ablation opens one component-aware FutureSeed successor only if it
beats native `[S,M]` by at least `.005` balanced accuracy, removes at least 20
wrong-key swaps, adds no total errors, loses at most `.005` in either
direction, and loses at most `.01` joint exact. If no mode meets every
condition, the residual tail is assigned to the intra-layer recurrence and
the next experiment must use a distinct owner-preserving live transition.

This diagnostic cannot authorize selecting an ablated checkpoint, changing a
gate, tuning momentum, or transferring to Sudoku. It only chooses the FS or
GDN research branch.

## 5. Result

Pending exact pushed-source GPU diagnostic.

The first launch attempt stopped before run-directory creation or model load
because the read-only external checkout contained untracked Python bytecode
created by P059. The corrected launcher still rejects every tracked source
change, while source hashes remain enforced by the model loader, and disables
future bytecode writes. This is an orchestration-only pre-model event, not a
science result.

R1 then wrote a non-science `abort.json` before model construction because the
shell-verified GPU name and UUID were not exported to the Python checker. R2
exports those same fixed values; mechanism, cases and decision thresholds are
unchanged.
