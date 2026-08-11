# P-BIND-001: Address/Binding Interference Diagnostic

## 1. Metainfo

- Status: approved, implementation in progress
- Date: 2026-08-12
- Benchmark: official 9x9 Sudoku and directional MQAR L1024
- Compute: one AIStation task-mode A800 80GB
- New trainable parameters: zero
- Parent Sudoku checkpoint SHA256:
  `6339c3cb2b5fc5230a581d6633716483e35ff8e4522f06a9d7aaf26512f023da`
- Frozen MQAR endpoint: source `77e5539`, balanced accuracy `0.7475`,
  joint exact `0.339`

## 2. Hypothesis

The recent FS/GDN failures all activated their proposed mechanism, yet did not
close hard-board exactness. They do not distinguish address/binding
interference from insufficient loop convergence. The next architecture should
therefore be selected by a zero-parameter diagnostic of the actual GDN2 state
edit, not by another generic gate or router.

## 3. Method

Wrap the pinned official `chunk_gdn2` function transparently and return its
outputs unchanged. In FP32, replay the exact token transition

`e_t = w_t*v_t - (b_t*k_t)^T [Diag(exp(g_t)) S_(t-1)]`

and record the Frobenius norm of `k_t e_t^T` as surprise. Measure normalized-key
Gram anisotropy, effective rank, condition, top-surprise versus matched-recency
write survival, and top-16 surprise concentration. Correlate Sudoku cell
surprise with next-loop correction and final failure. Reconstruct the frozen
directional MQAR L1024 endpoint exactly, then measure wrong-key valid-value
swaps and surprise/survival versus write horizon.

Sudoku uses 32 deterministic official boards in each range `46-50`, `51-55`,
`56-60`, and `61-64`, five loops, and the final high-state stream. MQAR uses
the full 1000-example endpoint score for identity and the first deterministic
128 validation examples for internal-state diagnostics.

## 4. Integrity Contract

The exact pushed SHA in a clean detached worktree must prove:

1. one visible CUDA index0 with the registered UUID;
2. pinned FLA SHA `9c8e42e...d85e` and `ChunkGDN2FunctionBackward`;
3. wrapped versus unwrapped Sudoku and MQAR logits are bit-exact;
4. model state hashes and parameter counts do not change;
5. FP32 recurrence replay is finite and agrees with the official terminal
   state within declared low-precision tolerance;
6. MQAR initialization, data hashes, balanced accuracy, joint exact, and both
   directional accuracies exactly match the frozen endpoint;
7. no fallback, CPU model execution, optimizer update, data-order change, or
   concurrent GPU model/eval occurs.

Any miss invalidates the diagnostic; it does not select a mechanism branch.

## 5. Pre-Registered Branch Decision

Exactly one branch opens, in priority order:

1. **Receiver-native surprise cache** if hard-Sudoku corrected-cell AUROC is
   at least `0.65`, top-quartile correction lift is at least `1.50x`, and
   surprise-selected writes survive at least `0.10` less than recency.
2. **Bounded native PGDN conditioner** if branch 1 fails, median key effective
   rank is at most `0.50` or median anisotropy is at least `4.0`, and at least
   `75%` of records satisfy either condition.
3. **Clustered delta memory** if branches 1-2 fail, long-horizon surprise-write
   survival is at least `0.20` below short-horizon survival and top-16 writes
   contain at most `40%` of surprise.
4. **Loop dynamics/training signal** if none passes.

No threshold may be changed after reading the diagnostic.

## 6. Allowed Successor

If surprise opens, the first science field is directional MQAR L1024 trained
from scratch at the frozen D128/L2/H4/K32, 10-epoch condition. Use K16
receiver-native canonical evidence rebuilt per layer/loop. Compare only
surprise admission and matched recency. Do not transport producer-basis K/V,
carry whole state across macro loops, or introduce a Sudoku selector.

The fixed quality gate is balanced accuracy at least `0.85` and at least
`+0.10` over `0.7475`, joint exact at least `0.60`, surprise beating recency by
at least `0.05` balanced accuracy or joint exact, wrong-key swap fraction down
at least `0.10`, and end-to-end overhead at most `25%`. Any core miss kills the
mechanism without K, temperature, seed, LR, loss, width, or duration rescue.

If another branch opens, it receives one equally explicit from-scratch MQAR
mechanism registration before code launch. No HYPIC/PIC quality claim, generic
Raven/V-lifetime wrapper, or copied SDM/KATA implementation is allowed.

## 7. Commands

The exact contract and formal commands are supplied by
`scripts/run_gdn2_binding_diagnostic.sh contract|full <out_dir>` from a clean
detached worktree.

## 8. Results

The first contract attempt from source `843b02b` exited before model
construction because the launcher omitted the pinned Zoology checkout from
`PYTHONPATH`. It used no GPU memory and consumed no science gate. The run
contains a `non_science_harness_failure` `abort.json`; the launcher-only fix
adds exact SHA and clean-worktree checks for Zoology before import.

Scientific results pending.

## 9. Decision

Pending the zero-parameter diagnostic.
