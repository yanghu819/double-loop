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

The R2 contract then stopped in the pre-model provenance check because
`inspect.getfile` reported Torch Dynamo's wrapper file for `chunk_gdn2`. Direct
inspection showed `inspect.unwrap(chunk_gdn2)` resolves to the expected pinned
`fla/ops/gdn2/chunk.py`. This is archived as a second non-science checker
failure; R3 validates the unwrapped implementation path while retaining the
wrapper module in provenance.

R3 reached the synthetic official backward contract, then its hand-built test
inputs exposed a BF16/FP32 mismatch between `q/k/v` and `g/b/w`. Triton rejected
the mixed dot during compilation before the formal Sudoku model ran. This is
archived as a third non-science harness failure. R4 makes all six synthetic
kernel inputs BF16, matching production autocast, while the diagnostic replay
remains FP32.

The first full run reached the formal Sudoku forward and then stopped in the
first read-only Gram statistic: production autocast downcast the recorder's
`einsum`, and CUDA `eigvalsh` does not accept BF16. No result or branch decision
was emitted. The incident is archived as a non-science diagnostic-precision
failure. The next source disables autocast for the complete committed-edit
replay and explicitly casts Gram inputs to FP32; model forward autocast remains
unchanged.

The next run completed Sudoku capture and then exposed a historical provenance
gap before MQAR training: the August 4 A100 endpoint did not save a checkpoint,
and exact frozen source `77e5539` on the current A800 runtime deterministically
produces init/model parameter hashes `b03e287...e17d9` / `3e8fe03...e4c44`
instead of the historical `5595ecb...28ae` / `3b0c133...1a368`. Source, pinned
FLA/Zoology, model parameter count, and both data hashes remain exact. Bytewise
endpoint reconstruction is therefore impossible rather than silently assumed.

Before observing any new training metric, R3 registers one runtime-fork
reconstruction: the current hashes above are frozen; balanced accuracy must be
at least `0.70`, joint exact at least `0.25`, and both past/future accuracy at
least `0.68`. A pass freezes the resulting checkpoint as the matched parent for
new MQAR arms. The successor quality bar is unchanged: balanced at least
`0.85` and at least `+0.10` over the historical `0.7475`. A reconstruction miss
invalidates MQAR branch evidence and receives no seed/epoch/LR rescue.

The strict R6 contract passed from exact pushed source `74f928e` on the only
visible A800. It confirmed zero new parameters, bit-exact wrapped/unwrapped
Sudoku logits, unchanged model-state hashes, the pinned official FLA source,
and `ChunkGDN2FunctionBackward`. The formal Sudoku capture then completed
without changing logits or model state.

On the combined official 51-64 blank batches, exact committed-edit surprise
does **not** identify cells corrected by a later loop: AUROC is `0.522970`,
Pearson is `0.015239`, and top-quartile lift is `1.052392`. Surprise-selected
writes survive `0.144955` less than recency-selected writes, but the other two
registered surprise gates miss decisively. A receiver-native surprise cache is
therefore not opened by Sudoku evidence.

The same capture exposes a strong address-geometry symptom. Median key Gram
effective-rank fraction is `0.479188`, median anisotropy is `6.456761`, median
condition number is `402.9106`, and median coherence is `0.924599`. These are
descriptive Sudoku evidence only: the preregistered PGDN branch additionally
requires the affected-record fraction from the validated L1024 MQAR carrier.

The sole preregistered runtime-fork reconstruction completed all ten epochs but
failed every carrier floor. Balanced accuracy is `0.3245`, joint exact is
`0.0010`, and past/future accuracy is `0.3275/0.3215`, versus fixed minimums
`0.70/0.25/0.68/0.68`. Source `77e5539`, Zoology/FLA SHAs, both data hashes,
parameter count, and the newly frozen runtime initialization hashes all match;
the failure is a quality-regime mismatch, not an integrity or harness error.
No checkpoint or MQAR internal diagnostic is accepted.

Artifact SHA256 values are:

- CUDA contract: `fb75afeca0456ee2824d2200caf50f15cd1b29efb31d3361a7ac777eb0eb1d32`
- Sudoku diagnostic: `2571472623f0d2616995e198effe24b2e7541d4486177044e796585032bd37bb`
- MQAR reconstruction score: `a19226f2aea4b17ecc4fa2e2123d1850691a8c1ed1e3acad3e8e6099a5c33404`
- MQAR training metrics: `b085f498717cee4ee4174fb15384aeccef5c139fc14043d384290c0313eb8cc5`
- Scientific abort: `b3c2a6df172184a3f2f42263b69b2a1bdffae096f31f091057b9d53120b42968`

## 9. Decision

Close `P-BIND-001` at its registered MQAR reconstruction gate. The Sudoku
capture is valid and rules out surprise-cache admission on this parent, but it
cannot substitute post hoc for the missing MQAR affected-record statistic.
Consequently none of the four mechanism branches is authorized by this run.

Do not retry seed, epoch count, LR, loss, width, or the same A800 runtime fork.
The only admissible recovery is provenance work: reproduce the historical A100
runtime and exact initialization, or preregister a different independently
validated directional-binding carrier. Until then, the anisotropic Sudoku
geometry is a concrete hypothesis for a bounded native PGDN conditioner, not a
quality claim or launch permission.
