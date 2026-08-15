# P-GDN3-058: Bidirectional Cycle-Consistency Memory

## 1. Metainfo

- Status: preregistered
- Date: 2026-08-16
- Branch: `codex/gdn3-cycle-consistency-memory-20260816`
- Decision field: directional MQAR L1024 wrong-key binding regime
- Frozen control: deterministic P-REPRO-001 replay B
- Fixed model/data: D128/L2/H4/K32/V32, four associations, 10,000 train
  and 1,000 validation examples, 10 epochs, batch32, seed123
- GPU: one AIStation A100-SXM4-80GB, CUDA index 0 only

## 2. Evidence And Hypothesis

The frozen native GDN2 plus FutureSeed control reaches
balanced/future/past/joint `.494/.454/.534/.041`. It makes 2,024 errors, of
which 1,546 are valid values assigned to the wrong key. `99.61%` of those
swaps choose the adjacent write owner. Deleting the competing layer-0 write
repairs `98.25%` of swaps but destroys the competitor's own answer. Values are
present; instance ownership merges in the recurrent state.

P057 shows that changing the hidden representation before the native scan
breaks the co-adapted Q/K/V/gate system. The new hypothesis is narrower: leave
that learned system exact, but test whether a candidate value is consistent
with the key that requested it. A value-to-key memory supplies an answer-free
cycle check: a correct value should reconstruct the query owner, while a value
belonging to the adjacent write should reconstruct the competing owner.

## 3. Mechanism

Each layer performs three pinned-official GDN2 scans:

```text
y, S_main = GDN2(q, k, v, g, b, w; S_main_in)
r, S_rev  = GDN2(y, v, normalize(k), mean(g), w, b; S_rev_in)
d_owner   = normalize(q) - normalize(r)
y_refined = GDN2(q + tanh(a_head) * d_owner,
                 k, v, g, b, w; S_main_in)
```

The main terminal state is from the native first scan, so its write/update is
unchanged. The third scan only rereads that same trajectory. `a_head` starts at
exactly zero. Its native storage inputs are stop-gradient; the correction can
train the parent only through the query-owner residual. This preserves the
complete native output/state gradient at zero while allowing the opened cycle
path and reverse incoming state to receive end-to-end credit. Both main and
reverse terminal states receive the same native cross-layer FutureSeed
normalization and receiving-layer gate. The candidate adds exactly eight
parameters and one 4x32x32 reverse state per layer. It adds no task rule, token
selector, search, label signal, noncausal path or fallback.

## 4. Why Existing Failures Do Not Cover It

- Direct key splitting, whitening, gauges and RLS replace or distort the
  co-adapted native address geometry. P058 preserves it exactly.
- Companion memories P047/P049/P056 produce an additive answer. P058's second
  state reconstructs an owner key and can only alter the native query.
- P006/P053 use parallel value memories; P058 is a dual relation, value to
  key, with explicit cycle consistency.
- P041/P007 read state into a learned controller. P058 uses the algebraic
  mismatch between the original query and the reconstructed owner.
- P057 forms a local event before the scan. P058 leaves the parent input and
  live main commit untouched.

## 5. Strict CUDA Contract

Exact pushed source in a clean detached worktree must prove:

1. one visible registered A100 at CUDA index 0, pinned FLA SHA
   `9c8e42e762fce087c27b673af4922795d9edb85e`, and no fallback;
2. exactly six `ChunkGDN2FunctionBackward` paths across two layers and Triton
   short convolution provenance;
3. exactly eight new parameters, 4,096 reverse state values per layer, three
   official scans per layer, and no change to the main state size;
4. zero gates preserve full logits, arbitrary nonzero incoming-state output,
   main terminal state and every parent gradient exactly;
5. all eight zero gates receive finite nonzero gradients, and opened gates
   materially affect logits and depend on the reverse incoming state;
6. exact within-layer token-scan causality conditioned on fixed incoming main
   and reverse FutureSeed states, plus head-permutation equivariance; and
7. frozen data, initialization, source and artifact hashes remain exact.

## 6. Fixed Science Gate

After contract success, run one candidate-only 10-epoch/batch32/seed123 arm
from the frozen initialization. Do not repeat the control.

Activation requires all eight gates at absolute value `>=1e-3`; each layer's
cycle-query relative RMS must lie in `[.01,.50]`; board/token variation and
refined-output change must be material; reverse output/state must be finite,
noncollapsed and bounded relative to main state; native main and reverse
FutureSeed routes must activate.

Quality requires balanced accuracy `>=.65` and control gain `>=.10`, future
and past accuracy each `>=.62`, joint exact `>=.15` and gain `>=.10`, total
errors at least `20%` lower, and wrong-key swap fraction at least `.10` lower.

Cost requires elapsed, post-warm wall and independently warmed step each
`<3.50x` control and peak allocation `<2.00x`. These ceilings are frozen for
three official scans and two recurrent states per layer.

## 7. Falsifiable Prediction And Kill Rule

If ownership verification is the missing operation, future and past accuracy
must improve together, wrong-key swaps must fall without breaking
native-correct cases, and joint exact must open. Any integrity, activation,
quality or cost miss closes bidirectional cycle memory, including gate,
reverse decay/state, number of rereads, layer, seed, LR, loss, batch, epoch and
Sudoku rescue.

## 8. Results

The first contract attempt stopped before training because the checker required
an arbitrary random reverse initial state to remain visible in the terminal
state after 256 random tokens. That retention condition was not preregistered
and conflicts with the intended decayed recurrence. The attempt is preserved
as a non-science abort. The checker now retains the measured terminal
dependency but gates the intended property directly: with the cycle path open,
model output must have finite nonzero gradient to the reverse initial state.
No mechanism, model tensor, dataset, optimizer, training budget or science gate
changed. The registered endpoint remains pending.

The second contract attempt then showed a BF16 parent-gradient difference of
`9.765625e-4`: output and terminal-state credit had been split across two
otherwise identical official main scans. The exact gradient gate was kept. The
third reread now stop-graduates native storage inputs and adds only its
cycle-query correction to the original main output. At zero gate the first
scan again receives the exact native joint output/state gradient; once open,
the owner residual and reverse state remain differentiable. Scan/state/parameter
counts and every registered science or cost gate are unchanged. This is a
pre-science implementation correction, not a quality rescue.

The third contract attempt reached the causality check, where the checker
perturbed suffix tokens and demanded identical full-model prefix logits. That
condition is invalid for both candidate and native FutureSeed: layer 1 is
intentionally initialized by layer 0's terminal state, which summarizes the
whole sequence. The corrected contract holds both incoming FutureSeed states
fixed and perturbs only the suffix of one layer's hidden sequence; its three
official recurrent scans must then preserve the prefix exactly. This tests for
an illicit noncausal token scan without rejecting the registered cross-layer
FutureSeed mechanism. No model or endpoint setting changed.

## 9. Decision

Pending the single registered endpoint.
