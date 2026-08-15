# P-GDN3-052: Committed-Edit Interference Credit

## 1. Metainfo

- Status: registered; not yet launched
- Decision field: validated directional MQAR L1024 wrong-key binding regime
- Parent: frozen deterministic P-REPRO-001 initialization and replay-B score
- Candidate: D128/L2/H4/K32/V32 pinned-official GDN2, native FutureSeed,
  10 epochs, batch32, seed123
- Control: frozen P-REPRO-001 replay B; do not rerun it
- Sweep: none

## 2. Mechanism Hypothesis

Frozen replay localizes `76.38%` of native errors to valid values assigned to
the wrong key. Of those swaps, `99.55%` stay inside the correct future/past
direction, `99.61%` choose the adjacent owner by write rank, and `74.58%` are
within 128 query tokens of that owner. The memory usually recovers the right
two-value set but loses instance ownership between nearby writes.

P-GDN3-051 rejects a fixed one-token address/payload lag. P-GDN3-045 rejects
global key-spectrum whitening: it penalized all addresses equally and
collapsed retrieval. The remaining falsifiable hypothesis is narrower. Only
writes that make a large *actual committed edit* need extra pressure against
recent addresses, because their off-target energy scales as
`||e_t||^2 (k_i^T k_t)^2`. Credit should therefore follow the exact official
edit rather than globally reshaping every key.

## 3. Exact Intervention

The native forward and inference graph remain unchanged. During training only,
for each layer, recompute native post-Triton-convolution K from a detached copy
of the layer input. Capture the exact committed residual already produced by
the pinned official chunk forward:

```text
e_t = w_t * v_t - (b_t * k_t)^T [Diag(exp(g_t)) S_(t-1)]
s_t = RMS(e_t) / causal_prefix_mean(RMS(e_[0:t]))
G_t = sum_(max(0,t-128) <= i < t) s_i k_i k_i^T
M_t = sum_(max(0,t-128) <= i < t) s_i
I_t = K * (k_t^T G_t k_t / M_t)
L_credit = mean s_t * ReLU(I_t - 1)
L = L_CE + 0.03 * mean_layers(L_credit)
```

K is L2-normalized exactly as required by the official recurrence. The random
isotropic overlap floor is one after multiplying by K, so only the collision
tail above that floor receives credit. Both `e_t` and the layer input are
detached. Auxiliary gradients can reach only native K projection and K short
convolution parameters. There are zero new parameters, recurrent values and
inference scans; training adds one K projection/short-conv recomputation per
layer and one bounded 128-token prefix-Gram calculation.

## 4. Distinction And Prediction

This is not P045 global whitening: P052 is causal, local, committed-edit
weighted, and hinge-truncated at the random floor. It is not surprise cache or
P-FS2-007: no event is selected, copied or replayed, and inference stores no
extra evidence. It is not P051 lag, decoupled K, Raven control, a side bank or
a task-aware loss. Native Q/K/V/g/b/w, recurrent state, FutureSeed transfer and
logits are unchanged.

If nearby high-energy writes cause ownership swaps, the credit must reduce
both total errors and the *count* of valid-value wrong-key swaps while retaining
past retrieval. If conditional swaps fall only because arbitrary errors rise,
or if the committed-edit weights are not variable, the mechanism is falsified.

## 5. Fixed Data And Provenance Contract

- train/test examples: `10,000/1,000`;
- sequence length/pairs: `1,024/4`, mixed future and past directions;
- epochs/batch/seed: `10/32/123`;
- causal window/credit coefficient: `128/.03`;
- matched initialization SHA256:
  `7402e46c48cbd47070d65262d1a1b62a32ac55a4d644a6fb9644b96b0914850f`;
- frozen control score SHA256:
  `df8ae1c212590666440f12029d60c74a6d09cbc059b23e58bcf864c00ae98854`.

Formal source must be an exact pushed/read-back SHA in a clean detached
worktree. Only CUDA index0 on the registered A800-SXM4-80GB, UUID
`GPU-ad227769-8993-9351-affb-604e46d6e196`, may be visible.
Require pinned FLA SHA `9c8e42e762fce087c27b673af4922795d9edb85e`,
clean Zoology SHA `1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb`, Triton
short convolution, two official GDN2 layers and no fallback.

## 6. Strict CUDA Contract

Before science, prove all of:

1. candidate tensors exactly equal the frozen initialization, with exactly
   661,584 parameters and 4,096 recurrent values/layer;
2. eval logits, train logits and nonzero-incoming outputs/states are bit exact
   to native GDN2;
3. the combined objective retains exactly two
   `ChunkGDN2FunctionBackward` nodes, while auxiliary-only backward contains
   zero official recurrence nodes;
4. auxiliary gradients are finite and nonzero in both K projections and K
   convolutions and nowhere else;
5. both exact committed edits are finite and variable, collision tails exceed
   the random floor, and the credit is finite and nonzero;
6. simultaneous head permutation and global K/edit rescaling preserve the
   objective, while repeated temporal addresses strictly increase it;
7. parameter/state/inference-scan deltas remain zero, training-conv delta is
   exactly two, and no CPU/eager/alternate model path exists.

Any contract miss writes an integrity abort and closes this implementation.

## 7. Activation, Quality And Cost Gates

Endpoint activation is conjunctive:

- exactly two active layers with fixed window128 and coefficient `.03`;
- finite unweighted credit in `[1e-5,10]`, weighted credit `>=1e-6`;
- committed-edit RMS `>=1e-4`, surprise coefficient of variation `>=.05`,
  and nonzero board variation in both layers;
- at least 5% of local interference lies above the random floor, valid fraction
  is `>.99`, and local mass is positive in both layers;
- native FutureSeed has exactly one active receiving route.

All quality checks are conjunctive against frozen replay B:

- balanced accuracy `>=.60` and gain `>=+.10`;
- future accuracy `>=.58` and gain `>=+.10`;
- past accuracy regresses by no more than `.03`;
- joint exact `>=.10`;
- total query errors fall by at least 15%;
- wrong-key valid-value swap count falls by at least 20%.

Candidate/control elapsed, post-warm wall and independently warmed-step ratios
must each be `<1.35x`; peak allocation must be `<1.25x`. These thresholds are
fixed before launch and reflect one training-only O(T*K^2) prefix statistic.

## 8. Kill And Next Decision

Any integrity, activation, quality or cost miss closes exact committed-edit
interference credit. Do not rescue coefficient, window, hinge/floor,
normalization, detach boundary, token/layer scope, seed, LR, loss, batch, width,
depth, data or duration. A complete pass authorizes one hard-Sudoku transfer
with the same objective. A miss means local collision-aware training credit is
insufficient; the next experiment must change learned ownership in the live
transition rather than add another geometry loss or cache.

## 9. Submission Record

- Formal source/run: pending pushed SHA and strict CUDA contract.
- Endpoint: pending.
- Decision: pending.
