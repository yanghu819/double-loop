# P-GDN3-035: Chunk-Local Bi-Axis GDN2

## 1. Metainfo

- Status: discarded; completed_rejected
- Date: 2026-08-14
- Benchmark: directional MQAR L1024, four associations
- Compute: A100-SXM4-40GB index0, UUID
  `GPU-31166d8c-9fe5-d953-dc44-d0d549969ada`
- Seed: 123 only
- Model: D128/L2/H4/K32/V32 pinned-official GDN2 plus native FutureSeed
- Control: contemporaneous native GDN2+FutureSeed in the same process

## 2. Hypothesis

GDN2 persistently controls lifetime only along K rows. Its V gate changes the
current write but cannot later forget one stored value subspace while retaining
another under the same address row. A nonpositive right-axis decay therefore
adds a separable K-by-V lifetime field that neither an output gate nor larger V
width can reproduce.

P-GDN3-015 tested the algebra through one sequence-global moving frame, but one
step drove its inverse scale to 3505. P-GDN3-016 bounded that frame by subtracting
a common contraction from K decay; the resulting extra forgetting destroyed all
quality. P-GDN3-035 changes neither group count nor task settings. It restores
the physical state every 64 tokens, so V decay is direct and nonexpansive without
the destructive common K contraction.

A fresh GPU VJP audit also corrects P-GDN3-009's boundary claim. A loss depending
only on chunk2 produced nonzero gradients in chunk1 V, K decay and incoming state
(`4.22e-4`, `4.57e-3`, `1.26e-4`). Counting only the terminal grad-fn type had
missed the dependency chain. Chained official chunk calls are therefore a valid
correctness-first implementation.

## 3. Configuration

The physical recurrence is

`S_t = A_t D^K_t S_(t-1) D^V_t + k_t (w_t * v_t)^T`.

Each layer predicts eight V-group logits with one zero-initialized bias-free
`D128 -> H4*G8` projection. The exact-zero, one-sided map is

`gV = clamp(log(2) - softplus(raw), -log(4)/64, 0)`.

Within each 64-token anchor, `C_t=exp(cumsum(gV))`, writes become `v/C_t`, the
unchanged pinned official `chunk_gdn2` runs once, and outputs/final state are
multiplied by `C_t`. The restored physical state initializes the next anchor.
Consequently every local frame and inverse are bounded by four, while physical
V decay can accumulate across the full sequence. The candidate adds exactly
8,192 parameters over two layers, no state values, no extra tokens, no reverse
scan, no cache, and no task logic.

Both arms use 10,000 train and 1,000 validation examples, L1024, four mixed
past/future associations, ten epochs, batch32, AdamW LR1e-3/WD0.1, BF16 and the
same data/order/initial parent parameters.

## 4. Environment

- Required CUDA visibility: exactly index0 and one registered UUID
- Required FLA source SHA:
  `9c8e42e762fce087c27b673af4922795d9edb85e`
- Required recurrent path: official `ChunkGDN2FunctionBackward`, with 16 chained
  native chunks per layer at L1024
- Formal source: exact pushed SHA in a clean detached remote worktree
- No CPU model smoke and no concurrent GPU model/evaluation

## 5. Commands

The exact contract and endpoint commands are frozen by their committed scripts:

```bash
python experiments/zoology_mqar/check_chunk_local_biaxis_cuda.py \
  --expected-gpu-uuid <registered-uuid> --output <contract.json>

python -m experiments.zoology_mqar.gdn2_chunk_local_biaxis_endpoint \
  --output-dir <run-dir> --max-epochs 10 --batch-size 32
```

## 6. Launch Gates And Artifacts

Before training, the CUDA contract must prove all of:

1. one correct GPU and exact pinned FLA source;
2. candidate parent parameter hash equals the native control hash;
3. exact +8,192 parameters and no state-width change;
4. zero-decay full-model output matches native within fixed BF16 tolerance;
5. direct opened Bi-Axis recurrence output/state agreement within 0.08;
6. two official chunk backward nodes and nonzero chunk2-to-chunk1 V/K-decay/V-decay VJP;
7. finite nonzero first-order gradients in both new projections;
8. nonpositive V decay, local scale at least 0.249 and inverse at most 4.01;
9. no NaN, OOM, fallback, source drift or concurrent model process.

Formal artifacts must include control/candidate score, cases and checkpoint,
comparison JSON, source/config/log hashes and GPU timing/memory.

The strict R6 contract passed on clean detached pushed SHA
`61920ce2e78b5c4149f07db0619e62705da5350f`. It proved exact zero-decay
identity, +8,192 parameters, two pinned-official backward nodes, direct
recurrence agreement and nonzero cross-chunk gradients into first-chunk V,
K-decay and V-decay. Contract JSON SHA256 is
`7cbabc7a8046931b448c5a415f8fb7ecd24bcdd76d851aef5105db63102685a6`.

## 7. Registered Result Gates

Activation requires two active layers, 16 anchors/layer, mean absolute V decay
at least `1e-4`, active fraction at least 0.05, nonzero group/board/token
variation, finite terminal state and the local frame bound.

Quality requires every check:

- balanced accuracy at least 0.35 and at least +0.10 over control;
- future and past accuracy each at least +0.07 over control;
- joint exact at least 0.03 and at least +0.03 over control;
- fewer total errors;
- wrong-key valid-value swap fraction no more than 0.05 above control.

This reference implementation intentionally exposes 16 official calls. Its
elapsed, post-warm wall and warmed-step ratios must each stay below 4x control,
and peak allocation below 1.5x. A full pass authorizes one fused chunk kernel,
whose production target remains near the original GDN2 pipeline.

## 8. Decision

The fixed same-process endpoint completed both arms. Native control versus
candidate balanced/future/past/joint accuracy is
`0.09975/0.0930/0.1065/0` versus `0.01375/0.0120/0.0155/0`. Total errors rise
`3601->3945`. Conditional wrong-key swaps fall `0.131630->0.038530`, but this
is broad retrieval collapse: the candidate learns much later, ends near random,
and has no joint exact examples.

The failure is not inactivity or instability. Both layers activate with V-decay
magnitudes `0.006751/0.014278`, active fractions `0.3132/0.6683`, nonzero
group/board/token variation, finite terminal state and local scale/inverse
exactly within `[0.25,4]`. The learned solution therefore changes the intended
physical lifetime but suppresses useful long-range retrieval.

Elapsed, post-warm wall, warmed-step and allocation ratios are
`2.309/2.321/7.356/1.206x`; the warmed-step cost gate independently fails.
Comparison, candidate score, checkpoint and manifest SHA256 are respectively
`3f1558662096c4fce72e0b1208e3b8ce847ddcccb2b2c61261f6079d82531718`,
`78447503de5f926fc0d56a8e0069ab4c5ad95e4412207b280ab13875aea9b13d`,
`ad8f4fe8656bcca4fc83070abbbe67e51e98c9698e11019c7564dfb95470f003`
and `b69c58df37f93f1225d64fc0baf896905b9b6c0010fc09cb76ee5d5160401dd9`.

Close this implementation without group, cap, decay-map, initialization,
fusion, seed, optimizer or duration rescue. Together with P015/P016, this
rejects grouped persistent V-axis lifetime as the missing binding mechanism;
it does not reject future fused chunking for a different recurrence.

## 9. Submission Record

Not applicable. This is an architecture experiment, not a submission.
