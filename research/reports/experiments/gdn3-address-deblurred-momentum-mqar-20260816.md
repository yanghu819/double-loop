# P-GDN3-069 Address-Deblurred Momentum

## 1. Research Question

Does temporal mixing in the Q/K address front end cause the narrow
adjacent-owner tail left by P059 Momentum DeltaNet?

## 2. Evidence And Hypothesis

P059 reaches balanced/future/past/joint
`.94425/.95150/.93700/.82400`, with 223 errors. Of those, 151 are valid values
assigned to a wrong key; every one is an adjacent write owner and 150 have the
same direction. P063 shows that adding an independently learned prediction key
destroys the co-adapted coordinate system, P067 shows that a separate erase key
destroys retrieval, and P068 shows that replacing dense Momentum by an
owner-local projection removes the useful global integration.

The remaining distinct hypothesis is that the owner coordinate is coherent
but blurred before the recurrence: the external layer applies independent
width-four causal depthwise convolutions to Q, K and V. Adjacent address
contamination is exactly the observed error topology. Removing temporal mixing
only from the address streams should preserve learned value formation and the
full successful Momentum transition while sharpening local ownership.

## 3. Fixed Mechanism

Keep P059's external Momentum DeltaNet at exact SHA `c6e77fa`, all Q/K/V
linear projections, V Triton short convolution, alpha/mu/beta/eta gates,
output correction, one primary scan, stacked `[S,M]` state and native
FutureSeed. Replace Q and K short convolution by parameter-free pointwise
`SiLU(W_q x_t)` and `SiLU(W_k x_t)`. This removes exactly 1,024 parameters per
layer, 2,048 total. Persistent state and scan work are unchanged.

This is not direct erase/write-key decoupling: read, residual prediction,
Momentum update and state commit still share one K address. It is not a stable
position anchor, lagged key, second state, cache, selector, reverse scan or
task-specific rule.

## 4. Strict CUDA Contract

Before formal training, the exact pushed SHA in a clean detached worktree must
prove one visible registered A800, exact external Momentum and host FLA source
hashes, two native Momentum chunk backward paths, two pointwise Q paths, two
pointwise K paths, two Triton V convolutions, exactly 597,624 parameters,
unchanged 8,192 recurrent values per layer, exact shared-parent tensor loading,
finite nonzero gradients through Q/K/V, all four Momentum gates and the native
FutureSeed edge, address token variation, nonzero incoming-state dependence,
causal prefix identity, head permutation equivariance and no fallback.

Synthetic pointwise output must equal `silu(x)` exactly and be independent of
changes to neighboring tokens. The V path must remain a width-four Triton
convolution. Any integrity miss closes the implementation before training.

## 5. Prediction And Gates

The single fixed directional MQAR L1024 candidate uses D128/L2/H4/K32/V32,
10 epochs, batch32, seed123 and the exact P059 matched initialization/data.
It passes only if:

- both address-deblur layers and one native FutureSeed route are active;
- balanced accuracy is at least `.955` and improves by at least `.01`;
- future and past each regress by at most `.005`;
- joint exact is at least `.84` and improves by at least `.01`;
- total errors are at most 178;
- wrong-key swaps are at most 105 and conditional share at most `.60713`;
- adjacent-owner swaps fall by at least 20%; and
- elapsed, post-warm, warmed-step and allocation ratios are each below `1.10x`.

Any miss closes Q/K address deblurring. There is no one-stream/partial bypass,
residual tap, convolution width, activation, mix, scale, seed, LR, loss, batch,
model width/depth or duration rescue. Only a full pass authorizes Sudoku
transfer.

## 6. Result

The exact pushed/read-back source `b42e6295a44274bb1f0cc6cd024fad1726b409cc`
ran from clean detached worktree
`/huyang2/double-loop/worktrees/p-gdn3-069-b42e629` on the registered A800
`GPU-c1d7c624-a393-befa-3807-7e00602d65ca`. The strict CUDA contract passed:
both native Momentum chunk backward paths were present, Q/K were exactly
pointwise and neighboring-token independent, V remained a width-four Triton
convolution, the parameter delta was exactly `-2,048`, `[S,M]` state geometry
was unchanged, the native FutureSeed edge and all recurrence gradients were
finite and nonzero, and shared-parent loading was exact. Contract JSON SHA256
is `7d34444f44d894dc29a083150e503739ab8a41cc355b6a543e68eb0226b6fb9a`.

The sole formal endpoint completed all ten epochs and failed quality sharply:

| Metric | P059 control | P069 candidate | Delta |
|---|---:|---:|---:|
| balanced accuracy | .94425 | .36275 | -.58150 |
| future accuracy | .95150 | .35650 | -.59500 |
| past accuracy | .93700 | .36900 | -.56800 |
| joint exact | .82400 | 0 | -.82400 |
| total errors | 223 | 2,549 | +2,326 |
| wrong-key valid-value swaps | 151 | 1,491 | +1,340 |
| adjacent-owner swaps | 151 | 1,459 | +1,308 |

This is broad retrieval failure, not a cleaner address tail. Of 3,777
control-correct queries, 1,405 become wrong-key swaps and 999 become other
errors; only 52 old swaps become correct. Both pointwise address paths are
active and token-varying, the native FutureSeed gate is `.50932`, and state
and Momentum remain bounded. Therefore dead activation or numerical
instability cannot explain the quality loss.

Cost ratios for elapsed/post-warm/warmed-step/allocation are
`1.13908/1.13439/.60612/.99998x`. The isolated warmed step is faster because
two convolutions were removed, but full training is slower and misses both
registered wall-time gates. GPU telemetry contains 163 samples: compute-active
mean/peak utilization is `48.75%/82%`, observed memory peaks at `3,016 MiB`,
and power peaks at `227.83 W`.

Comparison, candidate cases and checkpoint SHA256 are respectively
`485616caf7dc535df6c5c19c32cd0e16377443a3e2c9a390b516cd5bc5bee602`,
`f06c9e8c015f51deecab78c6967cd11eed8047c530ac1403509aa9adfd1094cf`,
and `7a92c75e30faf859a1c91eab2327b6fb409bdec89eff8b747768eda2bff8c497`.

## 7. Decision

Discard P-GDN3-069. The result falsifies the claim that Q/K short convolution
is merely harmful adjacent-owner blur. It is part of the useful learned local
address representation, and deleting it destroys the P059 optimization
transition despite preserving the complete recurrent core. Close full or
partial Q/K bypass, residual tap, convolution width, activation, interpolation
and every training-setting rescue. The remaining owner tail must be addressed
without breaking the co-adapted Q/K front end.
