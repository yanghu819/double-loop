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

Pending.

## 7. Decision

Pending the sole registered endpoint.
