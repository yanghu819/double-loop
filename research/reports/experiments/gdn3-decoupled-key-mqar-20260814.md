# P-GDN3-031: Function-Preserving Decoupled-Key GDN2

## 1. Metainfo

- Status: preregistered; implementation and GPU acquisition in progress
- Decision field: directional MQAR L1024 wrong-key binding before Sudoku
- Fixed setting: D128/L2/H4/K32/V32, native FutureSeed, 10 epochs, batch32, seed123
- Resource: first verified single authorized AIStation GPU; CUDA index0 only

## 2. Evidence And Hypothesis

The external `GDN_decouple_k` repository at commit
`c7667fd11d95d3d147f59bb3d4492989909b69ae` proposes separate normalized keys
for erasing and writing. Its reported CPU MNIST comparison is not a matched
GDN2 experiment, and the repository has no license, so this work uses only the
mathematical idea and copies no code.

Standard GDN2 uses one normalized key both to locate content to erase and to
place the new payload. The falsifiable hypothesis is that this aliasing causes
some valid-value/wrong-key swaps, while the rest of the working GDN2 transition
should remain unchanged. Unlike P-GDN3-030, the test must begin exactly on the
native GDN2 function and preserve channel-wise erase/write gates. A quality
gain must therefore come from learned key specialization rather than replacing
the working transition at initialization.

## 3. Mechanism

Retain native GDN2 Q/V projections, decay `g`, channel erase gate `b`, channel
write gate `w`, output path, K32xV32 state, and native terminal FutureSeed.
Rename the existing normalized key `k_w` and add `k_e` with an identical
projection and Triton short-convolution initialized by exact parameter copy.
For `D=Diag(exp(g))`, apply

```text
S_t = D S_(t-1)
    - k_e [(b * k_e)^T D S_(t-1)]
    + k_w (w * v)^T.
```

This is one pinned official `chunk_dplr_delta_rule` call with
`a=D(b*k_e)`, `b_factor=-k_e`, additive `k=k_w`, and additive `v=w*v`.
When `k_e=k_w`, the recurrence is exactly native GDN2. The D128/L2 candidate
has 695,376 parameters, exactly 33,792 more than native, retains 4,096 state
values per layer, and adds no state or scan.

## 4. Novel Boundary

P030 used a random independent erase direction, a new scalar beta, and a
symmetric contractive transition; it removed native channel-wise erase
behavior and collapsed retrieval. P031 instead starts on the exact native
manifold, retains `g/b/w`, and gives optimization only one new degree of
freedom: whether the erase address should diverge from the write/read address.
P005/P007 were gate/controller wrappers, P029 used sequential transformations,
and P028 forced a paired write. None tested this direct function-preserving
decoupling.

## 5. Registered Contract

Before training, the executable CUDA contract must prove:

- exactly one visible GPU at CUDA index0 with the registered name and UUID;
- pinned FLA SHA `9c8e42e762fce087c27b673af4922795d9edb85e`, exact DPLR source tree,
  official `ChunkDPLRDeltaRuleFunctionBackward`, and Triton short convolution;
- exact 695,376 parameters, +33,792 over native, unchanged state and one scan;
- candidate parent parameter hash exactly equals the locked native model hash;
- erase projection and convolution are bit-exact copies of write key at init;
- tied DPLR output/state match official GDN2 and explicit FP32 recurrence within
  relative RMS 0.03, and full two-layer logits within 0.10;
- finite nonzero gradients through Q, both key paths, V, decay, erase/write
  gates, initial state and native FutureSeed, with distinct erase/write key
  gradients;
- erase-key dependency, nonzero incoming-state dependency, head equivariance,
  sampled transition spectral norm <=1.25, and no fallback.

## 6. Registered Science And Cost Gates

Run exactly one candidate from scratch on the fixed 10,000/1,000 directional
MQAR L1024 set. Reuse locked historical and same-runtime controls; do not rerun
or select a control.

The candidate passes only if all conditions hold:

- two active layers and one active native FutureSeed route;
- exact initial key tie; per layer trained `1-cos(k_e,k_w)>=1e-4`, key relative
  RMS >=0.01, and projection/conv parameter deltas >=1e-5;
- channel erase variation and writes are finite/nonzero; sampled transition
  spectral norm <=1.25; terminal-state RMS in `[1e-4,1e4]` with board variation;
- balanced/future/past accuracy each >=0.85 and joint exact >=0.60;
- balanced accuracy >=historical `0.7475+0.10` and >=current-runtime+0.10;
- wrong-key swap fraction at least 0.10 below both references and fewer total
  errors than both;
- fit, post-warm wall and warmed-step ratios <=1.75; peak allocation <=1.50.

Any contract, activation, stability, quality or cost miss closes this exact
mechanism. There is no key angle, initialization, gate, normalization, kernel,
seed, LR, loss, batch, epoch, width, depth or duration rescue. A full MQAR pass
is required before one hard-Sudoku transfer.

## 7. Results

Pending exact pushed source, clean detached worktree, CUDA contract and the
single fixed science run.

## 8. Decision

Pending. A pass authorizes one matched hard-Sudoku gate. A miss closes direct
decoupled-key GDN2 and returns the program to a different scalable recurrent
state topology.

## 9. Provenance

- Idea source: `yanghu819/GDN_decouple_k` commit
  `c7667fd11d95d3d147f59bb3d4492989909b69ae`; mathematical inspiration only.
- Candidate branch: `codex/gdn3-decoupled-key-mqar-20260814`.
- Formal source/run/checkpoint hashes: pending.
