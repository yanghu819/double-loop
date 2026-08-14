# P-GDN3-031: Function-Preserving Decoupled-Key GDN2

## 1. Metainfo

- Status: completed; rejected at the fixed directional-MQAR quality gate
- Decision field: directional MQAR L1024 wrong-key binding before Sudoku
- Fixed setting: D128/L2/H4/K32/V32, native FutureSeed, 10 epochs, batch32, seed123
- Resource: one A100-SXM4-80GB at CUDA index0, physical UUID
  `GPU-d2877fe4-641c-fe64-2a74-8abca47c292f`

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

The exact pushed source `004dd0553a160ea397bbbe6cc51b237ec62dbb89`
completed the contract and single registered endpoint. The contract passed:
two of two carrier layers were official `GatedDeltaNet2`, autograd contained
two official `ChunkDPLRDeltaRuleFunctionBackward` nodes, all Q/K-write/K-erase/V
short convolutions used the Triton path, and the FLA/DPLR source hashes matched
the registration. Initial erase/write projection and convolution error was
zero, tied output/state parity passed, all required gradients were finite and
nonzero, parameter delta was exactly 33,792, and persistent state/scan deltas
were zero. The contract JSON SHA256 is
`22f895e1fed7ea4088299958c60a41a57deec2781096039acb6225706055f26f`.

The fixed endpoint result is:

| metric | historical native FS | current-runtime native FS | decoupled key |
| --- | ---: | ---: | ---: |
| balanced accuracy | 0.747500 | 0.306250 | **0.011000** |
| future accuracy | 0.741500 | 0.311500 | **0.008000** |
| past accuracy | 0.753500 | 0.301000 | **0.014000** |
| joint exact | 0.339000 | 0 | **0** |
| errors / 4,000 queries | 1,010 | 2,775 | **3,956** |
| wrong-key valid-value swaps | 815 | 1,271 | **141** |
| swap fraction among errors | 0.806931 | 0.458018 | **0.035642** |

The apparently lower conditional swap fraction is a collapse artifact, not an
addressing gain. Only 44 of 4,000 queries are correct. Predictions concentrate
on token 165 (`1,914`) and token 177 (`1,743`); 3,626 of 3,956 errors (`91.66%`)
fall in the broad 160--191 value-token range. The model therefore emits a few
globally common value tokens rather than retrieving the value associated with
the queried key. Reporting swaps without total errors would reverse the true
conclusion.

The mechanism did activate. Layer 0/1 erase-write cosine is
`0.048316/0.060033`, separation is `0.951684/0.939967`, and key relative RMS is
`1.379631/1.371106`. Projection delta RMS is `0.018718/0.024485` and convolution
delta RMS is `0.024441/0.025138`. Sampled transition spectral maxima remain
bounded at `1.006299/0.998570`; terminal state RMS is `1.764684/0.279975` with
nonzero board variation. Thus the endpoint is not an inactive-mechanism or
numerical-instability failure.

Systems gates also pass. Candidate fit, post-warm wall, warmed-step, and peak
allocation ratios are `1.0823x/1.0856x/1.6035x/1.2235x`, all below their frozen
limits. Warmed throughput is 1,271.84 examples/s. Training plus validation took
99.95 seconds, and the full contract-to-decision run completed from
2026-08-14T03:39:14Z to 03:44:59Z. The score/decision SHA256 is
`1b631993bbc5dd1e483bd0178083b93f0b9f10c358208ebc7cd767156ba8e538`.

## 8. Decision

Reject and close direct decoupled-key GDN2. Starting from exact native GDN2 did
not rescue the mathematical idea: optimization rapidly made erase and
write/read keys nearly orthogonal. The erase operation then no longer targets
the rows populated by the write/read address, so the recurrent state remains
bounded while useful retrieval disappears. This independently agrees with the
P028/P030 boundary that a lower wrong-key error fraction can be obtained by
destroying the usable linear address channel.

No Sudoku transfer was launched. There will be no angle/tie regularizer,
key-scale, alternate initialization, kernel, seed, LR, loss, batch, epoch,
width, depth, or duration rescue. A tie or cosine constraint would be a new
coupled-address hypothesis rather than evidence that direct decoupling works.
The next GDN3 candidate must preserve a shared address anchor or change the
recurrent state topology for a separately falsifiable reason.

## 9. Provenance

- Idea source: `yanghu819/GDN_decouple_k` commit
  `c7667fd11d95d3d147f59bb3d4492989909b69ae`; mathematical inspiration only.
- Candidate branch: `codex/gdn3-decoupled-key-mqar-20260814`.
- Formal source: `004dd0553a160ea397bbbe6cc51b237ec62dbb89`; local GitHub
  SSH readback matched before transfer.
- Clean detached worktree:
  `/huyang2/double-loop/worktrees/p-gdn3-031-004dd05`.
- Formal run:
  `/huyang2/double-loop/runs/p-gdn3-031-decoupled-key-l1024-20260814T0338Z-004dd05`.
- AIStation HTTPS access to GitHub timed out, so the already-pushed exact source
  was transported as a verified Git bundle. Bundle SHA256:
  `cceaeded3b28aeb2255ea3e3cc399a58f83f20f180df6e88505cec5dc8cc58ea`.
  The remote bundle and branch ref were verified before creating the detached
  worktree; the local GitHub SSH readback remains the authoritative remote
  provenance claim.
- Candidate checkpoint SHA256:
  `02f61cfbae4fc88625df3acd68f9c75a2eda06a7ab711ce06bb6df0bec3db802`.
- Candidate config/metrics/score SHA256:
  `a7caba05a75d0a03a73296990ed339517d7e22b4c9fd5ac8e4296222413dbe59` /
  `832f1c13ef483d536a7986feb5e7b08d380e269798776d2c455c442b24a8c8b1` /
  `a2afa9ed5ec548447c4f09458ce97ab114cd6347b004f73a3548fbb36da250ea`.
- Compact tracked evidence:
  `runs/p-gdn3-031-decoupled-key-l1024-20260814T0338Z-004dd05`.
