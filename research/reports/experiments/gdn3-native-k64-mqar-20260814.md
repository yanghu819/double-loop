# P-GDN3-032: Native Coherent K64/V32 Main-State Expansion

## 1. Question

Does doubling the native GDN2 address-row dimension from K32 to K64, while
holding D128/L2/H4/V32, the official recurrence, the single scan, and native
FutureSeed fixed, reduce directional-MQAR binding interference enough to close
the L1024 task?

## 2. Motivation

The direct `GDN_decouple_k` test P-GDN3-031 falsified unconstrained separate
erase and write/read keys: the branches became nearly orthogonal and retrieval
collapsed. P-DIAG-ADDR-001 then found severe K32 producer/receiver key-Gram
collapse but no evidence that cache admission, write survival, or receiver
reprojection was the causal repair. This leaves one bitter-lesson hypothesis:
the learned main state may simply need more coherent address rows.

P-CAUSAL-014 is not this test. It held K32 fixed and expanded V32 to V64;
quality regressed, so payload-axis capacity is closed. P-GDN3-014 is also not
this test. It tried to graft zero rows onto a trained Sudoku parent and failed
the exact parent-function migration gate before training. No native K64/V32
GDN2 has been trained from scratch on the validated L1024 binding regime.

## 3. Mechanism

Instantiate the unmodified pinned official FLA `GatedDeltaNet2` with
`hidden_size=128`, four heads, `head_dim=64`, and `expand_v=0.5`. Per head the
state is K64xV32. The native recurrence remains

```text
S_t = (I - k_t (b_t * k_t)^T) Diag(exp(g_t)) S_{t-1}
      + k_t (w_t * v_t)^T
```

Read, erase, and write therefore retain one learned normalized key namespace.
There is one official chunk scan per layer, one native terminal-state
FutureSeed edge, no cache, replay, selector, auxiliary state, second write,
second scan, or task rule.

## 4. Falsifiable Prediction

If K32 address-row capacity is the binding bottleneck, the extra learned rows
should reduce correct-value/wrong-key swaps and raise both directional
accuracies, not merely change the composition of failures. The fixed endpoint
must reach balanced/future/past accuracy at least 0.85, joint exact at least
0.60, balanced accuracy at least 0.10 above both historical and current-runtime
native K32 references, at least 0.10 lower wrong-key swap fraction than both,
and fewer total errors than both.

If those conditions fail, K-axis main-state expansion is closed. There is no
K48/K96, head-count, state-size, seed, LR, loss, batch, epoch, width, depth, or
duration rescue.

## 5. Fixed Protocol

- Directional MQAR L1024, four key/value pairs, 10,000 train and 1,000 test
  examples with the locked data hashes.
- D128/L2/H4/K64/V32, batch32, ten epochs, seed123, native FutureSeed.
- One candidate only. Reuse the locked historical K32 and contemporaneous
  P-FS2-007 native K32 references; do not rerun or select a control.
- Candidate parameters: exactly 770,384. Persistent recurrent state: exactly
  8,192 values per layer, twice K32. Logical scan count remains one.
- Fit, post-warm wall, and warmed-step ratios must each be at most 2.00; peak
  allocation ratio must be at most 1.50 versus the current-runtime control.

## 6. CUDA Contract

Before science, require exactly the registered task-mode A100 at CUDA index0,
the pinned FLA wheel/source/ops hashes, exact Zoology SHA, two official
`GatedDeltaNet2` layers, two `ChunkGDN2FunctionBackward` nodes, Triton Q/K/V
short convolutions, and no fallback. Assert exact K64/V32 geometry, 8,192 state
values per layer, 770,384 parameters, finite nonzero gradients in the added
Q/K/decay/erase rows plus V/write and native FutureSeed, a finite nonzero-state
path, fixed data hashes, and genuine future dependency.

Any contract or systems-integrity miss aborts before training. Any completed
quality or cost miss rejects the mechanism without rescue.

## 7. Results

R1 source `e80baf7e` exited while sourcing the launch environment because the
GPU name containing spaces was not quoted. It occurred before contract model
construction and allocated zero GPU memory, so it is retained as a non-science
orchestration abort. R2 changes only that quoting; mechanism, data, budget and
all gates remain frozen.

R2 source `f6f9951d` then exited before model construction because its checker
used the newer NUL-delimited generic Python-tree hash while the frozen GDN2 ops
hash was defined by the existing undelimited checker. The locked source files
were unchanged and only a 3 MiB CUDA context was created. R3 aligns the hash
algorithm with the existing GDN2 contracts; no mechanism or gate changes.

R3 source `94c0da43` passed provenance, data and model construction, then exited
inside the synthetic nonzero-state checker because the harness supplied a BF16
initial state while pinned official `chunk_gdn2` requires FP32 recurrent state.
No formal training ran. R4 changes only that synthetic state dtype; the model,
data, budget and all decision gates remain frozen.

R4 source `daff830afd54fb1ceb07a12c9125b726a675c184` passed the strict
contract and completed the one registered endpoint. The contract observed two
official `ChunkGDN2FunctionBackward` nodes, Triton Q/K/V short convolutions,
exact K64/V32 geometry, 8,192 state values per layer, and exactly 770,384
parameters. Both layers had finite nonzero gradients in the added address rows;
native FutureSeed dependency and gradient were also finite and nonzero. The
contract JSON SHA256 is
`a30a9e298d21121b7a71daf91af029041281b046c1b57be8d802fb9496220da1`.

| metric | historical K32 | runtime K32 | native K64/V32 |
| --- | ---: | ---: | ---: |
| balanced accuracy | 0.747500 | 0.306250 | **0.045500** |
| future accuracy | 0.741500 | 0.311500 | **0.044500** |
| past accuracy | 0.753500 | 0.301000 | **0.046500** |
| joint exact | 0.339000 | 0 | **0** |
| errors / 4,000 queries | 1,010 | 2,775 | **3,818** |
| wrong-key valid-value swaps | 815 | 1,271 | **294** |
| swap fraction among errors | 0.806931 | 0.458018 | **0.077004** |

The extra address rows therefore change the error composition but do not solve
binding. They reduce wrong-key swaps by 977 versus the runtime K32 reference,
while adding 1,043 total errors. Only 182 of 4,000 queries are correct. This is
the same diagnostic trap exposed by P028-P031: a lower conditional swap rate
can result from broader retrieval failure.

Optimization is also materially slower. Historical K32 reaches validation
accuracy `0.1965` by epoch4 and `0.71575` by epoch6. K64 is only `0.0110` at
epoch4 and `0.02725` at epoch6, ending at `0.0455`. The mechanism is active,
not dead: the native FutureSeed gate is `0.506501`, its raw RMS is `0.064316`,
and every registered added-row gradient passes. Raw coherent address capacity
has diluted or delayed learnability rather than providing useful separation.

All cost gates pass. Fit, post-warm wall, warmed-step and peak-allocation ratios
are `1.34535x/1.34482x/1.69907x/1.35364x`; warmed throughput is 1,200.31
examples/s. Training plus validation takes 124.24 seconds. The complete run
finishes from `2026-08-14T05:41:26Z` to `05:49:44Z`; decision SHA256 is
`c6153f3b15eed55c7479d4517f97f9772d18dbe1527e91b1333ad397969294b5`.

## 8. Decision

Reject and close native K-axis state expansion. K64/V32 passes provenance,
activation, official-kernel and cost gates, but misses every absolute quality
gate and performs far below both K32 references. The observed swap reduction
is not accompanied by lower total error, so severe K32 Gram anisotropy is not
evidence that adding address rows will improve usable memory.

There will be no K48/K96, state-size, head-count, seed, LR, loss, batch, epoch,
width, depth or duration rescue. No Sudoku transfer is authorized. The next
candidate must improve how evidence is bound or credited in a coherent address
namespace, rather than merely increasing its dimensional capacity.

## 9. Provenance

- Branch: `codex/gdn3-native-k64-20260814`.
- Formal source and GitHub readback:
  `daff830afd54fb1ceb07a12c9125b726a675c184`.
- Clean detached worktree:
  `/huyang2/double-loop/worktrees/p-gdn3-032-e80baf7`.
- Formal run:
  `/huyang2/double-loop/runs/p-gdn3-032-native-k64-l1024-r4-20260814T054524Z-daff830`.
- Candidate checkpoint/config/score SHA256:
  `916189db960875ed26be5219dde3cd64a876078a43f71873912497550ccd7153` /
  `1c7659b0cb6702a8c1a79a441bb5a62c8f82649fbe06192c97edefc8d2c7da7c` /
  `daccd1ed10444b0c2da879da90e23e75c65014c5d94f7ac268ebb89b8adc4d60`.
- Contract/decision/source-snapshot/formal-log SHA256:
  `a30a9e298d21121b7a71daf91af029041281b046c1b57be8d802fb9496220da1` /
  `c6153f3b15eed55c7479d4517f97f9772d18dbe1527e91b1333ad397969294b5` /
  `3b863e9397628d4359aa2e359255896abe9daa04faeb36ac0db5c944981ba7f4` /
  `6fb3a7cd70d02156cf52722fe96b4ecd0a5d5dd04504e947b930ba3a129b8fca`.
- R3 non-science abort/contract-log SHA256:
  `0e80811f942aba3c9d97c6b57f74852d62fac967e1589670f7332fd97e0ff432` /
  `4cb6dbd8d832eb5f86411cf750891fcb67a6b11e54c51834937c539c9ddb81b5`.
- Compact tracked evidence:
  `runs/p-gdn3-032-native-k64-l1024-r4-20260814T054524Z-daff830`.
