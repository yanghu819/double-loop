# P-FS2-006: Receiver-Native Inherited-vs-Live Contrast

## 1. Metainfo

- Status: failed at strict recurrence-parity gate; no quality verdict
- Parent: D256/L12/H8/K32/V32 position-QK official GDN2 plus native terminal FutureSeed, exact step3000
- Task: hard official/full-diversity 9x9 Sudoku, 51-64 blanks
- Intervention: zero-parameter causal diagnostic only

## 2. Mechanism Hypothesis

FutureSeed may carry useful evidence into a receiving layer but lose it when
that layer's live token recurrence overwrites or aliases the same address. A
receiver-native signal must compare inherited and live evidence in the
receiver's basis. This is narrower than P-GDN3-007: P007 read the incoming
state and used a learned controller before the scan; it did not test the
same-address discrepancy after the token's official committed write.

This also differs from producer codecs and caches. No producer-basis K/V is
transported, no token is selected, and no second recurrent core is introduced.
The frozen incoming state `F` is retained read-only during each official scan,
so any eventual implementation must count it as a second side-memory bank.

## 3. Exact Intervention

For official GDN2 token `t`, replay in FP32:

```text
D_t = Diag(exp(g_t)) S_(t-1)
e_t = w_t * v_t - (b_t * k_t)^T D_t
S_t = D_t + k_t e_t^T
o_t = q_t^T S_t
r_t = q_t^T F
d_t = r_t - o_t
```

`q` and `k` use the pinned kernel's L2 normalization and `q` uses its
`1/sqrt(K)` scale. Thus the live read is explicitly post-write. RMS-match
`d_t` to `e_t`; a virtual zero-init receiver parameter would produce
`delta_v=rho*tanh(alpha)*d_hat`. Discovery obtains its exact zero-init
cotangent through the unchanged official backward graph.

Discovery aggregates exactly one global `[11,H,V]` direction over all ranges,
edges and boards. It cannot select a range, layer, edge, head, token or board.
Holdout fixes the actual budget causally at every current token by enforcing
the causal target `RMS_HV(w*delta_v)=0.01*RMS_HV(e)`. One scalar per board/token
preserves the discovered direction's relative strength across heads. The
endpoint recalibrates against the actual BF16 value received by the kernel and
requires every token-local relative error to be <=10%; the achieved board-level
ratio is archived. No future token participates in scaling an earlier
intervention.

## 4. Independent Data Contract

The parent checkpoint records `seed52`, `Random(1052)`, training microbatch32,
grad-accum4 and curriculum `46-50:500,51-55:3500,...`. Replaying all draws
through step3000 produces exactly 384,000 samples and a Python RNG state equal
to `checkpoint["rng_python"]`. It observed 335,814 unique rows; 3,451,620
strict-unseen rows remain in 51-64.

Frozen data artifacts:

- index SHA256 `8675395b75899aa6c9f3a899791df9b71f042d87a168426eafdf89269258bdd7`;
- manifest SHA256 `7be5a07d15b7b9dbd3127491e6d75cda43877756d218dd452fd5b2056b36dde5`.

For each of 51-55, 56-60 and 61-64, fixed seed `5205101` selects 128
discovery rows and seed `5205102` selects 128 holdout rows from the remaining
unseen pool. Base row IDs and hashes are archived. No official test row is
used and discovery/holdout cannot overlap.

## 5. Integrity Contract

Run only on one visible A10080 at CUDA index0 from a clean detached pushed SHA.
Require parent checkpoint SHA
`6339c3cb2b5fc5230a581d6633716483e35ff8e4522f06a9d7aaf26512f023da`,
parent source `9f2ee8d1738032bc5f09b55db0b81d507780b376`, pinned FLA SHA
`9c8e42e762fce087c27b673af4922795d9edb85e`, 12 official layers and exactly
180 `ChunkGDN2FunctionBackward` nodes per discovery graph. The imported FLA
tree must match all files in pinned wheel SHA `0280db31...` with aggregate tree
SHA `dd792d8b...`; official train inputs/labels are pinned by content SHA. FP32 replay output
and terminal-state relative RMS must each be <=`2e-3`; after BF16 quantization,
every token-local 1% holdout-budget relative error must be <=`0.10`. Model parameter SHA must remain unchanged, every model
`.grad` must remain `None`, and no optimizer or checkpoint may be created.

Tensor unit tests cover independent recurrence replay, post-write contrast,
deterministic derangement, autograd/finite-difference virtual gradients, exact
committed RMS budget and deterministic paired bootstrap. They do not replace
the official CUDA contract.

## 6. Fixed Falsifiers And Gate

The disjoint holdout evaluates baseline, full contrast at `+epsilon`, full
contrast at `-epsilon`, frozen-only `q^T F`, and board-shuffled
`q^T F_other-q^T S_t`. One fixed seeded board derangement is reused across
every receiver call in a microbatch so the sham preserves cross-layer,
depth-cycle and loop coherence. All conditions are required:

- mean paired equal-five-loop CE improvement >=`0.01`;
- deterministic paired-bootstrap 95% lower bound >`0`;
- full contrast improves and sign reversal worsens in at least two ranges;
- full contrast beats frozen-only by >=`0.005` CE;
- shuffled-F improvement is at most 25% of full;
- no discovery board carries more than 20% of total direction energy.

Any miss closes this mechanism without edge/range selection, epsilon, rho,
memory, seed, loss, batch, duration or Sudoku continuation rescue.

## 7. Cost And Scaling

The diagnostic separately uses fixed B64 execution microbatches and a combined
B256 four-arm holdout on A10080, under a 2400-second wall budget. A future trainable
candidate would add exactly `11*8*32=2,816` parameters and retain one frozen
K32xV32 state bank per receiving path. Before Sudoku use it must pass a
from-scratch directional MQAR test and, when fused, satisfy elapsed `<1.30x`
and allocation `<1.10x` against a contemporaneous control.

## 8. Next Decision

R1 on pushed SHA `dc972a38` was stopped before science when review found that
the shuffled-F sham changed donor boards across receiver calls. It produced no
score; its orchestration classified the externally terminated endpoint as a
non-science timeout abort. R2 on pushed SHA `71a6fd53` reuses one fixed donor
derangement across every receiver call in a microbatch and passes the source,
GPU, data, parent, wheel and tensor-test contracts.

R2 then fails before discovery aggregation: the first captured official call
has FP32-replay output relative RMS `0.0025230150`, above the registered
`0.002` ceiling; terminal-state relative RMS is `0.0011426633`. The endpoint
exits nonzero, launcher status is `25`, `abort.json` marks an integrity failure,
and no `score.json`, direction, holdout metrics or quality verdict exists.
This closes the replay-based receiver-live diagnostic without tolerance,
precision, epsilon, batch, seed or duration rescue. It does not falsify the
receiver-live mechanism because its intervention was never admitted.

R2 artifacts:

- run: `/huyang2/double-loop/runs/p-fs2-006-receiver-live-contrast-r2-20260813T145000Z-71a6fd5`;
- run log SHA256: `fe11beb63c806c73674e60e9e3e949f4493b296e1465a17c40d7554e71963585`;
- abort SHA256: `0fe7de3d79f1fbfbb9f4614aac23f08426a8312f51dc2bca7aacf8fa86304db9`;
- source/provenance SHA256: `e6976a166c35e12db4d3f8db11551f22ea00af33fa67ffde1be1a62569396349` / `1109da69accf756be1e1b4e0033e89d0b7b8b25aef008363493a05a772db8c09`;
- final GPU: index0, target UUID, zero MiB, no compute app.

## 9. Submission Record

Not applicable.
