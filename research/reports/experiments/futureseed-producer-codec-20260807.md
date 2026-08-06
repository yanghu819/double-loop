# P-FS3-002: Shared Producer Update Codec

## 1. Metainfo

- Status: discarded; strict matched decision complete
- Date: 2026-08-07
- Branch: `codex/fs3-producer-codec-20260807`
- Formal source SHA: `39b19273ba449ce382d7ff13fd517c0c6ecd5399`
- Benchmark: official/full-diversity hard 9x9 Sudoku
- Compute: AIStation task-mode GPU1 only
- GPU UUID: `GPU-53e9f3b4-2966-65d3-6614-09c540921519`
- Seed: 52 only
- Parent mechanism: position-QK GDN3 plus native FutureSeed
- Parent checkpoint:
  `/huyang2/double-loop/models/gdn3-position-qk-d256l12-s12000-20260806T131527Z-9f2ee8d/checkpoints/train_state_step003000.pt`
- Parent checkpoint SHA256:
  `6339c3cb2b5fc5230a581d6633716483e35ff8e4522f06a9d7aaf26512f023da`
- Frozen matched control:
  `p-fs3-001-terminal-s3100-20260806T200859Z-3e167b6`
- Formal candidate:
  `p-fs3-002-codec-s3100-20260806T220246Z-39b1927`

## 2. Mechanism Question

P-FS3-001 showed that a live, bounded residual based on analytic orthogonality
does not improve hard exact. Earlier FutureSeed2 experiments also closed raw
state carry, static KxV masks, scalar content trust, and fixed longer-radius
readout. P-FS3-002 asked whether a producer should instead learn a compact
message from what its recurrent state actually wrote.

For receiver layers2+, let `T` be the producer terminal state and `I` the exact
incoming seed consumed by that producer. The shared codec:

1. normalizes `T` and the actual producer update `T-I` by terminal-state RMS;
2. scores every K row from five generic row statistics and compresses the KxV
   update to one V-dimensional payload by softmax pooling;
3. decodes a bounded KxV residual from local terminal/update values, the shared
   payload, and their interaction;
4. adds the residual before unchanged native unit normalization and head gate.

The row scorer and cell decoder are shared across every layer and head and are
equivariant to independent K-row and V-column permutations. The bottleneck is
the existing V width, not a tunable rank. The final decoder is zero initialized,
making the complete model bit-exact to terminal FutureSeed at initialization.
The mechanism adds exactly 59 parameters and leaves all 12 pinned official-FLA
GDN2/Triton recurrent layers and position-QK addressing unchanged.

## 3. Registered Test

The terminal control was frozen from P-FS3-001 and was not rerun. The candidate
alone performed an exact step3000-to3100 continuation from the same parent,
preserving optimizer, RNG, data order, architecture, effective batch128, and
seed52. Its only semantic changes were:

```text
future_seed_content_mode=producer_codec
resume_allow_future_seed_content_upgrade=true
```

Activation required finite nonzero payload and residual RMS, residual relative
RMS at least `1e-4`, and nonzero between-board residual variation.

Primary quality required hard official51-64 macro loop5 exact `>=+0.02` with no
hard-range blank regression greater than `0.01`. The alternate route required
mixed loop5 exact `>=+0.03`, non-regressive official61-64 exact, and stronger
same-board loop3-to5 correction. Time and peak-allocation overhead each had to
remain below 10 percent. Any miss discards the mechanism without rescue.

## 4. Preflight And Integrity

The final formal source came from pushed SHA `39b1927` in a clean detached
worktree. The strict CUDA contract passed:

- CUDA index0 exposed only the registered GPU UUID;
- pinned FLA source SHA was
  `9c8e42e762fce087c27b673af4922795d9edb85e`;
- all 12 layers were official `GatedDeltaNet2`, with
  `ChunkGDN2FunctionBackward` and Triton convolution;
- zero-init output and all 12 terminal states were bit-exact to terminal
  FutureSeed;
- parameter delta was 59 and checkpoint migration named exactly the six codec
  tensors;
- all six parameters received finite nonzero gradients after opening the final
  decoder;
- independent row/column permutation error was `2.3842e-07`, below `2e-6`;
- no NaN, OOM, fallback, second compute app, source drift, or data drift occurred.

Contract log SHA256 is
`f9eb9a2e8b633beb5b6f2f6dfa7a3add6a9015cf843086b5b8ed806b0cf931a2`.

The first intended one-step full-stack probe exposed an orchestration defect:
the explicit `HOLE_STAGES` sum controlled the endpoint, so `--steps=3001` did
not truncate it. It was stopped exactly at step3006 and received a non-science
`abort.json`; none of its metrics were used. Pushed SHA `39b1927` bounded the
probe with `46-50:500,51-55:2501`. The corrected probe exited0, accepted exact
resume, exercised the official CUDA path, saved exactly step3001, and showed
loop5 residual relative RMS `0.001530` with nonzero board variation.

The formal run exited0. Its metrics and step3100 checkpoint were complete, the
source patch was empty, and GPU memory returned to zero after completion.

## 5. Formal Results

### 5.1 Optimization And Activation

Control/candidate train CE is `0.858617/0.858225`. Parameter count changes
`11,485,760 -> 11,485,819`, exactly the registered 59 parameters.

The codec is live at loop5:

- payload-code relative RMS: `0.336215`;
- producer-update relative RMS: `0.895528`;
- residual relative RMS: `0.002382`;
- residual between-board standard deviation: `0.0000655`;
- row-attention entropy/max: `0.908478/0.033911`;
- row-attention between-board standard deviation: `0.000317`.

Activation therefore passes. The row scorer remains close to uniform over 32
K rows (`1/32=0.03125`), which is important mechanism evidence: the learned
single-payload bottleneck changes content but does not discover selective
address routing in 100 matched steps.

### 5.2 Mixed And Official Quality

Mixed exact is identical across all loops. Control/candidate loop5 exact is
`0.025391/0.025391`. Candidate mixed blank accuracy across loops1-5 is
`0.5120/0.5367/0.5440/0.5453/0.5446`, versus control
`0.5151/0.5369/0.5448/0.5455/0.5456`.

Official 512-board loop1-to5 metrics are:

| Range | Arm | exact loops1-5 | blank loops1-5 |
| --- | --- | --- | --- |
| 51-55 | control | 0/0/0.001953/0.001953/0.001953 | 0.532345/0.566677/0.573945/0.573623/0.573766 |
| 51-55 | codec | 0/0/0.001953/0.001953/0.001953 | 0.532130/0.565782/0.572907/0.573086/0.573014 |
| 56-60 | control | 0/0/0/0/0 | 0.473182/0.498061/0.501699/0.503140/0.503861 |
| 56-60 | codec | 0/0/0/0/0 | 0.472256/0.495796/0.501390/0.502007/0.501527 |
| 61-64 | control | 0/0/0/0/0 | 0.504319/0.578523/0.589207/0.591954/0.591923 |
| 61-64 | codec | 0/0/0/0/0 | 0.504624/0.577821/0.588444/0.589848/0.589298 |

Hard51-64 macro loop5 exact is unchanged
`0.000651->0.000651`. Loop5 blank deltas for 51-55/56-60/61-64 are
`-0.000752/-0.002333/-0.002625`. Neither quality route passes.

### 5.3 Same-Board Loop Dynamics

All 256 case IDs, labels, and data hashes match independently for each hard
range. Mean wrong-cell trajectories are:

| Range | control loops1-5 | codec loops1-5 | control L3-to5 | codec L3-to5 | codec L5 better/equal/worse |
| --- | --- | --- | --- | --- | --- |
| 51-55 | 25.74/24.35/23.92/23.86/23.93 | 25.75/24.34/23.91/23.86/23.83 | -0.016 | 0.082 | 98/72/86 |
| 56-60 | 29.70/28.20/28.08/28.04/28.02 | 29.75/28.30/27.84/27.87/27.82 | 0.066 | 0.023 | 90/88/78 |
| 61-64 | 31.86/27.20/26.61/26.51/26.43 | 31.68/27.10/26.64/26.44/26.40 | 0.184 | 0.246 | 94/64/98 |

The codec slightly changes which boards improve, but it does not strengthen
late correction consistently: candidate loop3-to5 correction is weaker on
56-60, and loop5 has more regressions than improvements on 61-64. This cannot
activate the alternate route when exact is unchanged.

### 5.4 Cost

Matched fresh-process continuation time is `825.97s -> 1002.01s`, so effective
throughput falls `15.497 -> 12.774` boards/s and elapsed overhead is `+21.31%`.
Peak allocated memory is `13,186.4 -> 15,105.5 MiB` (`+14.55%`); peak reserved
memory is `14,288 -> 16,082 MiB` (`+12.56%`). Both registered cost limits fail.

## 6. Decision And Artifacts

P-FS3-002 is discarded. It passes identity, gradient, activation, official
kernel, exact-resume, data, and execution-integrity gates. It fails both quality
routes and both cost limits. Do not rescue with scorer features, payload count,
rank, decoder scale, seed, LR, loss, batch, model width/depth, or duration.

Primary frozen artifacts:

- candidate metrics SHA256:
  `9e4202425916692c50c48d2942a23bcf976869d741052bec74137476a0c71a4d`;
- candidate step3100 checkpoint SHA256:
  `aaa6fb08df96dd0bba4046173f5675fc183b11dd5fa82cb9c974eb2dd8544907`;
- matched comparison JSON SHA256:
  `813759efbabb5c133b4316ffcd25d5b2f4caaec0e6ed5fc6aa82d39a4ede97a4`;
- hardest same-board HTML SHA256:
  `ea2ed842fbf9a3c6389e1b6107905fe229a2cd7941e994068700509466cc90b1`;
- comparison directory:
  `/huyang2/double-loop/runs/p-fs3-002-comparison-20260806T222100Z-39b1927`.

The comparison HTML renders the two hardest shared boards per hard range with
control and codec loop1-to5 predictions, and the JSON contains every official,
activation, cost, same-board, provenance, and artifact-hash field used here.

## 7. Mechanism Boundary

Learning from the actual producer update is not sufficient when all K address
rows are collapsed to one nearly uniform V payload. The result closes this
single-code, permutation-equivariant codec, not learned producer communication
in general. A successor must preserve multiple address-conditioned pieces of
producer state or change the generic recurrent memory/update itself. It must
earn a new contract and board-level gate; it is not authorized as a nearby
payload-count or decoder-width sweep.
