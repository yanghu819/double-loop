# P-GDN3-007: Closed-Loop State-Feedback Update

## 1. Metainfo

- Status: discarded after clean matched completion; no rescue
- Date: 2026-08-07
- Branch: `codex/gdn3-state-feedback-update-20260807`
- Benchmark: official/full-diversity hard 9x9 Sudoku 51-64 blanks
- Compute: AIStation task-mode GPU1 only
- GPU UUID: `GPU-53e9f3b4-2966-65d3-6614-09c540921519`
- Seed: 52 only
- Parent mechanism: D256/L12 position-QK GDN3 plus native terminal FutureSeed
- Parent checkpoint:
  `/huyang2/double-loop/models/gdn3-position-qk-d256l12-s12000-20260806T131527Z-9f2ee8d/checkpoints/train_state_step003000.pt`
- Parent checkpoint SHA256:
  `6339c3cb2b5fc5230a581d6633716483e35ff8e4522f06a9d7aaf26512f023da`
- Parent source SHA: `9f2ee8d1738032bc5f09b55db0b81d507780b376`
- Frozen matched control:
  `p-fs3-001-terminal-s3100-20260806T200859Z-3e167b6`
- Formal source SHA:
  `20322df78757104ee6884ebc460277f9a0a2a935`
- Formal run:
  `p-gdn3-007-state-feedback-s3100-20260807T034337Z-20322df`
- Comparison:
  `/huyang2/double-loop/runs/p-gdn3-007-comparison-20260807T040800Z-20322df`

The implementation, checker fix, launcher, configuration, and preregistration
were pushed before formal execution. GPU execution used a clean detached
worktree at the exact source SHA above.

## 2. Evidence Boundary

Five active matched mechanisms now fail to improve hard exact:

1. orthogonal FutureSeed innovation residual;
2. shared producer-update compression;
3. address-local producer-update routing;
4. coherent erase/write scalar coupling;
5. a complete independent recurrent residual expert.

The first three alter transfer content, the fourth alters one aggregate update
degree of freedom, and the fifth adds a private address/update/state stack.
P-GDN3-006 proves that a second state can become active without changing the
main solver's board decisions; its parallel official-GDN2 expert more than
doubles elapsed time and still weakens the two hardest late-loop corrections.
This closes another transfer residual, scalar prior, or expert-width/count
experiment.

## 3. Mechanism Hypothesis

The remaining structural gap is closed-loop state interaction. In the current
position-QK path, official GDN2 reads its incoming KxV state, but current write
address, payload, erase, and write gates are projected from position/current
content before the kernel. The carried state does not directly condition the
decision that edits that state.

Hard iterative correction may require each token to compare its proposed edit
with the memory already stored at its current query. If this is the missing
mechanism, a state-derived controller should become board- and token-dependent,
alter all four write components, and improve late-loop full-board closure. If
it activates but exact and same-board correction remain unchanged, then
open-loop update control is not the local bottleneck and this direction is
closed without hidden-size or scale rescue.

## 4. Candidate

Before the unchanged pinned official `chunk_gdn2`, each receiving layer uses
its normalized position query to read the incoming state:

`R[b,t,h,v] = Q_unit[b,t,h,k] @ S_in[b,h,k,v]`.

One head-shared controller per layer maps V32 to hidden16 with SiLU, then to
four zero-initialized residuals: K32 write-address, V32 payload, K32 erase-logit,
and V32 write-logit. The effective official-kernel inputs are:

- `K' = K + dK`;
- `V' = V + dV`;
- `b' = sigmoid(b_raw + db)`;
- `w' = sigmoid(w_raw + dw)`.

The final projection is exactly zero, so parent output and recurrent states are
bit-exact at migration, including nonzero incoming states. The controller is
shared across heads and therefore head-permutation equivariant. It adds exactly
2,560 parameters per layer and 30,720 total, no recurrent state, no extra scan,
no second FLA layer, and no Sudoku-specific logic. Layer0 has no incoming
FutureSeed and remains exactly inactive; layers1-11 are the registered receiving
paths.

## 5. Frozen Configuration

- D256/L12/H8/K32/V32, channel multiplier4
- position-QK addressing, random traversal
- native terminal FutureSeed, loop5, equal CE at every loop
- 12 pinned official-FLA GDN2/Triton layers
- head-shared V32->16->128 state-feedback controller per layer
- official full-diversity curriculum and exact step3000 parent
- BF16, effective batch128, optimizer/RNG/data order exact resume
- LR0.0015, weight decay0.001, seed52
- candidate-only step3000-to3100 continuation

The frozen terminal continuation is not rerun. There is no controller hidden
size, output scale, layer subset, feedback target, seed, optimizer, loss,
duration, width, or depth follow-up.

## 6. Launch Gates

The strict GPU1 CUDA contract must prove:

1. exactly CUDA index0 and the registered UUID;
2. pinned FLA SHA `9c8e42e762fce087c27b673af4922795d9edb85e`;
3. 12 exact official `GatedDeltaNet2` layers and 12
   `ChunkGDN2FunctionBackward` terminal graphs;
4. exact 30,720-parameter migration and no unexpected parameter;
5. bit-exact full-model output and all 12 terminal states at zero output
   projection;
6. per-layer exact output/state identity under synthetic nonzero incoming
   state;
7. finite nonzero gradient in all 12 zero output projections, then finite
   nonzero gradient in all 12 input and output projections after one synthetic
   opening step;
8. exactly 11 active receiving layers in a full native-FutureSeed pass;
9. finite nonzero state-read RMS, between-board variation, token variation,
   and K/V/b/w relative changes after opening;
10. head-permutation equivariance and a changed controller output when incoming
    states are shuffled across boards;
11. exact-resume optimizer/RNG/data-order migration and a complete step3001
    checkpoint plus metrics JSON;
12. no fallback, NaN, OOM, source drift, data drift, or hidden CPU model path.

A failed contract or production probe closes this implementation. It does not
authorize a smaller controller or target subset.

Contract R1 on pushed SHA `dc64679987cea801aec6b62760e9f0f4da95dc73`
reached the final synthetic equivariance check after exercising the model, then
the checker called the private update helper outside the formal CUDA BF16
autocast context and raised a BF16-input/FP32-weight dtype error. No registered
identity, gradient, official-kernel, activation, science, or cost assertion
failed. The process exited naturally, GPU allocations cleared, and the exact
log plus non-science abort remain under
`/huyang2/double-loop/artifacts/launch/p-gdn3-007/`. R2 changes only those three
synthetic checker calls to use the same autocast context as formal forward; the
mechanism, parent, configuration, and every gate remain frozen.

Contract R2 on pushed SHA `20322df` exits status0. Its log SHA256 is
`6ab99ab9f146d223c71e0eff5c9a562765c00a0296d2970e687b6bdc66f8f514`.
It verifies the exact 30,720-parameter delta, full output and all 12 terminal
states at bit identity, 12 official GDN2/`ChunkGDN2FunctionBackward` paths,
two-stage gradients, zero head-permutation error, state-shuffle dependency,
and exactly 11 receiving paths. The exact-resume step3001 probe also exits
status0; its metrics/checkpoint SHA256 values are
`0a45bd55ed18d858a449c7fc7fb34a6bc9364861e5b37d9eaeb3a7979b1c8186` and
`bef43f2da7074401cc5df7eb2b88c36f7e1ad5508cec1336b0808bee696e3732`.
The probe opens finite state reads and all K/V/b/w edits, so the formal
candidate was authorized without changing the registered mechanism or gate.

## 7. Science And Cost Gates

At step3100, activation requires all of:

- mean enabled receiving-layer fraction exactly `11/12`;
- state-read RMS and between-board std finite and nonzero;
- controller residual relative RMS and token std `>=1e-4` and finite;
- K, V, erase, and write relative changes each finite and nonzero;
- all 11 receiving layers active, no fallback.

Quality passes by exactly one of:

1. hard51-64 macro loop5 exact improves by at least `+0.02`, with every hard
   range losing no more than `0.01` blank accuracy; or
2. mixed loop5 exact improves by at least `+0.03`, official61-64 does not
   regress, and same-board loop3-to5 wrong-cell correction is stronger.

Because this adds no second recurrent kernel or recurrent state, independent
warmed elapsed time and peak allocated memory overhead must each remain below
`25%` versus the frozen control. Timing instability, OOM, or fallback kills
the run regardless of quality.

Any activation, quality, or cost miss discards this exact state-feedback
controller. Do not rescue controller hidden size, output scale, target subset,
layer sharing, seed, LR, loss, batch, model width/depth, or duration.

## 8. Required Readout

Report and archive:

- mixed and official51-55/56-60/61-64 loop1-5 exact/blank/wrong cells;
- train CE and same-board loop3-to5 correction counts;
- state-read, board/token variation, K/V/b/w change, and controller-weight
  activation metrics;
- independently warmed throughput, peak allocation/reservation, and timing
  stability;
- config, source, parent, checkpoint, metrics, log, and comparison hashes;
- same-board loop1-5 visualization using the frozen control case IDs.

## 9. Decision

Discarded. The formal candidate exits status0 on the exact pushed SHA with the
registered GPU, source, parent, optimizer/RNG/data order, and official-FLA
kernel contract. There is no NaN, OOM, fallback, source drift, or data drift.

### Activation

All 11 receiving paths are active. At loop5, enabled fraction is `11/12`,
state-read RMS/board std is `0.725926/0.013616`, hidden RMS is `0.233113`, and
controller residual relative RMS/token std is `0.018716/0.007774`. K/V/erase/
write relative changes are `0.038044/0.009582/0.005715/0.006785`; controller
input/output weight RMS is `0.093749/0.013349`. The activation gate passes and
a dead controller cannot explain the quality result.

### Exact And Blank Metrics

| Range | Arm | Exact loops1-5 | Blank loops1-5 |
|---|---|---|---|
| 51-55 | control | 0/0/0.001953/0.001953/0.001953 | 0.532345/0.566677/0.573945/0.573623/0.573766 |
| 51-55 | candidate | 0/0/0.001953/0.001953/0.001953 | 0.530591/0.566856/0.572334/0.574160/0.573623 |
| 56-60 | control | 0/0/0/0/0 | 0.473182/0.498061/0.501699/0.503140/0.503861 |
| 56-60 | candidate | 0/0/0/0/0 | 0.472805/0.498370/0.503140/0.504204/0.504478 |
| 61-64 | control | 0/0/0/0/0 | 0.504319/0.578523/0.589207/0.591954/0.591923 |
| 61-64 | candidate | 0/0/0/0/0 | 0.506486/0.577455/0.588444/0.590183/0.590489 |
| mixed | control | 0.017578/0.023438/0.025391/0.025391/0.025391 | 0.515087/0.536904/0.544841/0.545540/0.545610 |
| mixed | candidate | 0.017578/0.023438/0.023438/0.023438/0.023438 | 0.508828/0.536939/0.543862/0.544876/0.544456 |

Hard51-64 macro loop5 exact stays `0.000651`; delta is exactly0 versus the
registered `+0.02` requirement. Mixed loop5 exact regresses
`0.025391->0.023438`, delta `-0.001953` versus the alternate `+0.03`
requirement. Official51-55/56-60/61-64 loop5 blank deltas are
`-0.000143/+0.000618/-0.001435`; therefore the hardest range is also
regressive. Train CE improves only `0.858617->0.855740`.

### Same-Board Dynamics

Across all 256 matched boards per range, control/candidate mean wrong-cell
trajectories are:

- 51-55: `25.742/24.348/23.918/23.855/23.934` versus
  `25.762/24.289/23.922/23.906/23.910`; loop3-to5 correction
  `-0.0156->+0.0117`;
- 56-60: `29.695/28.195/28.082/28.043/28.016` versus
  `29.711/28.219/27.961/27.938/27.898`; loop3-to5 correction
  `0.0664->0.0625`;
- 61-64: `31.855/27.203/26.613/26.508/26.430` versus
  `31.727/27.293/26.668/26.473/26.488`; loop3-to5 correction
  `0.1836->0.1797`.

The late correction is weaker in both 56-60 and 61-64, so the alternate route
fails independently of mixed exact.

### Cost

The 100-step continuation takes `825.970s` control versus `995.478s`
candidate; throughput falls `15.497->12.858` effective boards/s and elapsed
overhead is `+20.52%`. Peak allocated memory is
`13186.4->14597.8MiB` (`+10.70%`) and reserved memory is
`14288->15738MiB` (`+10.15%`). Both registered 25% cost ceilings pass. The
failure is scientific, not a cost-only rejection.

### Provenance And Artifacts

- formal metrics SHA256:
  `66084b6fb83dfe0d6dffc2a6828a10564f222535fd009b03a5dd05feb5794329`;
- formal checkpoint SHA256:
  `d1fb913a03abf887bc88899c459b0ff6bdb7c9e128195f6de826df8bc0d6ef84`;
- formal config/log SHA256:
  `8cebdb1b8bbd441ec92a5b5136dd3e14884a9475914b76b627e84f06eece58b3` /
  `ba4ccae442582b055600edbd6aabbec6a41e89b02f8f7ef60fedeec7000ca3fb`;
- source snapshot SHA256:
  `236bec0a682ba5484201205f867e66993857e06f27287fc45a0e1d95d2606c61`;
- comparison JSON/HTML SHA256:
  `90c3bdcd5b077694feb4cf09b674d27a929fb8648f0e2f5036cf7f63da8beaef` /
  `4f0a06d83e5c3c75fbc83d7ac8361cc462f1569a05ef38df62f368aaa21c82bf`;
- repository copy:
  `research/reports/visualizations/gdn3-state-feedback-update-20260807/`.

The visualization contains loop1-5 predictions for the same hardest boards in
the frozen control and candidate. This closes the exact state-feedback
controller. Do not rescue controller hidden size, output scale, target subset,
layer sharing, seed, LR, loss, batch, model width/depth, or duration. The next
high-information candidate must alter the scalable recurrent state transition
itself rather than append another zero-init residual around unchanged GDN2
edits.
