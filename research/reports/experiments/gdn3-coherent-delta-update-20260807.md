# P-GDN3-005: Coherent Delta Erase/Write Update

## 1. Metainfo

- Status: discarded after the sole matched step3000-to3100 candidate
- Date: 2026-08-07
- Branch: `codex/gdn3-coherent-delta-20260807`
- Benchmark: official/full-diversity hard 9x9 Sudoku
- Compute: AIStation task-mode GPU1 only
- GPU UUID: `GPU-53e9f3b4-2966-65d3-6614-09c540921519`
- Seed: 52 only
- Parent mechanism: D256/L12 position-QK GDN3 plus native FutureSeed
- Parent checkpoint:
  `/huyang2/double-loop/models/gdn3-position-qk-d256l12-s12000-20260806T131527Z-9f2ee8d/checkpoints/train_state_step003000.pt`
- Parent checkpoint SHA256:
  `6339c3cb2b5fc5230a581d6633716483e35ff8e4522f06a9d7aaf26512f023da`
- Parent source SHA: `9f2ee8d1738032bc5f09b55db0b81d507780b376`
- Frozen matched control:
  `p-fs3-001-terminal-s3100-20260806T200859Z-3e167b6`

- Formal source SHA: `e21bccf55313a6792825a3c8d1afec9434832cdb`
- Formal run: `p-gdn3-005-coherent-delta-s3100-20260807T002142Z-e21bccf`
- Formal source worktree:
  `/huyang2/double-loop/worktrees/p-gdn3-005-e21bccf`

The formal source was pushed before launch and the AIStation run used a clean
detached worktree at that exact SHA.

## 2. Mechanism Boundary

P-FS3-001 through P-FS3-003 all learned active FutureSeed content residuals.
The address-local variant even preserved every KxV payload and improved the
61-64 blank tail, yet none changed hard macro or mixed exact, and all missed
the cost gate. This closes nearby transfer-side routing as the next step.

P-GDN3-004 supplies the complementary clue. Stable position-QK addresses
create genuine late-loop correction and nonzero hard exact, but the run stalls
below its step3000 opening gate. The remaining failure can lie inside the
generic recurrent state edit rather than in how a terminal state is passed to
the next layer.

## 3. Hypothesis

The official GDN2 layer predicts an erase gate `b` over K coordinates and a
write gate `w` over V coordinates independently. The underlying delta update,
however, is one state-edit event: erase the old value addressed by K, then
write the new V target. If their aggregate strengths drift apart, the update
can under-erase stale state or under-write its replacement. This predicts slow
soft correction without enough stable board-level fixed points to open exact.

The falsifiable prediction is that a small learned positive coupling should
reduce the erase/write gate gap and improve hard exact or mixed exact. If the
coupling remains inactive, learns a negative mean, fails to contract the gap,
or only changes soft accuracy, the hypothesis fails.

## 4. Candidate

For every token and head, after the official projections but before the
unchanged official kernel, define:

```text
b0 = sigmoid(b_raw)
w0 = sigmoid(w_raw)
c  = 0.5 * (mean_K(b0) + mean_V(w0))
m  = tanh(a[layer, head])
b1 = clamp(b0 + m * (c - b0), 0, 1)
w1 = clamp(w0 + m * (c - w0), 0, 1)
```

Each of 12 layers has eight scalar `a` parameters, for exactly 96 added
parameters. They are zero initialized and excluded from weight decay. At
`m=0`, every gate, output, and terminal recurrent state must be bit-exact to
the parent, with and without an incoming recurrent state. The candidate keeps
the pinned official `chunk_gdn2`, position-QK addressing, full KxV state,
native terminal FutureSeed, one forward traversal, and linear complexity.

This is not gate-temperature tuning. There is no fixed scale, auxiliary loss,
Sudoku rule, search, repair, extra scan, new state slot, or second seed.

## 5. Frozen Configuration

- D256/L12/H8/K32/V32, channel multiplier4
- 12 pinned official-FLA `GatedDeltaNet2` layers and Triton short convolution
- position-QK addressing, random traversal
- native terminal FutureSeed, loop5, all-loop CE
- official full-diversity curriculum and exact step3000 parent
- BF16, effective batch128, optimizer/RNG/data order exact resume
- LR0.0015, weight decay0.001, seed52
- candidate-only step3000-to3100 continuation

The frozen control is the existing exact terminal continuation to step3100;
it is not rerun.

## 6. Launch Gates

The strict CUDA contract must prove:

1. exactly CUDA index0 and the registered UUID;
2. pinned FLA SHA `9c8e42e762fce087c27b673af4922795d9edb85e`;
3. 12 exact official `GatedDeltaNet2` layers and
   `ChunkGDN2FunctionBackward`;
4. exactly 96 inserted parameters and the exact migration set;
5. bit-exact full output and all 12 terminal states at zero initialization;
6. bit-exact per-layer output and terminal state with nonzero incoming state;
7. finite nonzero gradient for every layer's coupling parameter;
8. gate range `[0,1]` and strict gap contraction under a positive opened mix.

Only after the contract passes may a single exact step3000-to3001 full-stack
probe run. It must accept the declared migration, preserve optimizer/RNG/data
order, save a complete checkpoint and metrics JSON, use the official CUDA
path, and show a finite opened coupling after the optimizer step. Any contract
or integrity miss blocks the formal run.

## 7. Science And Cost Gates

Activation at step3100 must satisfy all of:

- mean coupling `m > 0`;
- mean absolute coupling `>= 1e-4`;
- finite nonzero coupling gradients;
- post/pre erase-write gap RMS `< 1`.

Quality passes by exactly one of:

1. hard51-64 macro loop5 exact improves by at least `+0.02`, with each hard
   blank range losing no more than `0.01` blank accuracy; or
2. mixed loop5 exact improves by at least `+0.03`, official61-64 does not
   regress, and same-board loop3-to5 wrong-cell correction is stronger.

Elapsed overhead and peak allocated memory must each remain below `10%` versus
the frozen control. Timing must use the existing independent warmup protocol.
Any miss discards the mechanism without coupling scale, seed, LR, loss, batch,
width, depth, duration, or nearby update rescue.

## 8. Commands

Strict contract:

```bash
CUDA_VISIBLE_DEVICES=0 \
XDG_CACHE_HOME=/huyang2/double-loop/.cache \
TRITON_CACHE_DIR=/huyang2/double-loop/.cache/triton \
TORCHINDUCTOR_CACHE_DIR=/huyang2/double-loop/.cache/torchinductor \
TORCH_EXTENSIONS_DIR=/huyang2/double-loop/.cache/torch_extensions \
TMPDIR=/huyang2/double-loop/.cache/tmp \
FLA_DISABLE_BACKEND_DISPATCH=1 FLA_CONV_BACKEND=triton \
/opt/conda/bin/python \
experiments/rwkv_fs_sudoku/check_gdn3_coherent_delta_cuda.py
```

Exact full-stack probe:

```bash
CUDA_VISIBLE_DEVICES=0 GDN3_FULL_STACK_PROBE=1 \
scripts/run_gdn3_coherent_delta_arm.sh
```

The formal command is the same launcher without
`GDN3_FULL_STACK_PROBE=1`.

## 9. Execution Evidence

The strict CUDA contract completed with status0 at:

`/huyang2/double-loop/artifacts/launch/p-gdn3-005/contract-e21bccf.log`

It verified the pinned FLA source SHA, 12/12 official `GatedDeltaNet2` layers,
`ChunkGDN2FunctionBackward`, exact 96-parameter insertion, exact zero-init
model-output and terminal-state identity, exact parity with nonzero incoming
states, finite nonzero gradients in all 12 layers, and a `0.75` gap ratio when
the test mix was manually opened to `0.25`.

The exact-resume full-stack probe also completed with status0:

- run: `p-gdn3-005-coherent-delta-probe-s3001-20260807T001746Z-e21bccf`
- metrics SHA256:
  `f18b7f11f4d16b69f13cfca3b681efc73d93917d6973b07956b27e5d5c217ec1`
- checkpoint SHA256:
  `d998b6632bf09aaa33c50cfbcafcaa5de5a73aee5ba73f4294482848ee5a262f`

After one resumed step, all 12 layers had finite nonzero mixes with mean
absolute value `0.00149647`, and the probe's evaluation gap ratio was below
one. The probe established migration and activation only; it was not used for
the science decision.

The formal continuation exited cleanly with status0. It used only CUDA index0
and the registered GPU UUID, remained at the clean source SHA, and produced no
NaN, OOM, fallback, source drift, data drift, or checkpoint-integrity error.

Formal artifact hashes:

| artifact | SHA256 |
|---|---:|
| metrics JSON | `2fb908dc4d76e9e545a672aa17559c7bd53964fb24eb10e829b7470c138fde82` |
| step3100 checkpoint | `52b15d67bc83d6d918b70a045ae5f7b98029b1fdf4bff2a004cd6154238656b3` |
| config | `6688b0ba56977d54013f69c588a93a9967981ebe96f051b3d7fd94dcabca6b5e` |
| run log | `5be9318969a4a1870210ea5b501421a76a46c29fdb07f9fabe4025347d8855cf` |
| source snapshot | `4571ce5504cae52ef158ff8983b2842a5a321da6a5bbe5fa79e6536749e64d88` |

## 10. Matched Results

The mechanism activated but falsified its own directional prediction. The
train mean/absolute mix was `-0.0024063/0.0121813`; evaluation loop5 mean
absolute mix was `0.0122610`. The train and loop5 post/pre gate-gap ratios were
`1.0019307` and `1.0019815`, respectively. The learned signed coupling
slightly *expanded* the erase/write gap instead of contracting it.

Exact closure did not change:

| metric | frozen control | candidate | delta |
|---|---:|---:|---:|
| hard51-64 macro loop5 exact | 0.000651 | 0.000651 | 0.000000 |
| mixed loop5 exact | 0.025391 | 0.025391 | 0.000000 |
| train CE | 0.858617 | 0.855293 | -0.003324 |
| mixed loop5 blank | 0.545610 | 0.544806 | -0.000804 |

Mixed exact across loops1-5 was identical for both arms:
`0.017578/0.023438/0.025391/0.025391/0.025391`.

Official fixed-board loop metrics:

| range | arm | loop1 exact/blank | loop2 exact/blank | loop3 exact/blank | loop4 exact/blank | loop5 exact/blank |
|---|---|---:|---:|---:|---:|---:|
| 51-55 | control | 0/0.532345 | 0/0.566677 | 0.001953/0.573945 | 0.001953/0.573623 | 0.001953/0.573766 |
| 51-55 | candidate | 0/0.531772 | 0/0.566248 | 0.001953/0.571510 | 0.001953/0.572692 | 0.001953/0.573157 |
| 56-60 | control | 0/0.473182 | 0/0.498061 | 0/0.501699 | 0/0.503140 | 0/0.503861 |
| 56-60 | candidate | 0/0.475721 | 0/0.496723 | 0/0.502179 | 0/0.502488 | 0/0.502968 |
| 61-64 | control | 0/0.504319 | 0/0.578523 | 0/0.589207 | 0/0.591954 | 0/0.591923 |
| 61-64 | candidate | 0/0.505998 | 0/0.578401 | 0/0.589542 | 0/0.591527 | 0/0.591557 |

The candidate's loop5 blank deltas for 51-55/56-60/61-64 were
`-0.000609/-0.000892/-0.000366`. Thus every official range regressed softly,
although all regressions stayed within the secondary per-range tolerance.

All 256 case IDs and data hashes matched in every range. The same-board mean
wrong-cell comparison was:

| range | control loop1/2/3/4/5 | candidate loop1/2/3/4/5 | loop3-to5 control/candidate | loop5 better/equal/worse | late stronger/equal/weaker |
|---|---|---|---:|---:|---:|
| 51-55 | 25.742/24.348/23.918/23.855/23.934 | 25.730/24.367/23.949/23.941/23.938 | -0.0156/0.0117 | 92/78/86 | 73/109/74 |
| 56-60 | 29.695/28.195/28.082/28.043/28.016 | 29.680/28.211/27.941/27.883/27.883 | 0.0664/0.0586 | 95/86/75 | 73/108/75 |
| 61-64 | 31.855/27.203/26.613/26.508/26.430 | 31.859/27.270/26.430/26.324/26.320 | 0.1836/0.1094 | 97/83/76 | 79/89/88 |

The candidate had lower loop5 mean wrong cells in 56-64, but its loop3-to5
correction was weaker in both ranges. This does not satisfy the alternate
route, which also required a mixed-exact gain.

Cost also failed:

| metric | frozen control | candidate | change |
|---|---:|---:|---:|
| train seconds | 825.9695 | 918.2878 | +11.177% |
| effective boards/s | 15.4969 | 13.9390 | -10.053% |
| peak allocated MiB | 13186.424 | 15605.996 | +18.349% |
| peak reserved MiB | 14288 | 16764 | +17.329% |
| parameters | 11,485,760 | 11,485,856 | +96 |

The paired comparison and same-condition loop visualization are archived at:

`/huyang2/double-loop/runs/p-gdn3-005-comparison-20260807T004600Z-e21bccf`

Its JSON/HTML/builder/manifest SHA256 values are, respectively:

- `fd8655a9b87cd6953dfd0dd3b34de1f11cae2a50d0278cf6256eea3083b8d741`
- `e3a6d57500c14f9fd91a8084d4e9cef517b1851b6b02968a1d37359a041a7758`
- `03bee7d708cc3573308d1a77b6aed5837b9180bc49a803692abd714537220e9a`
- `68d0e61d22eda72b0c37d742ec55cdbfd92b3dfc78054c7ae666d1bd7ef04a3e`

## 11. Decision

Discard P-GDN3-005. The activation gate passed, but the learned mix was signed
negative on average and the observed gap ratios exceeded one, directly
falsifying the registered coherence mechanism. Hard macro and mixed exact were
unchanged, the alternate same-board condition failed, elapsed overhead exceeded
10%, and peak allocation exceeded 10%.

No mix constraint, scale, seed, LR, loss, batch, width, depth, duration, or
nearby gate rescue is authorized. The slight CE gain and local 56-64 wrong-cell
reduction are insufficient because neither produced additional board-level
closure. The next mechanism must increase generic learned recurrent
memory/address/state capacity rather than impose another scalar aggregate gate
prior.

## 12. Submission Record

Not applicable. This is a mechanism experiment, not a benchmark submission.
