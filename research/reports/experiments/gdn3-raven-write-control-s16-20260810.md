# P-GDN3-019: Kernel-Minimum Persistent Raven Write Control

## 1. Metainfo

- Status: discarded after complete matched endpoint
- Date: 2026-08-10
- Branch: `codex/gdn3-raven-write-control-s16-20260810`
- Benchmark: official/full-diversity hard 9x9 Sudoku 51-64 blanks
- Seed: 52 only
- Parent checkpoint: `/huyang2/double-loop/models/gdn3-position-qk-d256l12-s12000-20260806T131527Z-9f2ee8d/checkpoints/train_state_step003000.pt`
- Parent SHA256: `6339c3cb2b5fc5230a581d6633716483e35ff8e4522f06a9d7aaf26512f023da`
- Frozen control: `p-fs3-001-terminal-s3100-20260806T200859Z-3e167b6`

## 2. Hypothesis

P-GDN3-018 posed a new Raven/GDN composition question but its fixed S8
controller could not compile in pinned official chunk GSA. The failure says
nothing about persistent sparse retrieval because no model output or score
was produced. P-GDN3-019 asks the same question at the official kernel's
minimum valid slot dimension, S16. This is a production constraint, not a
slot search: S16 is the only authorized setting and there is no S24/S32 arm.

The falsifiable claim remains that Raven's persistent sparse state can retain
and retrieve selected long-lived content while the stronger position-QK GDN2
data plane commits that content into its live V state. If every controller,
cross-layer state path and V adapter activates but exact closure stays flat,
the two-plane Raven/GDN hypothesis is closed at this parent and budget.

## 3. Fixed Mechanism

Each of 12 D256 position-QK GDN2 blocks projects hidden content to D64 and
runs one pinned official Raven with H4/K16/V16, S16 and top1. The preceding
block's normalized packed Raven terminal state seeds the next controller.
A zero-initialized D64-to-V256 adapter injects Raven retrieval into the main
GDN2 V payload before the unchanged single official GDN2 chunk call.

Main Q/K, decay, erase/write gates, output gate, K32xV32 state, native
terminal FutureSeed, optimizer, RNG, data order, BF16, batch and loss remain
parent-exact. Additions are fixed at 643,344 parameters and 24,576 Raven state
values across 12 blocks. There is no kernel edit, padding, backend fallback,
slot/top-k/width sweep or task-specific logic.

## 4. Strict CUDA Contract

Exact pushed source in a clean detached worktree must prove one admitted GPU
at CUDA index0, pinned FLA SHA
`9c8e42e762fce087c27b673af4922795d9edb85e`, 12 official GDN2 and 12
official Raven chunk paths, S16/top1 in every controller, and exact
parameter/state deltas. Zero adapters must preserve full output and every
main terminal state exactly, including nonzero incoming main states.

All 12 adapters need finite nonzero first-stage gradients. After synthetic
adapter opening, all Raven Q/K/V/router paths need finite nonzero gradients.
Controller-state shuffling must alter the write path; packed states must
round-trip and remain finite. Any provenance, identity, graph, gradient,
state-dependency, production-fit or fallback miss closes P-GDN3-019.

## 5. Exact Step3001 Probe

Resume the exact parent for one step with only the registered semantic content
upgrade. Require 12 active controllers/adapters, 11 incoming state paths,
incoming RMS at least `1e-4`, V residual relative RMS in `[1e-4,0.5]`, finite
board/token variation, normalized slot entropy at least `0.50`, maximum slot
mass share below `0.80`, and main terminal RMS at most `4x` the parent. The
probe is migration and activation evidence only.

## 6. Matched Decision

Only after contract and probe pass, run one candidate-only exact
step3000-to3100 continuation. Do not repeat the frozen control.

- Primary: hard51-64 macro loop5 exact improves by at least `+0.02`; each
  official hard-range blank regression is no worse than `0.01`.
- Alternate: mixed loop5 exact improves by at least `+0.03`, 61-64 does not
  regress, and same-board loop3-to5 wrong-cell correction is stronger.
- Stability: slot entropy is at least `0.50`, maximum mass share is below
  `0.80`, V residual relative RMS is at most `0.5`, and main state RMS is
  within `4x`.
- Cost: independent warmed elapsed overhead is below `60%`; peak allocated
  memory overhead is below `30%` versus the frozen control.

Any miss discards P-GDN3-019. No controller width, slot/top-k, injection
target/scale, gate, initialization, seed, LR, loss, batch, main width/depth or
duration rescue is authorized.

## 7. Commands

The exact contract, probe and formal commands are archived with the admitted
task launch logs. All runs set `CUDA_VISIBLE_DEVICES=0`, bind the exact GPU
UUID and use `/opt/conda/bin/python` with project-local persistent caches.

## 8. Results

Exact pushed source `9abc2929256657e06399b62fe62cdd9c6ec19d9f`
ran from a clean detached worktree on one task-mode A100-SXM4-40GB at CUDA
index0, UUID `GPU-93aad99c-9d1c-f2fb-1f10-fed39dde185c`. The strict contract
completed with status0. It proved 12 official GDN2 paths, 12 official Raven
paths at S16/top1, their official backward functions, exact zero-adapter parent
output and all 12 main-state identities including nonzero incoming states,
exact +643,344 parameters and +24,576 controller-state values, finite gradients
through every adapter and Raven projection, 11 state-transport dependencies,
packed-state round trips, and no fallback. Contract-log SHA256 is
`76d8c6c7ed6969a2e7099d6b4fbb975b2af4ba0f3c8f47bb48502b933dd04806`.

The exact step3001 probe also completed with status0. Loop5 V residual relative
RMS was `0.009200`, normalized slot entropy `0.9328`, maximum slot mass share
`0.1493`, and all 11 receiving paths were live. Probe metrics/checkpoint SHA256
values are
`8e97469bff444913e8aa79436408f3c6b9e58a96bb3aa227d604b2bac9e14688`
and
`2b8ab976ab13c6ec3a00f0838bb47585fc3a952184be53f6519cb154b68d824e`.
This passes migration, activation and production-fit gates; its tiny evaluation
is not used as science evidence.

The formal run
`p-gdn3-019-raven-write-s16-s3100-20260810T100445Z-9abc292` exited naturally
with status0. At loop5 all controllers remain stable and nontrivial: V residual
relative RMS is `0.066201` (`0.027680..0.115102` across layers), board/token
variation is `0.004768/0.019113`, minimum normalized slot entropy is `0.670635`,
maximum slot mass share is `0.370819`, all 11 incoming paths remain live, and
maximum main-state RMS is `11.8991`, inside the fixed `4x` bound.

Candidate loop1-to5 fixed-condition readout:

| Range | Exact L1/L2/L3/L4/L5 | Blank L1/L2/L3/L4/L5 | Mean wrong cells L1/L2/L3/L4/L5 |
|---|---|---|---|
| mixed | `0.015625/0.023438/0.023438/0.023438/0.023438` | `0.514947/0.535506/0.543408/0.543512/0.543303` | not a single fixed blank range |
| 51-55 | `0/0/0.003906/0.003906/0.003906` | `0.537214/0.567000/0.573873/0.573730/0.573981` | `25.65/24.21/24.06/23.99/23.94` |
| 56-60 | `0/0/0/0/0` | `0.473731/0.497581/0.504272/0.504959/0.505336` | `29.72/28.49/28.05/27.96/27.93` |
| 61-64 | `0/0/0/0/0` | `0.502610/0.576234/0.588383/0.590244/0.590031` | `31.88/27.32/26.71/26.59/26.59` |

Activation does not translate into closure. The frozen-control to candidate
official loop5 exact/blank results are:

- 51-55: `0.001953/0.573766 -> 0.003906/0.573981`;
- 56-60: `0/0.503861 -> 0/0.505336`;
- 61-64: `0/0.591923 -> 0/0.590031`.

Hard51-64 macro exact therefore moves only `0.000651 -> 0.001302`
(`+0.000651`, far below `+0.02`). Mixed loop5 exact regresses
`0.025391 -> 0.023438`; train CE also worsens `0.858617 -> 0.862443`.
Across the identical 256-board case banks, mean wrong cells over loops1-5 are
`25.65/24.21/24.06/23.99/23.94`,
`29.72/28.49/28.05/27.96/27.93`, and
`31.88/27.32/26.71/26.59/26.59` for the three hard ranges. Loop3-to5
correction is stronger than control on 51-55 and 56-60, but weaker on 61-64
(`0.1172` versus `0.1836` wrong cells). The alternate route fails independently
because mixed exact regresses and the hardest range is not non-regressive.

The controller adds `5.60%` parameters but recurrent execution is expensive.
Fresh matched 100-step throughput falls `15.497 -> 7.451` effective boards/s;
elapsed overhead is `+107.99%`, above the fixed `60%` ceiling. Peak allocated
and reserved memory rise `+17.54/+17.72%` and remain within the `30%` allocation
ceiling. No separate timing-CV benchmark can change the binding quality and
elapsed-cost decision.

Formal metrics/checkpoint/config/log/source-snapshot SHA256 values are
`641ba615dd9ab3eda68009e97ee3f5395def50fbbff3fc86bbbb1a3f466d27e5`,
`b5c4db3364aadffb2b2827107a2b317d1265ea829539ce1586680f20c3f25553`,
`6c372f43f98737900e0bd90c1332111c3abd82437c286970bc33c667f79582b4`,
`7a0e224cd6ebb6def165dc68f166de7d23692aa12f44ad7cd6b4b77320fb1274`,
and
`9e3fc4438dec4dd1b07a4ea26ce76f5f9d06c925195c3dc4dea2fda578cd3cd1`.
Machine comparison, abort record and hardest-shared-board loop visualizations
are archived at
`/huyang2/double-loop/runs/p-gdn3-019-comparison-20260810T105107Z-9abc292`;
comparison JSON SHA256 is
`21ca61f8bd82d4f58957a7873744d2de63e992d44a4ceda56672a1f1895af3b2`.

## 9. Decision

Discard P-GDN3-019. The experiment establishes that a real pinned-official
Raven sparse memory can be attached to GDN as a persistent cross-layer write
control plane without migration ambiguity, dead paths, slot collapse or state
explosion. It does not establish a useful hybrid: the hard improvement is one
board-equivalent in one range, mixed exact regresses, hardest-range late
correction weakens, and elapsed cost more than doubles.

This closes the registered compact persistent Raven-write family at the fixed
parent and 100-step budget. Do not run S24/S32, top-k, controller-width,
injection-target/scale, gate-initialization, seed, LR, loss, batch, main-width,
depth or duration rescues. No successor is launched automatically; a future
experiment requires a different mechanism hypothesis and a new explicit
decision.
