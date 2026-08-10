# P-GDN3-019: Kernel-Minimum Persistent Raven Write Control

## 1. Metainfo

- Status: approved; strict CUDA contract pending
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

Pending.

## 9. Decision

Pending.
