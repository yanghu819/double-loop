# FutureSeed2 Selective State Gate

## 1. Metainfo

- Plan: `P-FS2-002`
- Status: completed, hypothesis rejected
- Date: `2026-07-28`
- Machine: AIStation `GPU1` only
- Branch: `codex/futureseed2-selective-gate-20260728`
- Baseline tag: `baseline/futureseed-four-carrier-20260727`
- Canonical model code: `42102bd65a28d60bde6b09ef93343692740582a1`
- Parent checkpoint:
  `/huyang2/double-loop/models/gdn2-futureseed-d192l10-s12000-20260726T131301Z-42102bd/checkpoints/train_state_step009000.pt`
- Parent SHA256 prefix: `606caf52`

## 2. Hypothesis

The frozen FutureSeed1 baseline uses one scalar gate per layer and head for the
entire imported `K x V` GDN2 state. Useful and harmful memory entries therefore
cross the layer boundary together. The rejected `P-FS2-001` experiment showed
that rotating or mixing the state basis damages late-loop correction.

FutureSeed2 should instead preserve the learned state coordinates and select
which state entries cross the boundary:

```text
gate(l,h,k,v) = sigmoid(base_logit(l,h) + delta(l,h,k,v))
seed(l,h,k,v) = normalized_state(l-1,h,k,v) * gate(l,h,k,v)
```

`delta` is initialized to zero, so the model is exactly FutureSeed1 before the
first optimizer step. This asks one narrow question: is indiscriminate seed
transport, rather than state-space mismatch, limiting the 56-64 blank tail?

## 3. Configuration

- Strict official FLA `GatedDeltaNet2`, chunk recurrence and Triton convolution.
- D192/L10/H6/K32/V32, expand-v 1, loop5, CE on every loop.
- BF16, microbatch 32, accumulation 4, effective batch 128, seed 52.
- Official full-diversity Sudoku curriculum and 512 fixed test boards per
  51-55, 56-60, and 61-64 blank range.
- FutureSeed scale 1, fixed update, unit-RMS normalization.
- Only new parameters: `(L-1) x H x K x V = 55,296` zero-initialized gate
  deltas, excluded from weight decay.
- No noise, feedback, scratch memory, margin loss, repair, search, selector,
  oracle inference, or task-specific rule.

The already completed matched FutureSeed1 identity continuation from step9000
to step9100 is reused as control:

`/huyang2/double-loop/runs/futureseed2-identity-s9100-20260728T1450Z-57455e4`

## 4. Environment

- AIStation row: `GPU1`
- Required device: `CUDA_VISIBLE_DEVICES=0`
- Persistent root: `/huyang2/double-loop`
- Python: `/opt/conda/bin/python`
- Cache, models, runs, and artifacts remain under the persistent root.
- CPU model smoke and GPU2 are forbidden.

## 5. Commands And Decision Rules

Integrity gates:

1. CUDA zero-delta gate must be bit-exact to FutureSeed1.
2. Activated gate must equal a direct Torch formula and produce finite nonzero
   state/base-logit/delta gradients.
3. Full stack must load the frozen checkpoint with exactly one expected missing
   parameter: `reasoner.future_seed_selector.gate_delta`.
4. Official FLA class, chunk backward, Triton convolution, clean detached SHA,
   checkpoint hash, and no-fallback checks must pass.

Formal probe:

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/run_futureseed2_selective_arm.sh state
```

Budget: one 100-step continuation, expected 12-20 minutes. No control rerun and
no gate-rank, scale, seed, LR, loss, or training-length sweep.

Success:

- mean loop5 exact over the three official ranges improves by `>= +0.015`; or
- 56-60 or 61-64 exact improves by `>= +0.02` while 51-55 regresses by no more
  than `0.01`.

Mechanism alive: gate delta RMS or seed relative change is `>= 1e-4`.

Kill:

- fail any integrity gate;
- NaN/OOM/wrong GPU/fallback;
- wall overhead `>20%` without quality gain;
- hard exact is neutral or worse despite a live gate.

The next decision is binary. A positive result makes selective transport the
first FutureSeed2 candidate for longer scaling. A negative result rejects
static state-entry selection and does not authorize a dynamic-gate sweep.

## 6. Artifacts

- CUDA selective-gate check:
  `/huyang2/double-loop/artifacts/launch/futureseed2-selective-module-check-5914b2e.log`
- Strict official-FLA preflight:
  `/huyang2/double-loop/artifacts/launch/futureseed2-selective-fla-preflight-5914b2e.json`
- Two-step full-stack smoke:
  `/huyang2/double-loop/runs/futureseed2-selective-state-smoke2-20260728T1558Z-5914b2e`
- Formal run:
  `/huyang2/double-loop/runs/futureseed2-selective-state-s9100-20260728T1604Z-5914b2e`
- Local side-by-side visualization:
  `runs/futureseed2-selective-20260728/index.html`
- Machine-readable comparison:
  `runs/futureseed2-selective-20260728/comparison.json`

## 7. Results

### Integrity

All fail-closed gates passed on GPU1:

- Zero-initialized seeded state was bit-exact to FutureSeed1.
- Activated gate matched the direct Torch formula with max absolute error `0`.
- State, base-head-logit, and gate-delta gradients were finite and nonzero.
- Official GDN2 output/state/gradient max absolute errors were
  `5.99e-5 / 2.76e-4 / 2.34e-4`.
- The runtime used `fla.layers.gdn2.GatedDeltaNet2`,
  `ChunkGDN2FunctionBackward`, and Triton q/k/v convolution in all ten layers.
- Checkpoint migration reported exactly
  `reasoner.future_seed_selector.gate_delta` missing, no unexpected parameter,
  and a successful optimizer-group expansion.
- The formal run used clean detached source
  `5914b2e4929e8f86b718a51f1b753de163b2cb8a`, the frozen step9000 checkpoint,
  GPU1, the same optimizer/RNG/data/eval contract, and no fallback.

### Matched result

| metric | FutureSeed1 identity | FutureSeed2 selective | delta |
|---|---:|---:|---:|
| train CE at step9100 | 0.6435 | 0.6493 | +0.0058 |
| mixed loop5 exact | 0.2520 | 0.2383 | -0.0137 |
| official 51-55 loop5 exact | 0.3672 | 0.3672 | +0.0000 |
| official 56-60 loop5 exact | 0.1309 | 0.1211 | -0.0098 |
| official 61-64 loop5 exact | 0.1973 | 0.1387 | -0.0586 |
| mean official hard exact | 0.2318 | 0.2090 | -0.0228 |
| train wall seconds | 659.9 | 675.6 | +2.4% |
| parameters | 5.462M | 5.517M | +1.01% |
| peak allocated CUDA memory | 8.095 GiB | 8.211 GiB | +1.43% |

The mechanism was active:

- gate delta RMS: `0.00950`;
- within-head state-entry gate standard deviation: `0.00224`;
- imported-seed relative change: about `0.00525`.

The 61-64 exact trajectories were:

```text
FutureSeed1:          0 -> 0 -> .0566 -> .1855 -> .1973
FutureSeed2 selective:0 -> 0 -> .0469 -> .1172 -> .1387
```

The largest damage appears in late correction rather than initial opening. On
shared 64-blank case `b0048`, FutureSeed1 changes wrong cells
`32 -> 12 -> 3 -> 0 -> 0`, while static selective FutureSeed2 changes
`36 -> 13 -> 6 -> 6 -> 5`. This is not global collapse: both variants solve
shared case `b0197` by loop4. Static selection changes which examples converge,
but not in a consistently useful direction.

## 8. Conclusion

Reject static state-entry selection as FutureSeed2. It fails both success
rules, despite a live gate and low systems overhead. Do not continue to
step9300 and do not sweep gate rank, scale, seed, LR, loss, or training length.

The useful mechanism conclusion is narrower than "selection never works." A
single learned `K x V` mask is shared by every input, but the same state entry
can carry useful evidence on one board and harmful evidence on another. The
shared-case visualization supports this interpretation: one board remains
solvable while another loses late-loop convergence. Any future selection
mechanism must therefore test content-dependent routing as a new hypothesis,
not tune this static mask.

The frozen strict-official GDN2 plus FutureSeed1 checkpoint remains the strong
baseline. No solver-specific rule, search, repair, selector, oracle inference,
CPU model smoke, GPU2, or silent fallback was used.

## 9. Submission

Not applicable. No experiment tag because the primary score regressed and the
mechanism claim was rejected.
