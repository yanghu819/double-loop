# FutureSeed2 Selective State Gate

## 1. Metainfo

- Plan: `P-FS2-002`
- Status: pre-registered
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

Pending.

## 7. Results

Pending.

## 8. Conclusion

Pending.

## 9. Submission

Not applicable. No experiment tag unless the established strong-score rule is
met.
