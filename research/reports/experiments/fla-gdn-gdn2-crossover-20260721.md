# Official FLA GDN versus GDN2 Crossover Gate

## 1. Metainfo

- Plan ID: `P-LA-004`
- Status: in progress
- Started: 2026-07-21 08:54 CST / 2026-07-21T00:54:16Z
- Machine: AIStation `GPU1`, NVIDIA A800-SXM4-80GB; GPU2 forbidden
- Source branch: `codex/fla-gdn2-kda`
- Checkpoint-training source SHA: `c3342b8326a6e00e012cd6972d8c0e295d82c0b3`

## 2. Hypothesis And Decision

P-LA-003 rejects the idea that KDA/GDN2 lose because their recurrence erases
FutureSeed. GDN2 retains `1.80x` the token81 state influence of GDN. Meanwhile,
from step400 to step500, GDN2 CE falls `0.0181` while GDN falls only `0.0094`.
The high-information hypothesis is now that GDN's step500 advantage is faster
opening under a shared recipe, while GDN2 may cross over with enough hard data.

Prediction: after an identical additional 500 optimizer steps on 51-55 blanks,
GDN2 should close or reverse the CE and official 46-50 exact gaps if the old
ranking was an early-budget artifact. If GDN still leads by at least `0.01` CE
and `0.05` 46-50 exact at step1000, keep GDN as the empirically stronger
common-recipe backbone and stop this architecture branch.

## 3. Configuration

- Resume exact GDN and GDN2 step500 checkpoints, including optimizer and Python,
  CPU-Torch, and CUDA RNG state.
- Continue both to global step1000 with executed curriculum
  `46-50:100,51-55:900`.
- Keep D192/L10/H6/D32, expand-v2, short-conv4, loop5, all-loop CE, effective
  batch128, BF16, LR `0.0015`, weight decay `0.001`, FutureSeed scale1 fixed
  unit-RMS transfer, seed52, and all evaluation seeds unchanged.
- Evaluate the same fixed holes53 batch and official 46-50/51-55/56-64 batches.
- Compare CE slope, exact, blank accuracy, loop1-to5 gain, wall time, VRAM, and
  strict FLA runtime provenance.

This is one paired continuation decision, not a hyperparameter sweep. KDA is not
continued because GDN2 was already the closer and faster alternative at step500.

## 4. Budget And Kill Criteria

- GPU1 only; run arms sequentially, expected total training under `110m` plus
  evaluation.
- Stop an arm if the resume step, optimizer/RNG state, source, wheel/backend, or
  visible GPU does not match exactly.
- Stop if either 100-step continuation exceeds `15m`, loss is non-finite, or GPU
  utilization remains low while memory is occupied.
- No rescue through LR, seed, loss, gate, head geometry, noise, repair, search,
  selector, or task-specific rule.

## 5. Paper Claim Gate

GDN2 crossover would show that more expressive Linear Attention needs more
optimization before its generic state update pays off, correcting the premature
"GDN is better" claim. No crossover would support the narrower claim that
simple GDN is the best sample-efficient FutureSeed carrier under the tested
matched small-state budget. Neither outcome alone establishes a universal
architecture ranking.

## 6. Results

Pending.
