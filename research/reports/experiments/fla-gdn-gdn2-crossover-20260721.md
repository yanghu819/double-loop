# Official FLA GDN versus GDN2 Crossover Gate

## 1. Metainfo

- Plan ID: `P-LA-004`
- Status: completed; crossover inconclusive, no GDN2 task or efficiency win
- Started: 2026-07-21 08:54 CST / 2026-07-21T00:54:16Z
- Machine: AIStation `GPU1`, NVIDIA A800-SXM4-80GB; GPU2 forbidden
- Source branch: `codex/fla-gdn2-kda`
- Checkpoint-training source SHA: `c3342b8326a6e00e012cd6972d8c0e295d82c0b3`
- Continuation source SHA: `87e66cf91c086283e6981235c93152579fb93dc4`
- Completed: 2026-07-21 13:51 CST / 2026-07-21T05:51:36Z

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

### Matched step1000 readout

| Metric | GDN | GDN2 |
|---|---:|---:|
| Parameters at the shared outer geometry | 5.980M | 6.946M |
| CE, step500 | **0.9981** | 1.0101 |
| CE, step1000 | **0.9413** | 0.9539 |
| Mixed loop1 exact | 0.01953 | **0.02148** |
| Mixed loop5 exact | **0.02344** | **0.02344** |
| Mixed loop5 blank accuracy | **0.53183** | 0.52795 |
| Fixed holes53 loop5 exact | **0.01758** | **0.01758** |
| Fixed holes53 loop5 blank accuracy | **0.51872** | 0.51641 |
| Segment seconds per optimizer step | **5.15** | 5.70 |
| Peak allocated VRAM | **7281 MiB** | 9382 MiB |

GDN2 briefly crosses the noisy training CE at step700 (`0.9293` versus GDN
`0.9433`), then returns to `0.9451/0.9420/0.9539` at steps800/900/1000. GDN
ends at `0.9349/0.9366/0.9413` for those steps. The useful conclusion is not
that one isolated point wins; the two curves enter the same range after step600,
whereas the step500 comparison made the opening gap look much larger.

### Official blank ranges at loop5

| Blanks | GDN exact / blank | GDN2 exact / blank |
|---|---:|---:|
| 46-50 | **0.99805 / 0.99983** | 0.99219 / 0.99971 |
| 51-55 | 0 / **0.55372** | 0 / 0.55057 |
| 56-64 | 0 / **0.49090** | 0 / 0.48583 |

The 46-50 exact gap shrinks from `0.1055` at step500 to only `0.00586` at
step1000. Neither arm opens full-board closure at 51 or more blanks. Thus the
old large easy-range ranking was mainly a finite-budget opening effect, but
GDN2 still supplies no harder-range win after the extra training.

Across loops, GDN exact moves `0.01953 -> 0.02344` and blank accuracy moves
`0.50705 -> 0.53183`. GDN2 moves `0.02148 -> 0.02344` and
`0.50963 -> 0.52795`. On the displayed same-seed hard case, wrong-cell counts
are `26 -> 22 -> 25 -> 24 -> 24` for GDN and
`22 -> 21 -> 19 -> 19 -> 19` for GDN2. That case shows a cleaner GDN2 early
correction, but both stop changing meaningfully after loop3 and the aggregate
full-board score is tied.

### Runtime and interruption audit

The first GDN2 continuation reached and saved exact step600 before the GPU1
lease expired. Its `abort.json` records platform expiry rather than model
failure. GPU1 then remained queued for about two hours. The second leg resumed
the complete step600 model, optimizer, Python/Torch/CUDA RNG state and finished
to step1000. Logs from both legs are joined in the aggregate curve. Queue time
is excluded from model timing. No GPU2 or CPU smoke was used.

The post-run strict gate was rerun on the restarted GPU1 pod. The installed FLA
source hashes match pinned wheel SHA256
`65f57bf2aa937991fc497bd63f42883d263ead8646f21f2492735ccddd82d0eb`.
Observed backward nodes are `ChunkGatedDeltaRuleFunctionBackward`,
`ChunkKDAFunctionBackward`, and `ChunkGDN2FunctionBackward`; every Q/K/V short
convolution reports Triton, and all initial-state gradients are finite and
nonzero. There is no Torch recurrent fallback or reverse scan.

## 7. Decision

The preregistered clean crossover condition does not pass: GDN2 neither beats
GDN CE at step1000 nor beats its 46-50 exact. The strong no-crossover condition
also does not pass because the opening gap is only `0.00586`, far below the
required `0.05`. The correct label is therefore **inconclusive crossover with
no positive GDN2 result**.

This double-check retracts the broad statement "GDN is better than GDN2/KDA."
What survives is narrower and useful: under the shared D32/expand-v2 recipe,
GDN reaches the same full-board exact while GDN2 uses about `10.8%` more
measured seconds per continuation step, `16.2%` more parameters, and `28.9%`
more peak allocated memory. GDN is the pragmatic backbone for this recipe, not a proven
architecture-ceiling winner.

The official classes use different native defaults: KDA/GDN2 normally use a
larger head dimension and expand-v1, while this interface-matched experiment
forces D32/expand-v2. A native-geometry comparison would answer a different
question and is not justified as another immediate table. This experiment also
keeps FutureSeed enabled in both arms, so it ranks FutureSeed carriers; it does
not measure the causal gain of FutureSeed.

## 8. Artifacts

- Aggregate comparison, inline SVG curves, hard-case iframes, and screenshot:
  `runs/fla-gdn-gdn2-crossover-20260721-87e66cf/`
- GDN continuation:
  `runs/fla-official-gdn-fs-d192l10-resume500-s1000-20260721T010000Z-87e66cf/`
- Interrupted GDN2 step500-to600 leg:
  `runs/fla-official-gdn2-fs-d192l10-resume500-s1000-direct-20260721T023725Z-87e66cf/`
- Completed GDN2 step600-to1000 leg:
  `runs/fla-official-gdn2-fs-d192l10-resume600-s1000-direct-leg2-20260721T045147Z-87e66cf/`
- Exact checkpoints remain under `/huyang2/double-loop/models/` and are not
  committed.

## 9. Paper Claim

This run is negative evidence against claiming that a newer Linear Attention
equation automatically makes FutureSeed stronger. Together with P-LA-003, it
supports a cleaner mechanism statement: FutureSeed influence survives more
strongly in GDN2, but persistence alone does not produce better global closure.
Optimization, state geometry, and the downstream task still determine whether
that future context is useful.
