# GDN2 Long FutureSeed Causal Control

## Metainfo

- Plan: `P-CAUSAL-001`
- Status: preregistered
- Preregistered: 2026-08-03 02:45 CST / 2026-08-02 18:45 UTC
- Machine: AIStation GPU1, NVIDIA A800-SXM4-80GB
- GPU2: forbidden
- CPU model smoke: forbidden
- Seed: 52 only

## Question

The completed 12,000-step official-FLA GDN2 plus native FutureSeed run reaches
mixed loop5 exact `0.3379` and official 51-55/56-60/61-64 exact
`0.4492/0.1543/0.2637`. Its loop1 exact remains `0.0234`, while loop5 improves
with training. Does FutureSeed still change the learning curve or final
frontier at this scale, or would an otherwise identical causal GDN2 plus loop
eventually reach the same result?

This is the one causal control required by the main claim. It is not a carrier,
loss, schedule, width, seed, or checkpoint sweep.

## Mechanism Hypothesis

FutureSeed initializes each deeper recurrent layer with a terminal state from
the preceding layer. It therefore exposes future-token information through
the same causal recurrent kernel instead of adding a reverse scan or a
noncausal mixer. If this state initialization removes a real optimization and
information bottleneck, the no-FutureSeed model should learn the same task more
slowly or reach a lower hard-board frontier under matched data and compute.

If the no-FutureSeed control catches the 12k FutureSeed endpoint within normal
evaluation noise, the defensible claim must shrink to short-budget optimization
acceleration. If it remains materially behind, FutureSeed has long-scale causal
value on this proxy.

## Configuration

- source `configs/sudoku/gdn2_scale_nofs_12000.env`;
- exact canonical P-SCALE-037 model, official data, curriculum, random order,
  optimizer, LR, weight decay, BF16, batch, loop5, and every-loop CE;
- official FLA `GatedDeltaNet2`, pinned source, chunk backward, Triton Q/K/V
  convolutions, backend dispatch disabled, and no fallback;
- D192, 10 layers, 6 heads, K/V dimension 32, expand-v 1;
- 12,000 optimizer steps and effective batch 128;
- `FUTURE_SEED_SCALE=0` is the only scientific difference;
- no noise, scratch state, feedback, special loss, repair, search, selector,
  oracle rollout, task rule, architecture rescue, or second seed.

## Frozen FutureSeed Reference

The exact FutureSeed checkpoint curve used for decisions is:

| Step | holes53 exact/blank | holes58 exact/blank | holes64 exact/blank |
|---:|---:|---:|---:|
| 500 | `0.0000/0.5133` | `0.0000/0.3830` | `0.0000/0.4356` |
| 1000 | `0.0000/0.5599` | `0.0000/0.4448` | `0.0000/0.5315` |
| 3000 | `0.0215/0.6271` | `0.0000/0.4598` | `0.0000/0.5916` |
| 6000 | `0.1777/0.7871` | `0.0117/0.4853` | `0.0059/0.6628` |
| 9000 | `0.4824/0.8605` | `0.0371/0.5551` | `0.1230/0.8110` |
| 12000 | `0.5781/0.8894` | `0.0645/0.5873` | `0.2363/0.8456` |

## Predictions And Decisions

1. Step500 must reproduce the established no-FutureSeed regime: materially
   worse CE and 46-50 opening than FutureSeed, with exact official data,
   initialization shell, and CUDA provenance. A mismatch in data, parameters,
   source, or kernel invalidates the run.
2. The primary endpoint is the matched step12000 difference on mixed exact and
   mean official 51-64 exact. FutureSeed has a persistent frontier advantage if
   either difference is `>= +0.03` without relying on worse clue validity or a
   different evaluator.
3. FutureSeed has compute-compression value if no-FutureSeed needs at least
   `20%` more optimizer steps to cross the frozen FutureSeed step6000 or
   step9000 hard-readout level.
4. If no-FutureSeed is within `0.02` on both mixed and mean official 51-64 exact
   at step12000, the long-scale frontier claim is rejected; retain only the
   already established short-budget acceleration claim.

## Kill Criteria

- Stop immediately on wrong GPU, dirty or non-detached source, wrong data hash,
  parameter/init mismatch, nonofficial FLA class/source, missing official chunk
  backward, non-Triton convolution, fallback, NaN, OOM, or nonfinite gradients.
- If step100 exceeds 15 minutes, stop by exact PID and archive the systems
  failure; do not silently change batch, kernel, model, or device.
- Do not stop merely because hard exact is zero at step3000; that would make the
  asymptotic question unanswerable.
- At step6000, an early scientific stop is allowed only if no-FutureSeed already
  exceeds the frozen FutureSeed step9000 readouts on all three fixed hard holes.
  That would be sufficient evidence against a FutureSeed compute advantage.
- Otherwise continue unchanged to step12000. Do not rescue either outcome with
  more steps, a chosen checkpoint, loss/noise/gate tuning, or another seed.

## Claim Enabled

A positive result supports: native FutureSeed is a cheap way to improve the
learning efficiency or finite-compute frontier of a causal linear-attention
recurrent backbone, while preserving the official forward/backward kernel.
This single task is still not enough for a universal claim; Maze or language
evidence remains required.

