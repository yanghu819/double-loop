# GDN2 Long FutureSeed Causal Control

## Metainfo

- Plan: `P-CAUSAL-001`
- Status: complete; preregistered step12000 endpoint archived
- Preregistered: 2026-08-03 02:45 CST / 2026-08-02 18:45 UTC
- Machine: AIStation GPU1 only; A800 prior legs, A100-SXM4-80GB task-mode current leg
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

## Step 500 Execution Gate

The first execution leg used clean detached source
`53670e0a3c8f8367bc921d88034bdd6d4ff82891` on GPU1 only. Strict checks passed
for the pinned official FLA source, `GatedDeltaNet2`,
`ChunkGDN2FunctionBackward`, Triton short convolution, Torch/CUDA reference
alignment, native FutureSeed cache plumbing, and no fallback. A full-size
two-step CUDA smoke also passed before the formal launch. Step100 completed in
about ten minutes, below the 15-minute systems kill criterion.

The complete fixed-batch step500 readout is:

| Holes | no-FS loop1 exact/blank | no-FS loop5 exact/blank | FS loop5 exact/blank | FS minus no-FS blank |
|---:|---:|---:|---:|---:|
| 50 | `0.0000/0.4046` | `0.0000/0.4051` | not in frozen hard table | n/a |
| 53 | `0.0000/0.2751` | `0.0000/0.2727` | `0.0000/0.5133` | `+0.2406` |
| 58 | `0.0000/0.2546` | `0.0000/0.2539` | `0.0000/0.3830` | `+0.1291` |
| 64 | `0.0000/0.2649` | `0.0000/0.2639` | `0.0000/0.4356` | `+0.1717` |

Train CE is `1.22484`; loop1/loop5 losses are `1.22892/1.22484`. The control
has not opened full-board exact even at holes50, and its later loops do not yet
improve hard blank accuracy. This is a large short-budget FutureSeed advantage,
but it does not answer the preregistered long-scale endpoint, so training must
continue unchanged.

The guard waited for both the step500 fixed evaluation and atomic train-state
save, then stopped exact process group `2857`. The 72,202,195-byte checkpoint
has SHA256
`13fb1a5c93b9a497c2925dbe482bebe0e3474022c06cb7d9c08144ef1ef05921`.
`abort.json` records `infrastructure_lease_rollover` with
`scientific_failure=false`, and GPU1 returned to zero allocated memory. Resume
only through `configs/sudoku/gdn2_scale_nofs_resume_12000_from00500.env`, which
enables exact semantic-contract, source-SHA, checkpoint-hash, optimizer, data
RNG, and training RNG restoration. No scientific variable changes.

## Step 1000 Science Gate

GPU1 resumed the exact step500 state under clean detached source
`ef58ed858c02745b60092e94541a88eb140ae977`. The strict official-FLA CUDA
preflight passed again on the newly allocated physical GPU, including official
source/wheel hashes, disabled backend dispatch, Triton convolution, Torch
reference alignment, adapter checks, native FutureSeed-stack checks, and real
backward. The resume log confirms `step=500 reason=eval_checkpoint`; step600
switches to the preregistered 51-55 curriculum and all FutureSeed and optional
mechanism diagnostics remain zero.

The complete step1000 fixed-batch readout is:

| Holes | no-FS loop1 exact/blank | no-FS loop5 exact/blank | FS loop5 exact/blank | FS minus no-FS blank |
|---:|---:|---:|---:|---:|
| 50 | `0.0000/0.3885` | `0.0000/0.3887` | not in frozen hard table | n/a |
| 53 | `0.0000/0.2783` | `0.0000/0.2778` | `0.0000/0.5599` | `+0.2821` |
| 58 | `0.0000/0.2601` | `0.0000/0.2609` | `0.0000/0.4448` | `+0.1839` |
| 64 | `0.0000/0.2625` | `0.0000/0.2649` | `0.0000/0.5315` | `+0.2666` |

No-FutureSeed train CE is `1.59664`; after the stage transition it stays near
`1.61` from steps600-900. Hard loop5-minus-loop1 blank gains are only
`-0.0004/+0.0008/+0.0025`, and exact remains zero everywhere. The
FutureSeed advantage therefore does not vanish after another 500 optimizer
steps; it becomes larger on every frozen hard readout than at step500. This is
strong evidence for short-budget optimization/information value and against a
mere first-few-step initialization artifact. It still does not establish a
12k frontier difference, so the unchanged no-FutureSeed trajectory continues.

## Step 2500 Infrastructure Rollover

The unchanged no-FutureSeed trajectory continued from step500 to step2500 on
GPU1. Train CE remained on the same high plateau after entering the 51-55
curriculum: steps2000/2100/2200/2300/2400/2500 are
`1.5942/1.5891/1.6033/1.6063/1.5855/1.5833`. This is process evidence, not a
scientific endpoint; the preregistration explicitly forbids stopping only
because hard exact may remain closed at step3000.

The lease guard waited for both the complete step2500 log record and atomic
train-state file, then stopped exact process group `913`. The checkpoint is
72,248,723 bytes with SHA256
`f9030611801826f018faf20c934ce4445444cd966cabf190e88d81ae51283a55`.
An independent hash/size check passed, all training processes exited, and GPU1
had no compute process. `abort.json` records
`infrastructure_lease_rollover`, `scientific_failure=false`. Resume only via
`configs/sudoku/gdn2_scale_nofs_resume_12000_from02500.env`; no model, data,
optimizer, loop, loss, seed, kernel, or evaluation variable changes.

## Step 3000 Science Gate

GPU1 restarted on a new physical A800 and checked out clean detached source
`a79b65a09a10d676d09edb36d41de3f5abf0eeb7`. The complete official-FLA CUDA
preflight passed in 163 seconds with `fallback=false`. Exact restoration from
the hashed step2500 model, optimizer, scheduler, data RNG, and training RNG
state passed; the log reports `step=2500 reason=periodic`.

The frozen same-step comparison is:

| Holes | no-FS loop1 exact/blank | no-FS loop5 exact/blank | FS loop1 exact/blank | FS loop5 exact/blank |
|---:|---:|---:|---:|---:|
| 50 | `0.0000/0.3941` | `0.0000/0.3956` | `0.8384/0.9960` | `1.0000/1.0000` |
| 53 | `0.0000/0.2943` | `0.0000/0.2927` | `0.0000/0.5627` | `0.0215/0.6271` |
| 58 | `0.0000/0.2569` | `0.0000/0.2583` | `0.0000/0.4475` | `0.0000/0.4598` |
| 64 | `0.0000/0.2534` | `0.0000/0.2533` | `0.0000/0.5177` | `0.0000/0.5916` |

No-FutureSeed versus FutureSeed train CE is `1.5946` versus `0.8270`. Mean
loop5 blank accuracy over holes53/58/64 is `0.2681` versus `0.5595`, a
FutureSeed gap of `+0.2914`. No-FutureSeed loop5-minus-loop1 blank change is
only `-0.0016/+0.0014/-0.0001` on holes53/58/64. This is not an exact-score
edge case: FutureSeed fully solves the fixed holes50 batch at loop5 while the
matched no-FutureSeed model has zero exact and only `0.3956` blank accuracy.

The step3000 gate therefore establishes a large finite-compute optimization
and information gap, and shows that extra loops still cannot compensate for
missing FutureSeed at this budget. It does not prove a 12k frontier gap. Per
the preregistered rule, the unchanged no-FutureSeed trajectory continues; no
rescue, second seed, loss change, or mechanism change is allowed. The complete
compact comparison is archived in
`research/reports/experiments/futureseed-gdn2-long-causal-control-step3000.json`.

## Step 4500 Infrastructure Rollover

The unchanged no-FutureSeed trajectory crossed from the `51-55` curriculum
into `51-60` at step4100 exactly as configured. Train CE at
steps4100/4200/4300/4400/4500 is
`1.6122/1.6094/1.6105/1.6142/1.6092`. This remains a flat delayed-opening
warning, not a new scientific endpoint; the next registered fixed evaluation
is step6000.

The checkpoint-complete guard waited for both the step4500 log marker and the
atomic train-state file, then stopped exact process group `763`. The checkpoint
is 72,291,667 bytes with SHA256
`203db6e47bfd32d7b70e2769fdaafc50ed0ce3dce75bc2a0aefbf31a8b0de385`.
Independent size/hash checks passed, all launcher/train/guard processes exited,
and the GPU compute-process list was empty. `abort.json` records
`infrastructure_lease_rollover`, `scientific_failure=false`, and stop time
`2026-08-03T01:58:30Z`.

Resume only through
`configs/sudoku/gdn2_scale_nofs_resume_12000_from04500.env`, which preserves
the model, optimizer, scheduler, data RNG, training RNG, curriculum, loop
supervision, and every other scientific variable. A 10-second utilization
audit observed low A800 SM occupancy (`16-27%`) but only 11.6/80GB allocated
and unchanged step throughput; changing microbatch or accumulation inside this
matched causal control would invalidate parity with the frozen FutureSeed run.

## Step 6000 Science Gate

The exact step4500 state resumed in AIStation task mode on one visible GPU1,
an NVIDIA A100-SXM4-80GB. Clean detached training source is
`b312b02b54941d8fd03ceefa5396317bd55c82e2`; the strict official-FLA CUDA
preflight passed in 146 seconds with backend dispatch disabled, Triton short
convolution, official `GatedDeltaNet2`, real backward, and no fallback. The
formal process remains PID/PGID `2347` and continued beyond step6500 after this
read-only gate.

The fixed 512-board-per-hole comparison is:

| Holes | no-FS loop1 exact/blank | no-FS loop5 exact/blank | FS loop1 exact/blank | FS loop5 exact/blank |
|---:|---:|---:|---:|---:|
| 50 | `0.0000/0.4067` | `0.0000/0.4097` | `0.9192/0.9972` | `1.0000/1.0000` |
| 53 | `0.0000/0.2974` | `0.0000/0.2998` | `0.0000/0.6090` | `0.1777/0.7871` |
| 58 | `0.0000/0.2638` | `0.0000/0.2647` | `0.0000/0.4596` | `0.0117/0.4853` |
| 64 | `0.0000/0.2635` | `0.0000/0.2642` | `0.0000/0.5272` | `0.0059/0.6628` |

Over holes53/58/64, FutureSeed/no-FutureSeed loop5 exact mean is
`0.06510/0.00000`, blank mean is `0.64507/0.27624`, and loop1-to-loop5 blank
gain is `+0.11316/+0.00135`. Train CE is `0.79736/1.60346`. Thus FutureSeed
does not merely shift one-pass calibration: it both opens the optimization and
makes later recurrent loops useful under the same optimizer-step budget.

A separate strict-CUDA paired diagnostic evaluated the same 128 boards per
official 51-55/56-60/61-64 range. FutureSeed/no-FutureSeed loop5 exact is
`0.07031/0.00000` and blank accuracy is `0.63346/0.27529`. No-FutureSeed removes
only `0.12/0.13/0.12` mean wrong cells from loop1 to loop5; FutureSeed removes
`7.24/3.39/8.23`. Same-puzzle loop1/3/5 visualizations and complete case JSON
are archived under
`runs/futureseed-causal-step6000-paired-viz-20260803T170049Z-6be5bc2/`.

This is strong finite-compute causal evidence and already exceeds a marginal
`+0.03` mechanism effect. It is not the preregistered asymptotic endpoint.
Because no-FutureSeed did not exceed the frozen FutureSeed step9000 gate, the
registered decision is to continue unchanged to step9000 and step12000. Do not
compare wall time across arms: the reference and control used different lease
segments and physical GPUs; optimizer-step and fixed-data comparisons are the
valid evidence.
## Step 9000 Science Gate

The unchanged no-FutureSeed control reached the exact step9000 checkpoint on
one visible GPU1, an NVIDIA A100-SXM4-80GB. Training remains PID/PGID 2347
under clean detached source b312b02b54941d8fd03ceefa5396317bd55c82e2.
The 72,372,755-byte checkpoint has SHA256
555e1373bb4c83c80f2157ee8cea7a86ba29fe8db645f5f71578e887b94bd597.
The frozen FutureSeed checkpoint is step9000 with SHA256
606caf5229590f157d7a0f952423c719f4c17688003309a7bd0664e579588dd7.

The fixed 512-board-per-hole comparison is:

| Holes | no-FS loop1 exact/blank | no-FS loop5 exact/blank | FS loop1 exact/blank | FS loop5 exact/blank |
|---:|---:|---:|---:|---:|
| 50 | 0.0000/0.4133 | 0.0000/0.4145 | 0.9697/0.9988 | 1.0000/1.0000 |
| 53 | 0.0000/0.3031 | 0.0000/0.3040 | 0.0000/0.6421 | 0.4824/0.8605 |
| 58 | 0.0000/0.2752 | 0.0000/0.2744 | 0.0000/0.4747 | 0.0371/0.5551 |
| 64 | 0.0000/0.2738 | 0.0000/0.2727 | 0.0000/0.5567 | 0.1230/0.8110 |

Over holes53/58/64, FutureSeed/no-FutureSeed loop5 exact mean is
0.21419/0.00000, blank mean is 0.74222/0.28372, and loop1-to-loop5 blank gain
is +0.18439/-0.00033. Train CE is 0.65240/1.59670. The no-FutureSeed model has
therefore not opened after 9000 steps; later loops still reproduce nearly the
same local operating point.

A strict-CUDA paired diagnostic from visualization source
d24674c26369309288b42bf57249097ed8e26b03 evaluated the same 128 boards per
official 51-55/56-60/61-64 range. It verified official FLA GatedDeltaNet2,
source 9c8e42e, official layer forward, chunk/Triton execution, backend
dispatch disabled, one CUDA device, and byte-verified checkpoints. Paired
FutureSeed/no-FutureSeed loop5 exact is 0.19531/0.00000 and blank accuracy is
0.74137/0.28221. No-FutureSeed removes 0.55/0.20/0.18 mean wrong cells across
loops; FutureSeed removes 8.56/6.84/17.47. A selected 64-blank board changes
36 -> 17 -> 6 -> 1 -> 0 wrong cells with FutureSeed while no-FutureSeed stays
near 50 wrong. Same-board visualizations are archived under
runs/futureseed-causal-step9000-paired-viz-20260803T202002Z-d24674c/.

The control used about 4.583 seconds per optimizer step between steps8100 and
9000, or 785.5 steps/hour, with 11,597 MiB observed steady training memory.
The paired visualization completed in about 52 seconds while the control
continued to step9100. The checkpoint JSON elapsed fields are 20,868 seconds
for no-FutureSeed and 10,613 seconds for FutureSeed, but those wall times are
not directly comparable because the arms crossed different physical GPUs and
lease segments.

No-FutureSeed has now consumed 9000 steps, 50% more than the frozen FutureSeed
step6000 gate, and still does not match its exact, blank-accuracy, or recurrent
correction level. This establishes a greater-than-1.5x optimizer-step
compression lower bound for reaching the FutureSeed step6000 hard level.
It remains an intermediate finite-compute result, not the registered 12k
frontier decision or a universal task claim. Continue unchanged to step12000;
do not rescue the control, add a seed, or tag this sub-0.50 gate.

## Step 12000 Endpoint

The unchanged no-FutureSeed control completed its exact step12000 checkpoint
on one visible GPU1. The 72,411,027-byte checkpoint has SHA256
`bab94971cd59ee3f46de487a7a73e75d21f8322170f8323613d3fa81eca25a89`.
The frozen FutureSeed checkpoint has SHA256
`66b805cf163a6b0eaae1ec6ed74f9f7a8b3cc6e21919b58268b8150ae5ae3b37`.
Both are seed52, official GDN2 D192/L10/H6/D32, loop5 with every-loop CE,
effective batch128, the same official data and curriculum, and exactly 12,000
optimizer steps. Native FutureSeed is the only scientific difference.

The fixed 512-board-per-hole endpoint is:

| Holes | no-FS loop1 exact/blank | no-FS loop5 exact/blank | FS loop1 exact/blank | FS loop5 exact/blank |
|---:|---:|---:|---:|---:|
| 50 | `0.0000/0.4119` | `0.0000/0.4127` | `0.9899/0.9998` | `1.0000/1.0000` |
| 53 | `0.0000/0.3111` | `0.0000/0.3133` | `0.0000/0.6673` | `0.5781/0.8894` |
| 58 | `0.0000/0.2785` | `0.0000/0.2785` | `0.0000/0.4855` | `0.0645/0.5873` |
| 64 | `0.0000/0.2770` | `0.0000/0.2773` | `0.0000/0.5722` | `0.2363/0.8456` |

Over holes53/58/64, FutureSeed/no-FutureSeed loop5 exact mean is
`0.29297/0.00000`, blank mean is `0.77412/0.28971`, and loop1-to-loop5 blank
gain is `+0.19911/+0.00083`. Train CE is `0.59694/1.58121`, a FutureSeed
advantage of `0.98427`. No-FutureSeed still cannot solve even holes50 after
12,000 steps; its extra loops continue to reproduce almost the same wrong
board.

A separate strict-CUDA paired diagnostic evaluated the same 128 boards per
official 51-55/56-60/61-64 range. It verified official
`fla.layers.gdn2.GatedDeltaNet2`, source `9c8e42e`, official layer forward,
chunk/Triton execution, backend dispatch disabled, one CUDA device, and exact
checkpoint hashes. Paired FutureSeed/no-FutureSeed loop5 exact is
`0.27865/0.00000`, and blank accuracy is `0.77248/0.28330`. Across the three
ranges, no-FutureSeed removes only `0.16/0.02/0.21` mean wrong cells from
loop1 to loop5; FutureSeed removes `10.34/6.93/19.32`.

The clearest 64-blank solved case changes
`39 -> 14 -> 2 -> 0 -> 0` wrong cells with FutureSeed while the matched causal
control changes `50 -> 51 -> 51 -> 51 -> 51`. The hardest shared 64-blank
failure remains useful negative evidence: FutureSeed changes
`28 -> 31 -> 28 -> 28 -> 28`, showing that FutureSeed is not an oracle and
that some boards remain outside the learned attractor. Same-board HTML and all
case JSON are archived under
`runs/futureseed-causal-step12000-paired-viz-20260804T002500Z-2ea9069/`.

Observed no-FutureSeed throughput from checkpoint mtimes over steps11100-12000
is `4.537` seconds/step (`793.5` steps/hour) with 11,597 MiB steady training
memory. The corresponding FutureSeed checkpoint interval is `5.766`
seconds/step (`624.3` steps/hour). These numbers are process records, not a
matched speed comparison: the arms crossed different physical GPUs and lease
segments. Optimizer steps and fixed paired data are the valid causal evidence.

The preregistered persistent-frontier criterion passes decisively: fixed hard
exact delta is `+0.29297` and paired hard exact delta is `+0.27865`, both far
above `+0.03`. FutureSeed is not merely shifting an early learning curve. In
this official GDN2 proxy it is necessary for opening the hard regime, and it
creates a recurrent state on which later loops perform real correction.

Stop the no-FutureSeed trajectory. Do not add more steps, a second seed, or a
rescue loss. No tag is created because the primary score is below `0.50` and
one task cannot support a universal claim. The next high-information gate is
the same strict native-FutureSeed versus causal comparison on a non-Sudoku
task such as Maze or language/retrieval.
