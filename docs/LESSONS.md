# Lessons

## 2026-06-03 GPU1 bootstrap

- AIStation development work for this repository is GPU1-only. GPU2 can be visible
  in the platform table, but experiment commands should bind to the GPU1 container
  and use `CUDA_VISIBLE_DEVICES=0` inside that container.
- Persistent experiment state belongs under `/huyang2/double-loop`; avoid `/root`,
  `/root/.cache`, and other reset-prone locations for environments, caches,
  artifacts, models, and run logs.
- Keep GitHub as the source of truth: run from a detached commit SHA on the GPU
  node, archive run metadata with that SHA, then commit and push tracking changes
  deliberately.
- The bitter lesson applies here: prioritize scalable search/training feedback and
  measured experiment loops over hand-built solver shortcuts.

## 2026-06-03 uv bootstrap

- GPU1's system Python can lack `ensurepip`/`python3-venv`, so `setup.sh` must
  bootstrap uv without requiring `python3 -m venv`. Prefer a local wheelhouse,
  then `pip --target` if pip exists, then the standalone uv installer into
  `/huyang2/double-loop/.cache/uv-bootstrap/bin`.
- The current GPU1 image already includes `/opt/conda/bin/python` with
  `torch 2.7.0+cu126` and CUDA on the A100. Reuse that configured GPU stack
  before downloading large torch wheels again.
- Avoid broad `pkill -f` process patterns. They can match the shell command that
  is trying to clean up the process and terminate the SSH session itself.
- Keep the experiment script syntactically valid on the reusable GPU Python
  stack. The current image uses Python 3.10, so avoid newer nested f-string
  syntax even if local tools can parse it.
- After remote patch transfer, inspect script tails and run the wrapper end to
  end. A syntactically valid shell script can still be semantically truncated
  before the metadata-recording step.
- Scale-up controls should be environment-driven in `run.sh`. Keeping model
  width/depth, curriculum, rollout, and optimizer knobs configurable lets GPU1
  runs increase useful compute without creating one-off wrapper scripts or
  broad low-signal sweep tables.
- Generated run/cache/model directories should not make a subsequent experiment
  look source-dirty. Dirty provenance should track source edits, while run
  metadata is committed after the experiment completes.
- A full-run smoke preflight must not inherit the parent `RUN_NAME`; otherwise
  the smoke and full jobs can write into the same tracking directory and blur
  config, logs, and scores.
- Experiment recorders should trust the run-local `config.json` for the run's
  source SHA and dirty flag. Recomputing dirty state after outputs are written
  can make a clean run look dirty just because tracking artifacts now exist.

## 2026-06-03 RWKV7 CUDA pivot

- The first 9x9 cliff run (`9x9-cliff-20260603T090939Z-d8ce276`) and the smaller
  rescue probe (`9x9-cliff-small-20260603T092602Z-d8ce276`) both hit the wrong
  bottleneck: the pure PyTorch recurrent scan had not emitted `step=0100` before
  the kill window, even while the A100 was doing work. Treat this as an
  implementation throughput failure, not as evidence about 9x9 reasoning quality.
- Do not respond to that signal with more batch/curriculum table filling. The
  next high-ROI question is whether a real CUDA RWKV7 WKV kernel can make the
  same FutureSeed hypothesis cheap enough to evaluate.
- RWKV7 `wind` CUDA keeps the useful `s0 -> sT` state interface needed by
  FutureSeed, but it imposes hard shape constraints: CUDA bf16, `T % 16 == 0`,
  and `head_dim` divisible by 16. For 9x9 Sudoku, pad 81 tokens to 96 and use a
  compatible shape such as `D_MODEL=128 HEADS=8 HEAD_DIM=16`.
- `torch.utils.cpp_extension.load` requires a `ninja` executable even when the
  container already has a working CUDA PyTorch. When reusing `/opt/conda/bin/python`,
  `setup.sh` still needs to install/link `ninja` under the repo-local `.cache/bin`
  and `run.sh` must prepend that directory to `PATH`.
- When installing PyPI packages with `pip --target`, console scripts are written
  under the target's `bin/` directory, not necessarily under the imported
  package path. For `ninja`, link `.cache/ninja-pylib/bin/ninja` into
  `.cache/bin/ninja` before trying to compile CUDA extensions.
- The `modded-nanogpt-rwkv` wind kernel is not the right A100 default on the
  current GPU1 image: CUDA 12.6 reaches `ptxas`, then fails because `movmatrix`
  is not recognized while assembling for sm_80. Use the official
  `BlinkDL/RWKV-CUDA` state-passing clampw kernel for GPU1 scale-up; keep wind
  as an explicit future option for a toolchain/GPU where that asm is supported.
- The CUDA state-passing 9x9 cliff run (`9x9-cliff-cuda-20260603T103352Z-92ee7b9`)
  reverses the earlier 9x9 kill signal: with `D_MODEL=128`, `LAYERS=8`,
  `HEAD_DIM=16`, `MAX_LOOPS=5`, and a 4-8 then 8-12 hole curriculum, train CE
  fell below 1.0 by step100 and final eval reached holes8 exact 1.0000, holes12
  exact 0.9629, and holes16 exact 0.8535. This supports continuing 9x9 CUDA
  scale-up; the K8 oracle gap is effectively zero, so rollout selector work is
  low priority for this branch.

## 2026-06-03 9x9 mechanism ablation

- The one-hour ablation budget was enough for three targeted GPU1 runs at
  source SHA `5e2253f`: no outer loop, no FutureSeed, and no training
  feature-diff noise. This was a mechanism test, not a sweep; each run answered
  whether a specific part of the successful 9x9 CUDA cliff run was structural.
- Removing the outer loop (`9x9-ablate-no-loop-20260603T111606Z-5e2253f`) did
  not break easy-distribution 9x9: full-board exact was 0.9922, holes12 exact
  was 0.9551, and train CE reached 0.0135 by step800. It did reduce hard-hole
  transfer: holes16 exact was 0.7715 versus the baseline loop5 0.8535. Treat
  loop compute as hard-constraint refinement, not as the only reason the model
  can solve 9x9.
- Removing FutureSeed (`9x9-ablate-no-future-seed-20260603T111747Z-5e2253f`)
  collapsed the run: train CE was still 0.4498 at step800, full-board loop5
  exact was 0.0664, holes12 exact was 0.0195, and holes16 exact was 0.0000.
  K8 oracle exact rose to only 0.2031 while the best selector was 0.0762, so
  the problem is not just selector choice. FutureSeed is carrying necessary
  cross-layer state for 9x9 scaling.
- Removing training feature noise (`9x9-ablate-no-feature-noise-20260603T112449Z-5e2253f`)
  improved this setup: train CE reached 0.0007 at step800, holes12 exact was
  0.9766, holes16 exact was 0.9004, and K8 selector gap was zero. For the next
  9x9 scale run, default to `NOISE_SCALE=0` or a much smaller value; do not
  spend budget on rollout selector work while FutureSeed is enabled and the
  oracle gap remains tiny.

## 2026-06-15 D320 effective-batch h120 scaling

- D320 width was not fairly judged by the earlier batch48 run. With microbatch
  `24` and gradient accumulation `3`, the effective-batch D320 route preserves
  the h96/h108 foundation and opens h120. The useful scaling move was general
  training infrastructure: checkpoint/resume, effective batch, bf16, and the
  CUDA statepassing RWKV kernel.
- The step12800 continuation confirms real but slow scaling. h120 loop6 exact
  moved from the old D320 step9800 checkpoint `0.0566` through
  `0.0820 -> 0.1016 -> 0.1191`, and final full eval reached `0.1270`. h108
  remained strong at `0.9355`; h132 stayed closed at `0.0`.
- Width is useful but not a shortcut. The D320 final score is still below the
  D256 late-continuation best `0.1777`, and the run used about `27GB` allocated
  / `32GB` reserved on an 80GB GPU with moderate utilization. Before jumping to
  larger width, improve throughput/effective compute for the current D320 path.
- The packed D320 continuation confirms that this was a real systems bottleneck.
  Microbatch `48` with accumulation `2` reached about `60-64GB` memory use and
  near-100% GPU utilization, while h120 checkpoint loop6 exact moved
  `0.1289 -> 0.1777` from step13200 to step13600. Final full eval reached
  h96/h108/h120/h132 loop6 exact `0.9941`/`0.9492`/`0.1738`/`0.0`. This nearly
  matches the D256 best regime, but does not beat it decisively or open h132.
- Continuing the packed D320 path to step16600 does beat the old platform:
  checkpoint h120 loop6 exact moves `0.1445 -> 0.1738 -> 0.1953` at
  steps 14600/15600/16600, and final full eval reaches h120 exact `0.2363`.
  This is enough evidence that h120 remains compute-limited under the clean
  FutureSeed+loop recipe.
- The step19600 continuation confirms the same direction: h120 checkpoint loop6
  exact moves `0.1934 -> 0.2266 -> 0.2520`, and final full eval reaches h120
  exact `0.2891`. This is still a clean scaling win, not a reason to add
  selector or Sudoku repair.
- The step22600 continuation is the first useful plateau signal for packed D320.
  h120 checkpoint loop6 exact moves only `0.2637 -> 0.2676 -> 0.2461`, and final
  full eval reaches h120 exact `0.2754`, below the step19600 final `0.2891`.
  h96/h108 remain strong at `0.9961`/`0.9492`, so this is not foundation
  collapse; it is low marginal ROI for the same h120 hard-stage continuation.
  Do not spend the next budget on another identical 3k-step extension. Change an
  effective scaling axis or test a simple FutureSeed/loop state update that lets
  late loops keep revising boards.
- The delayed loop-credit probe answers the simplest objective-shaping escape
  hatch. Resuming step22600 with `LOOP_LOSS=delayed` and `LOOP_LOSS_START=4`
  keeps h96/h108 alive, but h120 loop6 exact falls to `0.1602` at step23100 and
  h120 loop1/3/4/5/6 exact is only `0.0000`/`0.0781`/`0.1406`/`0.1563`/`0.1602`.
  Moving supervision credit to late loops does not create late correction; it
  erases useful h120 structure. Stop delayed loop-credit variants unless a
  different state dynamic gives a reason to revisit them.
- Loop remains essential at h120: final h120 loop1 exact was `0.0`, loop3 was
  `0.0664`, and loop6 was `0.1270`. This supports recurrence as the active
  mechanism, not one-pass prediction. It does not justify selector or repair
  work because the clean path still has positive compute slope and K-oracle has
  not shown selector headroom.
- For packed D320, the loop evidence is even clearer: final h120 loop1/3/4/5/6
  exact is `0.0000`/`0.0820`/`0.1562`/`0.1699`/`0.1738`. The remaining limit is
  not one-pass recognition; it is turning late-loop local fill into more valid
  full-board solves.
- At step16600, packed D320 h120 loop1/3/4/5/6 exact is
  `0.0000`/`0.1621`/`0.2188`/`0.2363`/`0.2363`. Loop remains essential, but
  loop6 no longer adds exact solves beyond loop5, so do not respond with a
  loop-count sweep. Buy more clean training or change the state update only if
  exact stalls.
- At step19600, h120 loop1/3/4/5/6 exact is
  `0.0000`/`0.2070`/`0.2773`/`0.2852`/`0.2891`. Loop is still the mechanism,
  but the marginal loop5-to-loop6 gain is small. Deeper loop counts remain low
  ROI.
- At step22600, h120 loop1/3/4/5/6 exact is
  `0.0000`/`0.1914`/`0.2656`/`0.2715`/`0.2754`. Loop remains necessary, but the
  last two loops are mostly polishing. The next mechanism question is not "more
  loops"; it is whether the state update can keep correcting wrong partial
  boards after loop4 without adding Sudoku-specific repair.
- Delayed loop credit made that loop curve worse, not better. At step23100,
  h120 loop3/4/5/6 exact is `0.0781`/`0.1406`/`0.1563`/`0.1602`; this rejects
  the theory that all-loop loss alone is the late-loop bottleneck.
- The learned loop update gate is the first simple state-dynamics change with a
  positive h120 signal after the packed D320 plateau. Resuming the step22600
  checkpoint with `LOOP_UPDATE_MODE=learned_gate` and init `0.95` reaches final
  h120 loop6 exact `0.3340`, with checkpoint step23600 h120 loop6 exact
  `0.3496`. This beats the clean step19600/22600 finals `0.2891`/`0.2754` and
  avoids the delayed-loss collapse `0.1602`. The learned gate is not just a
  no-op: final eval uses lower update gates in early loops and near-full update
  in loops 4-6. This supports simple learned state dynamics as a real mechanism.
- The same learned-gate run still has h132 exact `0.0` and h132 blank accuracy
  only about `0.17`. Do not overclaim it as opening the next frontier. It makes
  h120 late-loop correction better; it does not solve the larger-scale
  interaction problem. Next work should either continue this checkpoint with
  meaningful compute, or test one slightly richer learned update rule such as
  per-channel/state-conditioned gating. It should not pivot back to selector,
  repair, Sudoku priors, feature-noise tables, or loop-count sweeps.
- Continuing the learned-gate checkpoint to step24600 did not restore a useful
  slope. The checkpoint h120 loop6 exact was `0.3281`, below the prior
  learned-gate checkpoint `0.3496` and not above the prior full-eval `0.3340`.
  The run was stopped deliberately. Treat this as evidence that a single scalar
  learned update gate helped the plateau once, but same-direction extra steps
  are now low ROI unless paired with a real state-dynamics or scaling change.
- A new detached worktree can fail before GPU training if `.cache/bin/ninja` is
  not linked or on `PATH`. The failed
  `d320-mb48eff96-h120-s13600-20260615T205559Z-785f3cd` launch is an environment
  abort, not a model result. Future remote launchers should always link the base
  repo-local ninja or prepend `/huyang2/double-loop/.cache/bin` and
  `/opt/conda/bin` before loading the RWKV CUDA extension.
- Treat GPU halts as platform events when a periodic checkpoint exists. The
  interrupted `d320-effb72-h120-resume9800-s12800-20260615T1703Z-785f3cd`
  segment resumed cleanly into
  `d320-effb72-h120-resume10000-s12800-20260615T1801Z-785f3cd`; do not count the
  interrupted segment as a model result.

## 2026-06-17 Maze proxy

- The no-Hydra EqR maze runner works on GPU1 and gives a cheap proxy for
  recurrence on grid path propagation. It runs online generated mazes, archives
  config/logs/source metadata, and does not require Hydra or flash-attn.
- The first 15x15 perfect-maze probe is neutral for FutureSeed transfer:
  FutureSeed loop8 path F1 was `0.5912`, base loop8 path F1 was `0.5905`, exact
  was `0.0` for both, and loop gain was only `+0.0008` for FutureSeed versus
  `+0.0000` for base.
- The harder Maze21 pressure setting is the first useful non-Sudoku signal.
  With 21x21 perfect mazes and path length 80-140, the 1200-step FutureSeed run
  reached loop10 path F1 `0.8051`, while the matched base reached `0.7614`.
  More importantly, FutureSeed showed a larger loop gain (`+0.1177` versus
  `+0.0518`) and reduced over-predicted path mass more aggressively. This
  suggests the loop is doing real path refinement, not just producing a static
  one-pass mask.
- The 2400-step Maze21 FutureSeed long-viz run opens exact path solving:
  loop1 exact/path F1 is `0.0000`/`0.7817`, while loop10 exact/path F1 is
  `0.9961`/`0.9997`. Train exact stayed zero through step1000, appeared at
  step1400, jumped to `0.6250` by step1800, and reached `0.9531` at step2400.
  The earlier 1200-step high-F1/no-exact result was therefore not a hard
  mechanism ceiling; it was under-computed for exact global cleanup.
- Maze21 visualizations show the actual loop behavior. Loop1 tends to mark a
  broad connected path region with many false-positive corridors. Loops 2-4
  remove most wrong branches while preserving the true path, and later loops
  polish rare misses. In the largest-gain case 191, loop1 has F1 `0.5972` with
  69 false positives and 16 misses; loop10 has F1 `1.0000` with zero false
  positives and zero misses. This is useful evidence that recurrence is doing
  iterative correction rather than a cosmetic confidence pass.
- On harder Maze31, state dynamics rather than candidate availability is now the
  main bottleneck. Dense ranking learned an average true-path vs false-positive
  score gap but did not prune the mask; the follow-up confidence-aware state
  competition (`maze31-cross-confstate-d256l4-s800-20260617T215136Z-91b7671`)
  kept non-collapsed candidate weights at loop12
  keep/proposed/context `0.3486`/`0.4412`/`0.2102` and high
  context-vs-proposed RMS `0.9635`, yet loop1-to-loop12 path F1 moved
  `0.5852 -> 0.5849`, precision `0.4158 -> 0.4154`, and predicted path fraction
  `0.4643 -> 0.4651`. Ten hard visualized cases stayed at zero false negatives
  but about 288 false-positive path cells. Do not spend the next budget on
  ranking-loss variants, gate-bias/temperature sweeps, or extra candidate
  diversity. The next high-ROI mechanism is a stronger generic recurrent
  decision-boundary update, such as learned threshold or normalization state,
  that can actually turn score gaps into mask sparsity without maze repair or
  rule priors.
- The direct learned decision-boundary probe
  (`maze31-boundary-d256l4-s800-20260617T224859Z-d538baa`) shows that the
  boundary can move, but current training pressure moves it in the wrong
  direction. Loop12 path F1 was `0.5854`, loop gain `-0.0003`, precision
  `0.4177 -> 0.4173`, and predicted path fraction `0.4587 -> 0.4593`. The
  learned threshold was not tiny: abs mean was about `0.30`. But raw path
  fraction was only `0.4339` at loop12 and calibration expanded it to `0.4606`;
  prune-flip fraction was `0.0` while add-flip fraction was `0.0267`. Ten hard
  visualized cases got slightly wider, with false positives `283.5 -> 285.1`.
  This reframes the bottleneck: a learned threshold is trainable and can affect
  predictions, but CE plus path-weight pressure rewards high-recall expansion
  once the true path is covered. Do not sweep threshold scale, gate bias, seed,
  or ranking variants. The next useful mechanism should alter generic recurrent
  training pressure so later loops are rewarded for reducing excess predicted
  mass without maze repair or topology rules.
- The late-loop path-mass pressure probe
  (`maze31-boundary-mass-w1-d256l4-s800-20260617T234853Z-3e1bfc3`) is the first
  clear sign that training pressure can reverse the expansion tendency. Loop12
  path F1 moved `0.5849 -> 0.5879`, precision `0.4154 -> 0.4235`, and predicted
  path fraction `0.4651 -> 0.4437`. Soft mass diagnostics moved in the intended
  direction too: non-PATH PATH probability `0.2630 -> 0.2137`, excess PATH
  probability fraction `0.1717 -> 0.1051`. But recall fell `1.0000 -> 0.9727`,
  true-path PATH probability fell `0.7880 -> 0.6500`, and hard visualized cases
  traded false positives `285.5 -> 276.5` for false negatives `0.0 -> 11.7`.
  This is a weak positive mechanism result, not a solved method. Do not sweep
  mass weight. The next high-ROI direction is a stronger generic
  recall-preserving pruning objective, focused on reducing excess predicted mass
  while explicitly preventing true-path probability collapse.
- The current visualization selector mostly captured solved largest-gain cases.
  That is good for explaining what the loop fixes, but not enough for studying
  the rare remaining failures after loop10. The next visualization upgrade
  should reserve final-failure cases first, then fill remaining slots with
  largest-gain cases.
- The Maze proxy reinforces the clean scaling principle: continue with more
  compute, harder tasks, and simple recurrent state dynamics. Do not pivot to
  selector, path repair, or handcrafted maze priors while this scalable route is
  still producing clear gains.
- Maze31 is the first clear failure frontier for the current small maze model.
  With 31x31 perfect mazes, path length 160-260, hidden 192, 2 layers, train
  loops 6 and eval loops 12, the run finishes at loop12 path F1 `0.5094` and
  exact `0.0000`; loop1 path F1 is `0.5203`, so the loop gain is negative
  (`-0.0109`). Training F1 oscillates around `0.55` from step200 through
  step1800 and CE stays near `0.35`, unlike Maze21 where exact opens late.
- The Maze31 failure visualizations are qualitatively different from Maze21.
  Maze21 loop1 gave a broad but useful path guess that later loops pruned. In
  Maze31 D192, loop1 already misses too much of the true path and later loops
  mostly preserve a wrong mask. Case 71 drops from loop1 F1 `0.3478` to loop12
  F1 `0.1763`, with false negatives rising from 87 to 127. This rejects "just
  add eval loops" as the next answer for Maze31.
- The immediate high-ROI Maze31 question is effective capacity, not base
  comparison. A D320 short capacity probe can decide whether the frontier is
  representation capacity or whether the next useful axis is curriculum/state
  dynamics. Do not extend the exact D192 Maze31 configuration.
- The D320 Maze31 capacity probe was interrupted after step600 by the GPU1
  lease, before final eval, but the early training curve is still informative:
  path F1 stays around `0.56` with exact `0.0`, matching the D192 plateau rather
  than showing a capacity breakout. Do not rerun the same D320 hard-from-step-1
  setup just for a final number. The next high-ROI Maze31 test is data/curriculum
  scaling, not another same-distribution width repeat.
- Maze31 path-length curriculum is a negative result with a useful cause. The
  `80-140` warmup reached only path F1 `0.1634` at step600, and the tighter
  `120-200` bridge collapsed to path F1 `0.0164` at step400. Shorter paths make
  PATH labels too sparse and encourage conservative non-PATH predictions. This
  is worse than hard-from-start, which at least learns a broad path mask.
- Maze31 simple depth scaling also does not open the frontier. Hidden-192 with
  4 layers reaches loop12 path F1 `0.5364`, exact `0.0`, and loop gain `-0.0008`.
  Loop1 and loop12 have almost identical precision/recall, so the loop is not
  revising the state. Width-alone, depth-alone, and naive curriculum are all low
  ROI now; the next useful step is a simple state-dynamics change or genuinely
  larger effective compute, not another one-axis table entry.
- Joint width+depth scale gives only a local mask improvement on Maze31, not
  global solving. D256/L4 reaches loop12 path F1 `0.5649`, exact `0.0`, and loop
  gain `-0.0013`. Recall rises to `0.8061`, but precision is only `0.4376` and
  predicted PATH fraction is `0.3559` versus true `0.1932`. The failure is now
  sharper: the model can cover much of the true path, but it cannot prune wrong
  branches, and the recurrent loop is nearly inert.
- Fixed linear state dynamics changes the coverage bias but does not solve
  Maze31. Positive delta-carry (`scale=0.35`, decay `0.95`) raises loop12 path
  F1 to `0.5721` and makes loop gain slightly positive (`+0.0012`), but it
  increases predicted PATH fraction to `0.3697` and does not improve precision.
  Negative delta-carry (`scale=-0.35`) collapses predicted PATH fraction to
  `0.1217` and recall to `0.2846`, yielding path F1 `0.3481`. This supports the
  state-dynamics thesis but rejects fixed sign/scale carry transforms; next work
  should be a small learned/gated update that can adaptively keep or prune.
- The first learned-gate implementation was diagnostic: applying the gate after
  logits left it untrained because loop carry is detached between calls. The
  corrected readout-gated run proves the gate can learn (`H` gate mean moves
  `0.896 -> 0.876`, std nonzero), but it still does not reopen recurrence:
  loop12 path F1 is `0.5851`, loop gain is only `+0.0001`, and predicted PATH
  fraction rises to `0.4646` versus true `0.1932`. A simple per-token keep gate
  is not enough; do not sweep gate bias. The next generic state update needs
  richer comparison/competition between candidate path hypotheses, not another
  scalar coverage knob.
- State competition is a small move toward pruning but not yet a loop mechanism.
  The `state_compete` run lets H state choose between previous, proposed, and
  context-competed candidates. It improves the broad-mask operating point versus
  readout-gate (`pred_frac 0.4556` vs `0.4646`, precision `0.4198` vs `0.4157`,
  loop12 F1 `0.5874` vs `0.5851`), but loop gain is still `-0.0001`. The
  softmax remains dominated by the proposed candidate (`~0.961` at loop12), so
  it does not yet use later loops to revise. Next work should increase real
  recurrence pressure or make the competing candidate stronger; do not turn this
  into a seed/gate-bias table.
- Final-only recurrence pressure does not fix `state_compete` collapse. The
  Maze31 D256/L4 final-loop-only run with `train_loops=8` reaches loop12 path F1
  `0.5852`, exact `0.0`, and loop gain `-0.0001`. Loop1 and loop12 are nearly
  identical: precision `0.4163 -> 0.4162`, predicted path fraction
  `0.4626 -> 0.4629`, and candidate weights stay collapsed around proposed
  `0.961`. Casebook failures show broad false-positive path branches copied
  from loop1 to loop12. This rejects simple loss-pressure as the next path; the
  next high-ROI mechanism is a stronger generic context/alternative candidate
  that can create real comparison before the softmax competition.
- Stronger generic candidate generation fixes weight collapse but not loop
  correction. `state_compete_cross` builds context from previous/proposed/delta
  and channel interaction, then applies noncausal attention. On Maze31 D256/L4
  it reaches loop12 path F1 `0.5847`, exact `0.0`, and loop gain only `+0.0002`.
  The weights are no longer collapsed (`keep/proposed/context =
  0.370/0.389/0.241` at loop12) and the context is genuinely different from
  proposed (`context_minus_proposed_rms ~1.03`), but precision and pred_frac
  barely move from loop1 to loop12 (`0.4156 -> 0.4160`, `0.4632 -> 0.4622`).
  This localizes the bottleneck: candidate diversity alone is insufficient; the
  alternative state must learn to carry error-correcting information across
  loops, likely via a simple temporal/predictive state objective rather than
  more gate, bias, temperature, or seed sweeps.
- Naive temporal prediction is learnable but teaches self-copying, not
  correction. The `state_compete_cross` predictive probe trained context logits
  at loop `t` to match stop-gradient logits at loop `t+1` with weight `0.1`.
  The predictive loss fell from `0.0094` to `0.0018`, and eval next-logit MSE
  fell to about `0.001`, so the auxiliary task worked mechanically. But loop12
  path F1 was only `0.5837`, exact `0.0`, and loop gain was effectively `0`.
  Precision/predicted fraction stayed frozen at `0.4142/0.4664`, recall was
  `1.0`, and casebook failures copied `290` false positives from loop1 to
  loop12. The lesson is sharp: predicting the next recurrent output just
  distills the broad mask. The next high-ROI objective must be improvement-aware
  or residual/contrastive, predicting how a later loop is better than loop1
  rather than predicting the next logits themselves.
- A simple CE-improvement target is also too weak for loop correction. The
  `state_compete_cross` context-improvement probe asked context logits to beat
  detached current logits on currently wrong tokens (`weight=0.1`,
  `margin=0.01`). The target was satisfied: at loop12 the diagnostic loss was
  `0.0`, context CE beat current CE by about `0.590` on current errors, and
  candidate weights stayed diverse (`keep/proposed/context =
  0.366/0.323/0.311`, `context_minus_proposed_rms = 0.950`). But loop12 path F1
  was `0.5830`, exact `0.0`, loop gain was `-0.0006`, and precision/predicted
  fraction worsened slightly from `0.4144/0.4654` to `0.4135/0.4672`. The hard
  cases still copy broad false-positive masks, e.g. `288 -> 290` false
  positives with no misses. This rejects token-level CE advantage as the next
  mainline; the correction signal must be tied to pruning or uncertainty
  sharpening, not just making context more confident on already-wrong cells.
- Ranking inside the current PATH mask is better aligned but still too weak in
  the hard min/max form. The `state_compete_cross` context-ranking probe trained
  context PATH scores so true-path cells inside the current predicted PATH set
  outrank false-positive PATH cells (`weight=0.05`, `margin=0.25`). This moved
  the operating point in the right direction but barely: loop1 to loop12
  precision was `0.4167 -> 0.4171`, predicted path fraction was
  `0.4608 -> 0.4596`, recall fell `0.9938 -> 0.9923`, and loop gain was only
  `+0.0001`. Candidate competition stayed healthy
  (`keep/proposed/context = 0.380/0.380/0.240`,
  `context_minus_proposed_rms = 1.004`), but the ranking loss stayed around
  `0.83` and the loop12 hard margin stayed negative (`-0.011`). Mean positive
  and negative PATH scores were almost identical. The lesson: ranking is the
  right family of signal for pruning, but the current hard min/max objective is
  too sparse or too hard. Do not seed-sweep it; make the pruning signal smoother
  or denser, or expose uncertainty/candidate contrast directly in the recurrent
  state update.
- Dense pairwise ranking fixes the average score separation but still does not
  produce loop-time pruning. The dense context-ranking probe used all true-path
  versus false-positive PATH pairs inside the current predicted PATH set
  (`weight=0.05`, `margin=0.25`, mode `dense`). It reaches loop12 path F1
  `0.5857`, exact `0.0`, and loop gain `+0.0003`. Precision and predicted path
  fraction move in the desired direction but only slightly (`0.4162 -> 0.4166`,
  `0.4634 -> 0.4629`), with recall nearly unchanged (`0.9984 -> 0.9982`).
  Unlike hard ranking, dense ranking learns a real average score preference:
  loop12 positive PATH score `4.0795` versus false-positive score `4.0246`,
  mean margin `+0.0548`. But the hard margin remains very negative (`-1.234`),
  ranking loss stays around `0.815`, and hard-case false positives barely move
  (`286.4 -> 286.2` average). The lesson: average pairwise preference is not
  enough to change the recurrent operating point. The next state-dynamics work
  should expose uncertainty/candidate contrast to the state or learn a simple
  threshold/normalization mechanism; do not keep adding ranking-loss variants.
- The useful signal is diagnostic, not positive: the model is mostly learning a
  broad path mask. In the FutureSeed run, the true path fraction was `0.1785`
  while the predicted PATH fraction was `0.4209`; recall was almost `1.0` but
  precision was only `0.4236`. This explains why exact is zero and why loops do
  not matter yet.
- Do not use this 15x15 run to claim FutureSeed works on maze. The next maze
  experiment should make recurrence pressure real, for example by increasing
  path length/grid size or by tracking whether later loops reduce over-predicted
  PATH cells. A flat seed table at the same 15x15 setting would be low ROI.
- The 21x21 recurrence-pressure run does create a meaningful loop signal.
  With path range `80-140`, path-loss weight `1.5`, hidden `192`, layers `2`,
  and train/eval loops `6/10`, FutureSeed reaches loop10 path F1 `0.8051`
  versus base `0.7614`. FutureSeed's loop gain is `+0.1177`; base loop gain is
  `+0.0518`. This supports recurrence as a transferable mechanism beyond
  Sudoku.
- The Maze21 FutureSeed gain is not just more PATH recall. At eval, the true
  path fraction is `0.2070`; FutureSeed moves predicted path fraction from
  `0.3270` at loop1 to `0.2797` at loop10, while precision rises
  `0.5640 -> 0.7031`. Base improves too, but remains broader at loop10
  predicted path fraction `0.3198` and precision `0.6308`. This is the first
  clean non-Sudoku evidence that FutureSeed plus loop can refine an over-broad
  global hypothesis.
- Exact path solving is still not opened on Maze21 held-out eval: both arms have
  loop10 exact `0.0`. Do not overclaim. The next high-ROI maze question is
  whether more compute or a simple state-dynamics change can sharpen from
  high-F1 masks to exact single paths. A seed table at the same budget is lower
  ROI than increasing recurrence pressure or examining exact-failure cases.
- Guarded path-mass pressure preserves recall but kills pruning. The
  `maze31-boundary-massguard` probe changed the late-loop mass objective so that
  when true-path probability fell below the loop1 floor, the model optimized
  only the recall guard and stopped applying non-path mass pressure. It did
  preserve recall (`1.0000 -> 1.0000`), but loop12 exactly copied loop1 at the
  hard mask level: path F1 `0.5847 -> 0.5847`, precision
  `0.4152 -> 0.4152`, predicted PATH fraction `0.4653 -> 0.4653`, hard-case
  false positives `288.2 -> 288.2`, and false negatives stayed `0.0`. Soft
  non-path PATH probability moved only slightly (`0.2627 -> 0.2561`), while the
  guard was active on every eval sample at loop12
  (`path_mass_guard_violation_frac = 1.0`). The lesson: recall-preserving
  pruning cannot be a stop-pressure guard. If the guard disables false-positive
  gradients, loops learn to copy the broad mask. The next objective must keep
  positive-cell margins and false-positive mass pressure active at the same
  time, for example through a normalized margin or constrained/Lagrangian form,
  not another sweep of mass weight, threshold, margin, or seed.
- Simultaneous path-mass pressure with a true-cell probability floor avoids the
  two obvious failures but still does not create useful loop correction. The
  `maze31-boundary-massconstr` retry kept non-path PATH-mass pressure active and
  added a cell-level loop1 probability floor on true-path cells. It no longer
  froze perfectly like guarded pressure, and it did not collapse recall like the
  original soft mass run. But the effect was tiny: loop1 to loop12 path F1
  `0.5836 -> 0.5841`, precision `0.4144 -> 0.4151`, recall
  `0.9984 -> 0.9971`, predicted PATH fraction `0.4655 -> 0.4641`, and hard-case
  false positives/false negatives `288.1/0.5 -> 287.5/1.3`. The positive floor
  was active on `30.7%` of true-path cells at loop12, but non-path PATH
  probability slightly worsened (`0.2556 -> 0.2581`). Lesson: probability-floor
  regularization mostly creates small calibration tradeoffs, not a robust
  prune/keep decision. Do not sweep this floor mode. The next high-ROI direction
  should change the recurrent decision variable itself, for example a learned
  per-loop budget/normalization state or a contrastive boundary objective that
  directly separates true-path from false-positive PATH candidates while
  preserving true-path margins.
- Learned recurrent budget state is not enough when the training pressure still
  points at broad masks. The `maze31-budgetstate-massconstr` probe added a
  generic per-sample per-loop budget shift to the decision boundary, using only
  PATH margin, soft PATH mass, entropy, candidate disagreement, candidate
  weights, and loop index. The module was active and candidates stayed diverse
  (`keep/proposed/context = 0.376/0.444/0.181`, context-proposed RMS `1.204`),
  but the learned boundary moved in the wrong direction: loop12 threshold mean
  was negative (`-0.4865`), prune flips were exactly `0.0`, and add flips were
  about `0.0200`. Loop1 to loop12 improved only slightly in F1
  (`0.5852 -> 0.5861`) and precision (`0.4168 -> 0.4184`), while recall dropped
  (`0.9938 -> 0.9897`). Hard visualized cases confirm the failure mode:
  false positives barely changed (`283.4 -> 282.2`) while false negatives rose
  (`3.3 -> 5.6`). Lesson: boundary capacity is no longer the clean bottleneck.
  If the objective rewards coverage more than correction, a learned budget
  variable also becomes an expander. Do not sweep budget scale, threshold bias,
  seed, or margin; the next work should change the generic self-correction
  pressure itself.
