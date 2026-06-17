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
