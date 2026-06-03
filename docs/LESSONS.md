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
