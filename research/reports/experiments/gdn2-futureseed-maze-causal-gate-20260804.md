# GDN2 FutureSeed Maze causal gate

## Metainfo

- Plan: `P-CAUSAL-002`
- Status: preregistered; strict CUDA preflight pending
- Date: 2026-08-04
- Machine: AIStation task-mode GPU1 only
- Seed: 52 only

## Question

Does native FutureSeed improve a causal official-FLA GDN2 carrier outside
Sudoku, or is the large step12000 effect task-specific?

Official EqR Maze-Unique is already reproduced from the released checkpoint:
D16/B1 exact `0.827`, D64/B1 `0.892`, and D64/B128 top-1 `0.9444`. This gate
does not rerun that checkpoint. It tests the narrower causal mechanism claim in
our own matched recurrent carrier under the same fixed official Maze data and
path-aware evaluator.

## Mechanism Hypothesis

The 30x30 maze is serialized into 900 tokens. A causal GDN2 layer cannot use
later walls, the goal, or later path structure when updating an early token.
Native FutureSeed passes a terminal recurrent state from one layer into the
next layer's initial state, providing a cheap route for later-token information
without a reverse scan or noncausal mixer. If this information route is
general, it should lower CE and improve path precision/F1 or exact accuracy
relative to the same GDN2+loop model with the seed disabled.

## Experiment

One pair only, on official `maze-30x30-unique-1k`:

- strict official FLA `GatedDeltaNet2`, pinned SHA `9c8e42e`, chunk/Triton,
  backend dispatch disabled, no fallback;
- D128/L6/H4/D32, expand-v1, short-conv4;
- train loops2, eval loops5, every-loop plain token CE;
- path weight exactly `1.0`; no class reweighting or Maze-specific loss;
- batch32, 300 optimizer steps, BF16, seed52;
- no-FutureSeed: scale `0`; FutureSeed: scale `1`;
- 128 fixed test boards, reporting token accuracy, exact, path F1,
  precision/recall, predicted path fraction, FP/FN, CE, loop1-to-loop5 change,
  speed, memory, and same-board hard-case visualization.

The frozen data hashes are:

- train inputs: `2922455815232dcb636ee45f335e7ce834aa45d4fab743b96ac97c17284ca7e2`;
- train labels: `6b1a99761606b78f85fe43085b158457f22a3d6274b1e09f12f7b16814fdb5e4`;
- test inputs: `c501b3f75ec018e27fd6d00534b6738ebcc555da50536909e530787d16b1ce0f`;
- test labels: `20b21168459847fb260d986394098addf7cca94becc7e682dcb358967d5993dc`.

This is an opening gate, not a paper-scale result. Only a positive signal earns
one longer continuation with the same model and data.

## Prediction

FutureSeed should produce at least one of:

- path F1 `>= +0.03` over no-FutureSeed without increasing predicted path
  fraction or FP;
- train CE at least `0.10` lower together with better path precision/F1;
- nonzero exact or materially larger loop1-to-loop5 FP reduction while the
  control remains closed.

## Kill Criteria

- Stop before training on wrong GPU count, dirty/different source, wrong data
  shape/hash, nonofficial FLA class/SHA, non-Triton convolution, backend
  dispatch, fallback, OOM, NaN, or failed real CUDA backward.
- The full-size one-step preflight must fit below 75 GiB and finish; otherwise
  stop and redesign the compute budget rather than silently changing the arm.
- After both 300-step arms, stop this scale if path F1 delta is below `0.01`,
  CE delta is below `0.05`, exact is zero in both, and loop correction is
  neutral. Do not add a seed, class weight, loss, width, or temperature sweep.

## Claim Enabled

A positive result supports cross-task evidence that FutureSeed improves the
information and optimization path of causal linear-attention recurrence. It
does not yet claim to beat released EqR or solve Maze. A negative result bounds
the Sudoku evidence and redirects the next generalization test to a cleaner
language/retrieval task.
