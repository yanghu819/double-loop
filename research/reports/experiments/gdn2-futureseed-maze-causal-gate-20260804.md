# GDN2 FutureSeed Maze causal gate

## Metainfo

- Plan: `P-CAUSAL-002`
- Status: done; stopped by the preregistered invalid-carrier criterion
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

## Execution Integrity

- Source was the clean detached GitHub SHA
  `dd252b71b530ea4a0b34d215fb852adbd08c279b`.
- AIStation task mode exposed exactly one A100 80GB GPU; all model execution,
  including preflight, forward, backward, and evaluation, used GPU1.
- The full-size CUDA preflight completed with 1,606,897 parameters and peak
  memory `14.23 GiB`. After the first Triton compile, two additional training
  steps took about one second, so the registered pair was feasible.
- Both arms used official FLA `GatedDeltaNet2` at pinned SHA `9c8e42e`, official
  layer forward, chunk recurrence, Triton short convolution, and disabled
  backend dispatch. No fallback was observed.
- The launcher's optional human-readable `condition` string was empty because
  the runner expects `RWKV_MAZE_CONDITION`; arm identity remains unambiguous in
  each run name and the archived `future_seed_scale` value. This metadata typo
  does not alter execution.

## Results

| Readout | No FutureSeed | Native FutureSeed | FS - control |
|---|---:|---:|---:|
| Step-300 train CE | 0.26410 | 0.26494 | +0.00085 |
| Loop-1 token accuracy | 0.87011 | 0.87012 | +0.00001 |
| Loop-5 token accuracy | 0.87015 | 0.87012 | -0.00003 |
| Loop-5 exact | 0 | 0 | 0 |
| Loop-1 path F1 | 0.00092 | 0.00091 | -0.00001 |
| Loop-5 path F1 | 0.00171 | 0.00117 | -0.00054 |
| Loop-5 precision | 0.10156 | 0.07031 | -0.03125 |
| Loop-5 recall | 0.00086 | 0.00059 | -0.00027 |
| Loop-5 predicted-path fraction | 0.00017 | 0.00013 | -0.00004 |
| Loop-5 mean FP / FN | 0.05 / 116.81 | 0.05 / 116.84 | -0.01 / +0.03 |
| Loop-1 to loop-5 F1 gain | +0.00079 | +0.00026 | -0.00053 |
| Wall time | 166.2 s | 199.3 s | +19.9% |
| Peak CUDA memory | 14.17 GiB | 14.23 GiB | +0.05 GiB |

## Visual Diagnosis

The independently selected hardest-case indices are identical in both arms:
`5, 9, 11, 3, 4, 7, 8, 2`. Both models predict essentially no PATH cells at
loops 1, 4, and 5. On case 5, both leave all 135 true-path cells as false
negatives at every displayed loop. The apparent `0.87` token accuracy is just
the fraction of non-PATH cells, not maze-solving ability.

The paired same-board visualization is archived at
`runs/gdn2-maze-causal-comparison-20260804T0121Z-dd252b/visualizations/index.html`.

## Decision

The preregistered stop condition fires: both exact scores are zero, path-F1
delta is below `0.01`, CE delta is below `0.05`, and later loops are neutral.
This is **not** negative evidence about FutureSeed. The matched causal GDN2
carrier never learned Maze, so there is no working behavior for FutureSeed to
improve. It only falsifies this 300-step plain-token-CE carrier recipe.

Do not rescue it with class weights, extra loss terms, a longer no-FS run, a
second seed, width, or temperature sweeps. Official EqR's released checkpoint
still reaches top-1 exact `0.9444` on the same fixed test data, so the dataset
and evaluator are valid. The next cross-task causal gate should use a cleaner
language/retrieval proxy with an already verified no-FutureSeed GDN2 baseline
and a balanced query-level target.
