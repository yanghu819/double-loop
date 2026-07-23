# GDN Unseen-Data Coverage Continuation

## 1. Metainfo

- Plan ID: `P-SCALE-036`
- Status: done
- Date: 2026-07-23
- Machine: AIStation GPU1 A800 only
- Parent: clean FutureSeed-GDN exact step30000 train-state checkpoint
- Branch: `codex/gpu1-data-coverage`
- Control source SHA: `4ef23df55f3db517a7da5041f3385b9a959ba031`
- Matched-unseen source SHA:
  `648ee75f94c794e11174008e6913c4f807866c60`
- Formal worktree:
  `/huyang2/double-loop/.worktrees/pscale036-formal-4ef23df-20260723`
- Final matched-unseen worktree:
  `/huyang2/double-loop/.worktrees/pscale036-matched-resume-648ee75-20260723`

## 2. Hypothesis

The clean step30000 run consumed about one dataset-equivalent number of board
draws, but it sampled with replacement. Nominal compute therefore overstates
the number of independent relations actually observed.

If the remaining hard-Sudoku error is still data-limited, directing the next
1500 optimizer steps to rows never drawn in the first 30000 steps should beat
an equal-step continuation from the full pool. If it does not, additional
sampler engineering is low ROI and the next axis must change generic model
capacity or state formulation.

## 3. Configuration

Both arms resume the same model, AdamW state, feature-buffer state, and RNG:

- local Triton-recurrent GDN, D224/L12/H14/D16, expand-v4;
- native FutureSeed scale 1 and fixed unit normalization;
- five loops with equal CE supervision at every loop;
- BF16, microbatch 32, gradient accumulation 4, effective batch 128;
- official full-diversity 9x9 Sudoku;
- hard stage 51-64 blanks;
- no noise, scratch, feedback, selector, rollout, repair, search, or task rule.

The only difference is the training row pool:

- `control`: original full 51-64 candidate pool with replacement;
- `unseen`: a strict subset containing only rows never drawn before step30000.

## 4. Environment

- Remote root: `/huyang2/double-loop`
- Worktree: recorded after launch
- Python: `/opt/conda/bin/python`
- CUDA device: `CUDA_VISIBLE_DEVICES=0`
- GPU2 and CPU model smoke: forbidden

## 5. Commands

Build the unseen index only after exact RNG validation:

```bash
/opt/conda/bin/python scripts/build_unseen_official_sudoku_index.py \
  --data-dir /huyang2/double-loop/data/sudoku-extreme-full \
  --checkpoint /huyang2/double-loop/models/gdn-full-diversity-d224l12-s6000-20260711T1510Z-eeb38f5/checkpoints/train_state_step030000.pt \
  --output /huyang2/double-loop/data/sudoku-extreme-full/indices/unseen-after-step030000-seed52-b128-51-64.npy \
  --manifest /huyang2/double-loop/data/sudoku-extreme-full/indices/unseen-after-step030000-seed52-b128-51-64.json
```

Matched formal arms:

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/run_gdn_data_coverage.sh control
CUDA_VISIBLE_DEVICES=0 ./scripts/run_gdn_data_coverage.sh unseen
```

## 6. Artifacts

The exact RNG reconstruction gate passed before GPU training:

- checkpoint SHA256:
  `b3811baf739de916b5c5724a1bf1e6926fb9959093a2c793b5dadd1e4fac0f23`;
- index:
  `/huyang2/double-loop/data/sudoku-extreme-full/indices/unseen-after-step030000-seed52-b128-51-64.npy`;
- index SHA256:
  `60d65c91d9921109d50b485f5f746219ab7e68f4da2beee90a556a222546901b`;
- manifest:
  `/huyang2/double-loop/data/sudoku-extreme-full/indices/unseen-after-step030000-seed52-b128-51-64.json`;
- target 51-64 rows: `3,741,961`;
- unique target rows seen before step30000: `2,359,779` (`63.0626%`);
- strict unseen rows: `1,382,182`.

GPU1 CUDA smoke:

- run:
  `gdn-data-coverage-unseen-smoke-s30001-20260723T1-4ef23df`;
- result: one resumed BF16 optimizer step completed through the local
  `triton_recurrent` GDN path, followed by the full mixed, official
  blank-range, and case-bank evaluation; exit code `0`.

Formal control:

- run:
  `gdn-data-coverage-control-s31500-20260723T1030Z-4ef23df`;
- launch PID: `3514`;
- launch log:
  `/huyang2/double-loop/artifacts/launch/p036-control-20260723T1030Z/train.log`;
- state at launch: exact source SHA, `git_dirty=0`, GPU1 only.
- metadata archive:
  `/huyang2/double-loop/artifacts/p036-control-metadata-20260723T1030Z.tgz`;
- metadata archive SHA256:
  `ac529242f23cac8c024a905e7903cbec5f4c49c9984bef6d32efe07352563ea6`.

First formal unseen attempt:

- run:
  `gdn-data-coverage-unseen-s31500-20260723T1204Z-4ef23df`;
- worktree:
  `/huyang2/double-loop/.worktrees/pscale036-unseen-4ef23df-20260723`;
- launch PID: `5193`;
- launch log:
  `/huyang2/double-loop/artifacts/launch/p036-unseen-20260723T1204Z/train.log`;
- state at launch: exact source SHA, `git_dirty=0`, strict unseen index,
  GPU1 only.
- status: aborted by exact process group at step30200 before any checkpoint;
- reason: the all-unseen pool did not preserve the control blank-count
  distribution. The full 51-64 pool has mean `55.9917` blanks and 51-55 rows
  are `43.00%`; all strict-unseen rows have mean `56.2601` blanks and 51-55
  rows are only `32.01%`. Continuing would confound independent coverage with
  a harder curriculum.

Corrected formal unseen pool:

- strict unseen rows remain the source set;
- deterministically select `990,000` rows with selection seed `2052`;
- match the full control pool's exact per-blank-count proportions across
  blank counts 51 through 64 using largest-remainder integer allocation;
- record the input/output histograms and output index SHA256 in a schema-v2
  manifest;
- rerun from the original step30000 checkpoint, not from the invalid attempt.

The initial requested size of `1,000,000` was rejected before writing an
index: matching the control histogram would require 2,229 blank52 rows, while
only 2,219 strict-unseen blank52 rows exist. The feasible `990,000`-row build
passes every quota and has:

- output mean blanks `55.991736`, versus control `55.991735`;
- output index SHA256
  `75bd706d38de29cb23b04e8033066af3e29b31fad6d8f287049e64efb6997f3c`;
- exact RNG/checkpoint reconstruction match;
- source SHA `648ee75f94c794e11174008e6913c4f807866c60`.

Matched step30500 leg:

- run:
  `gdn-data-coverage-unseen-histmatched990k-s30500-20260723T1220Z-648ee75`;
- launch PID: `6351`;
- launch log:
  `/huyang2/double-loop/artifacts/launch/p036-unseen-matched-s30500-20260723T1220Z/train.log`;
- parent: original clean step30000 checkpoint;
- state at launch: clean detached source, `git_dirty=0`, GPU1 only.

Matched step30500 to step31500 leg:

- run:
  `gdn-data-coverage-unseen-histmatched990k-resume30500-s31500-20260723T133526Z-648ee75`;
- launcher PID: `428`;
- launch log:
  `/huyang2/double-loop/artifacts/launch/p036-unseen-matched-resume-20260723T133526Z/train.log`;
- parent: exact matched-unseen step30500 train-state checkpoint;
- step31000 and step31500 fixed evaluations were predeclared before launch;
- completed with exit code `0` on GPU1;
- metadata archive:
  `/huyang2/double-loop/artifacts/p036-unseen-matched-s31500-metadata-20260723T133526Z.tgz`;
- archive SHA256:
  `ef78c422bb0677fe1e65a2dc65e2d95a923acc6641bf5c20ff8add2d3fef19ad`.

## 7. Results

The matched control completed with exit code `0`:

| Control readout | Value |
|---|---:|
| mixed loop1 exact | `0.0234` |
| mixed loop2 exact | `0.0547` |
| mixed loop3 exact | `0.2988` |
| mixed loop4 exact | `0.4297` |
| mixed loop5 exact | `0.4590` |
| official 46-50 loop5 exact | `1.0000` |
| official 51-55 loop5 exact | `0.6094` |
| official 56-64 loop5 exact | `0.3086` |
| train CE | `0.5161` |
| train time | `5224.2s` |
| peak allocated CUDA memory | `44527.1MB` |

Relative to the clean step30000 parent, control changes mixed exact
`0.4805 -> 0.4590`, official 51-55 `0.6152 -> 0.6094`, and official 56-64
`0.3848 -> 0.3086`. Blind continuation from the original replacement pool
therefore does not extend the clean scaling curve at this endpoint.

The distribution-matched strict unseen arm also completed:

| Readout | Parent step30000 | Control step31500 | Unseen step31500 | Unseen-control |
|---|---:|---:|---:|---:|
| mixed loop5 exact | `0.4805` | `0.4590` | `0.4551` | `-0.0039` |
| official 46-50 loop5 exact | `1.0000` | `1.0000` | `1.0000` | `+0.0000` |
| official 51-55 loop5 exact | `0.6152` | `0.6094` | `0.6367` | `+0.0273` |
| official 56-64 loop5 exact | `0.3848` | `0.3086` | `0.3711` | `+0.0625` |

The separate 256-board case-bank batch repeats the direction:

| Case-bank readout | Parent | Control | Unseen | Unseen-control |
|---|---:|---:|---:|---:|
| 51-55 loop5 exact | `0.6289` | `0.5898` | `0.6406` | `+0.0508` |
| 56-64 loop5 exact | `0.3516` | `0.3086` | `0.3789` | `+0.0703` |

Across the two different hard held-out batches, control solves `237/768`
boards and unseen solves `287/768`: `+50` boards, or `+0.0651` absolute.
An unpaired binomial normal approximation gives a 95% interval of
`[+0.0178,+0.1124]`. This approximation does not use the favorable
same-board pairing because per-board formal-eval flags were not persisted.

Fixed-hole exact at the final unseen endpoint is:

- holes53: `0.4824`;
- holes60: `0.4824`;
- holes64: `0.4590`.

The unseen mixed loop1 through loop5 exact curve is
`0.0234/0.1289/0.3496/0.4355/0.4551`. The two matched-unseen legs take
`1898.3 + 3495.6 = 5393.9s`, versus `5224.2s` for control, only `3.2%`
more wall time. Both allocate `44527MB` CUDA memory.

Visual inspection shows both the value and the remaining limit of loops:

- official 56-64 case `b0048`, with 60 hidden cells, changes
  `36 -> 28 -> 10 -> 0 -> 0` wrong cells and
  `26 -> 23 -> 19 -> 0 -> 0` conflicts. This is real iterative correction,
  not merely higher local accuracy.
- official 56-64 hard case `b0041`, with 56 hidden cells, changes
  `27 -> 9 -> 3 -> 4 -> 6` wrong cells. Later loops undo part of the useful
  loop3 state, so recurrent stability is still unsolved.

## 8. Decision

The arm passes the predeclared mechanism gate: official 56-64 exact improves
`+0.0625` over matched control, official 51-55 is preserved, and the second
held-out batch repeats both positive directions.

The claim must remain narrow. Fresh independent rows prevent most of the
hard-tail degradation caused by blind replacement-sampled continuation, but
they do not push the scalar frontier beyond the step30000 parent: unseen is
still `-0.0039` behind control on mixed exact and `-0.0137` behind the parent
on official 56-64 exact. Therefore:

- effective independent coverage is a real scaling variable;
- nominal optimizer steps over repeated rows overstate useful data scale;
- late continuation on fresh rows is not by itself the final scaling recipe;
- do not sweep sampler seeds, subset sizes, or blank weights.

The next high-information data experiment should change the training regime
from the start: deterministic epoch-style fresh coverage on the same clean
backbone, with the existing replacement sampler as the single matched control.
That asks whether better data scaling extends the frontier rather than merely
reducing late-stage forgetting.

## 9. Publication Record

Not a submission artifact and no tag: the primary scalar score is below
`0.50`. The result supports a clean data-efficiency mechanism claim, but not
an overall new best checkpoint.
