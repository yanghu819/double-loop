# official-eqr-sudoku-quick5-20260623

## 1. Metainfo

- Plan ID: `P-EQR-010`
- Status: stopped after quick gate; not continuing 5-seed table fill
- Machine: AIStation `GPU1` only
- Start time UTC: `2026-06-23T11:17:56Z`
- Local branch: `codex/gpu1-experiment-tracking`
- Remote work dir: `/huyang2/double-loop/official_eqr_sudoku_repro_20260623`
- Source SHA for launcher/tracking:
  initial `265a5c09479d90586bebaf63252d232156640ae2`; resume
  `1e9cd09b6f828443595cb26c94121032a7a3c8dc`
- Remote launcher PID: initial `946`; resume `201`

## 2. Hypothesis

The official EqR Sudoku-Extreme released checkpoint should reproduce the README
quick 5-seed mean/std under the official codebase, data, checkpoint, and
`config/eval/sudoku_lite_noise05.yaml`. This is the baseline gate before any
FutureSeed-vs-EqR claim.

## 3. Configuration

- Official EqR upstream SHA: `aba94e9cde0f273ce644db5261cd6915ba6561f0`
- Dataset: official `sudoku-extreme-1k-aug-1000`
- Checkpoint: official HF `locuslab/EqR-model/sudoku-extreme/eqr.pth`
- Eval config: `config/eval/sudoku_lite_noise05.yaml`
- Eval settings: `halt_max_steps=64`, `different_init=128`,
  `convergence_top_k=4`, `noise_scale=0.5`, `init_std=1.0`,
  `global_batch_size=128`, `max_eval_steps=16`
- Seeds: reuse completed seed `0`; run missing seeds `1,2,3,4`
- Compile: `DISABLE_COMPILE=1`
- GPU: `CUDA_VISIBLE_DEVICES=0`

## 4. Environment

- GPU: `NVIDIA A800-SXM4-80GB`
- Python/Torch env: `/huyang2/double-loop/official_eqr_compare/.venv`
- Attention path: PyTorch SDPA compatibility fallback. The literal
  FlashAttention path is blocked on this host by wheel glibc and nvcc/Torch CUDA
  mismatch, but the same fallback reproduced official Maze exact metrics.

## 5. Commands

```bash
cd /huyang2/double-loop/.worktrees/official-eqr-sudoku-quick5-ca6128f
bash runs/official-eqr-sudoku-quick5-20260623/launch_official_sudoku_quick5.sh
```

Actual detached launch worktree:

```bash
cd /huyang2/double-loop/.worktrees/official-eqr-sudoku-quick5-265a5c0
nohup bash runs/official-eqr-sudoku-quick5-20260623/launch_official_sudoku_quick5.sh \
  > /huyang2/double-loop/official_eqr_sudoku_repro_20260623/artifacts/sudoku_quick5_20260623/launcher.nohup.log 2>&1 &
echo $! > /huyang2/double-loop/official_eqr_sudoku_repro_20260623/artifacts/sudoku_quick5_20260623/launcher.pid
```

## 6. Artifacts

- Remote launcher output:
  `/huyang2/double-loop/official_eqr_sudoku_repro_20260623/artifacts/sudoku_quick5_20260623`
- Remote per-seed logs:
  `/huyang2/double-loop/official_eqr_sudoku_repro_20260623/artifacts/sudoku_quick5_20260623/logs`
- Remote per-seed metrics:
  `/huyang2/double-loop/official_eqr_sudoku_repro_20260623/artifacts/sudoku_quick5_20260623/metrics`
- Local launcher:
  `runs/official-eqr-sudoku-quick5-20260623/launch_official_sudoku_quick5.sh`

## 7. Results

Pending. Seed0 was already reproduced in `official-eqr-sudoku-repro-20260623`.
This run fills seeds 1-4 and will aggregate all 5 seeds.

Launch check at `2026-06-23T11:18:16Z`: seed1 was running on GPU1 with about
`63762 MiB / 81920 MiB` memory in use and `100%` GPU utilization.

Progress check at `2026-06-23T11:45:54Z`: seed1 was still healthy at `2/16`
eval batches. Runtime was about `600s` per eval batch under GPU1 SDPA fallback,
with about `71954 MiB / 81920 MiB` memory in use and `100%` GPU utilization.
No per-seed metrics had landed yet, as expected before seed1 completion.

Progress check at `2026-06-23T12:06:53Z`: seed1 was still healthy at `4/16`
eval batches, with the same about `600s` per eval batch, `71954 MiB / 81920
MiB` memory in use, and `100%` GPU utilization. No FutureSeed or other training
job was launched while this official baseline gate was running.

Progress/interruption check at `2026-06-23T12:11-12:15Z`: seed1 advanced to
`5/16` eval batches with GPU1 still at `100%` utilization and about `71954 MiB /
81920 MiB` in use. A later helper call failed to acquire a shell link; row
status then reported GPU1 `Halt` with remaining time `-111`. This was an
AIStation/runtime interruption, not a model/eval error. GPU1 was reopened with
new work-platform id `881d650b-7eb9-4c95-9793-fdc1f1b47b99` and was still
`Pending` as of the last status check. After it reaches `Running`, inspect the
remote artifacts before relaunching; do not assume seed1 wrote metrics.

Post-restart inspection at `2026-06-23T13:00Z`: GPU1 was `Running`, SSH/CUDA
probe passed, and the GPU was idle. The quick5 artifact directory had no
`summary.json` and no files under `metrics/`; `logs/seed1.log` ended at `5/16`
eval batches. This confirms seed1 did not complete before the AIStation halt.
The launcher was made resume-safe so completed seed metrics are skipped on
future restarts and summary generation only happens after seeds `1,2,3,4` are
all present. The next launch should rerun missing seeds from the committed
resume-safe source SHA, still on GPU1 only.

Resume launch at `2026-06-23T12:28:06Z`: worktree
`/huyang2/double-loop/.worktrees/official-eqr-sudoku-quick5-1e9cd09` was created
at detached SHA `1e9cd09b6f828443595cb26c94121032a7a3c8dc`, an
`aistation_halt_abort_20260623T1215Z.json` artifact was written, and the
resume-safe launcher started as PID `201`. Follow-up check showed seed1
`evaluate.py` active with GPU1 at `63762 MiB / 81920 MiB` and `100%`
utilization. No other FutureSeed or method experiment is running.

Health check after resume at `2026-06-23T12:41Z`: seed1 reached `1/16` eval
batches at `600.10s/it`; GPU1 was using `71954 MiB / 81920 MiB` at `100%`
utilization. No metrics were expected yet because the launcher copies a
per-seed metric JSON only after the full seed completes.

Progress check at `2026-06-23T13:18:17Z`: seed1 reached `5/16` eval batches
after the resume launch, again at about `600s/it`, with GPU1 still at
`71954 MiB / 81920 MiB` and `100%` utilization. This reproduces the earlier
pre-halt progress point under the new detached source SHA; no metrics have
landed yet, as expected before seed completion.

Seed1 completed at `2026-06-23T15:08:28Z` and `metrics/seed1.json` was copied
from the official EqR `eval_preds` directory; seed2 launched immediately after.
Seed1 metrics:

| seed | top1 exact | top4 exact | majority exact | any-correct |
|---:|---:|---:|---:|---:|
| 1 | 0.99072265625 | 0.9866943359375 | 0.98681640625 | 0.99072265625 |

The official metric JSON stores values under a top-level `metrics` dictionary,
so the resume launcher aggregation was updated to read `data["metrics"]` when
present and fall back to flat JSON for older copied artifacts.

Seed2 was preempted intentionally at `2026-06-23T16:07:30Z` after reaching
`5/16` eval batches. GPU1 had only about `806s` remaining, while a full official
seed takes about `9600s`; continuing could not produce a valid per-seed metric.
Exact PIDs `201`, `510`, `579`, and `587` were terminated, GPU memory returned
to `0 MiB`, and remote artifact
`seed2_preempt_insufficient_lease_20260623T1607Z.json` records the event. The
next resume should start missing seeds from seed2 using the latest committed
resume-safe source SHA.

GPU1 reopened as work-platform `d6e90210-2156-4a6a-a7d0-256ca62bbfde` and was
validated with an idle `NVIDIA A800-SXM4-80GB`. Resume launch at
`2026-06-23T16:32:49Z` used detached source SHA
`95aa1dc93da84fe6b10acf66fcba4c165a20621a`, with completed `seed1.json`
present, and started missing seeds `2 3 4` as launcher PID `124`. Follow-up
check showed seed2 `evaluate.py` active with GPU1 at `63762 MiB / 81920 MiB`
and `100%` utilization.

Seed2 completed at `2026-06-23T19:12:57Z` and `metrics/seed2.json` was copied.
Seed2 metrics:

| seed | top1 exact | top4 exact | majority exact | any-correct |
|---:|---:|---:|---:|---:|
| 2 | 0.9931640625 | 0.9859619140625 | 0.984375 | 0.9931640625 |

The same launcher advanced to seed3, but GPU1 later halted before seed3
completion. Post-restart inspection at `2026-06-24T02:48:33Z` found
`seed1.json` and `seed2.json` only, no `summary.json`, no active eval process,
and `logs/seed3.log` stopped at `7/16` eval batches. Seed3 therefore remains
missing and must be rerun from the start with seed4.

GPU1 reopened as work-platform `783e31e7-1b0a-40f4-b1d9-6b38968f7671` and
validated as an idle `NVIDIA A100-SXM4-80GB`. Because GitHub push for local
tracking commit `d3af3de` temporarily failed with `Empty reply from server`,
the resume launch used already-pushed detached source SHA
`6af61e42f83056cd6e75a9841c4604ab3c6e83f9`; launcher code is unchanged for the
purpose of continuing seeds. Missing seeds `3 4` were relaunched at
`2026-06-24T02:54:55Z` as launcher PID `184`, with seed3 active on GPU1 at
`63765 MiB / 81920 MiB` and `100%` utilization.

## 8. Conclusions

Stopped early by decision on `2026-06-24`: the goal is not to fill a 5-seed
table. Official Maze released-checkpoint metrics were already reproduced
against the paper, and Sudoku quick seed0/1/2 are in the expected official
range. That is enough to establish the baseline/eval path for the next
decision. Continuing seeds 3/4 has low information gain relative to testing the
mechanism claim. Seed3 was stopped at about `3/16` eval batches; seed4 was not
run.

Next decision: run the single high-information official EqR FutureSeed probe:
causal cheap backbone vs causal cheap + reverse-causal FutureSeed, one run each,
same data and train budget, no seed sweep.

## 9. Submission Record

Not applicable.
