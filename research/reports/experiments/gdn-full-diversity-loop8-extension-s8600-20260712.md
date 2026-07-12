# Full-Diversity Loop-8 Compute Extension

## 1. Metainfo

- Plan ID: `P-SCALE-031`
- Status: discarded by the predeclared step8200 rule
- Planned: 2026-07-12 19:43 CST / 2026-07-12T11:43:21Z
- Launched: 2026-07-12 19:49 CST / 2026-07-12T11:49:14Z
- Early-stopped: 2026-07-12 20:15 CST / 2026-07-12T12:15:29Z
- Evaluation completed: 2026-07-12 20:16 CST / 2026-07-12T12:16:40Z
- Machine: AIStation `GPU1` A800 only
- Branch: `codex/gpu1-experiment-tracking`
- Experiment source SHA: `77481aa3dca1a654b08b16006943806674fe6394`
- Resume checkpoint: P-SCALE-030 exact step8000
- Train run: `gdn-full-diversity-loop8-resume8000-s8600-20260712T115200Z-77481aa`
- Eval/viz run: `gdn-full-diversity-loop8-step8200-evalviz-20260712T121529Z-77481aa`

## 2. Mechanism Hypothesis

P-SCALE-030 improved loop4/5 while loop1 stayed unchanged. That is direct evidence
that the current model learns corrections in later recurrent passes. The next
high-information question is whether the correction process is still limited by
the number of supervised passes, or whether the recurrent state has already
exhausted the useful information it can carry by loop5.

If recurrent compute is still the limiting resource, extending the same shared
block from five to eight supervised loops should continue reducing hard-board
errors. If loop6-8 merely copy loop5, then adding loops is not a useful scaling
axis and the next budget must move to a larger full-diversity backbone.

## 3. Intervention

- Resume the exact step8000 model, optimizer, scaler, feature buffer, and RNG.
- Keep native FutureSeed GDN D224/L12/H14/D16 and `GDN_EXPAND_V=4.0`.
- Keep the 3,831,994 independent-board dataset and 51-64 hard exposure.
- Change only `MAX_LOOPS=5` to `MAX_LOOPS=8`.
- Keep `LOOP_LOSS=all`, so every one of the eight loops receives CE supervision.
- Train 600 additional steps, with eval at steps 8200, 8400, and 8600.
- Do not add noise, scratch state, repair, search, selector, or task rules.

## 4. Prediction, Budget, And Kill Criteria

Prediction:

- loop6-8 should improve holes60 or holes64 exact by at least `+0.02` over loop5;
- official 56-64 exact should reach at least `0.18` while 51-55 remains `>=0.42`;
- hardest-case wrong-cell counts should continue falling after loop5.

Budget: one GPU-only fit step, then at most 600 resumed training steps on GPU1,
estimated 0.9-1.2 GPU hours plus evaluation.

Kill:

- stop immediately on wrong GPU, source/checkpoint mismatch, NaN, or OOM;
- at step8200, stop if both holes60 and holes64 have loop8-loop5 exact below
  `+0.01` and the visual cases show no error reduction after loop5;
- do not respond to failure by sweeping loop count, loop-loss weights, seed, LR,
  or schedule.

## 5. Paper Decision

Success supports the claim that loop count is a useful variable-compute axis:
the same parameters can spend more recurrent computation and solve more boards.
Failure establishes a clean compute ceiling at loop5 and redirects scaling to
larger generic state/model capacity. It does not by itself establish a
FutureSeed advantage over matched no-FutureSeed or EqR.

## 6. Results

The GPU-only fit gate completed a real optimizer step with eight loops, equal
per-loop loss weights of `0.125`, BF16 Triton forward/backward, and effective
batch128. Peak CUDA memory was `71,049 MB` allocated and `71,114 MB` reserved,
so the experiment fit the A800 without activation checkpointing. The fit used
`eval_n=8`; its score is intentionally not treated as quality evidence.

The formal run resumed the untouched P-SCALE-030 step8000 checkpoint. It reached
step8200 in `1126.7 s` and saved an exact `161 MB` train checkpoint. Fixed
512-board checkpoint evaluation was:

| Bucket | loop1 exact / blank | loop3 exact / blank | loop5 exact / blank | loop8 exact / blank | loop8-loop5 exact |
|---|---:|---:|---:|---:|---:|
| holes53 | `0.0176 / 0.5464` | `0.1387 / 0.6708` | `0.2500 / 0.6880` | `0.2520 / 0.6895` | `+0.0020` |
| holes60 | `0.0215 / 0.5557` | `0.1836 / 0.6906` | `0.2871 / 0.7056` | `0.2871 / 0.7065` | `+0.0000` |
| holes64 | `0.0234 / 0.5481` | `0.1953 / 0.6696` | `0.2754 / 0.6814` | `0.2793 / 0.6815` | `+0.0039` |

Both hard-bucket deltas were below the predeclared `+0.01` continuation rule,
and blank accuracy changed by less than `0.001`. Exact PIDs
`1258/1256/1257/1210` were terminated, `abort.json` was written, the step8200
checkpoint was retained, and GPU memory returned to `0 MiB`. No step8400 or
step8600 training was run.

The independent full evaluation from the step8200 checkpoint gives the complete
loop curve:

| Loop | exact | blank accuracy |
|---:|---:|---:|
| 1 | `0.0234` | `0.5546` |
| 2 | `0.0273` | `0.6412` |
| 3 | `0.1797` | `0.6829` |
| 4 | `0.2617` | `0.6966` |
| 5 | `0.2812` | `0.6972` |
| 6 | `0.2852` | `0.6972` |
| 7 | `0.2871` | `0.6975` |
| 8 | `0.2871` | `0.6975` |

Loop5-to-loop8 exact gain is only `+0.0059`; loop7-to-loop8 is exactly zero.
Official loop8 blank ranges are `46-50=1.0000`, `51-55=0.4180`, and
`56-64=0.1484`. These trail the P-SCALE-030 step8000 loop5 results
`0.4473/0.1621` on the two hard ranges, so extra supervised loops did not trade
more compute for a better operating point.

The case bank confirms the aggregate result. Error counts are shown at loops
`1/3/5/6/7/8`:

| Range / case kind | Mean wrong-cell trajectory |
|---|---|
| 51-55 solved by loop | `24.25 -> 1.75 -> 0 -> 0 -> 0 -> 0` |
| 51-55 almost solved | `20.75 -> 8.75 -> 1 -> 1 -> 1 -> 1` |
| 51-55 hard failure | `18.00 -> 9.25 -> 5.25 -> 5.25 -> 5.25 -> 5.25` |
| 56-64 solved by loop | `27.25 -> 6.25 -> 0 -> 0 -> 0 -> 0` |
| 56-64 almost solved | `21.25 -> 9.75 -> 1.25 -> 1 -> 1 -> 1` |
| 56-64 hard failure | `22.00 -> 11.25 -> 6.50 -> 6.00 -> 5.75 -> 6.00` |

One representative 57-blank failure goes `27 -> 13 -> 9 -> 6 -> 6 -> 6`.
Loop6 makes one last small correction; loops7/8 freeze. Another hard case changes
`6 -> 7 -> 6 -> 7`, showing late oscillation rather than reliable correction.

## 7. Conclusions

P-SCALE-031 is a useful negative result. The current shared recurrence extracts
most useful computation by loop5. More supervised unrolling is mechanically
trainable and occasionally fixes one additional board, but it does not produce
continued hard-case correction and is substantially more expensive: peak memory
rises to about `71 GB` and each optimizer step takes about `5.6 s` after startup.

The correct mechanism claim is now narrower: loops are essential from loop1 to
loop4/5, but loop count is not an unbounded scaling axis for this state. The
bottleneck is the information represented and updated by the generic recurrent
state, not the absence of loop6-8 loss. Do not run loop6/7/9/10 tables or longer
loop8 continuation.

The next high-ROI scaling test is one larger backbone trained on the proven
3.83M-board full-diversity regime with enough tokens for a delayed crossover.
That changes model capacity and data utilization together; it does not add a
Sudoku rule, repair, selector, or custom objective.

## 8. Artifacts And Visualization

- Metadata-light bundle SHA256:
  `0f4310c21ac40662c8d31af40ebccd6002e99ac36769bb066d67ed2659695247`.
- The aborted train run archives config, launch env/script, logs, step8200 fixed
  metrics, source SHA/patch, and `abort.json`.
- The eval run archives full JSON/Markdown/HTML, official metrics, dashboard,
  and 27 case HTML files covering loops `1/3/5/6/7/8`.
- Full source snapshots and the step8200 checkpoint remain under
  `/huyang2/double-loop`; checkpoints and datasets are not committed to Git.
- In-place dashboard:
  `http://127.0.0.1:8775/runs/gdn-full-diversity-loop8-step8200-evalviz-20260712T121529Z-77481aa/visualizations/index.html`.
- Dashboard artifact links initially resolved from the wrong subdirectory. The
  generic builder was fixed to resolve run-root artifacts as `../output/...`;
  Kimi WebBridge then opened the representative hard case successfully.

## 9. Submission Record

- Final score: `0.287109375` at `metrics.eval_clean.loop8.label_exact`.
- Decision: discarded as a loop-count scaling direction.
- No tag: score is below `0.50`, and the added loops fail the mechanism gate.
