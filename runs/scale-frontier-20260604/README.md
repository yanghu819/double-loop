# Scale Frontier Summary: 9x9 to 16x16

Purpose: identify the largest Sudoku board currently supported by the FutureSeed RWKV recurrent-loop paradigm on GPU1.

## Result

The original flattened FutureSeed RWKV loop supports 12x12 but not 16x16. The new explicit unit-state mechanism opens a 16x16 foothold under the tested budget.

| board | run | key result | decision |
| --- | --- | --- | --- |
| 9x9 | `scaleup-d192l10-h40-20260604T0844Z-a5b3b4a` | h40 loop5 exact `0.5332`, h44 `0.1914` | supported |
| 12x12 initial clean | `frontier-12x12-d256-h72-20260604T0920Z-f67e6a2` | h60 loop5 exact `0.5052`, h72 `0.0443`, h84 `0.0` | first clean 12x12 support |
| 12x12 clean h84 long-stage | `h84long-12x12-d256-l12-loop5-s2400-20260611T043811Z-f3e48c2` | h60/h72/h84/h96 loop5 exact `0.9297`/`0.7161`/`0.1172`/`0.0`; h84 loop1 `0.0`, loop3 `0.0625`, loop5 `0.1172` | h84 opened by clean curriculum/compute; h96 closed |
| 12x12 D320 capacity probe | `capprobe-d320-12x12-h84-s900-20260611T054106Z-d29bb4e` | D320/L12/loop5 batch48 fits around 49GB, but shortened 900-step curriculum gives h72/h84/h96 exact `0.0` and h84 blank_acc `0.5523` | width fits, but short budget is not enough evidence |
| 12x12 clean h96 frontier | `h96frontier-12x12-d256-l12-loop5-s3400-20260611T075445Z-be4046b` | h72/h84/h96/h108 loop5 exact `0.8750`/`0.4740`/`0.0104`/`0.0`; h96 loop1/3 `0.0`, loop4 `0.0078`, loop5 `0.0104` | h96 barely opened; h108 closed |
| 12x12 loop-time h96 | `looptime-h96-12x12-d256-l12-loop5-s3400-20260611T093013Z-cddd449` | `LOOP_TIME_SCALE=1`; h72/h84/h96/h108 loop5 exact `0.8542`/`0.4115`/`0.0052`/`0.0`; h96 loop1 blank_acc `0.5102` vs clean `0.4211` | stronger early loops, worse final consistency; stop loop-time scale sweeps |
| 12x12 compressed h96 focus | `h96focus-12x12-d256-l12-loop5-s2400-20260611T1046Z-edb4804` | shortened early curriculum plus 1400-step h96 stage; h72/h84/h96/h108 loop5 exact `0.9010`/`0.3828`/`0.0`/`0.0`; h96 blank_acc `0.7545` | hard-stage exposure alone is insufficient; full middle curriculum matters |
| 12x12 full-budget clean h96 | `h96fullclean-12x12-d256-l12-loop5-s4800-20260611T1212Z-4337212` | full middle curriculum plus 2400-step h96 stage; h72/h84/h96/h108 loop5 exact `0.9844`/`0.8828`/`0.3164`/`0.0`; h96 loop1/3/5 exact `0.0`/`0.0293`/`0.3164` | h96 supported by clean scaling; h108 still closed |
| 12x12 h108 frontier | `h108frontier-12x12-d256-l12-loop5-s6200-20260611T1400Z-e446f4a` | full h96 foundation plus 1600-step h108 stage; h84/h96/h108/h120 loop5 exact `0.9902`/`0.8848`/`0.1797`/`0.0`; h108 loop1/3/5 exact `0.0`/`0.0020`/`0.1797` | h108 opened; h120 still closed |
| 12x12 h120 loop6 all-loop | `h120loopall-retry-12x12-d256-l12-loop6-s7400-20260611T1640Z-8ccc4a9` | loop6, `LOOP_LOSS=all`, full h108 foundation plus 1600-step h120 stage; h96/h108/h120/h132 loop6 exact `0.9922`/`0.9160`/`0.0273`/`0.0`; h120 loop1/3/4/6 exact `0.0`/`0.0117`/`0.0273`/`0.0273` | h108 strongly supported; h120 barely opened but remains frontier |
| 12x12 h120 hard-stage curve | `h120curve-12x12-d256-l12-loop6-s9000-ckpt1k2k3k-20260612T0455Z-2e95bd8` | same D256/L12/loop6/all-loop path with 3200-step h120 stage and fixed checkpoint evals; h120 exact at +1k/+2k/+3k/final `0.0137`/`0.0332`/`0.0410`/`0.0449`; final h96/h108/h120/h132 exact `0.9902`/`0.9355`/`0.0449`/`0.0` | longer h120 training helps but remains a global-coupling cliff |
| 16x16 high-hole | `frontier-16x16-d192-h112-20260604T1006Z-f67e6a2` | high-loss at h64-h96; aborted | over-hard curriculum |
| 16x16 foothold | `frontier-16x16-foothold-d192-h64-20260604T1018Z-f67e6a2` | final CE `0.5792`, but h32 exact `0.0039`, h48+ exact `0.0` | not supported |
| 16x16 all-loop | `frontier-16x16-allloop-d256-l8-20260604T1234Z-75a8796` | `LOOP_LOSS=all`, D256/L12/loop8; final CE `1.0498`, h24-h48 exact `0.0` | naive per-loop CE is negative |
| 16x16 delayed+unit loss | `frontier-16x16-delayed-unit-d256-l8-20260605T0236Z-37a3d6e` | delayed loop CE + unit digit-mass loss; final CE `1.0642`, h24-h48 exact `0.0`, h32 blank_acc `0.3831` | loss-only rescue is negative |
| 16x16 unit-state | `frontier-16x16-unitstate-d192-h64-20260605T0349Z-e9eabcf` | explicit row/col/box state passing; h32 exact `0.6914`, h48 `0.2148`, h64 `0.0117` | 16x16 opened |
| 16x16 scaled unit-state | `frontier-16x16-unitstate-d256-l12-loop7-h80-b32-20260605T0658Z-f170ff7` | D256/L12/loop7, h48 exact `0.4180`, h64 `0.0430`, h80 `0.0`; batch48+ OOM | scale helps but h80 closed |
| 16x16 unit-memory | `frontier-16x16-unitmem-d192-l10-loop7-h80-b48-20260605T0803Z-79004cf` | persistent row/col/box unit memory, h48 exact `0.6992`, h64 `0.2305`, h80 `0.0` | structure beats direct scale |
| 16x16 unit-memory h80 curriculum | `frontier-16x16-unitmem-h80curr-d192-l10-loop7-b48-20260605T0850Z-0be64f4` | longer 64-80 stage; h64 exact `0.4297`, h72 `0.2031`, h80 `0.0625`, h88 `0.0039` | h80 opened |
| 25x25 unit-memory feasibility | `frontier-25x25-unitmem-feas-d160-l8-loop5-b32-20260605T0937Z-710bb46` | D160/L8/loop5; h50 exact `0.5156`, h75 `0.0742`, h100 `0.0` | 25x25 low-hole foothold |
| 25x25 unit-memory h100 curriculum | `upperbound-25x25-unitmem-h100curr-d160-l8-loop5-b32-20260607T050029Z-46a0b06` | h75 exact `0.1289`, h90 `0.0078`, h100 `0.0078`, h110 `0.0`; h100 blank_acc `0.9381` | h100 barely opened; global consistency frontier |

## Insight

The scale boundary is not raw CUDA throughput. 16x16 ran with high utilization and acceptable memory. The boundary is global consistency: the model can reduce local CE on 16x16, but it does not assemble complete valid boards.

Loop depth becomes more important at 12x12: loop1 exact is zero across evaluated holes, while loop5 reaches h60 exact `0.5052`. At 16x16, loop depth no longer rescues exact solve, which means the current recurrence no longer propagates enough structure over 256 cells.

The delayed+unit-loss probe rules out a loss-only patch. The unit digit-mass objective is satisfied by near-uniform predictions and did not create a full-board communication path; K8 oracle exact remained `0.0`, so this is not a selector failure.

The unit-state run changes the conclusion: the 16x16 cliff was not mainly hidden size or loop count. It was missing row/column/box communication. With explicit unit-state pooling and broadcast, the same D192/L10/loop5 frontier jumps from h32 exact `0.0039` to `0.6914`.

The D256/L12/loop7 harder-hole run shows that scale is useful but insufficient. It improves h48 exact to `0.4180` and h64 exact to `0.0430`, but h80 remains `0.0`. Loop depth matters more on hard holes, ending the h64-80 stage with loop-last CE `0.1867` versus loop1 CE `0.5612`, but eval exact saturates around loop5. Because batch64 and batch48 both OOM on the 80GB GPU, the current no-checkpointing unit-state design is close to its direct scaling limit.

The persistent unit-memory run is the strongest scaling insight so far. A smaller D192/L10/loop7 model with recurrent row/col/box memory beats the D256/L12 pooled unit-state run: h48 exact `0.6992` versus `0.4180`, h64 `0.2305` versus `0.0430`, and final CE `0.1226` versus `0.1867`. The bottleneck is therefore unit-level recurrent state, not just capacity.

The h80-focused memory curriculum opens the next frontier: h80 exact rises from `0.0` to `0.0625`, h72 reaches `0.2031`, and h64 reaches `0.4297`. h88 is barely nonzero at `0.0039`; h96 remains closed. That means h80 was partly a curriculum boundary, while h88/h96 are still global consistency frontiers.

The 25x25 conservative feasibility run establishes the first larger-board foothold. With D160/L8/loop5 memory mode, h50 exact reaches `0.5156` and h75 reaches `0.0742`; h100 remains `0.0`. The model can optimize 625-cell Sudoku under low/mid holes, so 25x25 is no longer a pure infeasibility case.

The 25x25 h100 curriculum probe pushes the frontier but also clarifies the upper bound of the current mechanism. With the same D160/L8/loop5 memory mainline and a `50-75:300,75-100:700` curriculum, h75 improves to `0.1289` and h100 becomes nonzero at `0.0078`. However, h100 blank accuracy is already `0.9381`, while exact remains near zero. Loop1 h100 exact is `0.0`, loop2 opens `0.0078`, and loops 3-5 mostly refine blank accuracy without improving exact. This is a global consistency bottleneck, not a local recognition or throughput bottleneck.

The 2026-06-11 clean h84 long-stage run updates the original flattened FutureSeed+loop picture. Without adding row/column/box unit-memory machinery, simply extending the 72-84 hard stage from the short probe to 1500 steps moves h84 loop5 exact from `0.0234` to `0.1172`, and h72 rises to `0.7161`. This is a real curriculum/compute scaling win. It also keeps the bottleneck visible: h96 is still `0.0`, h84 blank accuracy is `0.8895` while exact is `0.1172`, and K1/K4/K8 rollout oracle gaps are zero. The next clean scaling question is therefore capacity or loop/FutureSeed state dynamics, not selector search or Sudoku-specific repair rules.

The D320 short capacity probe is a useful negative guardrail. It fits in memory at batch48, but a shortened 900-step h84 curriculum leaves CE at `0.8789` and all exact metrics at `0.0`. That means width is not disqualified, but short probes are not enough to validate the capacity route. A fair D320 test needs the same 2400-step h84 curriculum or a schedule that reaches low CE much earlier.

The h96 frontier run pushes the clean path one step further. Moving curriculum pressure to `84-96` opens h96 from `0.0` to `0.0104`, while h84 rises to `0.4740` and h72 reaches `0.8750`. This is a real but thin opening: h96 blank accuracy is only `0.7769`, h96 CE stays around `0.34-0.50` late in training, and h108 remains `0.0`. Loop depth is again doing the hard part, with h96 exact still zero at loop1/3 and nonzero only at loop4/5.

The loop-time conditioning run answers a narrower mechanism question. Adding a learned projection of generic loop phase makes early loops much stronger: h96 loop1 blank accuracy rises from `0.4211` to `0.5102`, and loop1 training loss is consistently lower in later stages. But final exact gets worse: h96 falls from `0.0104` to `0.0052`, h84 from `0.4740` to `0.4115`, and h96 loop5 blank accuracy falls from `0.7769` to `0.7549`. The model did learn and use the phase signal, but the phase signal turned the early loop into a better predictor without improving late global consistency.

The compressed h96 focus run is a useful negative guardrail. It spends 1400 of 2400 steps directly in the 84-96 stage and reaches final CE `0.3850`, close to the full h96 baseline `0.3931`. But h96 exact falls from `0.0104` to `0.0`, h84 falls from `0.4740` to `0.3828`, and h96 blank accuracy falls from `0.7769` to `0.7545`. The one improvement is easier transfer at h72, from `0.8750` to `0.9010`. That means low CE and more hard-stage exposure are not enough; the full middle curriculum appears to build global consistency that the h96 stage cannot recover after being compressed.

The full-budget clean h96 run changes the 12x12 conclusion. Keeping a full middle curriculum and extending the 84-96 stage to 2400 steps moves h96 loop5 exact from `0.0104` to `0.3164`, while h84 jumps to `0.8828` and h72 to `0.9844`. The late h96 CE drop is also real: step4000 is `0.3226`, step4500 is `0.2721`, and step4800 reaches `0.1178`. This means the previous h96 cliff was not a hard mechanism ceiling; it was still substantially compute/curriculum limited.

The same run also keeps the role of loop clean. At h96, loop1 exact is `0.0`, loop3 is `0.0293`, loop4 is `0.2715`, and loop5 is `0.3164`. The solve does not come from a stronger single forward pass; it comes from recurrent refinement. K1 oracle and selector are identical, so selector work remains low ROI.

The h108 frontier run pushes the same clean path again. With a strong h96 foundation and a 1600-step 96-108 stage, h108 loop5 exact reaches `0.1797`, while h96 transfer reaches `0.8848` and h84 reaches `0.9902`. Loop remains the mechanism: h108 loop1 exact is `0.0`, loop3 is only `0.0020`, loop4 jumps to `0.1328`, and loop5 reaches `0.1797`. h120 stays closed at `0.0` exact and only `0.3439` blank accuracy, so it is a harder local-and-global frontier rather than a near-solved selector problem.

The h120 loop6 all-loop run moves that frontier but does not solve it. Adding one more recurrent loop and supervising every loop makes h108 robust: h108 exact rises from `0.1797` to `0.9160`, and h96 rises to `0.9922`. It also opens h120 from `0.0` to `0.0273`, with h120 blank accuracy rising from `0.3439` to `0.5866`. The h120 loop curve is the key readout: loop1 exact is `0.0`, loop3 is `0.0117`, loop4 reaches `0.0273`, and loops 5-6 only add tiny blank accuracy without improving exact. That means loop6 plus all-loop supervision improves the trajectory and the h108 foundation, but h120 still needs better hard-stage learning or state dynamics.

The h120 hard-stage curve run answers the immediate "train longer" question more directly. Extending the h120 stage to 3200 steps and evaluating the same fixed batches at +1k/+2k/+3k gives h120 loop6 exact `0.0137`/`0.0332`/`0.0410`, with final `0.0449`. h108 continues improving to final exact `0.9355`, while h120 blank accuracy rises to the low `0.62-0.64` range. This is a real compute/curriculum gain over the 1600-step h120 run, but the slope is shallow: another short extension is unlikely to turn h120 into a solved regime. The next clean experiment should either increase capacity/compute substantially or make the FutureSeed state update simple but more revisable across late loops.

## Decision

Treat 12x12 h108 as strongly supported by the original flattened FutureSeed+loop paradigm after loop6/all-loop scaling; h120 is open but still frontier territory, with best clean h120 exact now `0.0449`. Treat 16x16 as supported by the historical unit-memory mainline through h80. Treat 25x25 as supported at low holes, feasible at h75, and barely open at h100. h100 on 25x25, h88/h96 on 16x16, and h120/h132 on clean 12x12 are still frontier territory.

Next high-ROI work should change the mechanism:

1. For the clean FutureSeed+loop path, keep h120 as the next target but use only high-leverage changes: more h120-stage compute/capacity, activation checkpointing if it enables real scale, or one simple FutureSeed state revision mechanism. Do not run loop-time, feedback scale, compressed h96 schedule sweeps, or selector work.
2. Keep selector work paused until K-oracle improves; current K1 is already the main result on the clean h84 run.
3. For the historical unit-memory path, only revisit true row/column/box unit tokens if the clean path stalls again; do not add Sudoku-specific repair rules to the mainline.
4. Add activation checkpointing only if it enables a structurally different scaling experiment, not just a blind D384/L16 table entry.
