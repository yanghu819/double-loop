# Experiment Plans

Primary evidence target: show whether FutureSeed supplies cheap future context
for recurrent reasoners under matched compute, without solver-specific repair or
selector tricks.

## A. Current Candidate Plan

| ID | 状态 | 假设 | 方法 | 机器/资源 | 预估时长 | 期望 Δ | 实际结果 |
|---|---|---|---|---|---:|---|---|
| P-MAZE-003 | proposed | EqR's noncausal mixer masks FutureSeed; a causal Maze backbone should reveal the cheap-bidirectional mechanism. | RWKV/CUDA Maze path recovery with FutureSeed on/off, same compute, path-aware visuals. | GPU1 A100 | 60-120 min | FS removes directional/position bias and improves path F1 | run if P-MAZE-001/002 are neutral |

## B. Completed Plan

| ID | 状态 | 假设 | 方法 | 机器/资源 | 预估时长 | 期望 Δ | 实际结果 |
|---|---|---|---|---|---:|---|---|
| P-MAZE-002 | discarded | If objective alignment opens path recovery, FutureSeed should improve official EqR sample efficiency or final path F1. | Matched base vs FutureSeed under the same aligned official EqR objective. | GPU1 A100 | done | FutureSeed path F1 +0.03 or same F1 with fewer steps | Neutral: FutureSeed loop16 path F1 `0.4684` vs clean EqR `0.4682`, delta `+0.0001`; EqR mixer likely masks cheap future-context value. |
| P-MAZE-001 | done | Official Maze is currently blocked by token-loss/objective mismatch, not necessarily by model capacity. | Add optional generic class-balanced/path-token loss to official EqR; run clean EqR first; evaluate path F1 and hard-case visuals. | GPU1 A100 | done | loop16 path F1 opens above 0.10 without all-PATH collapse | Opened path recovery but as broad mask: loop16 path F1 `0.4682`, precision `0.3064`, recall `0.9998`, pred PATH frac `0.4325` vs true `0.1326`; continue to matched FutureSeed. |
| P-MAZE-000 | done | Official EqR e512 might solve Maze with longer training, and FutureSeed might improve it. | Official EqR clean vs FutureSeed, e512, token metrics plus path-aware visualization. | GPU1 A100 | done | path F1 opens | Negative: token acc ~0.868, exact 0, loop16 path F1 base 0.001273 / FS 0.0. Token accuracy is misleading. |

## D. Historical Lessons

- 2026-06-21: official EqR Maze token accuracy is dominated by non-PATH cells; path-aware metrics and visualizations are mandatory.
- 2026-06-21: a generic path-token-weighted objective opens official EqR Maze path recovery, but the failure shape becomes high-recall broad masks, not solved paths.
- 2026-06-21: FutureSeed is neutral on official EqR once path recovery is rewarded; treat EqR as a strong noncausal baseline/boundary, not as the backbone where FutureSeed should win.
- 2026-06-21: FutureSeed has a strong causal/RWKV Sudoku mechanism result, but official EqR Maze support is not established.
- 2026-06-21: no selector, repair, search, or maze-specific postprocessing should be used to rescue Maze scores.
