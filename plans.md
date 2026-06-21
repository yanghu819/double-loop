# Experiment Plans

Primary evidence target: show whether FutureSeed supplies cheap future context
for recurrent reasoners under matched compute, without solver-specific repair or
selector tricks.

## A. Current Candidate Plan

| ID | 状态 | 假设 | 方法 | 机器/资源 | 预估时长 | 期望 Δ | 实际结果 |
|---|---|---|---|---|---:|---|---|
| P-MAZE-001 | approved | Official Maze is currently blocked by token-loss/objective mismatch, not necessarily by model capacity. | Add optional generic class-balanced/path-token loss to official EqR; run clean EqR first; evaluate path F1 and hard-case visuals. | GPU1 A100 | 20-40 min | loop16 path F1 opens above 0.10 without all-PATH collapse | - |
| P-MAZE-002 | proposed | If objective alignment opens path recovery, FutureSeed should improve official EqR sample efficiency or final path F1. | Matched base vs FutureSeed under the same aligned official EqR objective. | GPU1 A100 | 40-80 min | FutureSeed path F1 +0.03 or same F1 with fewer steps | gated on P-MAZE-001 |
| P-MAZE-003 | proposed | EqR's noncausal mixer masks FutureSeed; a causal Maze backbone should reveal the cheap-bidirectional mechanism. | RWKV/CUDA Maze path recovery with FutureSeed on/off, same compute, path-aware visuals. | GPU1 A100 | 60-120 min | FS removes directional/position bias and improves path F1 | run if P-MAZE-001/002 are neutral |

## B. Completed Plan

| ID | 状态 | 假设 | 方法 | 机器/资源 | 预估时长 | 期望 Δ | 实际结果 |
|---|---|---|---|---|---:|---|---|
| P-MAZE-000 | done | Official EqR e512 might solve Maze with longer training, and FutureSeed might improve it. | Official EqR clean vs FutureSeed, e512, token metrics plus path-aware visualization. | GPU1 A100 | done | path F1 opens | Negative: token acc ~0.868, exact 0, loop16 path F1 base 0.001273 / FS 0.0. Token accuracy is misleading. |

## D. Historical Lessons

- 2026-06-21: official EqR Maze token accuracy is dominated by non-PATH cells; path-aware metrics and visualizations are mandatory.
- 2026-06-21: FutureSeed has a strong causal/RWKV Sudoku mechanism result, but official EqR Maze support is not established.
- 2026-06-21: no selector, repair, search, or maze-specific postprocessing should be used to rescue Maze scores.
