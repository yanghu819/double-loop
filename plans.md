# Experiment Plans

Primary evidence target: show whether FutureSeed supplies cheap future context
for recurrent reasoners under matched compute, without solver-specific repair or
selector tricks.

## A. Current Candidate Plan

| ID | 状态 | 假设 | 方法 | 机器/资源 | 预估时长 | 期望 Δ | 实际结果 |
|---|---|---|---|---|---:|---|---|
| P-EQR-006 | proposed | FutureSeed 的论文主张应该是 cheap bidirectional information mechanism：在官方 EqR codebase 中，它应能替代或压缩 EqR mixer 的信息交换，而不是作为完整 mixer 之后的无归因 patch。 | 官方 EqR matched-compute gate：先复现不改语义的 EqR mixer baseline，再比较一个贡献可分离的 FutureSeed replacement/compression 版本。必须同数据、同训练预算、同评估脚本，并报告 exact/token acc/path F1/precision/recall/pred_path_frac/FP/FN/速度/显存/参数量和 hard-case visuals。 | GPU1 only | gated | path F1 `>= +0.03` 且不靠 broad mask 膨胀，或同等 F1 下训练/推理计算节省 `>=20%` | task spec: `docs/NEXT_STAGE_OFFICIAL_EQR_TASK.md` |
| P-MAZE-005 | discarded | Maze failure may be a decision-boundary/calibration bottleneck: the causal RWKV may rank true PATH cells above false positives, but raw argmax cannot convert that ranking into a sparse mask. | Add a generic learned path-budget head and learned-budget decoder to the RWKV Maze runner. Train only with supervised path fraction, then decode PATH as top model-scored cells under the model's own predicted budget; compare no-FutureSeed vs FutureSeed under matched compute. | GPU1 A100 | ~45m | budget decoder reduces FP/pred frac without recall collapse; FutureSeed improves budgeted F1 or recall at matched budget | Negative but informative. The count head learned true PATH fraction accurately (`abs_err≈0.0095-0.0098`), but budget decoding overpruned true path cells: noFS budget loop8 F1 `0.3387`, FS `0.3289`, with FN about `78-80` per case. Raw loop gain stayed near zero. Conclusion: total path mass calibration is not enough; Maze bottleneck is true-vs-false PATH ranking / late-loop correction. |

## B. Completed Plan

| ID | 状态 | 假设 | 方法 | 机器/资源 | 预估时长 | 期望 Δ | 实际结果 |
|---|---|---|---|---|---:|---|---|
| P-MAZE-004 | discarded | Official Maze path-weight permits broad-mask shortcuts; a better paper proxy should make false positives costly without maze rules. | Add a generic PATH-vs-non-PATH margin objective plus per-sample PATH-mass calibration to the causal RWKV Maze runner; compare no-FutureSeed vs FutureSeed under matched compute. | GPU1 A100 | done | precision up, pred PATH frac/FP down, recall not collapsed; FutureSeed opens only if cheap future context was hidden by broad-mask objective | Weak mechanism signal, not a final benchmark: boundary objective narrows masks but overprunes. no-FutureSeed loop8 path F1 `0.4089`, precision/recall `0.3256/0.5828`; FutureSeed `0.4278`, precision/recall `0.3270/0.6280`, delta `+0.0189`. Loop gain remains near zero (`+0.0004` vs `+0.0008`). |
| P-MAZE-003 | discarded | EqR's noncausal mixer masks FutureSeed; a causal Maze backbone should reveal the cheap-bidirectional mechanism. | RWKV/CUDA Maze path recovery with FutureSeed on/off, same compute, path-aware visuals. | GPU1 A100 | done | FS removes directional/position bias and improves path F1 | Negative but informative: causal RWKV no-FS loop8 path F1 `0.4690`; RWKV+FS `0.4666`; both converge to broad masks with near-zero/negative loop gain. Official path-weight Maze is not exposing cheap-bidirectional value. |
| P-MAZE-002 | discarded | If objective alignment opens path recovery, FutureSeed should improve official EqR sample efficiency or final path F1. | Matched base vs FutureSeed under the same aligned official EqR objective. | GPU1 A100 | done | FutureSeed path F1 +0.03 or same F1 with fewer steps | Neutral: FutureSeed loop16 path F1 `0.4684` vs clean EqR `0.4682`, delta `+0.0001`; EqR mixer likely masks cheap future-context value. |
| P-MAZE-001 | done | Official Maze is currently blocked by token-loss/objective mismatch, not necessarily by model capacity. | Add optional generic class-balanced/path-token loss to official EqR; run clean EqR first; evaluate path F1 and hard-case visuals. | GPU1 A100 | done | loop16 path F1 opens above 0.10 without all-PATH collapse | Opened path recovery but as broad mask: loop16 path F1 `0.4682`, precision `0.3064`, recall `0.9998`, pred PATH frac `0.4325` vs true `0.1326`; continue to matched FutureSeed. |
| P-MAZE-000 | done | Official EqR e512 might solve Maze with longer training, and FutureSeed might improve it. | Official EqR clean vs FutureSeed, e512, token metrics plus path-aware visualization. | GPU1 A100 | done | path F1 opens | Negative: token acc ~0.868, exact 0, loop16 path F1 base 0.001273 / FS 0.0. Token accuracy is misleading. |

## D. Historical Lessons

- 2026-06-21: official EqR Maze token accuracy is dominated by non-PATH cells; path-aware metrics and visualizations are mandatory.
- 2026-06-21: a generic path-token-weighted objective opens official EqR Maze path recovery, but the failure shape becomes high-recall broad masks, not solved paths.
- 2026-06-21: FutureSeed is neutral on official EqR once path recovery is rewarded; treat EqR as a strong noncausal baseline/boundary, not as the backbone where FutureSeed should win.
- 2026-06-21: causal RWKV on official path-weight Maze also converges to the same broad-mask solution, so this proxy/objective does not isolate the missing-future-context mechanism.
- 2026-06-21: adding a generic PATH/non-PATH boundary objective makes broad masks costly, but the failure changes to overpruning true path cells. FutureSeed mildly improves recall/F1 under this pressure, yet loop gain is still near zero; do not turn this into loss-weight sweeps.
- 2026-06-21: FutureSeed has a strong causal/RWKV Sudoku mechanism result, but official EqR Maze support is not established.
- 2026-06-21: no selector, repair, search, or maze-specific postprocessing should be used to rescue Maze scores.
