# Maze31 FutureSeed feedback probe

Timestamp UTC: 2026-06-19T04:40:42Z
Git SHA: 7021726254209244b3f338ccdf06b32514ff13d7
Worktree: /huyang2/double-loop/worktrees/fs-feedback-c3065f3-20260619T042057Z
GPU row: GPU1 only

Mechanism hypothesis: current Maze31 FutureSeed loops mostly copy because each loop sees fixed maze input plus latent carry, but not an explicit previous belief. Feeding detached previous-loop logits back as a generic token-embedding expectation gives later loops a simple, non-rule-based state signal to revise.

Prediction: opening should be preserved by delayed Tversky; compared with delayed-Tversky baseline loop16 path_f1=0.5824 and near-zero loop gain, success means loop16-loop1 gain increases with FP down and FN flat or lower. Feedback diagnostics must show nonzero gate/RMS.

Budget: one GPU1 run, 1200 steps, batch16, eval512, no seed/weight sweep.

Kill criteria: if step600 still has path_f1=0.0 or CE remains around no-PATH attractor, stop exact PID and archive abort. If GPU util is low while memory high, stop and inspect.

Claim if successful: FutureSeed provides direction, loop provides compute, and generic previous-belief feedback can turn loops from copying into self-correction without maze rules, search, repair, or selector.
