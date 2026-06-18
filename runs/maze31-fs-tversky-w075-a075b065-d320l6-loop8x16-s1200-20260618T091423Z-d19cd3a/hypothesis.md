# Maze31 FutureSeed Tversky loop self-correction probe

Timestamp UTC: 2026-06-18T09:14:37Z
Source SHA: d19cd3a265ce0b0d275f4925d2e0f6705fce786b

Mechanism hypothesis: clean FutureSeed+loop can open hard Maze31, but ordinary CE/path-weight supervision rewards broad coverage. A generic late-loop Tversky objective puts soft TP/FP/FN in one differentiable target, so later loops should learn to reduce false-positive PATH mass without losing true-path cells. This is bitter-lesson compliant: no maze rules, no repair, no selector, no search, no seed sweep.

Prediction: compared with the clean D320/L6 FutureSeed run, loop16 should reduce predicted PATH fraction and false positives relative to loop1 while keeping recall/FN roughly flat. Useful positive signal is precision up plus pred_frac down without recall drop larger than about 0.02.

Budget: one GPU1 run, planned 1200 steps, batch16, eval512, train/eval loops 8/16. Kill at step600 if path_f1 is still 0.0, or if GPU utilization is low with high memory.

Decision: if loop16 still mainly adds coverage or creates FNs, loss-only dense FP/FN pressure is insufficient and the next direction must be modeling/state dynamics rather than more loss sweeps.
