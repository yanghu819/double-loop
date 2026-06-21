# Clean FutureSeed Long Scaling Maze31 Probe

Mechanism hypothesis:
If FutureSeed plus loop is a real scaling direction, longer clean training on the same hard Maze31 distribution should improve the operating point without hand-written repair, selector, search, maze rules, or extra pruning losses. If loop-time correction is an emergent effect of scale, loop16 should eventually improve precision or reduce false positives relative to loop1, not only add coverage.

Prediction:
The model should open by roughly step400-700, as the prior clean D320/L6 run did. At final evaluation, loop16 should beat the prior clean 1200-step baseline (`loop16 path_f1=0.5344`) and ideally approach or exceed delayed-Tversky/hardcorr scores (`~0.582-0.584`) without extra losses. The crucial readout remains loop1 vs loop16 precision/recall/pred_frac and visual FP/FN changes.

Budget:
One GPU1 run, 3000 train steps, no seed sweep, no gate/loss hyperparameter sweep.

Kill criteria:
If path_f1 remains exactly 0.0 at step700, stop by exact PID and archive abort. If GPU utilization stays low while memory remains high, stop and inspect. Otherwise run to final evaluation because the point is long clean scaling.

Claim if successful:
Clean FutureSeed+loop has a bitter-lesson-compatible scaling path on hard Maze31. If loop gains still remain flat, the result separates representation/optimization scaling from the missing recurrent correction mechanism.
