# Maze31 Clean D320/L6 Full Readout

Mechanism hypothesis: the previous clean D320/L6 run showed a late PATH opening at step400, while the matched `state_compete_cross` run stayed at no-PATH. If clean FutureSeed + loop is the better bitter-lesson direction, then with enough budget it should produce a meaningful loop1 -> loop16 readout and concrete case trajectories without extra loss tricks.

Prediction: by final eval, loop16 should either improve precision / reduce predicted PATH fraction over loop1, or prove that clean scale only opens a broad mask and still lacks pruning dynamics.

Budget: 1200 train steps, batch16, eval512, Maze31 path160-260, D320/L6, train/eval loops 8/16, GPU1 only.

Kill criteria: stop by exact PID if PATH is still unopened after step600, if OOM occurs, or if GPU utilization is persistently low while memory is high. Otherwise let it finish final eval and visualization.

Claim unlocked: decide whether the mainline should continue clean scaling on Maze31, or whether the next paper-relevant move must change recurrent state dynamics.
