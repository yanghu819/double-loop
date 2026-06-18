# Maze31 EqR Baseline D320/L6 Full Readout

Hypothesis: equal-compute EqR baseline with future_seed_scale=0 should be beaten by FutureSeed+loop if FutureSeed is a better mainline.
Prediction: compare against prior FS loop16 path_f1=0.5344 and FP/FN trajectory. If baseline matches or exceeds it, FutureSeed+loop is not yet superior on hard Maze31.
Budget: GPU1 only, 1200 steps, batch16, eval512, D320/L6, train/eval loops 8/16, path160-260.
Kill: stop if unopened after step600, OOM, or low util with high memory.
