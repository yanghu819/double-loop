# Learned Gate + Delayed Tversky Maze31 Probe

Mechanism hypothesis:
FutureSeed can provide an initial direction, but late loops fail when recurrent state is copied or mixed too rigidly. A minimal learned state gate should let each loop decide how much to preserve versus revise, while delayed Tversky supplies a generic late-loop pressure away from broad high-recall masks.

Prediction:
Loop16 should reduce predicted path fraction and false positives relative to loop1 while keeping recall high. If precision/pred_frac barely move, or the only improvement is recall sacrifice, learned scalar state gating is not enough for Maze31 self-correction.

Budget:
One GPU1 run, 1200 train steps, no seed or weight sweep.

Kill criteria:
If path_f1 is still zero at step600, stop by exact PID and archive abort. If GPU utilization stays low while memory is high, stop and inspect before burning more time.

Claim if successful:
A simple learned recurrent state update can convert FutureSeed direction plus loop compute into later-loop pruning, supporting the paper story that state dynamics controls sustained correction.
