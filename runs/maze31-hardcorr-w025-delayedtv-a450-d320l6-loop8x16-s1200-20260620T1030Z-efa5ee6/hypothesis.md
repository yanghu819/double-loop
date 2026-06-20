# Maze31 hard-decision correction probe

Hypothesis: FutureSeed opens hard Maze31, but later loops copy broad masks because previous losses reward coverage or soft calibration. A generic loop-pair hard-correction target should create a stronger decision-boundary signal: after opening, later loops push loop1 false-positive PATH margins below zero while preserving true-path margins.

Prediction: compared with delayed-Tversky baseline, loop16 should reduce pred_frac / false positives with recall roughly preserved. If path_f1 stays zero through step600, treat as opening failure. If precision rises only by increasing false negatives, treat as tradeoff not self-correction.

Bitter-lesson boundary: no maze rules, no repair, no selector, no search, no seed/weight sweep. This is a generic recurrent training pressure.
