# Maze31 delayed Tversky correction probe

Timestamp UTC: 2026-06-18T10:32:30Z
Source SHA: 5009fc1ff8d18c165f216652074f249467277d79

Hypothesis: always-on Tversky killed FutureSeed opening. Keep clean FutureSeed training until after the historical step400 opening point, then enable the same generic soft TP/FP/FN pressure at step450 to test whether loss-only correction can prune after opening.

Prediction: step400 should open with tw=0. After tw=0.75 starts, useful signal requires loop16 precision up and pred_frac/FP down without recall/FN collapse.

Kill criteria: stop if step600 is still unopened or if the opened trajectory collapses to no-PATH; do not sweep weight/alpha/beta.
