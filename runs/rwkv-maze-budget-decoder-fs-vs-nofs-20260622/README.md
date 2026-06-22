# RWKV Maze Budget Decoder FS vs noFS

## Result

| metric | noFS | FutureSeed | delta FS-noFS |
|---|---:|---:|---:|
| raw loop8 F1 | 0.465226 | 0.468430 | +0.003204 |
| raw loop gain | -0.000570 | 0.000020 | +0.000590 |
| raw precision | 0.311426 | 0.310783 | -0.000642 |
| raw recall | 0.933320 | 0.964022 | +0.030702 |
| raw pred path frac | 0.396326 | 0.410341 | +0.014015 |
| raw FP / case | 245.755859 | 254.736328 | +8.980469 |
| raw FN / case | 7.951172 | 4.318359 | -3.632812 |
| budget loop8 F1 | 0.338680 | 0.328948 | -0.009733 |
| budget loop gain | 0.002552 | -0.000094 | -0.002645 |
| budget precision | 0.336054 | 0.331849 | -0.004205 |
| budget recall | 0.343193 | 0.327744 | -0.015449 |
| budget pred path frac | 0.134475 | 0.130104 | -0.004371 |
| budget FP / case | 80.353516 | 78.230469 | -2.123047 |
| budget FN / case | 78.214844 | 80.025391 | +1.810547 |
| path count abs err | 0.009826 | 0.009504 | -0.000322 |

## Takeaways

- The learned budget head estimates path fraction accurately in both conditions, so total path mass is not the main missing signal.
- Raw argmax remains a broad high-recall mask with near-zero loop gain; extra loops do not prune false positives.
- Budget decoding reduces false positives but creates about 78-80 false negatives per case, which means true-path cells are not ranked above false-positive cells reliably.
- FutureSeed does not improve this Maze budget/ranking bottleneck in this probe; FS budget F1 is slightly lower than noFS.

## Files

- `summary.json`: parsed metrics and deltas
- `index.html`: side-by-side hard-case dashboard
- `remote/`: pulled run logs, JSON, and original visualizations
