# rwkv-maze-budget-fs-s800-20260622T0501Z-97404e1

Official Maze30 path recovery with a causal RWKV7 state-passing backbone.

- condition: `futureseed-budget-decoder`
- future_seed_scale: `1.0`
- loop1 path F1: `0.4684`
- loop8 path F1: `0.4684`
- loop gain: `+0.0000`
- precision/recall: `0.3108` / `0.9640`
- pred PATH frac: `0.4103`
- FP/FN per case: `254.7` / `4.3`
- budget decoder: `True`

No selector, search, repair, or maze-specific postprocessing is used.

- `rwkv_maze_probe.json`: metrics and config
- budget loop1 path F1: `0.3290`
- budget loop8 path F1: `0.3289`
- budget loop gain: `-0.0001`
- budget precision/recall: `0.3318` / `0.3277`
- budget pred PATH frac: `0.1301`
- budget FP/FN per case: `78.2` / `80.0`
- `visualizations/index.html`: hard-case loop visualization
