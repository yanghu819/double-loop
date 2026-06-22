# rwkv-maze-abortviz-probe-20260622

Official Maze30 path recovery with a causal RWKV7 state-passing backbone.

- status: `aborted`
- condition: `fs-dat-abortviz`
- future_seed_scale: `1.0`
- loop1 path F1: `0.4614`
- loop16 path F1: `0.4614`
- loop gain: `-0.0000`
- precision/recall: `0.3006` / `1.0000` -> `0.3006` / `1.0000`
- pred PATH frac: `0.4322` -> `0.4322`
- FP/FN per case: `272.1` / `0.0` -> `272.1` / `0.0`
- feedback mode: `pred`
- DAT weight: `0.5`
- budget decoder: `False`

## Abort

- reason: `broad_mask_low_precision_no_fp_drop`
- step: `100`

No selector, search, repair, or maze-specific postprocessing is used.

- `rwkv_maze_probe.json`: metrics and config
- `visualizations/index.html`: hard-case loop visualization
