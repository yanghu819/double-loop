# rwkv-maze-official-fs-s800-20260621T1244Z-1ad39c4

Official Maze30 path recovery with a causal RWKV7 state-passing backbone.

- condition: `future_seed`
- future_seed_scale: `1.0`
- loop1 path F1: `0.4673`
- loop8 path F1: `0.4666`
- loop gain: `-0.0007`
- precision/recall: `0.3111` / `0.9464`
- pred PATH frac: `0.4026`
- FP/FN per case: `249.8` / `6.4`

No selector, search, repair, or maze-specific postprocessing is used.

- `rwkv_maze_probe.json`: metrics and config
- `visualizations/index.html`: hard-case loop visualization
