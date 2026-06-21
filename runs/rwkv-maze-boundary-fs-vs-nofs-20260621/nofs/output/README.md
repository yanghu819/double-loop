# rwkv-maze-boundary-nofs-s800-20260621T151104Z-3df9e02

Official Maze30 path recovery with a causal RWKV7 state-passing backbone.

- condition: `boundary-nofs`
- future_seed_scale: `0.0`
- loop1 path F1: `0.4085`
- loop8 path F1: `0.4089`
- loop gain: `+0.0004`
- precision/recall: `0.3256` / `0.5828`
- pred PATH frac: `0.2360`
- FP/FN per case: `143.2` / `49.7`

No selector, search, repair, or maze-specific postprocessing is used.

- `rwkv_maze_probe.json`: metrics and config
- `visualizations/index.html`: hard-case loop visualization
