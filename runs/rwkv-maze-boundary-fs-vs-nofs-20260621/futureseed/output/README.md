# rwkv-maze-boundary-fs-s800-20260621T153154Z-3df9e02

Official Maze30 path recovery with a causal RWKV7 state-passing backbone.

- condition: `boundary-futureseed`
- future_seed_scale: `1.0`
- loop1 path F1: `0.4270`
- loop8 path F1: `0.4278`
- loop gain: `+0.0008`
- precision/recall: `0.3270` / `0.6280`
- pred PATH frac: `0.2527`
- FP/FN per case: `152.9` / `44.3`

No selector, search, repair, or maze-specific postprocessing is used.

- `rwkv_maze_probe.json`: metrics and config
- `visualizations/index.html`: hard-case loop visualization
