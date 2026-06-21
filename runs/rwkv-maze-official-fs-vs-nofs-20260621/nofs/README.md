# rwkv-maze-official-nofs-s800-20260621T1228Z-1ad39c4

Official Maze30 path recovery with a causal RWKV7 state-passing backbone.

- condition: `no_future_seed`
- future_seed_scale: `0.0`
- loop1 path F1: `0.4690`
- loop8 path F1: `0.4690`
- loop gain: `+0.0000`
- precision/recall: `0.3119` / `0.9580`
- pred PATH frac: `0.4065`
- FP/FN per case: `252.0` / `5.0`

No selector, search, repair, or maze-specific postprocessing is used.

- `rwkv_maze_probe.json`: metrics and config
- `visualizations/index.html`: hard-case loop visualization
