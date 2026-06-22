# rwkv-maze-budget-nofs-s800-20260622T0442Z-97404e1

Official Maze30 path recovery with a causal RWKV7 state-passing backbone.

- condition: `nofs-budget-decoder`
- future_seed_scale: `0.0`
- loop1 path F1: `0.4658`
- loop8 path F1: `0.4652`
- loop gain: `-0.0006`
- precision/recall: `0.3114` / `0.9333`
- pred PATH frac: `0.3963`
- FP/FN per case: `245.8` / `8.0`
- budget decoder: `True`

No selector, search, repair, or maze-specific postprocessing is used.

- `rwkv_maze_probe.json`: metrics and config
- budget loop1 path F1: `0.3361`
- budget loop8 path F1: `0.3387`
- budget loop gain: `+0.0026`
- budget precision/recall: `0.3361` / `0.3432`
- budget pred PATH frac: `0.1345`
- budget FP/FN per case: `80.4` / `78.2`
- `visualizations/index.html`: hard-case loop visualization
