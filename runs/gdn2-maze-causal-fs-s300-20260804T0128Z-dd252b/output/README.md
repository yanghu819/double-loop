# gdn2-maze-causal-fs-s300-20260804T0128Z-dd252b

Official Maze30 path recovery with a causal GDN2 backbone.

- status: `completed`
- condition: ``
- future_seed_scale: `1.0`
- loop1 path F1: `0.0009`
- loop5 path F1: `0.0012`
- loop gain: `+0.0003`
- precision/recall: `0.0547` / `0.0005` -> `0.0703` / `0.0006`
- pred PATH frac: `0.0001` -> `0.0001`
- FP/FN per case: `0.0` / `116.9` -> `0.0` / `116.8`
- feedback mode: `none`
- DAT weight: `0.0`
- budget decoder: `False`

No selector, search, repair, or maze-specific postprocessing is used.

- `rwkv_maze_probe.json`: metrics and config
- `visualizations/index.html`: hard-case loop visualization
