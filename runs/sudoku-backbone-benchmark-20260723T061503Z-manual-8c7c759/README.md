# FutureSeed Sudoku Backbone Benchmark

Decision: No backbone separates on hard full-board closure at this finite budget. GDN opening exact is 0.8770; RWKV is cheapest per optimizer step. Keep the existing clean GDN scaling line, but do not claim a universal architecture winner.

| Backbone | Params | Train CE | Mixed exact | Mixed blank | Sec/step | Peak |
|---|---:|---:|---:|---:|---:|---:|
| RWKV | 5.580M | 1.0432 | 0.00781 | 0.48666 | 3.87s | 9.04GiB |
| GDN | 5.980M | 1.0001 | 0.02148 | 0.50544 | 4.93s | 7.04GiB |
| GDN2 | 6.946M | 1.0114 | 0.01953 | 0.50131 | 5.45s | 9.08GiB |
| KDA | 5.852M | 1.0065 | 0.01758 | 0.50184 | 5.55s | 7.62GiB |


KDA peak memory was recovered from `sudoku-backbone-kda-memory-probe-s501-20260723T0920Z-8c7c759` with one matched optimizer step; its quality and speed remain from the original step-500 run.
Open `index.html` for official blank ranges, loop curves, kernel provenance, and same-puzzle visualizations.
