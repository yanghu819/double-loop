# FutureSeed Sudoku Backbone Benchmark

Decision: No backbone separates on hard full-board closure at this finite budget. GDN opening exact is 0.8652; RWKV is cheapest per optimizer step. Keep the existing clean GDN scaling line, but do not claim a universal architecture winner.

| Backbone | Params | Train CE | Mixed exact | Mixed blank | Sec/step | Peak |
|---|---:|---:|---:|---:|---:|---:|
| RWKV | 5.580M | 1.0432 | 0.00781 | 0.48666 | 3.43s | 9.04GiB |
| GDN | 5.980M | 0.9964 | 0.02148 | 0.50589 | 5.06s | 7.04GiB |
| GDN2 | 6.946M | 1.0078 | 0.01953 | 0.50163 | 5.44s | 9.08GiB |
| KDA | 5.852M | 1.0051 | 0.02344 | 0.50351 | 5.34s | 7.55GiB |

Open `index.html` for official blank ranges, loop curves, kernel provenance, and same-puzzle visualizations.

## Fairness Boundary

This is a shared-recipe comparison, not a state-, parameter-, or compute-matched
comparison. RWKV uses a `6 x 32 x 32` recurrent state; the three FLA arms use
`6 x 64 x 32` states because the suite fixes `expand_v=2`. Parameter counts and
seconds per step also differ materially. Use this result to choose a
provisional engineering carrier, not to claim that GDN is intrinsically better
than RWKV, GDN2, or KDA.
