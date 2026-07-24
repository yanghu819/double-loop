# FutureSeed Sudoku Backbone Benchmark

Decision: No backbone separates on hard full-board closure at this finite budget. GDN2 has the best 46-50 blank opening exact (0.7969); RWKV7 TimeMix is cheapest per optimizer step. Do not select or claim a universal architecture winner from this gate.

| Backbone | Params | Train CE | Mixed exact | Mixed blank | Sec/step | Peak |
|---|---:|---:|---:|---:|---:|---:|
| RWKV7 TimeMix | 5.093M | 1.0190 | 0.01953 | 0.49795 | 4.04s | 10.78GiB |
| GDN | 4.867M | 1.0509 | 0.00586 | 0.48869 | 4.89s | 5.99GiB |
| GDN2 | 5.462M | 1.0062 | 0.02344 | 0.50114 | 5.85s | 7.84GiB |
| KDA | 4.736M | 1.0179 | 0.01758 | 0.50121 | 5.58s | 6.45GiB |

Open `index.html` for official blank ranges, loop curves, kernel provenance, and same-puzzle visualizations.
