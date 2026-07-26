# FutureSeed Causal Four-Carrier Gate

Verdict: **SUPPORTED**. FutureSeed improves 46-50-blank full-board exact by at least 10 points on 4 causal carriers (rwkv, gdn, gdn2, kda). The cross-carrier causal gate passes.

| Carrier | b46-50 FS | b46-50 noFS | exact delta pts | FS time overhead | FS VRAM delta |
|---|---:|---:|---:|---:|---:|
| RWKV7 TimeMix | 72.85% | 0.00% | +72.85 | +9.3% | +0.29 GiB |
| GDN | 36.33% | 0.00% | +36.33 | +7.2% | +0.34 GiB |
| GDN2 | 79.69% | 0.00% | +79.69 | +17.2% | +0.34 GiB |
| KDA | 64.65% | 0.00% | +64.65 | +8.2% | +0.34 GiB |

Open `index.html` for loop curves, difficulty ranges, exact fairness checks, and same-puzzle visualizations.
