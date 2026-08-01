# Raven FutureSeed vs GDN2 FutureSeed

- Decision: Raven does not beat GDN2 on the primary hard-range exact metric.
- 51-55 exact delta: +0.0000
- 51-55 blank-accuracy delta: -0.0782
- Raven loop5-loop1 exact: +0.0000
- Raven/GDN2 sec per step: 1.275x
- Both arms use 1024 recurrent-state elements per head and the same source/data/seed/optimizer/loop loss.
- Raven uses 16 slots and top-2 routing; GDN2 keeps its established official short convolution.

## Matched endpoint

| Metric | GDN2 + FutureSeed | Raven + FutureSeed |
|---|---:|---:|
| Parameters | 5.462M | 4.650M |
| Train CE | 1.0186 | 1.1388 |
| Peak allocated VRAM | 8025 MiB | 7262 MiB |
| Seconds / step | 6.041 | 7.703 |
| 46-50 exact / blank | 0.7969 / 0.9918 | 0 / 0.7771 |
| 51-55 exact / blank | 0 / 0.5257 | 0 / 0.4475 |
| 56-64 exact / blank | 0 / 0.4680 | 0 / 0.4052 |

On the shared primary board, GDN2 has `23/21/22/22/22` wrong cells over
loops 1-5; Raven has `34/31/31/30/30`. Raven does make four corrections, but
starts much weaker and never turns them into a solved board. Do not rescue this
negative result with a slots, top-k, seed, learning-rate, or loss sweep.
