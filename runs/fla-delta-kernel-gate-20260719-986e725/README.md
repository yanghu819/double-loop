# FLA GDN2/KDA CUDA Gate

- GPU: NVIDIA A800-SXM4-80GB, `CUDA_VISIBLE_DEVICES=0`
- Source: `986e72507e42baa3bce1ebedfbe61a08ee8b8dce`
- Official FLA: `fe8fce9fc6984f22905f54cfa885dce1502baf26`, version `0.5.2`

Both official chunk kernels passed CUDA Torch-reference output, terminal-state,
backward, initial-state-gradient, sequence-split, and native FutureSeed stack
checks. The only missing parameter gradient is the expected first-layer
`future_seed_logit`, which has no preceding layer state to gate.

At the adapter test shape, GDN2 forward+backward measured `7.896ms`; KDA measured
`12.522ms`. These are hot-cache microbenchmarks, not end-to-end training times.
The KDA cold adapter check spent `708s` compiling/autotuning, so cold wall time
must not be treated as steady-state throughput.
