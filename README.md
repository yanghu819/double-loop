# Double Loop

This repository studies one mechanism:

```text
causal recurrent token mixer
+ FutureSeed cross-layer terminal-state initialization
+ repeated depth loops
```

FutureSeed is not a reverse scan. A layer's terminal recurrent state initializes
the next layer, so later layers receive a cheap summary of the full sequence.
The depth loop then spends more recurrent compute on the same problem.

Sudoku is a proxy for global constraint reasoning, not the product target. The
canonical task is official 9x9 Sudoku with 46-64 blanks.

## Canonical Baseline

The retained scale baseline is:

- local GDN Triton recurrent kernel;
- D224, 12 layers, 14 heads, head dimension 16, value expansion 4;
- native FutureSeed scale 1;
- five depth loops with CE on every loop;
- 3.83M independent official training boards;
- clean training only: no noise, search, selector, repair, or Sudoku rule.

The fixed recipe is
[`configs/sudoku/gdn_scale.env`](configs/sudoku/gdn_scale.env). Its archived
step30000 result reaches:

| Readout | Exact |
|---|---:|
| mixed official test, loop1 | 0.0234 |
| mixed official test, loop5 | 0.4805 |
| 46-50 blanks, loop5 | 1.0000 |
| 51-55 blanks, loop5 | 0.6152 |
| 56-64 blanks, loop5 | 0.3848 |

Run it on GPU1:

```bash
CUDA_VISIBLE_DEVICES=0 ./run.sh baseline
```

## Fair Backbone Benchmark

The benchmark compares `rwkv`, `gdn`, `gdn2`, and `kda` while fixing data,
sample order, outer dimensions, loops, loss, optimizer, effective batch, BF16,
training steps, and evaluator. Architecture-specific parameter count, wall
time, and peak memory are reported rather than hidden.

`gdn`, `gdn2`, and `kda` use exact official FLA layers from the pinned local
wheel. `rwkv` uses the local RWKV7 state-passing CUDA kernel. The launcher
refuses automatic kernel fallback.

```bash
CUDA_VISIBLE_DEVICES=0 ./run.sh baseline_preflight
CUDA_VISIBLE_DEVICES=0 ./run.sh benchmark rwkv
CUDA_VISIBLE_DEVICES=0 ./run.sh benchmark gdn
CUDA_VISIBLE_DEVICES=0 ./run.sh benchmark gdn2
CUDA_VISIBLE_DEVICES=0 ./run.sh benchmark kda
```

To run all four sequentially and build JSON, CSV, HTML, and same-puzzle loop
visualizations:

```bash
CUDA_VISIBLE_DEVICES=0 ./run.sh benchmark_suite
```

The exact contract is
[`configs/sudoku/backbone_benchmark.env`](configs/sudoku/backbone_benchmark.env).
In that contract, public `gdn` means official FLA GDN. Internal
`BACKBONE=gdn` remains the retained local GDN-Triton scale implementation.

## Reproduction

All persistent paths stay under the repository:

```bash
./down.sh
./setup.sh
CUDA_VISIBLE_DEVICES=0 ./run.sh baseline_preflight
```

`down.sh` validates a locally staged official Sudoku archive. It does not
silently download Hugging Face data on the server. `setup.sh` installs the
pinned FLA wheel from `wheelhouse/` and verifies its SHA256.

The canonical code paths are listed in
[`baseline/manifest.json`](baseline/manifest.json). Historical paths are not
deleted because negative results are evidence; their maintenance status is in
[`docs/DEPRECATIONS.md`](docs/DEPRECATIONS.md).
