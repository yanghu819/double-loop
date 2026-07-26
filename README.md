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

## Active Scale Candidate

The first quality candidate is now:

- official FLA GDN2 with strict chunk backward and Triton convolution;
- D256, 12 layers, 8 heads, head dimension 32;
- native FutureSeed scale 1;
- five depth loops with CE on every loop;
- 3.83M independent official training boards;
- clean training only: no noise, search, selector, repair, or Sudoku rule.

The fixed recipe is
[`configs/sudoku/gdn2_scale.env`](configs/sudoku/gdn2_scale.env). It trains for
12k steps while moving from 46-50 to 51-64 blanks, with formal checkpoints at
500/1000/3000/6000/9000/12000. This promotion is provisional until the hard
scale gate completes.

The strongest completed long-run reference remains the previous local GDN
D224/L12 step30000 result:

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

The previous reproducible GDN line remains available as:

```bash
CUDA_VISIBLE_DEVICES=0 ./run.sh gdn_legacy
```

## Fair Backbone Benchmark

The benchmark compares `rwkv`, `gdn`, `gdn2`, and `kda` while fixing data,
sample order, outer dimensions, loops, loss, optimizer, effective batch, BF16,
training steps, and evaluator. Architecture-specific parameter count, wall
time, and peak memory are reported rather than hidden.

`gdn`, `gdn2`, and `kda` use exact official FLA layers from the pinned local
wheel. `rwkv` uses the audited official RWKV7 TimeMix contract with explicit
initial/final state I/O and state-passing CUDA. The launcher refuses automatic
kernel fallback.

The accepted shared-initialization GPU1 step-500 rerun is:

| Backbone | Params | Train CE | Mixed loop5 exact | 46-50 exact | 51-64 exact | Sec/step | Peak |
|---|---:|---:|---:|---:|---:|---:|---:|
| RWKV7 | 5.093M | 1.0190 | 0.0195 | 0.7285 | 0 | **4.04** | 10.78 GiB |
| GDN | 4.867M | 1.0509 | 0.0059 | 0.3633 | 0 | 4.89 | **5.99 GiB** |
| GDN2 | 5.462M | **1.0062** | **0.0234** | **0.7969** | 0 | 5.85 | 7.84 GiB |
| KDA | 4.736M | 1.0179 | 0.0176 | 0.6465 | 0 | 5.58 | 6.45 GiB |

GDN2 is the strongest finite-budget opener in this state-matched table and is
therefore the active quality scale candidate; RWKV7 is the fastest. The
separate local GDN-Triton recipe remains the strongest completed long-run
reference. This table does not claim a universal architecture winner; all four
remain at zero full-board exact beyond 50 blanks at this short budget.
Full provenance, loop curves, and same-puzzle cases are in
[`research/reports/experiments/futureseed-sudoku-official-rwkv7-four-backbone-20260724.md`](research/reports/experiments/futureseed-sudoku-official-rwkv7-four-backbone-20260724.md).

## Causal FutureSeed Gate

The fair carrier table above enables FutureSeed in every arm, so it does not by
itself measure FutureSeed's contribution. The paired causal gate uses the same
source, byte-identical initialization, data order, objective, optimizer, state
geometry, five-loop budget, evaluator, and parameter count within each carrier.
Only the FutureSeed injection scale and artifact paths differ.

| Carrier | 46-50 exact FS | 46-50 exact noFS | FS time overhead | FS VRAM delta |
|---|---:|---:|---:|---:|
| RWKV7 | 0.7285 | 0 | +9.3% | +0.29 GiB |
| GDN | 0.3633 | 0 | +7.2% | +0.34 GiB |
| GDN2 | 0.7969 | 0 | +17.2% | +0.34 GiB |
| KDA | 0.6465 | 0 | +8.2% | +0.34 GiB |

Official source, CUDA/Triton backward, data hashes, per-pair arguments,
constructor parameter hashes, and same-trained-weights functional checks all
pass. This supports FutureSeed as a generic short-budget opening mechanism for
causal recurrent carriers. It does not yet establish hard closure: every arm
is still zero exact at 51-64 blanks. The full result and visualizations are in
[`research/reports/experiments/futureseed-causal-four-backbone-20260726.md`](research/reports/experiments/futureseed-causal-four-backbone-20260726.md).

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
