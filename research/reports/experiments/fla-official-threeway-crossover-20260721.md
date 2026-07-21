# Official FLA FutureSeed Three-Way Crossover Check

- Plan ID: `P-LA-005`
- Date: 2026-07-21
- Source SHA: `87e66cf91c086283e6981235c93152579fb93dc4`
- Machine: GPU1, NVIDIA A800-SXM4-80GB
- GPU2: unused

## Question

Was the step500 result evidence that GDN is intrinsically a better FutureSeed
carrier than KDA/GDN2, or only evidence that the shared small-state recipe
opens GDN earlier?

P-LA-004 already showed that GDN2 erased nearly all of its early opening gap by
step1000. KDA still had only a step500 result, so a broad GDN-over-KDA claim was
not supported. The highest-information check was one exact KDA continuation,
not another architecture, seed, LR, loss, or geometry table.

## Predeclared Decision

- If KDA reduced its step1000 CE gap versus GDN below `0.02` and its official
  46-50 exact gap below `0.05`, classify the old ranking as a finite-budget
  artifact.
- If it remained outside both boundaries, retain only the scoped conclusion
  that GDN is more sample-efficient under this shared D32 recipe.
- Stop on wrong GPU, checkpoint/provenance mismatch, NaN/OOM, or more than 15
  minutes per 100 optimizer steps.

## Matched Design

All three arms use:

- official `flash-linear-attention==0.5.2` layers;
- D192, 10 layers, 6 heads, head dimension 32, expand-v2, short-conv size 4;
- native FutureSeed scale 1 through official FLA `Cache` initial states;
- five depth loops with CE on every loop;
- effective batch 128, BF16, LR `0.0015`, weight decay `0.001`, seed 52;
- official full-diversity Sudoku data;
- executed curriculum `46-50:100,51-55:900`;
- identical fixed-hole and official test batches.

Each arm resumed its exact step500 model, optimizer, and Python/NumPy/Torch RNG
state. No noise, repair, search, selector, reverse scan, or task rule was added.

## Implementation Provenance

The post-run strict gate verifies:

- pinned wheel SHA256
  `65f57bf2aa937991fc497bd63f42883d263ead8646f21f2492735ccddd82d0eb`;
- installed FLA source marker
  `fe8fce9fc6984f22905f54cfa885dce1502baf26`;
- every audited installed source file matches the pinned wheel;
- official layer classes `GatedDeltaNet`, `KimiDeltaAttention`, and
  `GatedDeltaNet2`;
- actual backward nodes `ChunkGatedDeltaRuleFunctionBackward`,
  `ChunkKDAFunctionBackward`, and `ChunkGDN2FunctionBackward`;
- Triton Q/K/V short convolution and nonzero initial-state gradients.

There is no Torch recurrent fallback and no right-to-left scan mislabeled as
FutureSeed.

## Results

### Learning curve

| Step | GDN CE | KDA CE | GDN2 CE |
|---:|---:|---:|---:|
| 500 | 0.9981 | 1.0252 | 1.0101 |
| 600 | 0.9526 | 0.9676 | 0.9607 |
| 700 | 0.9433 | 0.9493 | 0.9293 |
| 800 | 0.9349 | 0.9545 | 0.9451 |
| 900 | 0.9366 | 0.9502 | 0.9420 |
| 1000 | **0.9413** | 0.9491 | 0.9539 |

The curves cross and fluctuate. KDA reduces its final CE gap versus GDN from
`0.0271` at step500 to `0.0078` at step1000. GDN2's final gap is `0.0126`.
This rejects a stable quality ordering inferred from step500.

### Quality and cost at step1000

| Metric | GDN | KDA | GDN2 |
|---|---:|---:|---:|
| Parameters | 5.980M | 5.852M | 6.946M |
| Fixed holes53 loop5 exact | **0.01758** | **0.01758** | **0.01758** |
| Mixed loop5 exact | **0.02344** | **0.02344** | **0.02344** |
| Mixed loop5 blank accuracy | **0.53183** | 0.52379 | 0.52795 |
| Continuation seconds/step | **5.15** | 5.65 | 5.70 |
| Peak allocated MiB | **7281** | 7809 | 9382 |

All three tie on the two pre-fixed full-board exact readouts. GDN retains a
small CE/blank-accuracy advantage and is about `9.8%` faster per continuation
step than KDA and `10.8%` faster than GDN2. That is an efficiency result under
this recipe, not an architecture ceiling.

### Official blank ranges at loop5

| Blanks | GDN exact / blank | KDA exact / blank | GDN2 exact / blank |
|---|---:|---:|---:|
| 46-50 | 0.99805 / 0.99983 | 0.99609 / 0.99975 | 0.99219 / 0.99971 |
| 51-55 | 0 / **0.55372** | 0 / 0.55042 | 0 / 0.55057 |
| 56-64 | 0 / **0.49090** | 0 / 0.48888 | 0 / 0.48583 |

Every model closes the easy-to-medium range and every model hits the same hard
full-board cliff. KDA/GDN2 do not produce a harder global-consistency win.

### Loop behavior

- GDN mixed exact: `0.01953 -> 0.02344`; same-seed wrong cells
  `26 -> 22 -> 25 -> 24 -> 24`.
- KDA mixed exact: `0.02344 -> 0.02344`; same-seed wrong cells
  `21 -> 17 -> 17 -> 18 -> 18`.
- GDN2 mixed exact: `0.02148 -> 0.02344`; same-seed wrong cells
  `22 -> 21 -> 19 -> 19 -> 19`.

The selected case shows KDA/GDN2 making a few early corrections, but it is one
case and does not override the aggregate result. None of the three keeps
improving full-board exact through later loops at this training budget.

## Decision

The predeclared crossover boundary passes for both KDA and GDN2. The step500
ranking was mostly a finite-budget artifact. The accurate conclusion is:

> Under a shared D32/expand-v2 recipe, official GDN is the cheapest
> FutureSeed-enabled carrier, while KDA and GDN2 reach essentially the same
> step1000 task quality at higher cost. No carrier solves hard closure here.

Do not claim that GDN is universally stronger. Do not launch native-geometry,
seed, LR, or loss tables from this result. The main line should return to the
already proven independent-data and longer-compute scaling path. The causal
FutureSeed claim comes from the separate matched FutureSeed/no-FutureSeed run,
because every arm in this comparison contains FutureSeed.

## Artifacts

- Three-way report and visualization:
  `runs/fla-official-threeway-crossover-20260721-87e66cf/`
- KDA continuation:
  `runs/fla-official-kda-fs-d192l10-resume500-s1000-20260721T061500Z-87e66cf/`
- GDN continuation:
  `runs/fla-official-gdn-fs-d192l10-resume500-s1000-20260721T010000Z-87e66cf/`
- GDN2 continuation:
  `runs/fla-official-gdn2-fs-d192l10-resume600-s1000-direct-leg2-20260721T045147Z-87e66cf/`
- Exact train checkpoints remain under `/huyang2/double-loop/models/` and are
  not committed.
