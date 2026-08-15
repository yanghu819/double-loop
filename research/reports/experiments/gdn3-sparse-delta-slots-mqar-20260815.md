# P-GDN3-050: Sparse Committed-Delta Pair Slots

## 1. Metainfo

- Status: completed; discarded at the single registered endpoint
- Run: `p-gdn3-050-sparse-delta-slots-l1024-20260815T062052Z-faef909`
- Source: exact pushed/read-back `faef909df85fc50285ca68d040cb3a0f879e09dc`
- Task: directional MQAR L1024, four future and four past queries
- Parent: D128/L2/H4/K32/V32 pinned-official GDN2 plus native FutureSeed
- Endpoint: candidate-only 10 epochs, batch32, seed123, frozen P-REPRO-001
  initialization and replay-B control
- GPU: A100-SXM4-80GB CUDA index0,
  `GPU-573c7ed1-1c51-8334-299b-edf2ff3440e6`

## 2. Mechanism Hypothesis

P-REPRO-001 proves that the protocol is exactly reproducible and that
`1546/2024 = 76.38%` of native errors retrieve a valid value under the wrong
key. P047 can recover the value set, while P049 shows that another dense
canonical-address matrix does not restore ownership. The falsifiable P050
hypothesis is therefore that committed edits need explicit pair isolation
inside live recurrent memory.

Each layer keeps its native GDN2 state, scan and FutureSeed. In parallel, the
exact committed edit already computed by the official GDN2 forward is written
to one of 16 learned address slots through pinned-official `chunk_gsa`. Its
factorized state has a key factor `[K,16]` and value factor `[16,V]`, so slot
`m` owns both sides of one update and cross-slot key/value products never form.
The local sparse read enters through a zero-initialized per-head gate. At the
layer boundary, the factor product is normalized and enters the receiver's
native KxV FutureSeed through a separate zero-initialized gate.

## 3. Exact Configuration

- one native official GDN2 scan plus one official GSA scan per layer;
- 16 slots, hard top-1 forward routing with a soft straight-through gradient;
- exactly 4,108 new parameters: two H4x16xK32 anchor tensors, eight local read
  gates and four receiving FutureSeed gates;
- exactly 4,096 additional factor-state values per layer;
- train/test examples `10000/1000`, sequence length 1024, four key/value pairs;
- no top-k, temperature, slot-count, router, seed, optimizer or duration sweep;
- no Sudoku logic, selector, search, repair, oracle or recurrence fallback.

## 4. Environment And Integrity Contract

The valid run used pinned FLA
`9c8e42e762fce087c27b673af4922795d9edb85e` and clean Zoology
`1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb`. The strict contract passed with
two `ChunkGDN2FunctionBackward` and two `ChunkGSAFunctionBackward` nodes,
bit-exact zero-gate full output, bit-exact output/state under finite nonzero
incoming main state, exact parameter/state/scan counts, finite nonzero gate and
anchor gradients, exact head-permutation equivariance, bounded transition,
finite state geometry and no fallback.

The fixed parent initialization SHA256 is
`7402e46c48cbd47070d65262d1a1b62a32ac55a4d644a6fb9644b96b0914850f`.
Train/test and warmup hashes match the frozen control exactly.

## 5. Commands And Registered Gate

The clean detached launcher executed:

```bash
CUDA_VISIBLE_DEVICES=0 \
  REPO_ROOT=/huyang2/double-loop/worktrees/p-gdn3-050-b607906 \
  EXPECTED_SOURCE_SHA=faef909df85fc50285ca68d040cb3a0f879e09dc \
  bash scripts/run_zoology_sparse_delta_slots.sh
```

Quality required balanced accuracy `>=.85` and `+.10` over `.494`, future and
past accuracy each `>=.82`, joint exact `>=.60`, fewer total errors and a
wrong-key swap-fraction reduction of at least `.10`. Activation required all
16 slots at both layers, entropy `>=.50`, max usage `<=.30`, live read/seed
gates and separated slot keys. Elapsed, post-warm and warmed-step ratios each
had to be `<2.0x`; peak allocation had to be `<1.60x`. Every check was fixed
before launch and conjunctive.

## 6. Artifacts

- run directory: `/huyang2/double-loop/runs/p-gdn3-050-sparse-delta-slots-l1024-20260815T062052Z-faef909`;
- comparison/score SHA256: `c13a7550a60751b1661b476e6667fe51c922446f3b5a397a9b683dc924cdaac7`;
- contract JSON SHA256: `e1ef247383fbe96ecee750fc6ae33d43927dc21f47dba3c1e036f68e6fc853ce`;
- candidate checkpoint SHA256: `9daa4614903bfcb590a043f12b26d34a95ad6c6eed0de2270edc59c8f7ab25be`;
- config/cases SHA256: `c3c13cd26de76b988080d66a8ebb3fa8c36cca4b8ba248699070ecc5f38adbf2` /
  `402c1ca3ddb097c12a90022530fcc9dbba77179d8ed29e33e96b091bb012db2c`;
- formal log/source snapshot SHA256: `7e0b91410b77fceddecf52f403f3c304e21216d923c01b0205ae41389735d761` /
  `12113cff59a65960010fec4135e4f0476d593d3e7a50cfb018b36638585ea644`;
- artifact manifest SHA256: `57d77d3e52817fb68b1f2e1254609b8e0c7951d8a6ef49e08928d5381a11a8d6`.

The launcher and endpoint both exited zero; final launcher status `2` is the
registered science-gate closure. There is no `abort.json` for the valid run.

## 7. Results

| metric | frozen control | candidate | delta |
| --- | ---: | ---: | ---: |
| balanced accuracy | `.494000` | `.044250` | `-.449750` |
| future accuracy / CE | `.454000 / 1.25939` | `.051000 / 4.05954` | `-.403000 / +2.80015` |
| past accuracy / CE | `.534000 / 1.22668` | `.037500 / 4.10436` | `-.496500 / +2.87767` |
| joint exact | `.041000` | `0` | `-.041000` |
| total query errors | `2,024` | `3,823` | `+1,799` |
| wrong-key valid-value swaps | `1,546` | `255` | `-1,291` |
| wrong-key fraction of errors | `.763834` | `.066702` | `-.697132` |

The sparse branch is live rather than numerically dead. Local read-gate absolute
means are `.014259/.010260`; the receiving sparse-FutureSeed gate is `.041901`.
Committed-edit RMS is `.041854/.047798`, sparse-output relative RMS is
`.228236/.117400`, and state statistics remain finite. Routing does not satisfy
the preregistered occupancy gate: layer 0 uses 15 slots and layer 1 only 10,
with entropy `.6882/.6730`. Thus hard allocation loses capacity while the
native task trajectory is still trying to form useful addresses.

Elapsed, post-warm, independently warmed step and peak-allocation ratios are
`3.2773/3.2511/1.5856/1.3873x`; both wall-time gates fail. Across the whole
launcher, 147 five-second GPU samples average `33.38%`; the 97 nonzero samples
average `50.59%`, peak at `87%`, peak memory is `5,157 MiB`, and peak power is
`292.59 W`. Whole-launch averages include source archiving, imports, compile,
contract and validation idle intervals.

## 8. Conclusions

Discard P-GDN3-050. The lower conditional wrong-key fraction is not a binding
win: the candidate makes 1,799 more errors and usually fails to retrieve a
valid value at all. Explicit pair slots are a useful representation idea, but
hard learned assignment in a second recurrent scan creates a new credit-routing
problem and slows the native learning trajectory enough to erase its baseline
capability.

This leaves the main diagnosis intact but sharpens it. The unresolved problem
is ownership-preserving address/value binding in the live transition, not a
lack of state bytes. A successor must preserve the native retrieval path while
making ownership intrinsic and differentiable; it must not rescue slot count,
temperature, anchors, gates, normalization, scan placement or training budget.

## 9. Submission Record

No Sudoku transfer or external submission is authorized. This checkpoint is
retained only as a closed mechanistic boundary.
