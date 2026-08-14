# P-GDN3-038: Content-Partitioned Full-State GDN2

## 1. Metainfo

- Status: discarded; completed_rejected
- Date: 2026-08-14
- Benchmark: directional MQAR L1024 wrong-key regime
- Fixed setting: D128/L2/H4/K32/V32, native FutureSeed, 10 epochs,
  batch32, seed123
- Arms: contemporaneous native GDN2 control, then one S2 slot-state candidate
- Resource: one task-mode GPU1, no concurrent model process

## 2. Evidence Boundary

P-GDN3-031/036 show that splitting erase and write keys breaks coherent row
ownership. P-GDN3-037 shows that scalarizing the exact committed residual also
destroys retrieval. P-FS2-009 improves aggregate accuracy but increases
wrong-key valid-value swaps, so receiver metric transport does not solve the
binding collision.

The nearest capacity tests do not cover the proposed topology. P022 divides a
fixed byte budget into H8/K16 banks with no learned admission and loses the
full K32 address space. P013 adds a zero-read companion bank to a mature
Sudoku checkpoint; its Q/K path is still exactly dead at the registered
step3001 gate. P006 is a residual side expert, P017 routes rows inside one
state, P019 uses a Raven controller around the existing state, and P025 writes
detached edits to a subordinate correction memory. None makes two full K32xV32
states first-class parts of the same live GDN2 transition and uses one shared
content hash for admission and read.

## 3. Mechanism

Each original H4 head owns exactly two independent K32xV32 recurrent states.
A bias-free D128-to-H4xS2 linear map produces per-token slot probabilities
`p`. The same probabilities are used at write and query time. Define slot mass
`rho=2p`. For either native erase or write gate `x in [0,1]`, route it as

`route(x,rho) = rho*x / (1 + (rho-1)*x)`.

This map is bounded in `[0,1]`, is exactly zero at `rho=0`, and is exactly the
native gate at uniform `rho=1`. Q, K, V and decay are repeated over the slot
axis; routed erase/write gates drive one pinned-official `chunk_gdn2` call with
the slot axis flattened into heads. The two official outputs are combined by
the same `p`, then pass through the unchanged native output norm/projection.
Native FutureSeed transports all H4xS2 terminal states.

The candidate adds 2,048 router weights plus eight duplicated per-slot
FutureSeed gates, exactly 2,056 parameters. State values/layer double from
4,096 to 8,192. Token count, physical scan count, loss, data and task logic do
not change. S is fixed at two; there is no temperature, top-k, auxiliary loss,
cache, selector or Sudoku logic.

## 4. Falsifiable Prediction

If wrong-key errors reflect collisions between bindings forced through one
state trajectory, the learned hash should specialize without collapsing,
produce distinct bounded slot states, and improve balanced accuracy by at
least 0.10 while reducing wrong-key swap fraction by at least 0.05. Both
future and past directions must improve, because this is a memory-binding
mechanism rather than a directional shortcut.

If the router activates and slots diverge but quality does not pass, content
partitioning into full states is insufficient. If it stays uniform/collapses,
the topology is not learnable under the fixed budget. Either outcome closes
the family; do not tune slot count, router form, temperature, gate map, init,
seed, LR, loss, batch, width, depth or duration.

## 5. Contract And Matched Initialization

Before formal training, exact pushed source in a clean detached worktree must
prove on GPU1:

1. CUDA index0, exact UUID, pinned FLA SHA and clean Zoology SHA;
2. exactly one official `ChunkGDN2FunctionBackward` per layer and six Triton
   short convolutions per arm, with no fallback;
3. exact +2,056 parameters, 8,192 state values/layer and one scan/layer;
4. candidate parent tensors bit-match a serialized control initialization;
5. a zero router gives bit-exact full output and, for finite nonzero incoming
   state duplicated across slots, bit-exact output and both terminal slots;
6. registered nonzero router initialization changes full output, gives finite
   nonzero gradients in both layers, and yields token/board variation plus
   distinct finite slot states;
7. swapping the two slots and router outputs commutes exactly with full model
   output;
8. the routed gate is exactly native at unit mass and bounded on mass `[0,2]`;
9. train/test hashes and control/candidate warmup batch hashes match.

The endpoint serializes one canonical native initialization, loads it into the
control, and maps every parent tensor into the candidate. Training RNG is
reset after warmup in the existing runner. This avoids the known confound in
which extra module construction shifts shared initial tensors.

## 6. Activation And Quality Gates

At the fixed endpoint both layers must have normalized router entropy in
`[0.10,0.95]`, mean max probability at least `0.55`, global mass of each slot
in `[0.10,0.90]`, token and board variation at least `1e-4`, state relative
difference at least `1e-3`, max slot-state cosine below `0.995`, and routed
erase/write change at least `1e-4`. Native FutureSeed must be active.

All quality checks are binding:

- balanced accuracy at least `0.50` and at least control `+0.10`;
- future and past accuracy each at least `0.45` and control `+0.07`;
- joint exact at least `0.05` and control `+0.04`;
- fewer total errors;
- wrong-key valid-value swap fraction among errors lower by at least `0.05`.

Carrier instability makes the same-process control authoritative; historical
scores are context only.

## 7. Cost And Kill Gates

Because the state and official head work deliberately double, elapsed,
post-warm wall and independently warmed-step ratios must each stay below
`2.75x`; peak allocation must stay below `2.00x`. Any integrity, activation,
quality or cost miss closes P-GDN3-038 and writes the structured abort. No
nearby rescue is authorized.

## 8. Required Readout

Archive both directional metrics, joint exact, CE curves, total errors,
wrong-key valid-value swaps, router entropy/mass/variation, slot-state
divergence/cosine/RMS, routed erase/write changes, native FutureSeed
diagnostics, state/parameter counts, elapsed/wall/warmed/allocation ratios,
source/config/checkpoint/score/log hashes and exact GPU provenance.

## 9. Decision

The exact pushed source was
`44f107196dead62a71e24828dba192ae55cedbc0`, run from a clean detached
worktree on A100-SXM4-40GB index0 UUID
`GPU-31166d8c-9fe5-d953-dc44-d0d549969ada`. The strict CUDA contract passed.
Both arms used the same serialized parent initialization and data hashes, and
the candidate retained exactly one pinned-official GDN2 scan per layer. The
candidate added the registered 2,056 parameters and doubled recurrent state
values/layer from 4,096 to 8,192 without a fallback.

The router and both full states genuinely activate. Layer router entropy is
`0.55347/0.38086`, mean maximum probability is `0.76690/0.89718`, and global
slot mass remains noncollapsed at `0.25747..0.74253` and
`0.17010..0.82990`. Token and board variation are finite, routed erase/write
changes are about `0.26..0.30` RMS, and mean slot-state cosine falls to
`0.86547/0.72373`. Layer1 state relative difference reaches `1.06652`.
However, the strict max-cosine activation gate misses at
`0.99977/0.99699`, showing that some examples still leave the slots nearly
identical.

Quality fails decisively. Control versus candidate balanced/future/past/joint
accuracy is `0.4940/0.4540/0.5340/0.0410` versus
`0.36525/0.3615/0.3690/0`. Future/past CE rises from
`1.25939/1.22668` to `2.20532/2.23182`, and total query errors rise
`2,024 -> 2,539`. The candidate does reduce wrong-key valid-value swaps
`1,546 -> 1,508` and their fraction among errors
`0.76383 -> 0.59393`, passing that isolated binding check. The reduction is
not closure: it comes with 515 additional errors and zero joint-exact cases.

Systems gates pass. Elapsed/post-warm/warmed-step/allocation ratios are
`1.1690/1.1710/1.6681/1.8344`, within the registered
`2.75/2.75/2.75/2.00x` limits. This therefore falsifies the fixed two-slot
content partition rather than exposing a runtime defect.

P-GDN3-038 is closed. A shared soft content hash can separate two full live
states and reduce conditional key swaps, but joint write/read routing delays
optimization and sacrifices general retrieval. Do not rescue slot count,
router form, temperature, gate map, initialization, seed, LR, loss, batch,
width, depth or duration. The next mechanism must avoid assigning one token to
one global memory trajectory while preserving native coherent erase/write/read
ownership.

Artifacts:

- run: `/huyang2/double-loop/runs/p-gdn3-038-slot-state-l1024-20260814T152219Z-44f1071`;
- contract SHA256: `092d1227ff45a19f9f4d7b5065c2abe4d302f92bad7f7cad0af79df46142e1b9`;
- comparison/decision SHA256: `aa4b75d43859ec519db55b8a1b91163fe873fdda575aa79c7fef061cc85b90c7`;
- control/candidate score SHA256: `e0cfc26013b92cd902af5f19915440520a545c6cfb576fc8ce0417de1bee03ad` / `dc9108a72c726db5e88e9bb63aa66f02fac46cedd92fdca36ea62fbe5393dbf6`;
- control/candidate checkpoint SHA256: `52d549ed70f6fa3426d6511f7472ab59173dc75f7f51ea4605b3be42cec3eae7` / `3bc9f6fdec07999659515f34408d5b26260119585fe7220458bcbd20cc5fab1c`;
- formal log SHA256: `54b8e2163caa41c3a40b2871fe1f17b4358ce9cf50cfbc1d13af98ff9f574207`;
- artifact manifest SHA256: `36df5e52eceea47aba634a6a95dacf499f18977b4ef9026376c1cffd770767a0`.
