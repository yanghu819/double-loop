# P-GDN3-006: Dual-State Residual Expert

## 1. Metainfo

- Status: discarded after a clean formal step3000-to3100 matched decision
- Date: 2026-08-07
- Branch: `codex/gdn3-residual-state-expert-20260807`
- Benchmark: official/full-diversity hard 9x9 Sudoku 51-64 blanks
- Compute: AIStation task-mode GPU1 only
- GPU UUID: `GPU-53e9f3b4-2966-65d3-6614-09c540921519`
- Seed: 52 only
- Parent mechanism: D256/L12 position-QK GDN3 plus native terminal FutureSeed
- Parent checkpoint:
  `/huyang2/double-loop/models/gdn3-position-qk-d256l12-s12000-20260806T131527Z-9f2ee8d/checkpoints/train_state_step003000.pt`
- Parent checkpoint SHA256:
  `6339c3cb2b5fc5230a581d6633716483e35ff8e4522f06a9d7aaf26512f023da`
- Parent source SHA: `9f2ee8d1738032bc5f09b55db0b81d507780b376`
- Frozen matched control:
  `p-fs3-001-terminal-s3100-20260806T200859Z-3e167b6`

The formal source was pushed before launch and executed from a clean detached
worktree at exact SHA `605b9c2ed58973aeeed97117def4bb6139c2428c`.

## 2. Closed Nearby Axes

The current parent has genuine late-loop correction and nonzero hard exact,
but four zero-init matched interventions fail to increase board closure:

1. orthogonal FutureSeed innovation residual;
2. a shared producer-update codec;
3. address-local producer-update routing;
4. coherent erase/write gate coupling.

All four activate and move soft state dynamics. The first three leave hard
exact unchanged from the transfer side. The fourth also leaves exact unchanged,
learns the opposite gate geometry from its hand hypothesis, and exceeds both
cost gates. This closes another scalar, residual, gate, or small-router repair.

Ordinary value-width expansion is also not the proposed test. Earlier
function-preserving expand-V work showed transient gains that did not persist
at its strong endpoint. A second raw-width point would be a table, not a new
mechanism question.

## 3. Mechanism Hypothesis

One position-addressed KxV matrix per layer must use a single learned address
and update geometry for persistent constraints, transient corrections, and
competing partial assignments. The hard boards may therefore need a second
learned recurrent subspace, not a stronger scalar applied to the first one.

The candidate adds a compact, independent official-GDN2 expert beside every
main GDN2 layer. It has its own learned position address, payload, decay,
erase, write, normalization, and terminal state. Native terminal FutureSeed is
applied independently across adjacent auxiliary experts. A zero-initialized
residual readout makes the full parent function exact at migration while
allowing optimization to recruit the new state only when useful.

This is a scalable learned-compute hypothesis: expert count and state capacity
can grow in principle, but this experiment tests exactly one fixed additional
expert. It adds no Sudoku rule, search, repair, selector, oracle, reverse scan,
auxiliary loss, or hand-coded state content.

## 4. Candidate

The main path remains exactly D256/H8/K32/V32 position-QK GDN2. Each of the 12
blocks receives one independent residual expert:

- down-projected content width128;
- separately down-projected canonical address width128;
- H8/K16/V16 official `GatedDeltaNet2` with short convolution4 and the pinned
  official `chunk_gdn2`/Triton backward;
- its own K16xV16 terminal state per head;
- native adjacent-layer terminal-state FutureSeed with its own head gate;
- a width128-to256 residual projection initialized to exact zero.

The expert state contains 2,048 values per layer versus 8,192 in the main
state, a fixed 25% state-capacity increment. Its independent Q/K projections
make this an additional address/update expert rather than an extra V payload
bank. No expert state is merged into, compressed from, or used to alter the
main state. The main terminal states and FutureSeed path remain bit-exact at
migration.

The declared parameter delta is exactly 2,479,488: 206,624 parameters per
expert across 12 blocks. The declared recurrent-state delta is 2,048 values
per layer and 24,576 values over all 12 auxiliary states. The CUDA contract
must reproduce both counts from the constructed model and exact migration set.

At zero residual readout, expert computation and terminal states are invisible
to the logits. The expert readout must nevertheless receive finite nonzero
gradient. After one optimizer step opens the readout, the inner expert and its
FutureSeed gate must receive finite nonzero gradients. This two-stage learning
path is part of the launch contract.

## 5. Frozen Configuration

- D256/L12/H8/K32/V32 main path, channel multiplier4
- position-QK addressing, random traversal
- native terminal FutureSeed, loop5, equal CE at every loop
- 12 main plus 12 compact pinned official-FLA GDN2/Triton layers
- auxiliary D128/H8/K16/V16 state expert, exactly one per block
- official full-diversity curriculum and exact step3000 parent
- BF16, effective batch128, optimizer/RNG/data order exact resume
- LR0.0015, weight decay0.001, seed52
- candidate-only step3000-to3100 continuation

The existing terminal continuation is the frozen control and is not rerun.
There is no K8/K24/K32, expert-count, residual-scale, depth, width, or duration
follow-up.

## 6. Launch Gates

The strict CUDA contract must prove:

1. exactly CUDA index0 and the registered UUID;
2. pinned FLA SHA `9c8e42e762fce087c27b673af4922795d9edb85e`;
3. 12 main and 12 auxiliary exact official `GatedDeltaNet2` layers and
   `ChunkGDN2FunctionBackward` on both paths;
4. exact parameter migration set and exact declared state/parameter delta;
5. bit-exact full model output and all 12 main terminal states at zero readout,
   with and without nonzero main incoming states;
6. nonzero, finite, board-varying auxiliary terminal states that cannot affect
   the zero-init model output;
7. finite nonzero zero-init residual-readout gradient;
8. after one optimizer update, finite nonzero gradients in every auxiliary
   Q/K/V/decay/erase/write path and each of the 11 receiving-layer auxiliary
   FutureSeed gates;
9. exact-resume optimizer/RNG/data-order migration and a complete step3001
   checkpoint plus metrics JSON;
10. no fallback, NaN, OOM, source drift, data drift, or hidden CPU model path.

A production-shape forward/backward fit must remain below the 80 GiB device
limit before the formal continuation is authorized. A failed gate closes this
implementation; it does not authorize a smaller expert or execution rescue.

## 7. Science And Cost Gates

At step3100, activation requires all of:

- auxiliary residual-output relative RMS `>=1e-4` and finite;
- auxiliary terminal-state RMS and between-board variation finite and nonzero;
- auxiliary FutureSeed incoming-state RMS finite and nonzero from layer2 onward;
- non-collapsed address geometry and finite nonzero main/expert output cosine;
- all 12 experts active, with no silent fallback.

Quality passes by exactly one of:

1. hard51-64 macro loop5 exact improves by at least `+0.02`, with every hard
   range losing no more than `0.01` blank accuracy; or
2. mixed loop5 exact improves by at least `+0.03`, official61-64 does not
   regress, and same-board loop3-to5 wrong-cell correction is stronger.

Because this experiment deliberately adds general recurrent compute and 25%
state capacity, its cost gate is broader than the scalar tests: independently
warmed elapsed time and peak allocated memory must each remain below `50%`
overhead versus the frozen control. Timing instability, OOM, or fallback still
kills the run regardless of quality.

Any science, activation, or cost miss discards this exact dual-state expert.
Do not rescue expert width/count, address projection, residual scale, seed, LR,
loss, batch, main width/depth, or continuation duration.

## 8. Required Readout

Report and archive:

- mixed and official51-55/56-60/61-64 loop1-5 exact/blank/wrong cells;
- train CE and same-board loop3-to5 correction counts;
- auxiliary output/state/FutureSeed activation and address-diversity metrics;
- parameter/state deltas, independently warmed throughput, peak allocation,
  peak reservation, and timing stability;
- config, source, parent, checkpoint, metrics, log, and comparison hashes;
- same-board loop1-5 visualization using the frozen control case IDs.

## 9. Decision

The model path, exact-resume migration, diagnostics, formal environment,
single-GPU launcher, and strict two-stage CUDA contract are implemented.
`py_compile`, shell parsing, CLI exposure, and `git diff --check` pass without
instantiating a model on CPU. GPU work remains blocked until the complete
commit is pushed, its remote SHA is read back, and every launch gate above
passes from a clean detached AIStation worktree.

Contract R1 at source `d1250b6194e043013589636ce684995105c060e4`
stopped after identity and two-stage backpropagation because the checker
addressed official FLA's two-layer `f_proj` as a single `Linear.weight`.
This is a pre-science checker implementation error: no probe or formal run
started. It is recorded in
`artifacts/launch/p-gdn3-006/contract-d1250b6.abort.json` (SHA256
`3b416b5a6167519c71606969e200cb314f3ffe4b3549b0c1620b3525b6036cb1`).
R2 checks both `f_proj[0]` and `f_proj[1]` explicitly; no model setting or gate
changed.

Contract R2 at source `6b17b1d6fee4ef5aadef9acde93bf6716e0fcc88`
then completed the per-layer two-stage gradient assertions but stopped because
the checker searched the final model-output graph for the custom chunk
backward node. The established official-FLA audit surface is each returned
terminal-state graph. This second pre-science observation error is recorded in
`contract-6b17b1d-r2.abort.json` (SHA256
`f54fa141cd4bd067bc77f82736c52effd926a8f554295a87ed1e1bd5f921648b`);
its log SHA256 is
`dd5ea5e5fd2946405dc94c2e5d1db694c844e2da04fcafbb67cab8e39c3a651b`.
R3 checks every main and auxiliary terminal-state graph independently. No
model, data, optimizer, or registered gate changed.

Contract R3 at source `a25b04f3ec0d08bcf11afb9670545a8d23cba8f2`
then stopped on the main layer0 graph because the checker intentionally froze
all main parameters but did not mark its synthetic content/address inputs as
differentiable. The missing graph was therefore expected test-fixture
behavior, not a missing kernel. This third pre-science error is recorded in
`contract-a25b04f-r3.abort.json` (SHA256
`bf0142838ff274ad5f8aade1e292f229a147cb1bed0c385a7b6c11ace06a8959`);
its log SHA256 is
`31e615d0d51308688882cde7a695add055545834d0a5aa6fa8ae0668c184b2ec`.
R4 marks only the synthetic inputs as requiring gradients, while keeping every
main parameter frozen and every registered model setting unchanged.

Contract R4 passed at pushed source
`605b9c2ed58973aeeed97117def4bb6139c2428c`. Its log SHA256 is
`28f5596b93101bb082c80445efea02748a97d708d2eafc1934eed488c2ec044f`.
It confirms the target GPU/FLA SHA, exact zero-readout output and 12 main-state
identity, exact identity with nonzero incoming main and auxiliary states, 12
main plus 12 auxiliary official `GatedDeltaNet2` layers, both sets of 12
`ChunkGDN2FunctionBackward` graphs, 2,479,488 inserted parameters, 24,576
auxiliary state values, all 12 readout gradients, all 12 inner Q/K/V/decay/
erase/write paths after one synthetic opening step, and all 11 receiving-layer
auxiliary FutureSeed gates. No probe or formal metric is inferred from this
contract.

The exact-resume full-stack probe then passed at the same pushed source as
`p-gdn3-006-dual-state-probe-s3001-20260807T015559Z-605b9c2` with status0.
The parent checkpoint/hash and source contract were accepted at step3000; the
new migration set contained only the registered expert parameters, no
unexpected parameters, and the optimizer groups expanded explicitly. The
production batch32/accum4 BF16 fit completed without NaN, OOM, fallback, or
source drift. It used 13,965,248 parameters and peaked at 19,335.76 MiB
allocated / 20,284 MiB reserved, well below the 80 GiB device limit.

After one real optimizer step, the fixed probe's loop5 expert residual
relative RMS was `0.0070933`, between-board residual std `0.0002498`, terminal
state RMS/std `0.0429976/0.0015680`, residual readout weight RMS `0.0014993`,
address contrast `-0.0044468`, and expert/main output cosine `-0.0034930`.
These values establish finite nonzero output, state, board variation, address,
and expert interaction; the eight-board probe is not used as a science score.
Probe artifact SHA256 values are metrics
`db3d5ca8207b6732643236d0634e3802660f3d4ae2c5b7cf0fb5c168761e4a0e`,
checkpoint
`5809e12bb27793c519a23df564b8082743a7e818e7d151efeb5d15528d21d2c1`,
config `779e71f63bf67eaab3cecf441c452b3ae04f5254c64e9f2ed8c038159d11f5ea`,
run log `2bb063da5bda87d1816c028a2ef881939a95e23703b23394ce7fd14047692ca4`,
and source snapshot
`faae93cb81b8e82542c89f8aecf06d2b511e962035fbc89e89f5cefdc20a62e5`.

With every launch gate satisfied and GPU1 idle, the formal candidate started
immediately from the same clean detached source and exact parent as
`p-gdn3-006-dual-state-s3100-20260807T020415Z-605b9c2`. Its wrapper PID/PGID is
`22830`, Python child `22917`, launch log is
`/huyang2/double-loop/artifacts/launch/p-gdn3-006/formal-20260807T020415Z-605b9c2.log`,
and status path is the adjacent `.status` file. The live log confirms
`gdn2_state_expert=dual_state` and exact resume at step3000 on CUDA index0 with
the registered UUID. No concurrent GPU evaluation is authorized while this
trajectory is healthy.

The formal continuation then exited cleanly with status0. It produced the
step3100 checkpoint and complete mixed/official metrics without NaN, OOM,
fallback, source drift, data drift, or GPU drift. The final candidate train CE
was `0.855923`, versus `0.858617` for the frozen control.

### Formal activation

The expert is not dormant. At loop5, residual relative RMS is `0.010408`,
between-board residual std is `0.001286`, terminal-state RMS/std is
`1.271940/0.171060`, auxiliary FutureSeed incoming RMS is `0.457982`, and the
readout weight RMS is `0.006379`. Address contrast is `0.002369` and
expert/main output cosine is `-0.002639`. Together with the strict contract,
these values pass the registered activation gate for all 12 experts.

### Formal quality decision

Mixed exact across loops1-to5 is:

| Arm | L1 | L2 | L3 | L4 | L5 |
|---|---:|---:|---:|---:|---:|
| frozen control | 0.017578 | 0.023438 | 0.025391 | 0.025391 | 0.025391 |
| dual-state expert | 0.017578 | 0.023438 | 0.023438 | 0.025391 | 0.025391 |

Mixed loop5 blank accuracy regresses `0.545610->0.543408`. Official loop5
results are:

| Range | Control exact/blank | Candidate exact/blank | Blank delta |
|---|---:|---:|---:|
| 51-55 | 0.001953 / 0.573766 | 0.001953 / 0.570150 | -0.003616 |
| 56-60 | 0 / 0.503861 | 0 / 0.497718 | -0.006143 |
| 61-64 | 0 / 0.591923 | 0 / 0.589298 | -0.002625 |

The hard51-64 macro loop5 exact is unchanged at `0.000651`. Therefore the
primary exact delta is `0`, and the alternate mixed exact delta is also `0`.
Both registered quality routes fail.

The same 256 boards in each hard range confirm that this is not hidden by an
aggregate tie. Candidate loop3-to5 wrong-cell correction is
`0.042969/0.015625/0.089844` for 51-55/56-60/61-64, versus control
`-0.015625/0.066406/0.183594`. Only the easiest hard range gains a small late
correction; both harder ranges weaken. At loop5, candidate better/equal/worse
board counts are `90/50/116`, `91/54/111`, and `97/41/118`.

### Formal cost decision

The candidate has `13,965,248` parameters versus `11,485,760` for control.
Matched 100-step elapsed time rises `825.970->1706.212` seconds, a
`+106.57%` regression, and effective throughput falls
`15.497->7.502` boards/s. Peak allocation rises
`13,186.42->19,356.85 MiB` (`+46.79%`) and peak reservation rises
`14,288->20,888 MiB` (`+46.19%`). Memory remains just inside the registered
50% limit, but the independent warmed elapsed gate fails decisively.

### Closure and artifacts

P-GDN3-006 is discarded. A fully active second recurrent expert lowers CE but
does not close another board, slightly regresses every official hard blank
range, weakens late correction on 56-64, and more than doubles continuation
time. Do not rescue expert width/count, address form, residual scale, seed, LR,
loss, batch, main width/depth, or duration. The next mechanism must obtain
closed-loop state-dependent addressing or updating more directly, without
duplicating a full recurrent stack behind a slow zero-init readout.

The complete paired comparison and same-board loop1-to5 visualization are
archived at
`/huyang2/double-loop/runs/p-gdn3-006-comparison-20260807T025309Z-605b9c2`
and mirrored under
`research/reports/visualizations/gdn3-dual-state-residual-expert-20260807/`.
Artifact SHA256 values are candidate metrics
`6c704bc1c8b5d83ca5b08c3c2d69bcdd6b4e5f7b5ea8034cc2847325ad417496`,
checkpoint
`f200d4dbedcd8dee75b743b1f8b8d672b76798e9730ba4e58a432e20de1024ca`,
config `b23e0549aae1c193487020f75e8d876dd0a6c51c1f222ed733e4c8f008f66515`,
run log `b7100b22452a8a28f6a517f6a774dae93e4397d2f9d38d97b8549d1fedc27829`,
source snapshot
`faae93cb81b8e82542c89f8aecf06d2b511e962035fbc89e89f5cefdc20a62e5`,
comparison JSON
`0207bfbb159d9180ad495cb9ce3978d2f5de0950ddb5a788820f79d1870e2df8`,
comparison HTML
`b42115c979bbfb39e2aa01630b89b72761d678883cadd436b6939a3e4ec2cf0b`,
and manifest
`9d25b90b4868262eaac2a0a9df90b9a4c469ba356bcd94d2fa0455f8e59e2559`.
