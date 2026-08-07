# P-GDN3-013: Paired Address-State Bank

## 1. Metainfo

- Status: implementation pushed; strict CUDA contract R2 pending
- Date: 2026-08-07
- Planned branch: `codex/gdn3-paired-address-bank-20260807`
- Benchmark: official/full-diversity hard 9x9 Sudoku 51-64 blanks
- Compute: AIStation task-mode GPU1 only
- GPU UUID: `GPU-53e9f3b4-2966-65d3-6614-09c540921519`
- Seed: 52 only
- Parent: D256/L12/H8/K32/V32 position-QK GDN3 plus native terminal FutureSeed
- Parent checkpoint:
  `/huyang2/double-loop/models/gdn3-position-qk-d256l12-s12000-20260806T131527Z-9f2ee8d/checkpoints/train_state_step003000.pt`
- Parent checkpoint SHA256:
  `6339c3cb2b5fc5230a581d6633716483e35ff8e4522f06a9d7aaf26512f023da`
- Parent source SHA: `9f2ee8d1738032bc5f09b55db0b81d507780b376`
- Frozen control:
  `p-fs3-001-terminal-s3100-20260806T200859Z-3e167b6`

## 2. Evidence Boundary

P005 changes erase/write coherence, P007 conditions updates on inherited state,
P008 adds a post-scan transition, and P009/P010 test transition composition.
P011 tests token-dependent payload routing among the existing heads. P012 opens
a content-dependent signed transition spectrum. All activate or fail their
registered systems boundary without materially improving hard exact.

P006 is the closest capacity result: it adds an independent D128/H8/K16/V16
parallel expert. That expert has a separately projected content and address
stream, a smaller independent payload, and a zero-read residual path; it is not
an additional address-state slot aligned to each full-width main position-QK
head. It activated but regressed all official blank ranges and cost 106.57%
more elapsed time. It therefore rejects an unaligned auxiliary core, not an
aligned multi-slot memory bank inside the main recurrence.

The untested boundary is recurrent address capacity. Each main H8 head owns one
K32xV32 trajectory. The block output can mix heads after the scan, but the live
recurrence cannot store the same full payload under two independently learned
addresses within a head namespace. P013 tests that topology directly without
changing data, training, traversal, task logic or physical sequence length.

## 3. Mechanism

For every parent head, create one aligned companion head inside the same
pinned-official `chunk_gdn2` call. At migration:

- companion Q and K are exact copies of parent position-QK tensors;
- companion V, decay, erase and write are exact copies of the parent tensors;
- a parent H8 incoming state is duplicated into base and companion H8 banks;
- native FutureSeed transports the resulting H16 state, duplicating each
  existing H8 FutureSeed gate across its base/companion pair;
- only the base H8 output enters the unchanged parent output path.

The companion addresses can differentiate through two zero-initialized,
head-shared, bias-free `V32 -> K32` projections per layer:

`q_comp = q_base + W_q v`

`k_comp = k_base + W_k v`

One zero-initialized scalar read gate per parent head combines the official
outputs before the unchanged output norm and projection:

`o = o_base + tanh(a) * o_comp`

At `W_q=W_k=a=0`, duplicated states and inputs make the two banks identical,
while `tanh(a)=0` leaves full model output exactly on the parent base path. The
read gates receive first-stage gradients; after they open, both address
projections must receive finite nonzero gradients and produce a differentiated
companion state.

The candidate adds exactly
`12 * (32*32 + 32*32 + 8) = 24,672` parameters. Persistent recurrent state
increases from `12*8*32*32 = 98,304` to `12*16*32*32 = 196,608` values per
board, an exact delta of 98,304. It adds no token, scan, second core, reverse
path, solver, selector, task rule or fallback. It still uses one official GDN2
call per layer and the original 81-token traversal.

## 4. Falsifiable Prediction

If single-trajectory address capacity is the missing bottleneck, the read
gates should open, companion Q/K should differentiate across boards, tokens and
heads, and base/companion terminal states should diverge while remaining
bounded. This should convert into stronger late-loop correction and at least
`+0.02` hard51-64 macro loop5 exact, rather than only moving blank accuracy.

If all 12 banks activate and diversify but exact remains flat, an aligned
second address slot is not sufficient at this checkpoint. If only read gates
activate while address projections remain dead, the optimization route is not
viable under exact migration. Either outcome closes the entire paired-bank
mechanism; do not tune bank count, head count, projection source/rank, gate
mapping/scale/init, state duplication, precision, seed, LR, loss, batch,
width/depth or duration.

## 5. Migration And CUDA Contract

Before any continuation, exact pushed source in a clean detached worktree must
pass all of:

1. CUDA index0 and UUID exactly match registered GPU1, with one compute
   application and no GPU2 visibility;
2. pinned FLA source SHA is
   `9c8e42e762fce087c27b673af4922795d9edb85e`;
3. all 12 layers are official `GatedDeltaNet2`, expose exactly one
   `ChunkGDN2FunctionBackward` each, and have no fallback;
4. exact parameter delta is 24,672, state-value delta is 98,304, and token,
   scan and core-count deltas are zero;
5. zero-initialized candidate gives bit-exact full output and bit-exact base
   terminal states versus the parent, while each companion terminal state is
   bit-exact to its paired base state;
6. the same identities hold for synthetic finite nonzero parent incoming
   states after exact base/companion duplication;
7. all 12 H8 read-gate tensors receive finite nonzero first-stage gradients;
8. after a synthetic read-gate opening, all 12 Q and K projections receive
   finite nonzero gradients, alter companion address/state/output, and leave
   base outputs/states unchanged;
9. paired-head permutation commutes with the shared address projections and
   duplicated FutureSeed gates to max error `<=3e-6`;
10. state tensors remain finite and have the exact registered H16xK32xV32
    geometry at every layer;
11. exact-resume model/optimizer/RNG/data-order migration completes from the
    registered parent and writes complete step3001 checkpoint and metrics;
12. no NaN, OOM, source/data drift, CPU model path, concurrent GPU process,
    task-specific dependency or silent fallback occurs.

Changing H8 to H16 may alter kernel launch numerics even when heads are
mathematically independent. Bit-exact base parity is therefore binding. A
tolerance relaxation, head-order rewrite or precision workaround is a contract
failure, not an implementation rescue.

Strict contract R1 on pushed SHA
`103da5c71cd19938fb6bdba12562e554601f1bea` exited naturally after the
identity and migration checks because its newly added state-cost assertion
compared a two-board batch total (`196,608`) with the registered per-board
delta (`98,304`). This is a checker accounting error, not a model, data,
identity or mechanism failure. The R1 log and non-science abort remain archived;
R2 changes only the assertion to count actual state values per board. Mechanism,
prediction and every registered gate remain unchanged.

## 6. Step3001 Production Probe

The exact one-step probe is migration and production-fit evidence only. It must
show all 12 paired banks enabled; finite nonzero read-gate absolute value;
finite nonzero companion Q/K residual relative RMS with board/token/head
variation; finite nonzero paired-state residual; finite paired address
contrast; complete checkpoint/config/source hashes; exactly one official
backward path per layer; and terminal RMS no more than `4x` the matching parent
readout. Failure closes P013 before a formal continuation.

## 7. Science, Stability, And Cost Gates

At step3100, activation and stability require all of:

- exactly 12 paired banks active;
- mean absolute read gate, companion Q residual relative RMS, companion K
  residual relative RMS and paired-state residual relative RMS each finite and
  `>=1e-4`;
- finite nonzero board, token and head variation in the differentiated address
  path;
- finite, noncollapsed base and companion address contrast;
- finite terminal-state RMS at every loop and no more than `4x` the frozen
  control at the matching loop;
- no fallback.

Quality passes by exactly one route:

1. hard51-64 macro loop5 exact improves by at least `+0.02`, with every hard
   range losing no more than `0.01` blank accuracy; or
2. mixed loop5 exact improves by at least `+0.03`, official61-64 does not
   regress, and same-board loop3-to5 wrong-cell correction is stronger in all
   three official ranges.

The registered state doubles and the official recurrence processes twice as
many heads, while token and call counts stay fixed. Independently warmed
elapsed overhead is capped below `100%` and peak allocated-memory overhead
below `70%` versus the frozen control. These ceilings cannot be relaxed after
measurement. OOM, unstable timing or either cost miss kills the candidate.

## 8. Required Readout

Report and archive mixed and official51-55/56-60/61-64 loop1-5 exact/blank/
wrong cells; train CE and same-board correction; read-gate, companion Q/K
residual, paired-state residual, address contrast and state geometry;
independently warmed throughput, allocation/reservation and timing; exact
config/source/parent/checkpoint/metrics/log hashes; and same-board loop1-5
visualization.

## 9. Decision

Implementation and preregistration are pushed. Strict CUDA contract R2 and the
exact step3001 probe remain pending. No formal GPU continuation is authorized
from an unpushed SHA or after a substantive contract miss.
