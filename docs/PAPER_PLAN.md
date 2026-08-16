# FutureSeed + Loop Paper Plan

## 2026-08-16 Owner-Local Projection Removes The Useful Global Integrator

P-GDN3-068 keeps P059's second-order Momentum, residual, owner key, state and
native `[S,M]` FutureSeed, but commits only the exact current-key projection of
Momentum. Its strict fused-kernel contract and all cost gates pass. Quality
collapses from `.94425` to `.01250` balanced accuracy and total errors rise
`223->3950`. The apparent adjacent-swap reduction `151->77` is caused by
`3,591` previously correct queries becoming unrelated wrong values.

This falsifies the idea that Momentum is merely a bag of separable owner edits.
Its dense global commit is the useful integrator. The remaining ownership tail
must be handled by an additive, scalable state organization that preserves the
parent path, not by projecting or erasing most of the velocity at every token.

## 2026-08-16 Independent Erase Destroys Learned Ownership

P-GDN3-067 tests a structured erase-then-delta transition using two ordered
pinned-official product microsteps. The first uses an independently learned
address and an exactly zero payload; the second retains the complete standard
same-key delta correction. Its strict contract and endpoint diagnostics prove
that both addresses, gates, states and native FutureSeed are active, stable and
equivariant.

The mechanism still collapses directional retrieval. P059 versus P067
balanced/future/past/joint is `.94425/.95150/.93700/.82400` versus
`.16850/.17600/.16100/.00100`; errors rise `223->3326`. The lower conditional
swap share is broad failure, not owner repair. This result sharpens the owner
constraint: retaining the standard write is insufficient if a preceding
independent address deletes useful state. The next paper candidate should keep
P059's successful second-order carrier and change how its velocity is committed
within the current owner subspace, without another learned deletion address.

## 2026-08-16 Global Least Squares Loses Directional Ownership

P-GDN3-066 replaces the recurrent transition with the exact pinned-official
Mesa operator and transports its complete `[Hkk,Hkv]` sufficient statistics
through native FutureSeed. Its strict contract is positive: both official
backward paths, all recurrence and seed gradients, incoming-state dependence,
head equivariance and exact-reference parity pass. It is also faster and
lighter than the P059 Momentum DeltaNet baseline.

Quality nevertheless falls to chance. Balanced/future/past/joint accuracy is
`.00975/.00650/.01300/0` versus `.94425/.95150/.93700/.82400`, with
`3,961` errors. Hkk is symmetric, PSD and solved accurately, but its effective
rank is only `1.079/1.169` out of 32. The paper should distinguish numerical
stability from useful memory geometry: a stable global normal equation can
compress the sequence into an almost one-dimensional statistic and erase the
local owner coordinates needed for directional retrieval. Native FutureSeed
cannot recover semantics absent from its producer state. Close Mesa and target
a live local edit that preserves ownership.

## 2026-08-16 Closed-Loop Ownership Requires Operator-Level Semantic Parity

P-GDN3-065 tests a complete Comba residual recurrence rather than another
wrapper: predict from the current state with an owner-collinear key, then write
only the unresolved value. This is the right mechanistic target after P064
showed that 16x sparse capacity cannot replace coherent future ownership.

The exact R5 source resolves all import and mixed-precision compile boundaries
and executes CUDA forward/backward, but the production chunk differs from its
fused-recurrent reference by `.239417` output RMS and `.258119` terminal-state
RMS. Both exceed the registered `.05` contract limit. Therefore no L1024
quality training is launched and no Comba, GDN3, or FS2 improvement is
claimed. The paper should treat chunk/recurrent agreement as architectural
evidence, not merely a systems detail. P059 Momentum DeltaNet with native
`[S,M]` FutureSeed remains the strongest positive result; a future ownership
transition needs clean-room semantics and parity from the beginning.

## 2026-08-16 Sparse Capacity Does Not Replace Coherent Future Ownership

P-GDN3-064 replaces the primary recurrence with the pinned official Sparse
Delta Memory and transports the complete 1,024-slot terminal bank through
native FutureSeed. It provides 131,072 recurrent values per layer, 16x P059,
and activates 66/214 slots in the two layers. The strict contract, sparse
backward paths, FutureSeed gradient and all cost gates pass.

Quality collapses despite that capacity. Balanced/future/past/joint accuracy is
`.26825/.02850/.50800/0` versus P059's
`.94425/.95150/.93700/.82400`; errors rise `223->2927` and wrong-key swaps
`151->1126`. The causal past direction partially learns, but the next layer
cannot use the terminal slot bank to resolve future writes. The paper should
separate memory quantity from state semantics: a compact derivative state can
be a better FutureSeed carrier than a much larger sparse snapshot when its
ownership coordinates are native to the receiving transition. Close sparse
slot/read/write/head rescue and test complete coherent recurrences instead.

## 2026-08-16 Momentum And State Are Not A Rotatable FutureSeed Basis

P-FS2-014 follows the clean P-DIAG-MOMFS observation that transported Momentum
alone retains almost all of P059's FutureSeed quality. It leaves the successful
second-order recurrence, owner key, stacked state, normalization and native
gate exact, then adds only one zero-initialized orthogonal S/M rotation angle
per receiving head. The strict contract proves exact parent behavior, complete
gradients and energy preservation with only four new parameters.

The endpoint rejects the tempting interpretation that state and Momentum form
a redundant two-dimensional coordinate plane. All angles activate and remain
stable, but balanced accuracy falls `.94425->.25950`, errors rise `223->2962`
and wrong-key swaps rise `151->1081`. The learned residual has substantial
head variation but effectively no board variation (`2.79e-9`), so it is a
global channel remapping rather than owner-specific evidence. The paper should
state the stronger semantic result: Momentum is the useful cross-layer future
signal, but it is useful *as a derivative*. Rotating it into the base state
destroys the live transition's learned meaning. Future FS2 mechanisms must
preserve component identity and improve credit or ownership transport without
treating `[S,M]` as exchangeable capacity.

## 2026-08-16 Independent Prediction Keys Break Momentum Ownership

P-GDN3-063 tests the most direct residual-tail interpretation of P059. The
native external Momentum recurrence already accepts a prediction key distinct
from the correction owner key. The candidate adds one independent prediction
projection and ShortConv per layer, initializes both exactly from K, and leaves
the successful owner/write K, recurrent `[S,M]`, scan and FutureSeed unchanged.
The A800 contract proves exact parent behavior and complete gradient paths.

The endpoint rejects unconstrained address decoupling. P/K cosine falls to
`.152/.143`, layer-0 state and momentum overflow, and balanced/joint accuracy
collapses from `.94425/.824` to `.009/0`. Wrong-key swaps appear to improve
`151->131`, but 3,744 native-correct queries break and only three swaps repair.
This is a useful negative boundary for the paper: the prediction/commit tie is
part of the stable owner geometry. The remaining P059 tail cannot be solved by
adding a free second key after initialization. Future claims should preserve
the learned owner coordinate and test bounded state organization or genuinely
new cross-layer second-order evidence.

## 2026-08-16 Local Reversibility Does Not Guarantee Trainable Long Memory

P-GDN3-062 evaluates the Momentum residual at a velocity lookahead and uses a
clean-room reversible Triton backward. The short contract is deceptively
strong: output/state parity is about `1e-7`, all synthetic gradients agree
within `1.5e-6`, and the L1024 forward has finite nontrivial lookahead and
residual changes. At L1024, however, nearly all full-model recurrent gradients
become NaN. Only final-layer V and the FutureSeed gate remain finite.

The paper should separate algebraic invertibility from stable credit transport.
The inverse at each token is locally well-defined, but repeatedly dividing by
decay factors reconstructs an exponentially ill-conditioned reverse history.
This is a useful negative systems result: strict production-length gradient
contracts are necessary for new linear-recurrent transitions, and a short
reference parity test is insufficient. P062 receives no quality claim and no
Sudoku transfer.

## 2026-08-16 Momentum Is The Effective Cross-Layer FutureSeed Component

P-DIAG-MOMFS-001 masks only the transported component of the exact trained
P059 checkpoint. Momentum-only replay remains at `.94375` balanced versus
`.94425` for native `[S,M]`, adding only two errors and two swaps. State-only
replay falls to `.48450` balanced because future accuracy collapses to `.031`,
while past accuracy remains `.938`. The direction asymmetry is mechanistically
clean: local causal recurrence already answers past queries, whereas future
queries depend on the cross-layer momentum carrier.

The paper should therefore distinguish state capacity from useful future
evidence. P059's benefit is not simply carrying twice as many numbers; the
second-order component encodes the transferable update trajectory. Its final
151 swaps remain adjacent-owner errors even when the base matrix is removed,
so the next GDN claim concerns coherent intra-layer owner geometry rather than
another FutureSeed component router.

## 2026-08-16 Second-Order State Nearly Closes Long-Context Binding

P-GDN3-059 replaces the primary first-order GDN2 transition with a complete
Momentum Delta recurrence trained from scratch, then lets native FutureSeed
carry both matrix state and momentum state across layers. It is not a
zero-init side path or short continuation. The exact external chunk kernel,
its fused recurrent reference, the pinned host FLA compatibility boundary and
all gradients pass a strict A800 contract.

The endpoint is qualitatively different from the failed wrappers. Balanced
accuracy rises `.494->.94425`, future/past accuracy reaches `.9515/.9370`,
joint exact rises `.041->.824`, and errors fall `2024->223`. Wrong-key swaps
fall `1546->151`; 1,474 baseline swaps become correct. The model also crosses
the retrieval transition two epochs earlier. This supports the paper's main
mechanistic claim: changing the live recurrent dynamics can preserve binding,
where adding more reads, side memories, local event encoders or post-hoc key
geometry generally reorganizes or destroys the native solution.

The result remains a strict gate miss. Wrong-key swaps comprise `.67713` of
the small residual error set, only `.08670` below the control share rather
than the preregistered `.10`. Therefore P059 gets no Sudoku transfer or nearby
hyperparameter rescue. The paper should present it as the strongest positive
architecture evidence and an honest residual boundary: second-order dynamics
solve most retrieval, but explicit scalable owner separation is still needed
for the final tail.

## 2026-08-16 Same-Weight Edge-Off Attributes Cycle Failure To Co-Adaptation

P-DIAG-CYCLE-001 removes the only eight learned cycle gates from the exact
trained P058 checkpoint while preserving every other tensor and the frozen
L1024 cases. This turns the reverse-state correction off at inference without
retraining. Balanced accuracy changes only `.07425->.07575`, errors
`3703->3697`, wrong-key swaps `380->388`, and joint exact remains zero. The
result is far inside the preregistered training-co-adaptation region.

The paper can therefore make a stronger distinction than "the cycle reread
was harmful." Zero initialization gave exact parent behavior before training,
but the opened side path subsequently reorganized the native GDN2 retrieval
map around itself. Removing the side path after training cannot recover that
map. Future architecture claims must protect native-correct retrieval during
learning or train a complete ownership-aware recurrence from scratch; optional
post-hoc validators and zero-gated wrappers are not optimization isolation.

## 2026-08-16 Cycle Consistency Diagnoses Ownership But Cannot Repair It

P-GDN3-058 keeps native GDN2 storage exact and adds a reverse official GDN2
relation from values back to normalized keys. A native candidate value is
checked by reconstructing its owner, and a zero-init per-head gate uses the
query-owner mismatch for one reread of the unchanged main trajectory. Both
states receive native FutureSeed. The strict R5 contract proves exact parent
output/state/gradient identity, causal official scans and complete reverse
state dependency.

The endpoint separates diagnosis from repair. Wrong-key valid-value swaps fall
from `1546` to `380`, so the reverse relation recognizes many owner mismatches.
Balanced accuracy nevertheless collapses `.494->.07425`, joint exact falls
`.041->0`, and errors rise `2024->3703`; 1,846 correct answers are destroyed.
Only seven/eight gates cross the activation floor and cycle evidence has weak
board specificity in layer 0. Thus a post-retrieval consistency certificate
can say that an answer belongs to the wrong key without supplying a safe path
to recover the right value. The paper should use this to close read-time
validators and motivate ownership-preserving live state formation or genuinely
receiver-native FutureSeed evidence.

## 2026-08-16 Local Raven Events Do Not Establish Global Ownership

P-GDN3-057 is the direct Raven/GDN hybrid test: a causal multiplicative
current/previous token-shift feature is formed before each unchanged native
GDN2 scan, while native FutureSeed carries the resulting terminal state across
layers. The strict contract proves exact zero-gate parent behavior, official
FLA/Triton execution, complete gradients and causal one-token dependence.

The endpoint rejects local event formation as the missing binding primitive.
Balanced accuracy collapses from `.494` to `.02475`, joint exact from `.041`
to zero, and errors rise `2024->3901`. Wrong-key swaps fall `1546->153`, but
1,848 native-correct queries become unrelated wrong values. The event is
bounded and all eight gates activate, yet board variation is effectively zero:
the adapter learns a shared phase change rather than owner-specific identity.
Together with post-GDN local attention and additive eligibility memory, this
closes local wrappers around the native transition. The remaining claim is
narrower: ownership must be preserved within the live memory edit without
breaking the co-adapted native Q/K/V/erase/write/read geometry.

## 2026-08-16 Eligibility Side Memory Removes Answers, Not Interference

P-GDN3-056 preserves native GDN2 and FutureSeed and adds one shared,
causal-eligibility companion state. The strict contract proves exact parent
behavior at zero gate, official FLA execution, causal history use and complete
gradient paths. At the endpoint all eight reads activate and wrong-key swaps
fall from `1546` to `639`.

That conditional improvement is deceptive. Balanced accuracy collapses from
`.494` to `.14825`, joint exact from `.041` to zero, and total errors rise
`2024->3407`. Paired transitions contain 1,680 correct-to-wrong changes versus
only 297 wrong-to-correct changes. The second-layer companion read grows to
`2.661x` native output RMS, so the side plane replaces the jointly learned
retrieval map instead of supplying a missing owner bit. Together with sparse
surprise replay, canonical states and protected FutureSeed planes, this closes
additive ownership memories. The next claim must concern joint event formation
before the native live transition, not another post-hoc memory complement.

## 2026-08-15 Block-Local Whitening Does Not Preserve Address Closure

P-GDN3-054 is the narrow scalable test of the decoupled-key intuition after
the exact dual-address oracle and independent-bank variants failed. It keeps a
single native ownership coordinate, maintains eight causal 4x4 inverse-
information blocks in Triton, and projects each corrected write back onto the
native erase-response constraint before one official DPLR scan. The strict
contract proves the statistic, state geometry, gradients and kernel path; this
is a real live-transition intervention rather than a dead wrapper.

The endpoint still collapses balanced accuracy from `.494` to `.018` and
joint exact from `.041` to zero. Errors rise `2024->3928`. Wrong-key swaps fall
`1546->190`, but 1,849 formerly correct queries become another wrong value and
only 24 swaps repair. Thus even an erase-response-preserving write transform
breaks the learned query/erase/write/read coordinate system. The block state is
stable and nondiagonal, but actual address motion remains only about `.008`
relative RMS; warmed-step cost is `2.007x`. Together with direct, anchored,
biorthogonal and redundant decoupling, this closes post-hoc key geometry as the
paper's explanation. Pair identity must be learned as part of state formation,
not imposed on a mature key basis.

## 2026-08-15 Cross-Loop Secants Do Not Restore Lost Ownership

P-FS2-013 tests the narrow convergence hypothesis on the fixed hard-Sudoku
parent. It preserves native terminal FutureSeed and adds only 88 zero-init
edge/head coefficients that extrapolate each producer terminal along its
bounded change from the preceding reasoning pass. The strict contract proves
parent identity, official FLA execution, complete gradient coverage and stable
geometry; the production path is therefore real rather than a dead adapter.

The endpoint closes this explanation. Hard51-64 macro and mixed loop5 exact are
unchanged at `.000651` and `.025391`; 61-64 blank falls by `.004182`, its
same-board loop3-to-loop5 correction weakens, and elapsed cost rises `18.36%`.
The learned residual itself shrinks from `.002687` of state RMS at loop2 to
`.00004161` at loop5. FutureSeed's remaining failure is not simply insufficient
momentum along its current trajectory. Once the recurrent state has merged
neighboring ownership, a secant can only extrapolate that lossy representation.
Future work must preserve new ownership evidence or change the live memory
transition, not add another interpolation/extrapolation wrapper.

## 2026-08-15 Correct Ownership Evidence Exists Across Heads But Is Not Selectable

P-DIAG-OWN-003 freezes the reproducible P-REPRO-001 endpoint and masks each
GDN2 head only at the official output projection. A label oracle over the four
final-layer only-head variants repairs `516/1546` wrong-key swaps and raises
balanced accuracy from `.4945` to `.64825`; an oracle over all variants reaches
`.8500`. Thus some alternative heads retain information that could correct a
substantial minority of bindings.

That observation does not justify a router. The preregistered answer-free
max-margin rule lowers swaps to `1300` but lowers balanced accuracy to
`.42325`, and `441/1546` swaps preserve the same wrong owner under every
final-head isolation. The model exposes no reliable confidence signal for
choosing the oracle head. This narrows the architectural claim: ownership must
be made intrinsic to the live within-head memory transition, not recovered by
post-hoc voting, confidence selection or another readout path. The diagnostic
also records a small fresh-process BF16/Triton boundary (`3990/4000`
predictions); all causal comparisons use one same-process baseline.

## 2026-08-15 Canonical Companion Addresses Do Not Establish Ownership

P-GDN3-049 isolates the hypothesis that P047 failed only because its semantic
certificate reused native addresses. It keeps native GDN2/FutureSeed intact and
adds a smaller H4/K16/V32 official companion state that stores complete native
V under one shared learned canonical address projection. The strict contract
proves exact parent behavior at zero gate, official-kernel provenance, full
rank, gradients, equivariance and active state in both layers.

The endpoint rejects the hypothesis. Balanced accuracy falls
`.03700->.03075`, future/past are `.04000/.03400->.03950/.02200`, total errors
rise `3852->3877`, and wrong-key fraction changes only
`.058930->.056229`. All activation and cost gates pass. Thus another dense
address domain is not equivalent to an ownership-preserving memory: values are
still superposed and can remain detached from the queried key. Together P047
and P049 say that the model can learn the candidate value set and can learn a
second address state, yet neither operation establishes pair identity. Future
work should change the organization of committed edits in live memory rather
than add semantic payload, dense address banks or coordinate wrappers.

## 2026-08-15 Reciprocal Payload Codes Suppress Errors By Losing Retrieval

P-GDN3-048 is the narrow ownership-in-payload test motivated by P047's nearly
pure wrong-key residual. Native `k` writes a bounded diagonal V code, native
`q` applies its reciprocal after the unchanged one-scan official GDN2, and only
eight scalar strengths are learned. The strict contract proves exact parent
identity, official-kernel provenance, finite gradients, bounded FP32 factors
and reciprocal/head-equivariant behavior.

The endpoint rejects the code. Balanced accuracy falls `.17475->.07500`,
future/past fall `.16100/.18850->.08600/.06400`, and errors rise `3301->3700`.
Wrong-key swaps fall `770->337`, but paired predictions show 627 correct
queries destroyed for only 228 wrong queries repaired. Thus the conditional
swap improvement comes from losing recognizable values, not binding them more
accurately. Production BF16 reciprocity also misses its fixed tolerance and
peak allocation reaches `1.2593x` despite only eight parameters. The paper
should report this with direct/anchored decoupled-key and Q/K gauge failures:
algebraically coherent address wrappers are not enough to create a learnable
ownership code. Do not transfer or tune this family on Sudoku.

## 2026-08-15 Semantic Certificates Recover Values, Not Ownership

P-GDN3-047 keeps native GDN2/FutureSeed as the predictor and adds a second
pinned-official state that writes a position-free token-identity payload under
the exact same native address and gates. Only eight scalar read gates are new.
The strict contract proves parent identity, four official backward paths,
nonzero gradients, exact token invariance and active certificate state in both
layers.

This produces the clearest binding decomposition so far. Balanced accuracy
rises `.17475->.48300`, future/past rise `.16100/.18850->.48650/.47950`, joint
exact reaches `.041`, and errors fall `3301->2068`. Yet wrong-key valid-value
swaps rise `770->1973`, accounting for `95.41%` of remaining errors. The model
has learned which values exist but not which key owns each value. The paper
should use this as direct evidence that extra semantic payload and lower CE are
not binding closure. Future GDN3 work must preserve ownership in the recurrent
state/update itself; appending another certificate, readout, cache or value
bank is now closed under the tested regime.

## 2026-08-15 Dynamic Address Frames Are Stable But Not Learnable

P-GDN3-044 tests the live-state operation left open by static metric and gauge
experiments. A token-dependent diagonal frame multiplies coherent native Q/K,
its exact log difference enters row decay so the existing K32xV32 state moves
between frames inside one official scan, and terminal rows return to a
canonical basis before native FutureSeed. The strict contract proves exact
identity, two official backward paths, zero state/scan expansion and exact
telescoping. At endpoint factors remain bounded (`.7101..1.4027`), condition is
below `1.965`, and canonical state is finite and variable.

The quality result is negative. Balanced accuracy falls `.36625->.0765`, both
directions fall below `.083`, joint exact goes to zero, and errors rise
`2535->3694`. The lower conditional wrong-key fraction accompanies broad
retrieval failure. Peak allocation and independently warmed-step ratios also
miss at `1.1462/1.4064x`. Together with P031/P036/P042/P043, this closes split
keys, forced Q/K similarity, static dual gauges and dynamic state frames. The
paper should state that stable coordinate transport is not sufficient for
binding; future work should target receiver credit or a different committed
memory edit, not another Q/K coordinate wrapper.

## 2026-08-15 Query Feedback Can Break Coherent Delta Learning

P-GDN3-041 keeps native normalized ownership/write/read key `k`, every K-wise
erase and V-wise write gate, the K32xV32 state, native FutureSeed and one
official scan. It changes only the live content estimate from `k` to
`k+lambda*q`, with an exactly zero-initialized bounded query term. The strict
contract and endpoint show that this is a real, stable intervention: all eight
feedback heads activate, lambda RMS is about `.32`, mixed alignment stays
`.692..1.312`, and sampled transition spectral norm remains below `1.057`.

That intervention destroys learning. Balanced accuracy falls
`.36625->.0145`, future/past fall `.3515/.3810->.0175/.0115`, joint exact falls
`.002->0`, and errors rise `2535->3942`. The apparent swap-fraction reduction
is again caused by broad retrieval collapse. Costs all pass, so this is a
quality boundary rather than an implementation or efficiency failure. The
paper should distinguish using the query to read a completed state from using
it inside the erase prediction: the latter disrupts the coherent native
learning dynamics even while ownership remains anchored to `k`.

## 2026-08-15 Exact Product Features Do Not Rescue Linear Binding

P-GDN3-040 preserves the complete learned native K32 address channel and adds
an exact uncompressed K8xK8 product complement in the same K96xV32 recurrent
state. It therefore isolates the strongest remaining interpretation of P024:
perhaps product binding failed only because the analytic feature map replaced,
rather than complemented, the native linear map. The strict contract proves
zero new parameters, one official scan/layer, exact native sub-block identity,
active and variable product addresses/state/read, and native whole-state
FutureSeed.

The result closes that interpretation. Balanced accuracy collapses
`.17475->.01225`, future/past collapse `.1610/.1885->.0155/.0090`, and errors
rise `3301->3951`. Product/native state RMS remains `.2700/.2652`, but its read
RMS is only `.1728/.2473`; the analytic block is active yet fails to acquire a
usable retrieval geometry. Swap fraction `.233263->.031891` is again a broad
retrieval-collapse artifact. Cost is within the preregistered 3x-state budget,
so the paper should report a quality boundary: fixed product features do not
solve wrong-key binding even as a direct-sum complement. Future work must keep
learnable native addressing and target credit or state organization without a
fixed analytic basis.

## 2026-08-14 Recurrent Address Context Can Destroy Native Binding

P-GDN3-039 keeps native GDN2 as the sole data plane and uses an independent
official Raven only to construct a recurrent residual for the shared Q/K
input. This is a clean organic Raven/GDN hybrid: V, gates, committed edit,
main K32xV32 state and native FutureSeed stay unchanged. The strict contract
and endpoint show that the hybrid is fully active, noncollapsed and affordable.
Raven slot entropy is `.766/.726`, address-residual relative RMS is
`1.738/2.991`, and Q/K projections change by roughly `1.9-6.3x`; elapsed and
warmed-step ratios are only `1.179x/1.361x`.

Quality decisively rejects the composition. Balanced accuracy falls
`.36625->.06275`, future/past fall `.3515/.3810->.0625/.0630`, joint exact
falls `.002->0`, and errors rise `2535->3749`. Wrong-key swap fraction falls
`.62091->.09149`, but only because the candidate loses general retrieval. The
paper may use P039 to distinguish recurrent address-context activation from
binding success. Together with P017/P019 it closes fixed-row Raven allocation,
Raven-to-V write control, and Raven-to-Q/K address composition under their
registered settings. Do not tune Raven slots, width, top-k, adapter or
cross-layer transport; a next successor must organize live state itself.

## 2026-08-14 Metric Pullback Improves Readability, Not Binding

P-FS2-009 keeps independently learned per-layer Log-SPD metrics and applies
the analytic receiver-read pullback `C_r^-1 C_p S` only to the native
FutureSeed edge. It is zero-parameter, bit-exact at identity, bounded and
cheap. Balanced accuracy improves `.1385->.17825` and total errors fall
`3446->3287`, so private-metric mismatch is a causal but small component of
cross-layer state readability.

The decisive binding result is negative. Joint exact reaches only `.002`, and
wrong-key valid-value swaps increase `602->748` (`.17470->.22756` among
errors). A static metric-derived coordinate map cannot reconcile tokenwise
normalized, layer-specific projections well enough to preserve bindings. The
paper may report the low-cost partial gain, but must not claim address closure
or transfer it to Sudoku. Close metric direction/scale variants and require
the next mechanism to alter collision-resistant live state organization.

## 2026-08-14 Receiver-Address Diagnostic Boundary

The frozen surprise-replay checkpoint retains `2044/2047` wrong-key valid-value
errors, but receiver-native key reprojection improves top-1 by only `0.00725`
overall and is negative on errors. Surprise-selected writes survive less than a
matched recency set, and exact survival has error AUROC `0.52717`. Key Gram
collapse is real (receiver effective-rank fraction `0.0423`, anisotropy `30.39`),
but it is not causally localized by these measurements. This closes sparse
cache, admission, receiver reprojection, and metric-plus-cache claims; future
architecture work must alter the coherent live recurrent transition and prove
quality from scratch on the directional L1024 binding regime.

## 2026-08-13 Write Reconstruction Is Not Query Binding

P-FS2-008 replaces P-FS2-007's sparse K16 replay with an all-token,
surprise-weighted receiver-native ridge residual while preserving the native
FutureSeed base state and both official recurrent scans. The solve is
numerically successful: it uses effectively `973.88/1024` tokens, reaches a
write-fit MSE ratio of `0.001132`, stays bounded, and has zero new parameters,
state or scans.

The actual retrieval result rejects the mechanism. Candidate balanced
accuracy is `0.2245`, below historical native FutureSeed `0.7475`. On the same
trained weights, the residual changes pooled query CE only
`2.617864->2.587832` (`0.988528x`) and increases wrong-key swaps from `906` to
`934`. Warmed training steps cost `2.1845x` the control. The paper may claim
that committed-edit surprise and receiver-native evidence are measurable, but
must not equate excellent ridge write reconstruction with improved FutureSeed
binding. The next target is the live query/address organization, selected only
after a zero-parameter key-geometry and overwrite diagnostic.

## 2026-08-13 Contractivity Alone Does Not Preserve Retrieval

P-GDN3-030 is a direct negative test of numerical stability. Its separate
erase/write DPLR uses one pinned-official scan and factors every live
transition as `sqrt(D)(I-beta rr^T)sqrt(D)`. Both learned directions activate,
the largest sampled spectral norm stays below one, terminal states remain
bounded, and every cost gate passes. Yet L1024 balanced accuracy falls from
the current native-FutureSeed reference's `0.30625` to `0.0205`, with errors
rising `2,775->3,918`.

The conditional wrong-key swap fraction falls to `0.041858`, lower than P028
or P029, while absolute retrieval is worst. The paper should state that
contractive geometry is useful protection against state explosion but is not
sufficient for learnable address/value binding. Combined with P-FS2-007, the
next thesis is sharper: surprise identifies useful evidence, but the receiver
must form it in a binding-preserving native basis rather than through sparse
sequential replay or a replacement recurrence that destroys the parent map.

## 2026-08-13 Sequential Delta Products Do Not Preserve Retrieval

P-GDN3-029 tests a dedicated pinned-official `GatedDeltaProduct(n=2)` rather
than another wrapper around GDN2. Both learned transform branches diverge,
their gradients and state dependence are active, and recurrent state remains
bounded. Yet directional L1024 balanced accuracy falls from the current
native-GDN2 FutureSeed reference's `0.30625` to `0.1370`, with errors rising
`2,775->3,452` and joint exact staying zero.

Wrong-key swap fraction still falls `0.458018->0.169177`. Together with P028,
this shows that adding sequential or analytically separated address updates
can change error composition without preserving a learnable value/read map.
The paper should treat swap reduction only as a mechanistic diagnostic and
motivate a stable state topology that jointly organizes erase and write, not
claim more within-token transformations as GDN3 progress.

## 2026-08-13 Atomic Binding Is Not Sufficient

P-GDN3-028 gives the cleanest separation yet between address-confusion rate
and end-to-end retrieval. Its atomic shared-payload paired-address update cuts
wrong-key valid-value swaps among errors from `0.458018` to `0.074980`, but
balanced accuracy collapses from `0.30625` to `0.04975` and total errors rise
from `2,775` to `3,801`. The model makes fewer wrong-key substitutions because
it has largely stopped learning a useful value/read mapping, not because it
closes bindings.

This negative result rules out a tempting claim that pairwise address
orthogonality alone solves the L1024 regime. It supports a sharper thesis:
GDN3 needs a scalable memory organization that jointly preserves linear
readout learnability and limits overwrite interference. Report the endpoint
as a causal boundary together with P020's address-metric gain and P-FS2-007's
surprise-admission gain; do not transfer P028 to Sudoku or tune the paired
address construction.

## 2026-08-13 Receiver-Native Surprise Replay Boundary

P-FS2-007 is the strongest new FutureSeed mechanism signal in the current
directional MQAR runtime, but it fails the preregistered endpoint. With both
main GDN2 scans unchanged and zero new parameters or persistent state, a K16
receiver-native replay chosen by exact committed-edit surprise raises balanced
accuracy from the contemporaneous baseline `0.30625` to `0.48825`; matched
recency replay collapses to `0.01525`. Surprise reaches joint exact `0.044`,
future/past accuracy `0.4880/0.4885`, and all activation/integrity checks pass.

The scientifically decisive result is the error transition: `99.853%` of the
remaining surprise-replay errors are correct values bound to the wrong key,
versus `45.802%` in the contemporaneous baseline and `80.693%` historically.
Thus exact surprise is a useful admission signal for preserving value evidence,
but sparse receiver-native replay does not resolve address binding or meet the
absolute `0.85` balanced and `0.60` joint gates. Its independently warmed step
also costs `1.438x` baseline. The paper may report this as positive mechanistic
evidence plus a closed cache boundary; it must not call the mechanism a quality
pass or justify K/admission/position/cache tuning. The next GDN3 should alter
the live joint address-state transition from scratch on the binding-error
carrier.

## 2026-08-13 Receiver-Live Diagnostic Integrity Boundary

P-FS2-006 does not contribute a mechanism or quality result. Its proposed
receiver-native signal compares the inherited FutureSeed read with the live
post-write read at the same receiver address, but the fixed FP32 recurrence
replay misses the official BF16/Triton output by `0.002523` relative RMS versus
the preregistered `0.002` ceiling. Terminal-state error is `0.001143`. The run
stops before discovery aggregation and produces no score. The paper may cite
this only as a measurement boundary: algebraically matching recurrence code is
not accurate enough to rank a 1%-budget causal intervention unless production
parity is demonstrated. It must not claim receiver-live FutureSeed failed.

## 2026-08-13 Online Inverse-Geometry Production Boundary

P-GDN3-027 tests the strongest remaining live-address hypothesis with a
receiver-local inverse-information matrix and an exact constrained committed
edit. The strict A100 CUDA contract passes, but production training is at
least `40.95x` slower than the contemporaneous official GDN2 control, against
the preregistered `<2.5x` ceiling. The run is therefore stopped during epoch 0
without a candidate quality verdict. The paper may use this as an engineering
boundary: exact online dense address whitening is algebraically feasible but
does not retain the linear-memory production advantage in this implementation.
It must not be presented as a negative accuracy result or as evidence against
all dynamic address mechanisms.

## 2026-08-13 Receiver-Edge Phase-Credit Boundary

P-LOOP-003 completes the causal localization sequence started by P-LOOP-001.
Its zero-parameter receiver-boundary audit reconstructs native FutureSeed gate
gradients to `3.71e-5` relative error without changing parameters, logits or
optimizer state. None of the three hard ranges satisfies the preregistered
cross-board and cross-depth phase-credit gate. The 51-55 aggregate opposition
is concentrated in receiver edge 8 and one board; 56-60 and 61-64 are instead
aggregate-aligned. P-LOOP-004 is therefore not admitted.

The paper-level claim is negative and useful: loop-specific scalar credit is
not a stable explanation for FutureSeed's hard-board closure failures. The
remaining near-unit orthogonal cotangent fraction is descriptive only because
simple orthogonal producer/receiver transport already failed its matched
quality test. Future work should change the live recurrent address/state
transition, with address binding tested from scratch before returning to the
Sudoku scale trajectory.

## 2026-08-12 FutureSeed Gradient-Conflict Boundary

P-LOOP-001 found genuine opening-versus-continuation conflict in native
FutureSeed gate gradients on two of three hard ranges. P-LOOP-002 turns that
correlation into a matched causal test without changing the forward graph,
loss, parameters or inference. The projection is strongly active on 57/100
steps, removes `83.16%` of the opposed opening component when active, and
preserves both the FS-subvector and global gradient norms to below `1e-7`
relative error.

The quality result is negative. Hard51-64 macro loop5 exact falls
`0.001302 -> 0`, mixed exact remains `0.025391`, all three official blank
deltas are negative, and same-board late correction weakens on 56-64. Cost and
integrity pass, so this closes scalar FutureSeed gate-gradient surgery rather
than blaming implementation or compute. The paper may use the pair to show
that aggregate loop-gradient conflict is real but insufficient for global
closure. It must not claim that gradient orthogonalization improves
FutureSeed. A next FS mechanism must change receiver-native content or resolve
macro-loop convergence, not tune this projection.

## 2026-08-12 Block-Causal Address Geometry Boundary

P-GDN3-026 closes the remaining online-Gram address-interference hypothesis.
Its exact strict contract verifies a causal, bounded, eight-parameter B64
exclusive-prefix query factor around one unchanged pinned-official GDN2 scan.
On the frozen P020 checkpoint, however, full-strength conditioning gives
effective-rank median gain `-0.000020`, anisotropy ratio `1.000088`, binding
margin median gain `0`, and only `36.57%` binding improvements. The geometric
admission fails before training, so no quality or Sudoku claim is made. This
negative result joins the cache/bank/hash/Raven/Bi-Axis boundaries: the paper
should not present another address wrapper as the next GDN3. The next justified
question is whether loop dynamics or the training signal, evaluated with a
same-runtime matched control, can turn already-active local correction into
global closure.

## Fixed Research Scope

As of 2026-08-05, the primary experimental scaffold is hard 9x9 Sudoku scaling.
The method program has two coupled targets: improve native FutureSeed state
transfer and derive a stronger, general official-FLA linear recurrent mechanism,
GDN3. Language and Maze results are supporting or historical evidence; they do
not replace Sudoku as the benchmark and cannot redirect the active experiment
queue. Every new mechanism must remain generic enough to transfer beyond Sudoku,
but it is selected and falsified on the fixed Sudoku scaling cliff.

The established GDN3 question is cross-layer state compatibility. FutureSeed
passes a terminal KxV state from a shallow layer into a deeper layer, while
ordinary GDN2 gives each layer an independently learned Q/K address basis. A
single address namespace shared by all layers was the first falsifiable
candidate for making that transported state readable without changing the
official recurrence.

P-GDN3-001 passes the registered quick falsifier. At matched step9100, mean
official 51-64 blank accuracy is `0.4479` versus normal GDN2 `0.2037` and the
older per-layer shared-address candidate `0.2844`; train CE is `1.1634` versus
`1.9370`, elapsed overhead is `6.9%`, and later loops reduce wrong cells on a
majority of fixed hard boards. Full-board exact remains zero. This remains
evidence that address compatibility matters, not that a globally shared basis
is the final architecture.

The full-diversity P-GDN3-002 trajectory closes the globally shared namespace
candidate. At step12000, full official 51-55/56-60/61-64 loop5 exact is
`0.3008/0.0996/0.0762`; hard-range macro exact is `0.1589` and mixed exact is
`0.2168`. These miss the registered `0.3191/0.40` routes and trail the sealed
D192/L10 canonical model in every hard range. Recurrent correction is still
real: official hard exact is zero at loop1 and opens in later loops, and one
same-board 64-blank failure reduces wrong cells `22 -> 16 -> 7 -> 6 -> 6`.
The paper can therefore claim a state-compatibility mechanism signal, but not
that forcing all layers into one Q/K basis is a scalable GDN3 design.

The read-only follow-up audit rejects post-hoc state rotation as the next move.
Across canonical step9000/12000 and P-GDN3-002 step12000, adjacent Q/K bases are
close to random orthogonal frames: normalized identity residual is about `1.42`,
optimal joint-Q/K Procrustes residual remains about `1.11`, rotation distance
from identity is about `1.41`, and some cross-basis singular values are below
`0.01`. A free learned bridge would therefore be large and ill-conditioned,
matching the failure mode already observed for arbitrary FS2 basis transport.

P-GDN3-003 tested the architectural hypothesis of making Q/K/V coordinates
exactly coherent at initialization, then let every layer specialize through
independent parameters. Only Q/K/V projections and their short convolutions are
copied from layer0 at construction; no parameters are tied or frozen, and the
native FutureSeed equation and official GDN2 recurrence are unchanged. This
tests whether compatibility is most useful as an optimization scaffold rather
than a permanent mature constraint, with zero added parameters or inference
operations.

The engineering gate passed. Exact initialization equality, distinct
parameter storage, unchanged parameter count, private output projections,
official FLA/Triton backward, finite gradients, and measurable specialization
after one optimizer step were all verified on the registered GPU1. However,
the full trajectory fails its step500 carrier gate: holes50 loop5 exact/blank
is `0/0.1321`, with CE `2.0089`, versus P-GDN3-002 `0.7778/0.9853` and CE
`0.0143`. Coordinate equality at birth therefore causes destructive deep-stack
symmetry rather than useful co-adaptation. P-GDN3-003 is closed without rescue.

P-GDN3-004 closes the full-size position-address scaling hypothesis. It removes
all cross-layer copying and uses the previously validated
position-address/payload separation: canonical position drives Q/K while
hidden content drives V and every state-edit/output gate. The formal SHA
`9f2ee8d` trajectory strongly passes the easy step500 gate, with h50 exact
`0.3838 -> 0.8990` across loops versus P-GDN3-002 loop5 `0.7778`. At
step3000, however, h53 loop5 exact/blank is only
`0.003906/0.582658`, missing both registered `0.02/0.60` alternatives.
The miss is informative rather than flat: h53 exact first appears at loop3,
h64 wrong cells fall `31.74 -> 26.28`, and step1000-to-step3000 hard blank
slopes are positive. Stable position addresses therefore improve optimization
and recurrent correction but do not solve hard global closure at this scale.
The exact process group was stopped and archived without rescue.

P-FS3-001 through P-FS3-003 close three candidate FutureSeed content
interfaces. Orthogonal innovation and single-payload producer compression are
active but leave hard exact unchanged. Address-local routing then preserves
all K rows and full V payloads and learns a substantially larger live residual
(`0.012122` relative RMS). It improves official61-64 blank accuracy by
`+0.005464` and reduces same-board wrong cells broadly, but hard macro and
mixed exact remain unchanged while 51-60 blank accuracy regresses and elapsed
overhead reaches `+15.66%`. This isolates a useful paper boundary: preserving
address-conditioned producer structure helps local hardest-tail correction,
but another FutureSeed-side residual/router is not the missing global-closure
mechanism. The next experiment should change the generic GDN recurrent
memory/state update itself, not tune these interfaces.

## Working Title

Future Seeds for Cheap Bidirectional Computation in Recurrent Reasoners

## Core Claim

FutureSeed is a small, generic state-conditioning mechanism that lets a causal or
recurrent backbone receive a learned summary of future context without full
bidirectional mixing at every layer. Looping then turns that initial direction
into extra computation. The current paper target is stronger than showing the
route exists: hard Sudoku scaling must reveal how FutureSeed and GDN3 memory
dynamics co-improve global convergence. The mechanism is judged under the same
code path, compute budget, and task metric, never by solver-specific repair,
search, rules, or selector tricks.

## Draft Abstract

Recurrent neural networks are attractive for reasoning because they can trade
test-time compute for accuracy, but causal recurrence is a poor fit for tasks
whose constraints are defined by both past and future context. We introduce
FutureSeed, a lightweight state-conditioning module that seeds recurrent
computation with a learned summary of future information. FutureSeed does not add
task-specific search, repair, or symbolic rules; it changes only the initial
direction of the recurrent state, leaving looped computation to refine the
answer. On synthetic constraint tasks, FutureSeed removes the position bias of a
causal RWKV backbone and opens hard Sudoku regimes where the same model without
FutureSeed collapses. We further use maze path finding as a diagnostic proxy and
show that standard token accuracy can be misleading when the desired path tokens
are sparse, motivating path-aware evaluation. Our experiments are designed to
separate three questions: whether FutureSeed supplies cheap bidirectional
information, whether loops perform real correction rather than copying a broad
mask, and whether the same mechanism improves official EqR-style recurrent
reasoning under matched compute. The resulting evidence supports FutureSeed as a
general mechanism for cheap bidirectional context in recurrent reasoners, while
also identifying objective alignment and late-loop correction as the main limits
for scaling to harder spatial reasoning tasks.

## Current Evidence

- **Important retained milestone:**
  [`Address/Payload-Factorized GDN2`](MILESTONE_GDN2_ADDRESS_PAYLOAD.md) is the
  canonical algorithm record for the position-Q/K carrier, its matched-compute
  evidence, its scaling boundary, and the claims that are not yet allowed.
- Strong cross-carrier causal gate: under byte-identical paired initialization,
  the same data/order/objective/optimizer/state/loop budget, and official
  kernels, b46-50 loop5 full-board exact with/without FutureSeed is
  `0.7285/0`, `0.3633/0`, `0.7969/0`, and `0.6465/0` for
  RWKV7/GDN/GDN2/KDA. FutureSeed adds no parameters, costs
  `7.2-17.2%` wall time and about `0.29-0.34 GiB`, and passes a separate
  same-trained-weights functional contract. This supports a generic
  short-budget causal opening claim, not hard closure: every arm is still zero
  exact at 51-64 blanks.
- New carrier-side mechanism evidence: under randomized Sudoku cell traversal,
  separating GDN2 memory address from payload raises equal-compute step9100
  mean official 51-64 blank accuracy from `0.2037` to `0.4333`. At matched
  step9300 it raises the same metric from `0.2753` to `0.5437` and lowers CE
  from `1.5707` to `0.9661`. Canonical position drives only Q/K; content still
  drives V and all memory-edit/output gates, with no new parameters or
  recurrent-kernel change. On matched 64-blank batch69, normal GDN2 changes
  `53->50->50->51->51` wrong cells while the split carrier changes
  `32->21->18->15->15`. This is persistent optimization and loop-correction
  evidence. A clean step9300->9600 hard-stage continuation first opens exact
  to `0.0059` in 51-55 and 56-60 through later loops, but mean hard blank only
  rises `+0.0364` and 61-64 exact remains zero. The paper can claim improved
  address learning and conditional recurrent correction, not robust global
  closure. The next evidence gate needs matched capacity/data scaling.
- Strong positive: RWKV9 Sudoku with and without FutureSeed, same CUDA
  state-passing backbone and same budget. Without FutureSeed, h12 loop5 exact is
  `0.0156` and early blank cells are much worse than late blank cells. With
  FutureSeed, h12 loop5 exact is `0.9492` and the early/late gap nearly
  disappears.
- Correction: one official EqR mixer-replacement probe was misnamed as
  FutureSeed. It actually replaced the token mixer with a forward+reverse
  recurrent scan. That is not FutureSeed and must not enter the main evidence
  chain. The side probe failed the e1024 gate: official mixer-base reached
  `accuracy=0.6644`, `exact=0.0249`, `lm_loss=0.7664`, while the scan
  replacement stayed at `accuracy=0.4231`, `exact=0`, `lm_loss=1.4029`.
- Strong boundary: official EqR Maze e256/e512 token accuracy is high but path
  F1 is near zero under plain token CE. A generic path-token-weighted objective
  opens path recovery to loop16 path F1 about `0.468`, but the model predicts a
  broad high-recall mask rather than a clean path.
- Official EqR + FutureSeed is neutral under the path-aware objective: loop16
  path F1 is `0.4684` versus clean EqR `0.4682`. This suggests EqR's mixer
  already supplies the noncausal interaction that FutureSeed is meant to cheaply
  add to causal/recurrent backbones.
- Causal RWKV on the same official path-weighted Maze objective is also neutral:
  no-FutureSeed loop8 path F1 is `0.4690`; FutureSeed loop8 path F1 is `0.4666`.
  Both arms converge to broad high-recall masks, so this objective does not
  isolate the future-context mechanism.
- A follow-up generic PATH/non-PATH boundary objective makes broad masks costly
  but exposes a different failure. no-FutureSeed loop8 path F1 falls to
  `0.4089`; FutureSeed improves to `0.4278` by preserving recall, but loop gain
  is still near zero. Treat this as weak evidence that FutureSeed can protect
  useful future-context signal under pruning pressure, not as a Maze win.
- A learned path-budget decoder separates mass calibration from ranking. Both
  no-FutureSeed and FutureSeed learn the total PATH fraction accurately, but
  budgeted decoding misses many true-path cells: no-FutureSeed budget loop8 F1
  is `0.3387`, FutureSeed is `0.3289`, and both produce about `78-80` false
  negatives per case. This shows the Maze bottleneck is not simply predicted
  mass; it is true-vs-false PATH ordering and late-loop correction.
- The official EqR causalized-mixer gate is negative as a FutureSeed result.
  With the same official code path, path-weighted objective, and e256 budget,
  causal base loop16 path F1 is `0.468067`; causal+FutureSeed is `0.468157`
  (`+0.000090`). Both arms still predict a broad high-recall mask with
  precision about `0.306`, recall about `1.0`, predicted PATH fraction about
  `0.433`, and roughly `270` false positives per case. FutureSeed has an early
  step500 optimization edge, but no final path-aware win.
- Important caveat: the current official EqR FutureSeed patch is a cross-level
  H/L latent injection. It is useful as an add-on boundary test, but it is not
  a clean "cheap bidirectional" implementation under causal attention because
  it does not introduce an independent future-token source. The neutral Maze
  causal/compression results should therefore not be over-read as disproving the
  broader FutureSeed mechanism.
- The official EqR H96 mixer-compression gate is also negative. H96 base loop16
  path F1 is `0.467337`; H96+FutureSeed is `0.467160` (`-0.000178`). Both arms
  keep recall at `1.0`, predicted PATH fraction around `0.434`, and roughly
  `271` false positives per case. Compression did not reveal hidden FutureSeed
  value on this Maze objective.
- The generic RWKV denoising-attractor feedback probe is negative. no-FutureSeed
  DAT at step300 has loop1/loop16 path F1 `0.4683/0.4680` and FP
  `268.3->269.0`; FutureSeed DAT has `0.4691/0.4688` and FP `265.9->266.4`.
  DAT loss decreases, but the feedback loop learns the same broad-mask fixed
  point. FutureSeed changes the operating point only slightly (`+0.0008`
  loop16 F1 over no-FS), not the loop correction behavior.
- Loop evidence is mixed: Sudoku scale-up shows loop matters; Maze visualizations
  often show later loops copying the same operating point. A paper claim about
  loops must report precision, recall, predicted mass, and false positives, not
  only final token accuracy.

## Experimental Roadmap

### E0. Quality-Matched Causal Efficiency

Hypothesis: if FutureSeed is a cheap future-context mechanism rather than only
an early curriculum shortcut, an efficient official causal carrier with
FutureSeed should reach the same full-board quality in materially fewer
optimizer steps or open 51-55 exact under the same clean scaling recipe.

Method: select one carrier from the strict four-way gate, then train one FS and
one noFS arm on the same independent-data schedule. Compare time/tokens to a
preregistered quality target and evaluate 46-50/51-55/56-64 exact at fixed
checkpoints. Do not tune the two arms separately and do not add noise, repair,
selector, search, or task-specific losses.

Decision:
- A publishable efficiency result requires at least `20%` less wall time or
  tokens to the same exact target, or a nonzero 51-55 exact frontier that noFS
  does not reach at matched compute.
- If both arms eventually converge at similar cost, narrow the claim to
  finite-budget optimization.
- If neither opens 51-55, stop carrier tables and treat global closure as a
  separate state-capacity/data-scaling problem.

### E1. Official EqR Maze Objective Alignment

Hypothesis: the official Maze failure is mostly an objective/metric mismatch.
Plain token CE is dominated by non-PATH cells, so the model can score about
`0.87` accuracy while predicting almost no path. A generic class-balanced loss
should make official Maze a meaningful proxy before we compare FutureSeed.

Method: patch the official EqR code path with an optional class-balanced token
loss or path-token weight. Train only the clean official EqR baseline first. This
is not a solver and does not encode maze rules; it aligns the supervised loss
with the path-recovery metric.

Prediction: if the proxy is viable, loop16 path F1 and predicted PATH fraction
should move away from zero by step500-1000. If it still predicts no PATH, Maze is
not ready for FutureSeed claims under this setup.

Decision:
- Result: passed the viability gate. Clean EqR reaches loop16 path F1 `0.4682`
  with recall `0.9998`, precision `0.3064`, and pred PATH frac `0.4325`.
- Interpretation: objective alignment works, but the task remains a broad-mask
  diagnostic rather than solved path finding.

### E2. Official EqR Maze FutureSeed Under Matched Objective

Hypothesis: once the Maze objective actually rewards path recovery, FutureSeed
should improve sample efficiency or final path F1 under the same official code
path and compute.

Method: matched base vs FutureSeed run under the E1 objective. Same official EqR
SHA, same data, same batch, same steps, same visualization runner.

Prediction: a real FutureSeed win should show better path F1, better recall at
similar predicted mass, or earlier opening at step500/1000.

Decision:
- Result: neutral. FutureSeed loop16 path F1 `0.4684`, clean EqR `0.4682`,
  delta `+0.0001`; hard-case false positives remain about `270` per case.
- Decision: do not claim FutureSeed beats official EqR. Use EqR as the strong
  noncausal baseline and boundary result.

### E3. Causal Maze Backbone With FutureSeed On/Off

Hypothesis: EqR's mixer already has noncausal mixing, so FutureSeed may be
masked. A causal/RWKV Maze backbone should expose the same future-context
failure mode seen in Sudoku.

Method: use the existing CUDA recurrent backbone, train Maze path recovery with
and without FutureSeed, and visualize loop trajectories. No selector, repair, or
maze-specific search.

Prediction: no-FutureSeed should show directional or position bias; FutureSeed
should improve path recall/F1 and reduce that bias.

Decision:
- Result on official path-weighted Maze: negative. RWKV no-FutureSeed reaches
  loop8 path F1 `0.4690`; RWKV + FutureSeed reaches `0.4666`; loop gains are
  near zero or negative.
- Decision: do not use official path-weighted Maze as a positive FutureSeed
  benchmark. It mostly measures broad PATH coverage and leaves hundreds of false
  positives per case.

### E4. Loop Correction Diagnostics

Hypothesis: FutureSeed gives a good initial direction, but top-conference-level
evidence needs to show whether loops actually correct errors.

Method: for the best Sudoku and Maze runs, report loop1/4/8/16 precision, recall,
predicted mass, false positives, false negatives, and hard-case visualizations.

Prediction: a strong loop story requires later loops to reduce false positives
without destroying recall. If later loops only copy loop1, the paper should not
overclaim loop correction.

Decision:
- Keep loop as a variable-compute mechanism if late-loop correction is visible.
- Otherwise position loop as compute reuse and future work for state dynamics.

### E5. Causal Maze Boundary Objective

Hypothesis: the path-weighted Maze objective permits broad-mask shortcuts. A
generic PATH/non-PATH boundary objective should make false positives expensive
without adding maze rules or postprocessing. If FutureSeed provides cheap future
context to a causal backbone, it should help preserve true path cells when the
model is pushed away from high-recall coverage.

Method: causal RWKV7 state-passing Maze runner, D128/L8, train loops 4, eval
loops 8, same official `maze-30x30-unique-1k` data. Compare
`FUTURE_SEED_SCALE=0` and `1` under matched compute. Objective is token CE with
PATH weight plus a generic PATH-vs-non-PATH margin BCE and per-sample PATH-mass
calibration.

Decision:
- Result: weak positive for FutureSeed, negative as a final benchmark.
  no-FutureSeed loop8 path F1 is `0.4089`; FutureSeed is `0.4278`.
  The gain is mainly recall (`0.5828 -> 0.6280`), while precision is almost
  unchanged (`0.3256 -> 0.3270`). Loop gain remains effectively zero.
- Interpretation: the objective can reduce broad-mask coverage, but it overprunes
  true path cells. Do not sweep boundary weights. Maze remains useful as a
  failure analysis tool, not yet as a positive FutureSeed benchmark.

### E6. Causal Maze Learned Budget Decoder

Hypothesis: the path-weighted Maze failure may be a calibration problem: the
causal RWKV logits may rank true PATH cells above false positives, while raw
argmax uses the wrong boundary.

Method: add a generic learned path-count head and decode PATH as the top
model-ranked cells under the model's own predicted budget. Compare no-FutureSeed
and FutureSeed under matched compute. No oracle true count, selector, search,
repair, or maze rules.

Decision:
- Result: negative but informative. The budget head learns path fraction well
  (`abs_err≈0.0095-0.0098`), but budget decoding trades false positives for false
  negatives. no-FutureSeed budget loop8 F1 is `0.3387`; FutureSeed is `0.3289`.
- Interpretation: the model has mass calibration but lacks ranking/correction.
  FutureSeed does not repair this Maze bottleneck. Do not sweep count/budget
  weights; the next useful Maze experiment must target generic ranking or
  self-correction directly.

### E7. Official EqR Causalized-Mixer Gate

Hypothesis: if FutureSeed cheaply supplies future-side information, removing
full noncausal attention from the official EqR mixer should create a condition
where FutureSeed helps a causalized EqR path recover Maze structure.

Method: official EqR upstream SHA `aba94e9`, same official
`maze-30x30-unique-1k` data, same path-weighted objective, same e256 budget,
same path-aware visualization. Compare causalized EqR no-FutureSeed against the
same causalized code path patched with FutureSeed. No selector, search, repair,
or maze-specific postprocessing.

Decision:
- Result: negative. causal base loop16 path F1 `0.468067`, causal+FutureSeed
  `0.468157`, delta `+0.000090`.
- Loop behavior remains broad-mask fixed point. Loop1 to loop16 gains are only
  about `+0.0018`, and false positives remain around `270` per case.
- Interpretation: causalizing EqR does not make this Maze proxy expose the cheap
  bidirectional value. The path-weighted objective still mostly rewards high
  recall broad masks. Do not sweep FutureSeed gate bias, seed, path weight, or
  causal-attention details.

### E8. Official EqR H96 Mixer-Compression Gate

Hypothesis: if FutureSeed cheaply substitutes for part of the noncausal mixer,
then a compressed official EqR model should benefit more from FutureSeed than
the full D128 model did.

Method: official EqR upstream SHA `aba94e9`, official `maze-30x30-unique-1k`,
path-token weight `8`, e256 budget, `hidden_size=96,num_heads=8`, matched base
vs FutureSeed, no selector/search/repair/postprocessing.

Decision:
- Result: negative. H96 base loop16 path F1 `0.467337`; H96+FutureSeed
  `0.467160`, delta `-0.000178`.
- Both arms are broad masks: precision about `0.306`, recall `1.0`,
  predicted PATH fraction about `0.434`, false positives about `271` per case.
- Interpretation: official EqR Maze path-weighted objective is not a positive
  FutureSeed benchmark under full, causalized, or compressed mixer settings.
  Stop hidden-size/gate/seed sweeps.

## Bitter-Lesson Boundaries

Allowed:
- Larger clean training budgets.
- Generic loss alignment to the task metric.
- Generic state dynamics such as learned FutureSeed gates or normalization.
- Causal/backbone comparisons under matched compute.

Not allowed:
- Maze repair, Sudoku repair, symbolic search, rollout selector, or best-of-K
  oracle at inference.
- Maze-specific hand rules.
- Broad seed/weight/temperature tables without a decision.
- Claiming success from token accuracy when the target metric is path recovery.

## Immediate Next Experiment

E1, E2, and the first E3 official-Maze variants are complete. They show that
token accuracy is misleading, broad-mask path recovery is easy to learn, and
extra decision heads or mass objectives do not yet make loops reliably repair
false positives.

The official-EqR mixer-compression gate is now complete and still broad-mask
neutral. Maze should not be used as positive FutureSeed evidence yet. The only
remaining Maze work worth running is a generic recurrent state/training
dynamics experiment that directly tests whether loops can leave the broad-mask
attractor without search, repair, selector, or maze-specific rules.

The denoising-attractor probe is now complete and negative. It did not reduce
false positives while preserving recall; both no-FutureSeed and FutureSeed arms
hit the step300 broad-mask kill rule. Maze should therefore remain failure
analysis unless the next mechanism changes the recurrent decision/state dynamics
more substantially while still staying generic.

P-MAZE-007 is complete. Abortable probes now dump hard-case
input/target/loop1/4/8/16 visuals before termination. The verification run is
not positive model evidence: at step100, loop1 and loop16 were the same broad
mask (`F1 0.4614 -> 0.4614`, precision `0.3006 -> 0.3006`, recall `1.0`,
FP `272.1 -> 272.1`). Its value is that future Maze failures are now auditable.

The next Maze mechanism is only worth running if it creates a genuinely
different state variable for true-vs-false PATH competition; another
loss-weight, corruption-mix, temperature, or seed sweep is explicitly low ROI.
For small runner probes, prefer a minimal Git archive of required source paths
when full worktree/source snapshot checkout stalls on historical tracked
artifacts.

The paper claim is allowed to proceed only if FutureSeed either improves hard
path F1 by `>= +0.03` without broad-mask inflation, improves official Sudoku
sample-efficiency/exact under a matched cheap backbone, or reaches the same
accuracy with `>=20%` lower train/inference compute. If the official EqR
baseline cannot be reproduced, stop there. If FutureSeed only increases token
accuracy or predicted-path coverage without improving exact/FP/FN, treat it as
a negative result and visualize the failure.

Do not run another weight/seed/temperature table for Maze.

The official-codebase bidirectional scan mixer replacement probe is complete and
is off-mainline. It should not be described as FutureSeed. The next paper
experiment should not be another old-patch gate or a longer run of that scan
replacement.

The highest-ROI next experiment is a simple generic mechanism change that keeps
the FutureSeed idea but avoids the stable-wrong attractor:

- Use official EqR as the strong noncausal ceiling and baseline.
- Keep FutureSeed as terminal-state seeding: a recurrent layer's final state
  seeds a later layer or loop. Do not add a hidden right-to-left scan under the
  FutureSeed name.
- Test either a gated/normalized FutureSeed state update or a minimal setting
  where FutureSeed supplies a future-conditioned initialization while learned
  capacity handles final decisions.
- Prefer Sudoku sample efficiency/exact for the first official-codebase gate
  because Maze path-weight repeatedly collapses to broad masks.
- Keep the mechanism generic: no Sudoku rules, no solver, no selector, no
  best-of-K oracle, no repair.
- Success means FutureSeed improves matched official sample efficiency without
  losing long-budget exact, or reaches the same quality with less recurrent
  compute. Failure means the current formulation is only an early optimization
  aid, not a standalone mixer replacement.

## 2026-08-04 Cross-Task Directionality Gate

`P-CAUSAL-007` supplies the first clean non-Sudoku causal mechanism result. In
the validated upstream Zoology shell with pinned official FLA GDN2, the matched
no-FutureSeed model reaches 93.05% on write-before-query retrieval but only
1.10% on query-before-write retrieval. Native cross-layer terminal-state
seeding reaches 99.55% and 99.30%, respectively, with identical parameters,
initialization, data, optimizer, kernel, and epoch budget. Future-query exact
changes from 0% to 98.60%.

This supports the narrow paper claim that FutureSeed is a cheap future-context
route for causal recurrent stacks. It does not yet support language quality,
asymptotic scaling, or wall-time efficiency. The next paper gate should test
one meaningful transfer axis rather than repeat seeds: OOD sequence/association
load or an established masked/retrieval language task with a warmed full
noncausal reference.

## 2026-08-04 Established-Text Gate Boundary

`P-CAUSAL-008` does not extend the paper claim to natural language. The matched
WikiText byte-MLM run produced masked accuracy `0.4247` for causal GDN2 and
`0.4218` for FutureSeed, but the proposed full-bidirectional attention ceiling
reached only `0.1879`. This fires the preregistered carrier-validity kill rule:
the attention reference did not learn the task, so the run cannot say whether
FutureSeed closes a meaningful future-context language gap.

The mechanism implementation itself passed the strict checks: scale 0 was
identical to causal GDN2, causal future dependency was zero, FutureSeed future
dependency and gate gradient were nonzero, both recurrent arms had identical
parameters and initialization, and official FLA/Triton executed without a
fallback. FutureSeed was nevertheless slightly worse in this exact invalid
carrier. Report that boundary honestly; do not present it as either positive
or negative language evidence and do not tune the failed attention shell.

The next paper gate must start by reproducing a validated, established
bidirectional masked-language implementation and recipe. It must visibly beat
a strict causal reference on masked recovery before any recurrent mixer is
substituted. Once that carrier is valid, compare matched official-FLA GDN2
scale 0 and native FutureSeed scale 1 with the same data, initialization,
training tokens, optimizer, width, depth, and metric. Until then, P-CAUSAL-007
remains the clean cross-task directionality evidence and language transfer is
an open question.

### P-CAUSAL-009 Established MLM Boundary

The exact Transformers BERT-mini carrier did learn on the fixed WordPiece
WikiText stream, but it reached only `0.0716` masked accuracy at 20.48M input
tokens and missed the preregistered `0.10` endpoint. No causal, GDN2, or
FutureSeed arm was run. This is not a language result for or against
FutureSeed. The paper should now prioritize the validated directional-MQAR
scaling and cost frontier; natural-language transfer remains a later gate that
requires a genuinely opened published recipe.

### P-CAUSAL-010 Scaling Carrier Boundary

The L64 matched GDN2 pair replicated the mechanism strongly: no-FutureSeed
past/future accuracy was `0.9965/0.0100`, while native FutureSeed reached
`0.9985/0.9920` and `0.981` joint exact. However, the registered
parameter-matched full SDPA reference reached only about `0.50` on both
directions in ten epochs, despite a verified nonzero future dependency. The
run therefore stopped before L1024. This is not a Transformer comparison.
Before drawing a scaling curve, reproduce the exact upstream Zoology MHA with
only its causal mask removed and allow the pre-existing official opening
budget; only an opened attention carrier can define the quality/cost ceiling.

### P-CAUSAL-011 Official-MHA Mask-Removal Boundary

The stricter carrier check also failed. The exact upstream MHA, with identical
parameters and initialization and only its causal mask removed, had verified
future dependency but finished 30 epochs at past/future accuracy
`0.4850/0.4845` and joint exact `0.048`. This closes the mask-removal carrier,
not bidirectional Transformers in general. The paper must not claim an
attention quality win from P-CAUSAL-010/011. The next core figure should first
establish causal GDN2 versus native FutureSeed length scaling at L64/L1024;
an attention quality/cost frontier remains conditional on a separate task and
published bidirectional recipe that independently opens.

### P-CAUSAL-012 Long-Context FutureSeed Boundary

The direct GDN2 endpoint gives strong but bounded evidence. At length64,
causal/FutureSeed past accuracy is `0.9965/0.9985` and future accuracy is
`0.0100/0.9920`. At length1024 under the same ten-epoch, four-association
protocol, causal GDN2 does not open either direction (`0.0110/0.0085`), while
FutureSeed reaches past/future accuracy `0.7535/0.7415`, directional exact
`0.590/0.572`, and joint exact `0.339`. The future delta is `+0.733`, but the
registered `0.80` future and `0.90` past thresholds and 80% L64-retention gate
are missed. The paper may claim a large long-context optimization and
information-routing benefit, not length-invariant quality.

The error structure gives the next mechanism question. At L1024, `82.4%` of
wrong future predictions and `78.9%` of wrong past predictions are valid
values belonging to another key in the same sample. FutureSeed transports the
value set but increasingly confuses bindings. Test one generic recurrent
address/state-capacity increase at L1024 before filling middle lengths. If it
reduces these swaps and opens the registered endpoint, then measure the full
length curve; otherwise treat terminal-state compression/update quality as the
limit. Do not rescue P-CAUSAL-012 with epochs, seed, LR, loss, or task rules.

Do not use the archived sequential throughput difference as a cost claim. The
two arms shared one Triton process and likely inherited different autotuning
cache state. The final cost-quality figure requires independent fresh-process
warmup and alternating execution order, plus a separately validated
bidirectional attention or bidirectional recurrent baseline.

### P-CAUSAL-013 Whole-Model Capacity Boundary

The first capacity intervention failed cleanly. At the exact L1024/K4 endpoint,
official-FLA GDN2 D256/L2/H8/D32 doubled recurrent-state values per layer but
increased total parameters `3.39x`. Both matched D256 arms stayed near chance:
causal past/future accuracy was `0.0145/0.0120`, and FutureSeed was
`0.0090/0.0115`, versus frozen D128 FutureSeed `0.7535/0.7415`. Validation CE
rose after epoch5 in both larger arms, while D128 FutureSeed had opened sharply
from epoch3. The FutureSeed path itself remained active and all official
FLA/Triton, data, initialization and dependency checks passed.

This result rejects the claim that undifferentiated width scaling is sufficient
under matched optimization compute. It does not show that larger recurrent
state is intrinsically harmful: width, parameter count and state capacity moved
together, and the larger model failed before binding diagnostics became
meaningful. The next high-information test is state-only scaling at the proven
D128 geometry through official GDN2 value expansion. Do not run middle lengths
or a D256 epoch/LR/seed rescue before that isolation test.

### P-CAUSAL-014 Value-State Axis Boundary

The isolation test also failed, but it makes the next question more precise.
With D128/L2/H4 and key dimension32 fixed, official GDN2 `expand_v=2` doubled
value-state scalars per layer from4,096 to8,192. It added22.49% parameters,
reduced warmed diagnostic throughput15.48%, and increased peak training memory
10.72%. Native FutureSeed balanced accuracy fell from the frozen `0.7475` to
`0.27675`; past/future accuracy was `0.2555/0.2980`, and joint exact was zero.
The matched causal arm stayed directionally valid with future accuracy0.0125.

This is not evidence that FutureSeed stops working: within the expand-v2
carrier it still adds `+0.25025` balanced accuracy over scale0. It is evidence
that widening the V/payload axis is a poor way to repair long-context binding.
The recurrent matrix remained K32 x V64, so the number of independent key
address directions did not increase. The main remaining paper mechanism
question is whether generic address separability or FutureSeed compression can
scale without changing the already-open D128 residual backbone. Do not present
state-element count as address capacity, and do not continue expand-v, epochs,
LR, or seeds.

### P-CAUSAL-015 Relative-Address Attention Boundary

Adding parameter-free standard RoPE to the frozen near-parameter-matched full
SDPA carrier did not open it. Strict preflight proved identical 539,136
parameters and initialized tensors, bit-exact `rope_scale=0`, exact data,
nonzero future dependency and finite CUDA backward. After 30 epochs, RoPE SDPA
reached past/future accuracy `0.4955/0.4820` and joint exact `0.046`, versus
plain SDPA `0.4980/0.4995` and FutureSeed `0.9985/0.9920` with joint exact
`0.981`. Its best aggregate validation accuracy was only `0.5030` at epoch 2.

All wrong predictions from all three bidirectional attention controls are
another valid value from the same sample. They learn the candidate value set
but not the key/value binding. Relative position alone does not repair that
algorithmic ambiguity in this two-layer shell. Close this MQAR proxy for an
attention quality/cost claim and do not rescue it with RoPE, optimizer, model,
epoch or seed tuning. The paper can retain the matched causal-versus-FutureSeed
directionality and long-context results, but any Transformer ceiling must be
established on a separate carrier that independently opens.

### P-CAUSAL-016 FutureSeed Context-Length Curve

The matched five-point curve is now strong paper evidence. Causal/FutureSeed
future-query accuracy at lengths `64/128/256/512/1024` is respectively
`0.0100/0.9920`, `0.0085/0.9810`, `0.0090/0.9780`, `0.0105/0.9850`, and
`0.0085/0.7415`. FutureSeed joint exact is
`0.981/0.955/0.932/0.942/0.339`; causal joint exact is zero throughout. All
arms use the same official-FLA GDN2 D128/L2/H4/D32 architecture, parameters,
data recipe, optimizer, training-token budget and CUDA kernels at each length.
All preregistered absolute and matched-delta gates pass.

The result changes the scaling diagnosis. There is no gradual quality dilution
through 512 tokens: FutureSeed future accuracy stays in `0.978--0.992`. The
single large drop is `0.2435` from L512 to L1024. At L1024, `82.4%` of wrong
future predictions select another key's valid value from the same sequence,
so the remaining boundary is address binding/compression rather than absence
of future content. Wider whole models and wider V state already failed; do not
reopen those axes or fill additional middle lengths.

The paper may now claim that native terminal-state seeding provides a compact
future-context route that scales cleanly through 512 tokens and remains highly
consequential at 1024. It still may not claim superiority to bidirectional
Transformers: all attempted attention carriers on this proxy failed their own
binding gate. The final cost-quality figure requires an independently opened
published bidirectional carrier. P-CAUSAL-016 peak-memory differences are
consistently about `1.50 MiB`, but its 20-step timing samples are too short and
variable for a precise throughput claim; rerun only a robust repeated timing
protocol when the valid comparison carrier exists.

### P-CAUSAL-017 Explicit Bidirectional Recurrent Boundary

A conventional two-pass recurrent ceiling was tested at L512: every layer ran
independent official-FLA GDN2 streams in forward and reversed order, flipped
the reverse output back, concatenated both streams and learned a linear fusion.
This baseline had 894,608 parameters versus 596,048 for causal and FutureSeed
GDN2. Strict preflight proved active future dependency, four official
chunk/Triton streams and nonzero reverse/fusion gradients.

The carrier nevertheless failed. Its past/future accuracy was
`0.0345/0.0120`, joint exact was zero and best aggregate validation accuracy
was only `0.02525`; frozen FutureSeed reached `0.9860/0.9850` and `0.942`
joint exact on the same L512 data. Robust cost measurement was stopped because
comparing speed against a chance-level ceiling would not answer the paper
question.

This cannot support a claim that FutureSeed beats bidirectional recurrent
models in general. It does support a more precise mechanism interpretation:
reverse visibility alone is insufficient when each causal stream still must
retain random associations over long distances. FutureSeed supplies a
trainable cross-layer memory route that helps both formally future and already
causally available long-range bindings. The paper still needs an independently
opened published bidirectional carrier before making a quality-cost claim.

### P-CAUSAL-018 Established Real-Text Carrier

The fixed WikiText masked-token carrier is now independently validated with the
official pretrained Google BERT-Tiny checkpoint. The same 4,416,698 checkpoint
parameters were evaluated with native bidirectional attention and with only the
attention semantics changed to strict causal. On 4,742 deterministic masked
targets, accuracy was `0.355546` versus `0.215732`, a `+0.139814` bidirectional
gain; CE was `4.033269` versus `5.453952`, a `1.420684` improvement. Strict
causal future dependency was exactly zero while native bidirectional dependency
was `3.567824`. Every preregistered carrier check passed.

This is carrier evidence, not a FutureSeed result and not a fair trained-model
quality comparison. It proves that the fixed text task exposes useful right
context under an established checkpoint, removing the invalid-carrier problem
that killed P-CAUSAL-008/009. The next paper experiment is one separately
preregistered, matched official-FLA GDN2 scale-0/scale-1 replacement with common
lexical initialization. Do not add a model-size, tokenizer, mask, seed, or
training-budget table before that causal gate.

### P-CAUSAL-019 Matched Real-Text FutureSeed Gate

The first fair real-text replacement is valid but weak. Two official-FLA GDN2
arms used identical D128/L2/H4/D32 parameters, frozen tied BERT-Tiny word
embeddings, fresh contextual modules, 160,000 identical corruption/window
pairs, 20.48M input tokens and the same optimizer. The only difference was
native terminal-state FutureSeed scale 0 versus 1. Strict preflight proved
scale-0 identity, exact causal non-leakage, active FutureSeed dependency,
official chunk/Triton execution and byte-identical provenance.

Causal/FutureSeed masked accuracy is `0.276466/0.278996`; CE is
`5.350222/5.262873`. The `0.087349` CE gain has paired-window 95% interval
`[0.06723,0.10778]` and the same sign at steps 1000 and 1250, but it misses the
registered `0.20` strong threshold. Accuracy gains only 12/4,742 targets and
its interval crosses zero. The correct paper wording is therefore not that
FutureSeed improves language-model accuracy.

The mechanism itself transfers: removing the suffix changes causal CE by
exactly zero but worsens FutureSeed CE by `0.439155`, with 95% interval
`[0.32460,0.56772]`. FutureSeed repairs 119 top-1 predictions while regressing
107. It has learned to use right context, but the one available L1-to-L2 state
transfer mostly improves probabilities rather than decisions. Retain this as
real-text directionality evidence and a scaling motivation. Do not tune or
extend the exact L2 endpoint; a future language test must change one meaningful
scaling axis, preferably depth/multiple transfer opportunities, under a new
preregistration.

### P-CAUSAL-020 Real-Text Depth Boundary

The registered depth test scaled only the official-FLA GDN2 stack from L2 to
L4 while keeping D128 state width, WordPiece/WikiText tensors, frozen lexical
table, optimizer, seed and 20.48M training tokens fixed. The scale-0 and
scale-1 arms each had 4,965,722 parameters. Strict preflight verified three
active terminal-state routes with nonzero gate gradients and zero causal future
dependency.

L4 causal/FutureSeed accuracy is `0.276677/0.284268`; its `+0.007592` paired
interval is `[0.00084,0.01426]`. CE is `5.323694/5.224210`, a `0.099484`
improvement with interval `[0.07965,0.11964]`. Suffix removal costs FutureSeed
`0.586523` CE, confirming stronger right-context use than L2. Yet CE advantage
increases by only `0.01213` over L2 and misses the registered `0.05` depth gain,
the `0.20` strong CE gate and the `+0.03` accuracy gate.

The paper may state that the real-text route persists across depth and becomes
statistically visible in top-1 accuracy. It may not state that depth scaling
solves masked language modeling or yields competitive quality. Close small
depth sweeps; a future language headline requires a materially larger
pretraining regime or another independently validated carrier.

### P-CAUSAL-021 Real-Text Data-Diversity Scaling

The previous fixed-token runs repeatedly remasked only 10,000 grouped text
windows. P-CAUSAL-021 replaced that schedule with 160,000 independent windows
from the full, hash-pinned WikiText-103 raw training split while holding the
official-FLA GDN2 D128/L4 architecture, initialization, optimizer, batch size,
1,250 steps and 20.48M input tokens fixed. This isolates data diversity from
model and compute scaling.

Causal/FutureSeed masked accuracy is `0.275833/0.292493`, a `+0.016660`
difference with paired 95% interval `[0.00973,0.02364]`. CE is
`5.300604/5.131930`, a `0.168674` FutureSeed advantage with interval
`[0.14567,0.19203]`. Relative to P-CAUSAL-020, the CE advantage grows by
`0.069190`, passing the preregistered data-diversity mechanism gate; the
accuracy advantage grows by `0.009068`, narrowly missing its separate `0.01`
gate. FutureSeed makes 182 repairs and 103 regressions, for 79 net repairs
versus P-CAUSAL-020's 36. Suffix removal costs FutureSeed `0.689943` CE and
causal GDN2 exactly zero, so the gain still comes from usable right context.

This is the clearest real-text scaling result so far, but it remains below the
strong paper gate of `+0.03` accuracy or `0.20` CE. The paper may claim that
independent language-data diversity amplifies the FutureSeed advantage at
fixed compute. It may not yet claim competitive masked-language quality. The
positive endpoint slope and passed mechanism gate authorize one clean joint
data-and-compute continuation at the same model size; they do not authorize a
model-size, seed, learning-rate or loss sweep.

### P-CAUSAL-022 Joint Data-and-Compute Language Scale

The preregistered continuation kept P-CAUSAL-021's D128/L4 official-FLA GDN2,
4,965,722 parameters, initialization, optimizer, seed, tokenizer, validation
and kernel fixed. It consumed 640,000 independent WikiText windows once over
5,000 steps, or 81.92M input tokens per arm. The only arm difference remained
native terminal-state FutureSeed scale 0 versus 1.

Causal/FutureSeed masked accuracy is `0.309363/0.373893`, a `+0.064530`
difference with paired 95% interval `[0.05492,0.07438]`. CE is
`4.618276/3.905131`, a `0.713145` advantage with interval
`[0.66992,0.75717]`. Both strong paper routes pass. The advantage also scales
smoothly from step1000 to step5000: accuracy delta grows
`0.00569->0.06453` and CE advantage `0.15974->0.71315`. Suffix removal costs
FutureSeed `2.12152` CE and causal exactly zero, directly tying the gain to
right-context use.

The aggregate visual audit contains 440 repairs and 134 regressions, for 306
net corrected targets across both halves of the text windows. Warmed diagnostic
throughput is 499k/475k tokens per second for causal/FutureSeed and peak
training allocation differs by about 19 MB. These support small implementation
overhead but are not yet a full bidirectional systems frontier.

P-CAUSAL-022 upgrades the real-text result from weak mechanism evidence to a
strong matched quality result: FutureSeed's benefit grows sharply under
ordinary independent-data and training-compute scaling. The paper should use
P019/P020/P021/P022 as a decision sequence, not as an ablation grid. Do not add
intermediate token budgets, multiple seeds or width/depth tables. The next
headline experiment must either move to a materially larger established
language regime or compare quality and robust cost against a valid
bidirectional carrier.

### P-CAUSAL-023 Frozen L128 Quality-Cost Frontier

Five fresh GPU1 processes per arm compared frozen P022 causal/FutureSeed GDN2
against the independently opened official BERT-Tiny checkpoint on the exact
same 4,742 masked targets. FutureSeed passes the registered quality frontier:
BERT/causal/FutureSeed accuracy is `0.355546/0.309363/0.374104`, and CE is
`4.033269/4.618317/3.905087`. It makes 441 repairs versus 134 regressions over
causal and has a net 88 correct-token advantage over BERT. Causal future
dependency remains exactly zero; FutureSeed dependency is `1.57446`.

The registered practical-cost route fails. Median batch-1 masked-recovery
latency is `1.361 ms` for BERT and `12.250 ms` for FutureSeed. Median batch-64
throughput is `5.888M` versus `0.667M` input tokens/s, and peak allocation is
`177.7` versus `378.9 MiB`. FutureSeed is therefore about `9x` slower and uses
`2.13x` the allocated memory at this short length. One BERT workload also has
CV `0.138`, so timing is formally non-claimable; the median cost miss is too
large for that caveat to reverse the decision.

Paper boundary: FutureSeed is a strong future-context and real-text quality
mechanism, but the current official-FLA implementation is not a cheaper BERT
replacement at length 128. Linear recurrent asymptotics remain a motivation,
not an empirical systems result. A credible cheapness claim now requires a new
long-context model with valid positional support and a quality-preserving
crossover curve. Do not rescue this frozen endpoint with systems or hyperparameter
tuning.

### P-FS3-001 Orthogonal Innovation Boundary

The exact matched step3000-to3100 test separates implementation activity from
architectural value. The candidate adds only 80 zero-initialized layer/head
scalars and preserves exact native FutureSeed output at initialization. Its
strict CUDA contract, position-QK path, pinned official-FLA/Triton backward,
checkpoint migration, and both arm exits all pass.

The mechanism activates by step3100: loop5 mean `abs(tanh(alpha))` is
`0.010348`, the orthogonal component carries `0.866` of terminal RMS, and
the injected residual is `0.010348` of terminal RMS. Nevertheless, official
51-64 macro loop5 exact remains `0.000651` in both arms, mixed exact remains
`0.025391`, and all three hard-range blank accuracies move slightly down.
Same selected-board correction is not consistently stronger. The small CE
change `0.858617->0.857070` is not a solve signal.

The paper must not claim that separating state innovation orthogonal to the
inherited seed improves FutureSeed. The useful negative result is narrower:
analytic direction separation is insufficient even when active and exact at
initialization. Close scale/floor/rank/seed/LR/loss/duration rescue. A future
content mechanism must learn a shared producer compression/update code and
earn a new board-level gate; it cannot be presented as a tuned P-FS3-001
variant.

### P-FS3-002 Single-Payload Producer Codec Boundary

P-FS3-002 tested a learned content interface rather than another analytic
direction or scalar gate. The shared 59-parameter codec observes the real
producer update `T-I`, softmax-pools all K address rows into one V payload, and
decodes a bounded KxV residual before unchanged native FutureSeed
normalization and gating. Zero initialization is bit-exact to the frozen
terminal control. Independent K/V permutation, gradient, exact-resume,
position-QK, and pinned official-FLA/Triton contracts all pass.

The mechanism activates cleanly. At loop5 the payload/update/residual relative
RMS is `0.3362/0.8955/0.002382`, with nonzero residual variation across
boards. It does not improve board closure: hard51-64 macro loop5 exact remains
`0.000651`, mixed exact remains `0.025391`, and all three hard-range blank
accuracies move slightly down. Across all 256 shared boards per range, the
codec changes individual trajectories but does not strengthen loop3-to5
correction consistently. The scorer is also nearly uniform over 32 K rows:
attention max is `0.0339` versus uniform `0.03125`.

The cost boundary is decisive as well. Matched 100-step throughput falls
`15.497->12.774` effective boards/s; elapsed, allocated-memory, and
reserved-memory overhead are `+21.31/+14.55/+12.56%`. The paper must not claim
that one shared producer payload improves FutureSeed. The narrower reusable
negative result is that a live learned codec can still erase the address
structure needed for global closure when it compresses the entire KxV update
to one nearly uniform message. A successor must preserve address-conditioned
multi-part state or change the generic recurrent memory/update, and cannot be
presented as a payload-count or decoder-width sweep.

### P-FS3-003 Address-Local Update Boundary

P-FS3-003 isolates the structural prediction left by the failed single-payload
codec. For every producer address row, it observes generic terminal/update
statistics and applies one shared bounded scalar to that row's full V update.
It never pools K rows, invents a payload basis, or adds a cell-wise decoder.
The 35-parameter feature-sized router is shared across all layers and heads;
its final projection is zero initialized, and its residual is elementwise
bounded by the actual producer update. Identity, migration, gradient,
permutation, exact-resume, official-FLA/Triton, source, data, and GPU contracts
all pass.

The router activates strongly. At loop5, mean row gain is `0.014254`, row and
board standard deviations are `0.003989/0.001069`, and update/residual relative
RMS is `0.898864/0.012122`. Compared with the frozen terminal control,
official61-64 blank improves `0.591923->0.597387`; its mean wrong cells fall
`26.43->26.06`, and 121/256 boards improve versus 78 regress. This confirms
the prediction that keeping address-local producer structure is better than
collapsing it to one payload on the hardest tail.

It still fails the registered claim. Hard51-64 macro loop5 exact remains
`0.000651`, mixed exact remains `0.025391`, and 51-55/56-60 blank changes
`-0.000895/-0.001888`. Elapsed overhead is `+15.66%`, despite only
`+5.86%` peak-allocation overhead. The paper may report the address-local
blank/wrong-cell signal only as a negative mechanism boundary, not as improved
FutureSeed closure. Close feature, router-width, gain, layer/head-specific,
seed, optimizer, loss, duration, and model-size rescue. A successor must alter
generic GDN memory/state dynamics rather than add a fourth transfer-side
content transform.

### P-GDN3-005 Coherent Delta Update Boundary

P-GDN3-005 moved the intervention from FutureSeed transfer content into the
generic recurrent state edit. It added one zero-initialized scalar per
layer/head and moved official GDN2's K-axis erase and V-axis write gates toward
their shared per-token/head strength before the unchanged pinned official
kernel. At D256/L12/H8 this added exactly 96 parameters and preserved exact
parent output and recurrent states at initialization, including nonzero
incoming states.

The CUDA, migration, gradient, exact-resume, source, data, and GPU contracts all
passed. The formal mechanism activated with loop5 mean absolute mix `0.012261`,
but the learned mean was negative and the erase/write gap ratio became
`1.001982`. Thus the candidate differentiated the gate strengths instead of
contracting them. Hard51-64 macro loop5 exact stayed `0.000651`, mixed exact
stayed `0.025391`, and official51-55/56-60/61-64 blank changed by
`-0.000609/-0.000892/-0.000366`. Same-board loop3-to5 correction was weaker on
both 56-60 and 61-64. The small train-CE gain
`0.858617->0.855293` did not close any additional board.

The systems result also fails: elapsed time and peak allocation increase by
`11.18%` and `18.35%`, despite only 96 added parameters. This is useful as a
negative paper boundary: exact zero-init and active learned gate coupling do
not imply a useful update geometry, and parameter count is not an execution-
cost proxy. Close positive-mix, scale, seed, optimizer, loss, duration, and
nearby gate-form rescue. The next paper candidate must test higher-capacity,
learned recurrent memory/address/state computation rather than another scalar
aggregate prior.

### P-GDN3-006 Dual-State Residual Expert Boundary

P-GDN3-006 tests the higher-capacity alternative directly. Every main
D256/H8/K32/V32 position-QK block receives an independent
D128/H8/K16/V16 pinned official-GDN2 expert with private address, update, and
terminal state, adjacent-layer native FutureSeed, and a zero-initialized
residual readout. The formal migration preserves the complete parent function
and main recurrent states exactly. The strict contract establishes gradients
through every one of the 12 expert cores and all 11 receiving auxiliary
FutureSeed gates.

The additional state is recruited, not dormant. Loop5 expert residual relative
RMS is `0.010408`, terminal-state RMS/board std is
`1.271940/0.171060`, auxiliary FutureSeed incoming RMS is `0.457982`, and
expert/main cosine is finite and nonzero. Yet hard51-64 macro loop5 exact stays
`0.000651`, mixed exact stays `0.025391`, and official51-55/56-60/61-64 blank
changes by `-0.003616/-0.006143/-0.002625`. Same-board loop3-to5 correction
weakens on both 56-60 and 61-64. The CE change
`0.858617->0.855923` again fails to predict board closure.

The systems boundary is stronger than the quality boundary. A 21.6% parameter
increase and 25% recurrent-state increase reduce throughput
`15.497->7.502` effective boards/s; matched elapsed time rises `106.57%`, while
peak allocation rises `46.79%`. The paper must not claim that a cold parallel
state expert improves GDN3 scaling. The reusable result is that simply
duplicating an independently addressed recurrent stack behind a zero-init
readout is both slow and insufficient for closure, even when every new path is
active. Close expert width/count/address/readout-scale and duration rescue. A
successor must let carried memory condition current addressing and state edits
directly, so added computation participates in a closed-loop update rather
than remaining a parallel residual subsystem.

### P-GDN3-007 Closed-Loop State-Feedback Boundary

P-GDN3-007 tests that proposed closed loop without adding another state or
scan. Each receiving position-QK layer reads its inherited KxV state with the
normalized query and uses a head-shared V32->16->128 controller to perturb K,
V, erase, and write inputs before the unchanged pinned official GDN2 chunk.
The zero-initialized migration is bit-exact and adds only 30,720 parameters.
The strict contract proves all 11 receiving paths, state dependency, head
equivariance, official-kernel provenance, and two-stage gradients.

The controller is strongly recruited: loop5 state-read RMS/board std is
`0.725926/0.013616`, residual relative RMS/token std is
`0.018716/0.007774`, and K/V/erase/write relative changes are
`0.038044/0.009582/0.005715/0.006785`. Nevertheless hard51-64 macro loop5
exact stays `0.000651`; mixed exact regresses `0.025391->0.023438`; and
official61-64 blank drops `0.001435`. Same-board loop3-to5 correction weakens
on both 56-60 and 61-64. A lower CE (`0.855740` versus `0.858617`) again does
not predict global closure.

The cost gate passes but is not free: throughput falls
`15.497->12.858` effective boards/s, elapsed rises `20.52%`, and peak
allocation rises `10.70%`. The paper should therefore state a sharper negative
boundary: conditioning precomputed update tensors on the layer's inherited
state is insufficient even when the feedback is active and affordable. It
still wraps the same single-pass linear delta transition and reads only the
incoming state, not the live state evolving within the token scan. The next
GDN3 hypothesis must change the scalable recurrent transition itself, with a
falsifiable advantage over this pre-scan controller; do not tune controller
hidden size, output scale, target subset, sharing, or continuation duration.

### P-GDN3-008 Post-Scan Consolidation Boundary

P-GDN3-008 moves beyond P007's fixed pre-scan read by adding a second
same-order pinned-official transition on producer layers0-10. It starts from
the first terminal state, reuses first-pass payload/erase/write gates, and uses
a zero-initialized head-shared V32-to-K32 projection of first-pass token outputs
as the correction address. The parent output is unchanged, and migration is
bit-exact for both zero and nonzero incoming states. The strict contract proves
23 official chunk backward paths and gradients through all 11 consolidation
projections.

The transition is active and moves local accuracy in the intended direction.
At loop5, correction-K relative RMS/board std/token std is
`0.120164/0.004845/0.036855`; official51-55/56-60/61-64 blank accuracy improves
by `+0.001325/+0.001853/+0.004609`. Hard macro exact, however, changes only
`0.000651->0.001302`, mixed exact remains `0.025391`, and same-board loop3-to5
correction weakens on 61-64. The lower CE (`0.855832` versus `0.858617`) again
does not imply exact closure.

The state dynamics reveal why a fixed extra sweep is not a scalable answer.
Terminal residual relative RMS rises from `1.1981e4` at loop1 to `4.7109e8` at
loop5, while loop5 state board std reaches `1.0649e10`. Downstream unit
normalization keeps inference finite, but it hides an explosively growing
memory representation. Throughput falls `15.497->12.151` effective boards/s;
elapsed and peak allocation rise `27.54%` and `22.95%`.

The paper may claim a qualified positive boundary: post-scan state transition
depth improves blank-token accuracy across every hard range. It must also state
the decisive negative result: an unconstrained repeated-write sweep neither
closes boards nor preserves stable state geometry. Close projection source or
scale, decay/gates, scan count, layer subset, and training rescue. The next
general GDN3 mechanism should make the live transition contractive or
normalized by construction while increasing useful state capacity; another
fixed scan, transfer router, pre-scan residual, scalar prior, or parallel expert
is not warranted.

### P-GDN3-009 External Chunk-Composition Boundary

P-GDN3-009 tests a norm-preserving live-state transport rather than another
pre/post-scan residual. It exposes the existing 81-token official GDN2 call as
its native 64+17 chunks and inserts a content-conditioned orthogonal rotation
of the KxV state between them. The controller adds 24,960 parameters and no
state, tokens, or additional recurrent transition.

The strict contract establishes a useful forward result: at zero angle, the
externally split path is bit-exact to the unsplit parent for full output and all
12 terminal states, including synthetic nonzero incoming states. The required
training graph does not compose, however. Every final terminal state exposes
one `ChunkGDN2FunctionBackward`, not the registered two. The experiment closes
before a step3001 probe or formal continuation.

This is an implementation boundary worth retaining in the paper plan. An
official recurrent operator can be forward-composable across cache boundaries
without exposing the state-gradient chain required to learn an inserted live
transition. Future mechanisms must either act within one differentiable
official call or prove an official-compatible initial-state gradient contract
before spending continuation compute. Do not present P009 as a Sudoku score or
rescue it with a custom backward, relaxed graph assertion, or alternate chunk
boundary.

### P-GDN3-010 Physical-Microstep Identity Boundary

P-GDN3-010 tests whether update rank can be increased inside one official
autograd function by inserting an auxiliary write before every parent token.
Both auxiliary K-residual and V projections are zero-initialized; auxiliary
decay/erase/write are fixed to `0/0/1`, so the intended mathematical update is
an identity before the unchanged parent update. The design adds no persistent
state or second recurrent core and keeps one pinned-official call per layer.

The strict contract rejects the migration before training. The 162-physical-
step path changes full-model output by max absolute `0.04559326171875` versus
the 81-step parent even with both auxiliary projections exactly zero. Thus a
mathematically zero write is not a bit-exact no-op after changing the physical
sequence geometry of the chunked operator. No gradient, step3001 or benchmark
claim is available.

This is a second systems boundary adjacent to P009. External calls preserve
forward values but lose the registered cross-call graph; a single expanded
call preserves the graph count but loses exact parent numerics. A future
in-token rank mechanism must either preserve the original 81-step geometry or
provide an explicitly audited kernel-level transition and migration contract.
Do not present P010 as a Sudoku score or tune auxiliary gates, order, count,
rank, scale, tolerance or duration.

### P-GDN3-011 Production-Precision Boundary

P-GDN3-011 tests token-dependent cross-head write routing while preserving the
original 81-token single official-GDN2 call. A shared V32-to3 descriptor
defines a two-dimensional head plane and a zero-initialized invariant angle;
the intended operation is an orthogonal rotation of V across the eight live
recurrent heads. It adds 1,152 parameters and no state, token or scan.

The pushed clean implementation passes its strict CUDA contract. Zero-angle
full output and all 12 terminal states are parent-exact, including synthetic
nonzero incoming states. All 12 official backward paths and two-stage
angle/plane gradients are present. Direct synthetic routing is
head-permutation equivariant and preserves FP32/BF16 payload norm inside the
registered tolerances.

The exact step3001 production probe exposes a different systems boundary. All
12 routes activate, with loop5 angle abs `0.059469` and routed-V relative RMS
`0.037769`, while plane errors remain below `1.5e-6`. But production FP32 norm
max error is `5.1444e-4`, above the fixed `1e-4` gate. Explicit FP32 tensors
were passed through operations still governed by the surrounding CUDA
autocast context; the direct checker ran outside that context. This is not a
Sudoku result and no formal continuation exists.

The paper should retain the methodological boundary: exact migration,
gradients and algebraic invariants must be tested under the same precision
context that training uses, after the mechanism activates. Do not present P011
as evidence against cross-head memory routing, and do not repair autocast,
plane, angle, target or tolerance within this registered experiment.

### P-GDN3-012 Signed-Transition Spectrum Boundary

P-GDN3-012 keeps the original 81-token, one-call pinned-official GDN2 path and
tests a missing transition family rather than another readout. A zero-init,
head-shared V32-to-K32 adapter changes erase from the parent's `[0,1]` range to
`b'=clamp(b+tanh(Wv),0,2)`. Values above one give a content-dependent negative
key-direction eigenvalue while exact zero initialization preserves parent
output and all carried states. The candidate adds 12,288 parameters and no
state, token, scan, second core or task logic.

The contract, one-step production probe and formal endpoint all pass their
identity, official-kernel, gradient, activation, stability and cost gates.
At mixed loop5, all 12 adapters are active, erase residual relative RMS is
`0.303028`, `8.707%` of channels use `b'>1`, the effective range is
`[0,1.84375]`, and terminal RMS remains bounded at `7.200136`. This is direct
evidence that signed recurrent modes are trainable in the pinned operator.

They are not sufficient for global closure at this checkpoint. Hard51-64
macro loop5 exact stays `0.000651`, mixed exact falls
`0.025391->0.023438`, and hard-range blank deltas are
`+0.000143/-0.001476/+0.000092`. Same-board loop3-to5 correction strengthens
only on 51-55 and weakens on 56-60 and 61-64. Train CE is effectively
unchanged. Throughput falls `15.497->13.287` effective boards/s; elapsed and
peak allocation rise `16.64%` and `6.04%`, inside the fixed ceilings.

The paper may claim a clean negative boundary: monotone nonnegative retention
is not the sole bottleneck, because a substantially used, stable negative-
retention regime does not improve full-board exactness. Do not tune adapter
source, nonlinear map, bound, scale, sharing or training duration. The next
general mechanism should increase address-state interaction or memory
organization rather than revisit transition spectrum strength.

### P-GDN3-013 Paired-Bank Optimization Boundary

P-GDN3-013 tests aligned recurrent address capacity without adding tokens,
scans or a second core. Each parent H8 position-QK head is duplicated inside
the same pinned-official call, giving an H16 base/companion state. Two
head-shared V32-to-K32 projections can differentiate companion Q/K, while a
per-head read gate returns companion output to the unchanged base path. The
candidate adds 24,672 parameters and 98,304 recurrent-state values per board.

Strict contract R2 establishes a useful systems result: an H8-to-H16 official
call preserves full model output, all base states, duplicated companion states
and nonzero incoming-state behavior bit-exactly. It retains one official
backward path per layer, and synthetic gate opening gives nonzero Q/K gradients
and companion-state differentiation while the base path remains exact.

The production optimization route nevertheless fails its registered one-step
gate. At exact step3001 the read gate reaches `0.001048`, but Q/K projection
weights, Q/K residuals and paired-state residuals remain exactly zero. With
both address projections and read gate zero-initialized, the first optimizer
step can train only the read gate; address learning is second order in this
composition. No formal Sudoku continuation exists, and the tiny 8-board probe
must not be interpreted as a score.

The paper should retain this as an optimization-accessibility boundary, not as
evidence against multibank recurrent memory. Exact migration plus a synthetic
two-stage gradient demonstration is insufficient when the registered
production checkpoint has not activated the actual state-diversifying path.
Do not repair the result with a second probe step or changed initialization.

### P-GDN3-014 Coupled-Row Identity Boundary

P-GDN3-014 removes P013's serial read gate and instead tries to double each
head's address-row axis inside one pinned-official transition. The proposed
K64 call repeats the normalized parent query and erase/decay fields, appends a
zero-initialized extra write-address bank, pads incoming state with zero rows,
and preserves H8/V32, tokens, scans and core count. This would expose the new
address projection to a direct first-order task gradient if exact migration
held.

It does not. On exact pushed SHA `078fe7e`, the strict GPU1 contract reaches
the zero-init end-to-end comparison and finds that external K32 normalization
plus a K64 zero-row bank changes full model output. This is a parent-function
identity failure before any production step, activation metric or Sudoku
score. No step3001 checkpoint or formal continuation exists.

The paper should use this as a numerical/operator boundary, not evidence
against address capacity: enlarging a recurrent row axis is not automatically
an exact embedding of the smaller pinned operator, even when added parameters
and incoming state are zero. Future scalable-memory candidates must prove the
full parent function under their actual physical state geometry; normalization
and scale arguments at the equation level are insufficient. The registered
normalization/map/scale/row-count family is closed without rescue.

### P-GDN3-015 Bi-Axis Value-Lifetime Boundary

P-GDN3-015 asks a genuinely different recurrent-memory question: official
GDN2 has persistent forgetting on K/address rows, while V channels only receive
a write-time gate. It adds eight grouped V-axis lifetimes per head and uses an
exact moving frame so the original 81-token pinned-official call and backward
graph remain unchanged. The clean pushed implementation passes strict parent
identity, nonzero-state identity, direct-gradient, official-kernel, recurrence-
reference and equivariance contracts with exactly 196,608 new parameters.

The exact one-step production probe falsifies the implementation before a
quality run. All 12 paths activate with loop5 log-decay magnitude `0.032549`
and nonzero group/board/token variation, but the cumulative scale minimum falls
to `0.000285` and inverse scale reaches `3505.29`; restored-state relative RMS
is `0.998929`. This is not evidence that V-axis lifetime is useless. It is
evidence that a sequence-global inverse moving frame is too ill-conditioned to
serve as its training parameterization, even on 81 tokens. No hard-Sudoku score
is claimed and no map/group/init/precision/scale rescue is permitted. Any future
paper candidate must implement an intrinsically bounded, chunk-local V-axis
transition rather than tune this wrapper.

### P-GDN3-016 Stable Bi-Axis Scientific Boundary

P-GDN3-016 removes P015's numerical confound without changing the scientific
question. It predicts a bounded grouped V-coordinate potential, funds the
common nonexpansive component through additional K contraction, transforms the
write frame, and uses exactly one unchanged pinned-official GDN2 call per
layer. The candidate adds 196,608 parameters and no recurrent state, token,
scan, second core, reverse traversal or task logic.

The strict contract and exact step3001 probe establish that this is a valid
implementation of the intended family. Zero initialization preserves full
output and all 12 carried states bit-exactly, including nonzero incoming
states; all 12 projections receive direct gradients; the direct physical
Bi-Axis recurrence agrees; and the frame and inverse remain in `[0.25,4]`.
The formal endpoint remains activated and bounded: mixed loop5 potential, physical V
decay and common K contraction are `0.564159/0.351812/0.351764`, while frame
min/max/inverse is `0.250029/3.998026/3.999530`.

Stability does not recover quality. Hard51-64 macro loop5 exact falls
`0.000651->0`, mixed exact falls `0.025391->0`, and official51-55/56-60/61-64
blank accuracy changes by `-0.220564/-0.181016/-0.297518`. On the matched
61-64 bank, all 256 candidate boards have more loop5 errors than control and
loop1-to5 correction shrinks `5.426->1.441` cells. CE degrades
`0.858617->1.498811`. Elapsed and allocation overhead are `+28.60/+21.55%`,
also outside their fixed ceilings.

The paper can therefore make a stronger negative claim than P015 allowed:
persistent grouped V-axis lifetime remains insufficient after its moving-frame
conditioning problem is removed. The model saturates the bounded gauge, and
the common K contraction required for nonexpansive physical V decay destroys
useful parent memory. This result closes the registered cap/group/map/max and
training-rescue family; it should not be softened into a tuning recommendation.

### P-GDN3-035 Chunk-Local Bi-Axis Boundary

P-GDN3-035 removes P016's common-K-contraction confound. It restores physical
state every 64 tokens and chains sixteen unchanged pinned-official chunks, so
the eight grouped V lifetimes are directly nonexpansive while every local
moving frame and inverse stays in `[0.25,4]`. A corrected GPU VJP also proves
that gradients cross chunk boundaries into earlier V, K-decay and incoming
state; the old P009 graph-count objection was a diagnostic error, not a kernel
limitation.

The fixed same-process L1024 result is decisively negative. Control/candidate
balanced/future/past/joint accuracy is
`0.09975/0.0930/0.1065/0` versus `0.01375/0.0120/0.0155/0`, while errors rise
`3601->3945`. Both layers are active and bounded, with decay magnitudes
`0.006751/0.014278`, so failure is not dead parameters or state explosion.
Conditional wrong-key swaps fall `0.131630->0.038530` only because general
retrieval collapses. The unfused reference also costs `7.356x` per warmed step.

Together P015/P016/P035 close grouped persistent V-axis lifetime as the missing
GDN2 binding mechanism: unstable global framing, funded bounded framing and
direct bounded physical decay all fail for independent reasons. Future work
should preserve one coherent erase/write/read ownership space and improve its
address separability or transition rank, not tune another lifetime gauge.

### P-FS3-004 Cross-Layer Basis-Alignment Test

P-FS3-004 tests a missing FutureSeed mechanism boundary rather than another
content or strength residual. Native FutureSeed transfers a producer KxV state
into a receiver whose recurrent K/V coordinates were initialized and learned
independently. Existing head gates and unit-RMS normalization regulate transfer
magnitude but cannot align those bases.

The candidate gives each adjacent layer edge and head independent
skew-symmetric K32 and V32 generators. Cayley transforms transport state as
`R_K S R_V^T`, preserving Frobenius geometry without adding state, tokens,
scans, recurrent cores, or task logic. Applying `(R-I)` residuals makes zero
generators a bit-exact parent identity with direct first-order gradients. The
fixed parameter delta is 87,296.

This is distinct from FS3 content/update routers, hidden readouts, within-layer
state rotations, and physical state-width expansion. Contract and production
evidence must establish exact migration, direct gradients on all 11x8 K/V
routes, bounded full-model state geometry, and one-step activation before any
Sudoku score exists. A clean negative result would show that coordinate
misalignment is not the current hard-board bottleneck; a positive result would
identify learned cross-layer state transport as a concrete FutureSeed advance.

The contract and one-step production boundary now pass on final pushed SHA
`0335534`. All edge minima are active: K/V rotation relative RMS
`0.005694/0.005998` and state residual relative RMS `0.014258`; board/head
variation is nonzero. FP32/storage norm error is `1.49e-6` and orthogonality
error is `2.21e-6`, so the real BF16 production path preserves the registered
geometry. This is valid implementation and activation evidence for learned
cross-layer coordinate transport.

It is not performance evidence. The step3001 probe used eight boards per tiny
diagnostic slice, and no matched step3000-to3100 hard51-64 continuation was
launched. The paper may state that orthogonal basis transport is feasible,
first-order trainable and stable, but must not state that it improves Sudoku
exactness, cost, or FutureSeed quality. The automated sequence stops here; any
future matched test is a separate explicit decision.

### P-GDN3-017 Raven Allocation Control-Plane Boundary

P-GDN3-017 tests a Raven/GDN hybrid without replacing the stronger recurrent
core. Raven contributes only token-dependent allocation over eight fixed K-row
slots; position-QK GDN2 remains the only live transition, and native terminal
FutureSeed remains the only cross-layer state path. The conserved
`8*softmax` allocation routes K by its square root and nonpositive row decay
directly, adding 196,608 parameters but no state, token, scan, second core,
reverse pass or task logic.

The implementation boundary is clean. On exact pushed SHA `e884b7d`, the
strict A100 40GB contract preserves zero-init full output and all 12 terminal
states bit-exactly, including nonzero incoming states; all 12 routers receive
direct gradients; allocation conservation, slot equivariance and protected
rows hold; and every layer executes one pinned-official GDN2 backward. The
exact step3001 production probe is active, diverse and stable, so the formal
negative cannot be attributed to migration or a dead route.

Training drives the allocation toward nearly hard single-slot selection. At
loop5, normalized entropy is `0.765180`, allocation spans
`0.000145..7.983757`, K/g relative change reaches `0.508998/1.127739`, and the
fixed `<7.5` maximum-allocation bound fails. This high-amplitude routing does
not close more boards: hard51-64 macro loop5 exact remains `0.000651`, mixed
exact falls `0.025391->0.023438`, and official blank deltas are
`-0.000179/-0.001510/-0.008455`. Same-board late correction improves only on
51-55 and weakens on 56-64.

The paper can therefore state a precise negative result: content-dependent
allocation over a fixed GDN K-row bank is trainable and computationally viable
(`12.622` effective boards/s on A100 40GB), but allocation alone is not the
missing hard-closure mechanism at this parent. The eight-slot softmax family is
closed without temperature, slot-count, top-k, scale or training rescue. A
future Raven/GDN hybrid must change stable recurrent state organization or
state interaction, not merely sharpen or soften this router.

### P-GDN3-019 Persistent Raven Write-Control Boundary

P-GDN3-019 tests the stronger Raven/GDN composition left open by P017. Instead
of using Raven only as a stateless allocation rule, every position-QK GDN2
block receives a compact pinned-official D64/H4/K16/V16 Raven with S16/top1,
its own persistent sparse state, and normalized adjacent-layer terminal-state
transport. A zero-initialized adapter injects retrieved Raven content into the
main GDN2 V write payload before the unchanged dense transition. This adds
643,344 parameters and 24,576 controller-state values while leaving the main
GDN state, native FutureSeed and benchmark contract unchanged.

The implementation result is clean. Exact pushed SHA `9abc292` passes the
strict A100 40GB contract with 12 official Raven and 12 official GDN2 backward
paths, exact parent output/main-state identity including nonzero incoming
states, finite gradients, state dependency and exact parameter/state deltas.
The step3001 probe and step3100 endpoint keep all controllers active. At loop5,
V residual relative RMS is `0.066201`, minimum slot entropy is `0.670635`,
maximum slot mass share is `0.370819`, and maximum main-state RMS is `11.8991`.
The negative result is therefore not caused by a dead adapter, collapsed sparse
bank, unstable state or backend fallback.

The quality result is nevertheless binding. Official 51-55 exact doubles from
one to two solved boards in the 512-board range sample, but hard51-64 macro
exact rises only `0.000651->0.001302`, far below the registered `+0.02` floor.
Mixed exact regresses `0.025391->0.023438`; 61-64 blank accuracy falls by
`0.001892`; and matched 61-64 loop3-to5 correction weakens
`0.1836->0.1172` wrong cells. CE also rises `0.858617->0.862443`.

Persistent sparse retrieval is also costly in this composition. Effective
throughput falls `15.497->7.451` boards/s: `+107.99%` elapsed overhead versus
the fixed `<60%` gate, despite only `+5.60%` parameters. Peak allocation rises
`17.54%` and passes its separate memory ceiling. The paper can therefore state
a precise boundary: an organic Raven control plane plus a dense GDN data plane
is executable, trainable and geometrically stable, but persistent retrieval
does not supply the missing short-horizon exact-closure mechanism at this
parent and more than doubles recurrent compute. This closes S16/top1 Raven
write control without slot, top-k, width, injection or duration tuning; it does
not claim that every long-context Raven/GDN composition is impossible.

### P-GDN3-020--023 Directional-MQAR Address Boundary

The historical directional-MQAR L1024 carrier did not reproduce on the current
A100 software/runtime stack: native FutureSeed reached balanced/joint only
`0.1735/0`, versus the historical `0.7475/0.339`. P020 therefore failed its
formal carrier-admission gate. Its one fixed exploratory Log-SPD candidate is
still mechanistically informative: a bounded learned Q/K metric raises
balanced accuracy to `0.48225` without changing state size or scan count. This
is evidence that address geometry is a real lever, but not a formal benchmark
win and not sufficient for transfer to Sudoku.

The three discriminators around that signal are negative. A second live edit
in P021 activates but destructively overwrites the shared state, ending at
balanced `0.00975`. A same-byte H8/K16/V32 bank organization in P022 preserves
4,096 state values and one official scan but reaches only `0.27975`, with
`2.55x` fit time. The fixed P023 combination of H8/K16/V32 and Log-SPD reaches
balanced/future/past/joint `0.3090/0.3225/0.2955/0.004`, below P020 by
`0.17325`, and misses the fit gate at `1.670x` control.

The error transition sharpens the mechanism boundary. P020 leaves fewer total
errors but `94.16%` of them are correct values assigned to the wrong key. P023
reduces that fraction to `48.19%` while increasing total errors to 2,764. The
model can learn much of the value set, yet static metric conditioning and bank
factorization do not establish reliable key-value binding. This closes extra
rank-one writes, same-byte head/K reshaping, and their fixed Log-SPD
combination. The next credible GDN3 direction, if research resumes, must be a
stable collision-aware associative state organization evaluated from scratch
on a reproducible binding carrier, rather than another static conditioner,
router, bank-shape change or zero-init Sudoku graft.

### P-GDN3-024 Dual-Hash Binding Test

P024 isolated one collision-aware topology without adding capacity. Every
existing H4/K32 address is split into two independently normalized K16 factors
and mapped through a fixed compact bilinear product sketch, while V and every
official GDN2 transition remain unchanged. Thus a strong retrieval requires
agreement in both factors, but
parameters, 4,096 state values and official scan count remain exact. This is
distinct from P022's costly H8 bank split and P013's separately gated companion
state.

The endpoint is a sharp negative. Both hash factors remain active and balanced,
and wrong-key valid-value swaps fall from P020 `94.16%` to `3.03%` of errors.
Balanced accuracy nevertheless collapses from `0.48225` to `0.0115`, with
future/past `0.0105/0.0125` and joint exact zero. The multiplicative sketch has
not separated useful bindings; it has destroyed the trainable linear address
channel strongly enough that almost every query becomes an arbitrary wrong
value. Peak allocation also reaches `1.27185x` control despite zero parameter
and state delta.

This closes fixed compact product binding without hash/partition/permutation,
soft interpolation, metric or training rescue. A collision-aware successor
must retain the native linear GDN2 state as a base path and introduce a bounded
correction memory whose contribution can be falsified independently.

### P-GDN3-025 Committed-Delta Correction-Memory Boundary

P025 preserves the parent H4/K32/V32 state and captures the exact committed
`v_new` residual from the pinned-official transition. A second official scan
writes that evidence into an independent H4/K16/V32 state. Fixed pairwise and
learned Cayley semi-orthogonal K32-to-K16 bases receive identical state bytes,
so their comparison isolates address organization from raw capacity.

The learned basis is a real mechanistic positive: balanced accuracy rises from
`0.01875` with the fixed basis to `0.36175`, basis movement is large while row
orthogonality error stays below `6e-7`, and wrong-key valid-value swaps fall
from P020 `0.94157` to `0.57618` of errors. The correction path is active,
bounded and within every preregistered systems gate.

The architecture result is still negative. Learned balanced/joint exact are
`0.36175/0.011`, below P020's direct main-state Log-SPD result by
`0.12050/0.033`, and all absolute quality gates fail. A learned address system
can recover much of the information destroyed by fixed compression, but an
independent compressed committed-delta bank is inferior to organizing the
main recurrent address itself. The paper should report this as causal evidence
for address geometry and a boundary against separate correction memory, not as
a GDN3 quality win.

### P-LOOP-001 Native FutureSeed Credit-Conflict Diagnostic

After the address/state families closed, P-LOOP-001 tested a different
explanation without changing logits or training: equal five-loop CE may send
opposed credit specifically into native FutureSeed gates. On three fixed
official hard batches it computed each loop loss gradient for the 96 FS gate
parameters, GDN2 address parameters, GDN2 edit parameters, and the shared
shell from the exact D256/L12 position-QK step3000 parent.

The result is localized rather than global. FutureSeed early-versus-late
cosine is `-0.904763` on 51-55 and `-0.718710` on 61-64; both have `0.40`
negative gradient-pair fraction, cancellation `0.529537/0.698487`, and genuine
loop1-to5 wrong-cell reduction. The 56-60 batch is strongly aligned and does
not pass. In contrast, GDN2 address and edit gradients have no negative pairs
on any range, and their early/late cosine remains strongly positive. The
remaining optimization conflict is therefore not evidence for more recurrent
capacity or another address wrapper.

The full cosine matrices sharpen the claim: loop1 opposes the continuation
loops in the two admitted ranges, while loops3-5 are nearly collinear. This
admits one narrow training-only FutureSeed experiment that projects a
conflicting opening gradient away from the continuation direction while
preserving the baseline FS-gradient norm. It does not justify global loss
reweighting, GDN gradient surgery, per-head selection, or a hyperparameter
sweep. The diagnostic establishes a causal target; the matched endpoint must
still show that removing this conflict improves exact-board closure.

### P-GDN3-031 Direct Decoupled-Key Boundary

P031 is the clean-room direct test of the mathematical idea in
`yanghu819/GDN_decouple_k`: use one normalized key to erase and another to
write/read. Unlike P030's independently initialized contractive transition,
P031 retains native GDN2 decay and channel-wise erase/write gates and starts
exactly on the native function. The erase projection and Triton convolution
are copied bit-for-bit from the write key; a single pinned-official DPLR scan
implements `DS-k_e[(b*k_e)^TDS]+k_w[(w*v)^T]`. The strict contract verifies
exact tied output/state parity, both key gradients, official backward/kernel
provenance, zero state/scan delta, and bounded transition geometry.

This stronger isolation produces a sharper negative than P030. On fixed
directional MQAR L1024, balanced/future/past/joint are
`0.011/0.008/0.014/0`, versus current native FutureSeed
`0.30625/0.3115/0.3010/0` and historical `0.7475/0.7415/0.7535/0.339`.
Errors rise from 2,775 to 3,956. The key branches do not remain useful
specialists: layer cosine falls to `0.0483/0.0600`, while key relative RMS rises
to `1.3796/1.3711`. The state remains numerically bounded, so this is not an
explosion or inactive-module result.

The apparent swap-fraction improvement is diagnostic of collapse. Conditional
wrong-key swaps fall `0.458018->0.035642`, but only 44 of 4,000 queries are
correct and 3,657 predictions are just tokens 165 or 177. Direct decoupling has
made erase nearly orthogonal to the rows populated by write/read, destroying
the usable linear address channel. The paper should therefore report total
errors and prediction concentration beside every swap fraction. P031 closes
unconstrained erase/write-key decoupling without angle, tie, scale,
regularization or training rescue and does not authorize Sudoku transfer. It
does not close mechanisms that preserve a shared base address while adding a
separately bounded correction topology.

### P-GDN3-032 Native K64 Address-Capacity Boundary

P032 asks whether P031 failed because the coherent native key namespace was
simply too small. It trains unmodified pinned-official GDN2 from scratch at
D128/L2/H4/K64/V32, preserving one normalized erase/write/read key, one scan,
and native FutureSeed. This doubles recurrent state from 4,096 to 8,192 values
per layer without a cache, wrapper, second bank, or task-specific operation.

The strict contract and cost gates pass, but quality collapses. Balanced/
future/past/joint accuracy is `0.0455/0.0445/0.0465/0`, versus contemporaneous
K32 `0.30625/0.3115/0.3010/0` and historical K32
`0.7475/0.7415/0.7535/0.339`. Wrong-key valid-value swaps fall from 1,271 to
294, yet total errors increase from 2,775 to 3,818. The paper should use this
as a second independent warning that a lower conditional swap fraction can
mean the model stopped retrieving rather than improved binding.

K64 also exposes an optimization boundary: its added address rows and native
FutureSeed gradients are active, but validation accuracy stays near chance
through epoch4 and reaches only `0.02725` at epoch6. Historical K32 is already
`0.71575` at that point. More coherent address dimensions therefore dilute or
delay learnability under the fixed budget; key-Gram anisotropy is not evidence
for raw K expansion. P032 closes K48/K96 and training rescue and does not
authorize Sudoku transfer.

### P-DIAG-CARRIER-002 Historical Carrier Reproduction

The historical directional-MQAR L1024 native FutureSeed score of `0.7475`
could not be reproduced despite matching source `77e5539`, data hashes,
initialization hash, pinned FLA/Zoology stack, A100 architecture and original
single-process order `causal_gdn2 -> future_seed_gdn2`. The exact reproduction
reaches balanced/future/past/joint only `0.10825/0.0990/0.1175/0` at L1024.
Its validation curve lacks the historical epoch-4-to-6 jump.

The same process reaches `0.99525/0.9920/0.9985/0.981` at L64, ruling out a
broken import, data generator or FutureSeed path. The evidence instead exposes
high sensitivity of long-sequence optimization to runtime trajectory. The
paper must therefore treat `.7475` as descriptive historical evidence rather
than an absolute gate. New L1024 mechanisms require a contemporaneous
same-source, same-process control and a registered relative improvement; no
architecture claim may mix scores from these runtime conditions.

### P-GDN3-033 Shared Log-SPD Boundary

P033 isolates whether tying P020's bounded metric across layers can align the
state namespace used by native FutureSeed. It adds one shared 2,108-parameter
trace-free Log-SPD factor and otherwise preserves coherent erase/write/read
keys, state size and one official scan. The strict contract proves the two
layers share exact storage and separately contribute finite gradient while the
zero point is exactly native.

The metric activates and remains bounded, but balanced accuracy falls from the
contemporaneous native control `0.17325` to `0.1250`; errors rise
`3307->3500`, and warmed-step time is `1.2738x`. Shared smooth geometry is not
equivalent to aligning independently learned layer projections. Close this
constraint without rescue.

The control is itself a useful replication: it is within `0.00025` of P020's
native control `0.1735`. Thus P020's per-layer candidate at `0.48225` retains a
credible `+0.30875` relative mechanism effect after P-DIAG-CARRIER-002 retires
the historical absolute gate. A Sudoku transfer can test task generality
without rerunning or tuning the MQAR experiment.

### P-GDN3-034 Per-Layer Log-SPD Sudoku Transfer

P034 performs that transfer without changing the mechanism. Each of the 12
position-QK layers receives its own bounded trace-free Log-SPD Q/K metric,
adding 50,592 parameters and no state or scan. The strict official-FLA
contract, exact-resume probe, and endpoint all pass integrity. At loop5 every
metric is active and bounded: raw RMS is `0.01554`, metric delta is `0.2590`,
and eigenvalues remain in `[0.7413,1.4189]`.

The mechanism effect does not become board closure. Hard51-64 macro exact is
unchanged at `0.000651`, mixed exact regresses `0.025391->0.023438`, and
same-board loop3-to5 correction weakens on 56-60. Blank accuracy does improve
on 51-55 and 61-64 by `0.000644/0.005769`, showing that the learned geometry is
not inert, but it cannot close the remaining globally inconsistent cells.
Throughput also falls `15.497->9.337` boards/s because the per-layer matrix
exponential is paid repeatedly through the five-loop model.

The paper should distinguish relative carrier replication from cross-task
transfer. P020 remains credible evidence that layer-specific Q/K geometry can
help directional MQAR under a matched runtime; P034 shows that this lever is
not sufficient for Sudoku loop closure. Close static metric variants on this
parent. A successor must target a different causal boundary, especially
FutureSeed loop credit or the live recurrent transition, rather than retuning
the metric.

### P-GDN3-036 Anchored Dual-Key Boundary

P036 tests the strongest ownership-preserving interpretation left by the
external decoupled-key idea. It retains the native coherent write/read key and
permits erase only a fixed-radius tangent correction, giving a structural key
cosine floor near `0.8944`, exact tied-parent initialization and one
pinned-official DPLR scan. The strict contract passes and both layer corrections
become strongly active without state explosion.

That constraint is still insufficient. Contemporaneous control/candidate
balanced accuracy is `0.36025/0.01375`, joint exact is `0.014/0`, and errors
rise `2559->3945`, even though mean erase/write cosine remains about `0.91`.
The lower conditional wrong-key swap fraction again reflects broad retrieval
collapse, not better binding. Together P031 and P036 close unconstrained and
bounded-tangent decoupled erase keys: useful GDN2 memory needs exact coherent
read/write/erase ownership, not merely nearby key directions. Do not tune the
angle, cap or regularizer; successors must alter a different scalable state or
transition boundary.

### P-GDN3-037 Committed-Residual Target Boundary

P037 keeps the exact native read/write key, every GDN2 projection, one
K32xV32 state and one scan, but replaces coordinate-wise erase plus V-wise
write with one scalar-gated committed prediction residual. The pinned-official
DPLR mapping is exact, byte-identical initialization and all gradients pass,
and the trained transition remains contractive with spectral norm below one.

The result rejects the simplification. Balanced accuracy falls
`0.36625->0.01525`, joint exact `0.002->0`, and errors rise `2535->3939`, even
though the committed residual is strongly active at roughly `1.07` relative
RMS and every cost gate passes. The much lower conditional wrong-key fraction
is another retrieval-collapse artifact. The paper should treat native GDN2's
K-coordinate erase and V-coordinate write gates as a useful learnable control
surface, not an incoherence to average into one beta. Close scalar residual
targets and preserve these degrees of freedom in future state organizations.

### P-GDN3-038 Content-Partitioned Full-State Boundary

P038 tests a genuinely different persistent topology: each native H4 head gets
two complete K32xV32 live states, and one learned content hash jointly routes
bounded erase/write admission and read weighting while preserving one official
scan per layer. The mechanism is not inert. Both routers specialize, mean
slot-state cosine falls to `.8655/.7237`, and conditional wrong-key swap
fraction improves `.76383->.59393`.

The topology still loses the task. Contemporaneous control/candidate balanced
accuracy is `.4940/.36525`, joint exact `.041/0`, and total errors
`2024->2539`. The shared hash reduces one error subtype but delays learning and
removes useful retrieval cases; some examples also leave the two slots almost
identical. This separates address-collision evidence from a viable solution:
global token-to-slot assignment is too coarse even when each slot preserves a
full native GDN2 state. Close slot count, routing temperature and training
rescues. A successor should preserve per-binding coherent ownership without
selecting a whole global trajectory for each token.

### P-FS2-010 Producer-Native Readout Is Not FutureSeed Closure

P-FS2-010 isolates a plausible coordinate-mismatch explanation for native
FutureSeed. The transferred KxV state remains the official initial state of
the receiving GDN2 scan; a single extra edge decodes that state with the
producer's own Q/output interface and fuses it into receiver hidden tokens.
The strict contract proves exact parent identity, two official GDN2 backwards,
zero new recurrent state or scans, and a fixed +12,288-parameter increment.

The negative result is mechanistically sharp. The readout activates strongly
and reaches its bounded residual cap, yet balanced accuracy falls
`.17475->.0280` and total errors rise `3301->3888`. Same-trained edge-off
accuracy is also `.0280`, so the candidate did not acquire a useful causal
readout; its backbone co-adapted to a destructive shortcut. The zero
transferred-state RMS board std is not evidence of identical content because
native FutureSeed normalizes each board to fixed RMS. Record this as a
diagnostic-design error and use pairwise content statistics in future runs.
The valid conclusion is narrower: a strongly active producer-native hidden
residual does not improve binding and can derail learning. Future FS2 work
must improve what is carried or credited, not add a stronger hidden residual
around the same terminal state.

### P-GDN3-042 Midpoint-Coherent Q/K Boundary

P042 directly tests whether drift between the native read-query and write-key
maps causes directional binding errors. It preserves their midpoint and learns
only a bounded per-coordinate scale on their difference. The K32xV32 state,
coherent erase/write/read ownership, one pinned-official scan and native
FutureSeed remain unchanged; the candidate adds exactly 256 parameters.

The strict contract passes and the mechanism is unambiguously active, but the
quality result is negative. Control/candidate balanced accuracy is
`.17475/.05725`, future/past accuracy is `.1610/.1885` versus
`.1000/.0145`, joint exact remains zero, and errors rise `3301->3771`.
Although wrong-key swaps among errors fall `.23326->.08778`, that change is a
broad-retrieval-collapse artifact. The independent warmed-step ratio also
fails at `1.6108x`.

Together with P031/P036, this result separates two address claims. GDN2 needs
exact coherent read/write/erase ownership, and it also needs the full learned
Q/K differential. P020's positive Log-SPD result changes geometry inside both
native maps without shrinking that differential; P042 shows why this
distinction matters. The paper should not motivate GDN3 as Q/K alignment or
decoupling. The open target is a scalable address geometry or state
organization that preserves both native maps and lowers binding interference.

### P-GDN3-043 Biorthogonal Q/K Gauge Boundary

P043 asks whether the positive address-geometry signal from P020 can be
isolated without splitting key ownership or shrinking the learned Q/K
differential. A symmetric trace-free bounded generator applies inverse-
transpose dual coordinates to Q and K, preserving the unnormalized bilinear
pairing while allowing normalized address and state-row geometry to change.
The candidate adds 4,216 parameters, no state or scan, and keeps native
FutureSeed.

The strict contract validates the intended mechanism: all eight heads learn
different bounded factors, raw pairing error stays near `.00205`, production
replay is exact, and terminal states remain finite and variable. Nevertheless,
matched balanced accuracy falls `.04850->.01975`, future/past accuracy falls
`.05450/.04250->.01800/.02150`, and total errors rise `3806->3921`.
Wrong-key valid-value swaps fall, but only as general retrieval deteriorates.
The warmed-step ratio also fails at `1.9149x`.

The paper should therefore avoid claiming that an algebraically clean address
gauge resolves GDN2 binding. Together with P031/P036/P042, P043 shows that the
open problem is not solved by separating keys, forcing Q/K agreement, or
changing their coordinate gauge while preserving pairings. The next credible
GDN3 direction must change scalable live-state organization or the committed
transition while retaining native coherent ownership and learnable Q/K maps.

### P-FS2-011 Receiver-Native Terminal-Read Credit Boundary

P-FS2-011 tests receiver-native credit without changing FutureSeed inference.
Detached receiver queries read both the inherited producer state and the exact
official receiver terminal state; normalized all-token MSE updates only the
producer state path and receiver FutureSeed gate. The strict contract proves
native inference identity, exact official-kernel provenance, zero added
parameters/state/scans and nonzero intended gradients.

The mechanism is active but destructive. Control/candidate balanced accuracy
is `.12375/.01775`, future/past accuracy is `.12150/.12600` versus
`.01650/.01900`, joint exact remains zero, and total errors rise `3505->3929`.
Wrong-key valid-value swaps fall from `543` to `185` only because useful
retrieval collapses. The candidate remains cheap at `.9984x` independently
warmed step and `1.0653x` peak allocation, so implementation cost does not
explain the negative result.

The paper may use this as a clean self-distillation boundary: receiver-native
coordinates alone do not make a terminal-state teacher causal for binding.
Agreement with a detached receiver endpoint can reproduce its mistakes or
flatten evidence. Do not present lower conditional swap fraction as an FS2
gain, do not transfer this loss to Sudoku, and do not tune nearby loss variants.

### P-GDN3-045 Coherent Key-Spectrum Credit Boundary

P045 tests the narrow statistical interpretation left after functional key
splitting and Q/K alignment fail. It keeps one exact native coherent key and
the independent learned query, leaves inference byte-identical, and applies a
fixed training-only covariance credit only to the native K projection and
Triton convolution. The strict contract verifies exact parent behavior,
official recurrent kernels and selective auxiliary gradients.

The endpoint rejects isotropic keys as binding credit. Balanced accuracy falls
`.4830->.0130`, future/past both collapse, joint exact falls `.0370->0`, and
errors rise `2068->3948`. Endpoint effective rank is `25.423/12.352` and
anisotropy `3.006/4.779`, so task training moves the second layer opposite to
the intended geometry. The apparent swap reduction is caused by broad failure.

The paper should therefore separate two claims: learned address geometry is
causal, but Euclidean key isotropy is not its objective. Together with
P031/P036/P042-P044, P045 closes direct erase-key decoupling, bounded coherent
tangents, Q/K alignment/gauges/frames and key-whitening loss. Do not present
these as routes to GDN3 or transfer them to Sudoku; the open mechanism must
change a scalable live-state organization or recurrent transition while
preserving coherent ownership.

### P-GDN3-046 Stable-Token Address Boundary

P046 tests the last positive-looking source-separation interpretation of the
external decoupled-key idea. It preserves GDN2's coherent erase/write key and
independent learned query, but gives both native Q and K one shared token-only
address residual at both layers. Repeated occurrences of the same semantic key
therefore receive an exactly identical learned anchor despite contextual and
positional drift. The candidate adds 16,384 parameters and no state or scan.

This is a genuine but insufficient retrieval signal. Balanced accuracy rises
`.36625->.42450`, future accuracy rises `.35150->.43350`, and total errors fall
`2535->2302`. Joint exact reaches only `.012`, however, and wrong-key
valid-value swaps increase `1574->1717`, from `.62091` to `.74587` of errors.
The stable anchor helps identify the candidate value set while making the
remaining binding ambiguity more dominant. An independently warmed step also
costs `1.37958x`.

The paper should report this as the strongest surviving signal from the
decoupled-key investigation, but not as a GDN3 win. Direct key splitting,
ownership-preserving tangents, Q/K alignment/gauges/frames, isotropy credit and
stable token anchors are now bounded. The next architecture must alter live
state organization or the committed transition so it can preserve distinct
bindings, not add another address residual.

### Native Replay Reliability And Binding Diagnosis

P-REPRO-001 removes an important measurement ambiguity. Two sequential native
GDN2 plus FutureSeed runs load one serialized initialization, use identical
directional-MQAR data and warmup batches, and reset RNG immediately before
training. They finish with the same trained-parameter hash, identical
validation curves and `4000/4000` identical query predictions. Their endpoint
balanced/future/past/joint accuracy is `.494/.454/.534/.041` in both arms.

This permits strict matched architectural comparisons under the current
pinned BF16/Triton protocol. More importantly, `1546/2024 = 76.38%` of native
errors are valid values retrieved under the wrong key. Combined with P047's
value-set gain and P049's failed dense canonical companion, the paper can state
a narrow mechanistic conclusion: the unresolved problem is ownership-preserving
binding in superposed recurrent memory, not merely more payload capacity or
more future context. The next GDN3 claim must therefore be tested through a
scalable live state organization with explicit pair isolation before any
Sudoku transfer.

### Sparse Pair Slots Expose A Credit-Assignment Boundary

P-GDN3-050 tests explicit ownership rather than another dense address wrapper.
Each exact official GDN2 committed edit is routed into one of 16 factorized
key/value slots by a second pinned-official GSA scan, while native GDN2 and
FutureSeed remain intact behind zero-initialized gates. The strict contract
proves exact parent identity, official-kernel provenance, live gradients,
bounded state and exact parameter/state accounting.

The endpoint rejects this realization decisively. Balanced/future/past/joint
accuracy falls from `.494/.454/.534/.041` to `.04425/.051/.0375/0`, and errors
rise `2024->3823`. Wrong-key swaps fall `1546->255`, but this is conditional on
near-total retrieval failure. Only `15/10` of 16 slots remain used after
training, and elapsed/post-warm wall time rises to `3.277/3.251x` control.

The paper should use P050 to distinguish representation from learnability.
Pair-isolated memory can suppress one error category while introducing hard
allocation credit failure and destroying the base learner. The open GDN3 route
is an ownership-preserving, differentiable live transition that retains native
retrieval, not a larger discrete side bank or a tuned slot router.

### P-FS2-012 Read-Only Dual-Plane Boundary

P-FS2-012 isolates a concrete FutureSeed failure hypothesis: producer future
evidence and receiver causal writes normally occupy the same mutable KxV
matrix, so later writes might destroy cross-layer evidence. The candidate keeps
the inherited state read-only, starts a separate official receiver state from
zero, and lets receiver-native queries read both planes. It adds no parameters
or recurrent transition and passes exact edge-off, gradient, official-kernel,
immutability and equivariance checks.

The endpoint rejects overwrite as the primary binding explanation. Frozen
native control versus candidate balanced/future/past/joint accuracy is
`.494/.454/.534/.041` versus `.4445/.451/.438/.009`; errors rise
`2024->2222` and wrong-key valid-value swaps rise `1546->1749`. The protected
plane is nevertheless essential to the trained candidate: disabling only that
read drops balanced/future accuracy to `.211/.011`. Time cost also reaches
about `1.99x` despite zero new parameters.

The paper should state the distinction explicitly. FutureSeed can transfer
usable evidence without preserving which key owns that evidence. State
retention and receiver-native decoding are insufficient; the next claim must
alter the coherent live commit or introduce an end-to-end ownership mechanism,
while retaining the native retrieval path and linear-memory advantage.

### Causal Lag And Same-Direction Ownership Boundary

P-GDN3-051 tests whether directional MQAR's serialized `key,value` layout
creates a one-token phase mismatch in GDN2's coherent erase/write commit. It
adds one zero-initialized causal lag scalar per head, preserves a single
official scan and exact parent behavior at zero, and never decouples erase from
write. The strict CUDA contract passes.

The endpoint rejects this account. Native versus lagged balanced accuracy is
`.4940->.2020`, joint exact is `.0410->0`, and total errors rise
`2024->3192`. The lower wrong-key count is broad retrieval collapse. Layer 0
prefers modest positive lag while layer 1 learns negative lag, and disabling
the mechanism after training does not recover quality. This closes fixed
temporal shift as an explanation rather than merely one parameterization.

A prediction-topology audit sharpens the paper's mechanistic claim. Of `1546`
native valid-value swaps, `99.55%` remain within the same future/past class and
`99.61%` select the adjacent owner by write rank; `346` events are reciprocal
two-cycles. The unresolved error is therefore instance ownership within an
already recovered direction-specific value set. This distinguishes the next
target from more payload capacity, protected FutureSeed copies, a global key
split, or a fixed commit delay.

### Exact Edit-Weighted Interference Credit Is Too Destructive

P-GDN3-052 asks whether only high-energy official committed edits should avoid
recently occupied key directions. It leaves inference bit exact and adds no
parameters or state; training alone receives a causal window128 hinge credit
weighted by detached exact committed-edit RMS. The strict contract proves that
the loss is active, target free, scale invariant and isolated to native K
projection/convolution gradients.

The candidate does not preserve the base learner. Balanced accuracy collapses
`.494->.013`, joint exact `.041->0`, and errors rise `2024->3948`. Wrong-key
swaps fall `1546->148` only because almost all retrieval fails. The credit is
still active at the endpoint, while all measured cost ratios are above `3.4x`.
This rules out local collision penalties as a practical substitute for learned
binding. A successful successor must preserve the native CE trajectory and add
an ownership signal inside a reversible or residual live transition, rather
than globally penalize the key geometry throughout training.

### Wrong-Key Swaps Are First-Layer Write Superposition

P-DIAG-EDIT-001 causally separates erase damage from write interference on the
frozen reproducible L1024 model. For each sample it turns off exactly one fixed
owner's erase or write gate at the recovered value commit token, without
training, new parameters, alternate kernels or labels inside the model.

Turning off the baseline wrong owner's layer-0 write repairs `1519/1546`
wrong-key swaps (`98.25%`); doing the same in both layers is identical, while
layer 1 alone repairs only 48. The intervention simultaneously preserves just
`5/1141` of that owner's own originally-correct queries. Erase-off repairs only
402 swaps and fails every preservation gate. The model is not primarily
forgetting the true value through excessive erase. It stores competing valid
payloads in a shared first-layer trajectory and cannot retrieve both
associations at once.

The label-mediated `.87425` balanced / `.705` joint selected result is an
oracle diagnostic, not a method. It proves that deleting the known competing
write exposes the right answer, but the identity of that write is the answer
itself. The paper should use this boundary to motivate a generic redundant
address-state topology trained from scratch, explicitly excluding owner
selectors, write suppression, erase-key rescue and global token-to-slot
routing.

### Redundant Independent Addresses Do Not Preserve Native Learnability

P-GDN3-053 tests the resulting foundational topology rather than a mature
checkpoint graft. Two full K32xV32 banks per head use independently learned
Q/K projections, receive the same native edit, run in one pinned-official H8
scan and contribute through a fixed mean. Both banks and the enlarged native
FutureSeed activate and remain non-collinear.

The result rejects raw redundant address capacity. Balanced accuracy falls
from `.494` to `.00875`, errors rise `2024->3965`, and validation stays near
chance for every epoch. The lower swap count is a consequence of losing
almost all retrieval. Time cost is also slightly above `2x`. The paper should
therefore separate the causal storage diagnosis from this failed remedy: two
bindings need separable ownership, but duplicating coordinate systems and
averaging their reads globally destroys the base optimization path. A viable
successor must preserve the native trajectory and add ownership in a bounded
live transition, not merely add or decouple keys.

### Exact Dual-Address Geometry Is Not a Binding Solution

P-DIAG-DUAL-001 asks the strongest possible version of the decoupled-key
hypothesis on the frozen reproducible L1024 model. It keeps native queries and
erase directions, but maps the four known owner writes to an exact dual basis.
The tied-DPLR control preserves the model, and the oracle pairing error is only
`3.62e-6`, so a failure cannot be attributed to an approximate solver.

The oracle instead lowers balanced accuracy from `.49425` to `.23025`, raises
errors from `2023` to `3079`, repairs only `327/1546` swaps and retains only
`512/1977` native-correct answers. This establishes a useful negative result:
GDN2 binding is distributed across a co-adapted query/erase/write/read system,
not recoverable by independently improving write-key separability. The paper
should use this ceiling to close post-hoc orthogonalization and motivate an
end-to-end ownership transition or a distinct loop-convergence mechanism that
preserves the exact native path.

## Local Binding Before Global Recurrent Transport

The latest ownership evidence motivates one architecture-level test rather
than another key wrapper. Native L1024 errors are dominated by valid values
from the adjacent write owner, and deleting that competing layer-0 write
repairs the queried binding while destroying the competitor. Direct key
decoupling and exact dual bases fail because query, erase, write and read
coordinates are co-adapted.

P-GDN3-055 preserves the entire native GDN2+FutureSeed path and adds a fixed
128-token causal SDPA branch at each layer. The intended division of labor is
local pair formation followed by global linear-recurrent transport. It is
trained from scratch on directional MQAR L1024 and must reduce wrong-owner
swaps at materially higher absolute quality before any Sudoku transfer. A
failure closes fixed-block local-attention hybrids rather than inviting a
window or optimization sweep.

The endpoint rejects that division of labor. Balanced accuracy is essentially
flat (`.4940->.4905`) and joint exact only moves `.041->.046`; future accuracy
improves `.454->.475` while past accuracy falls `.534->.506`. More decisively,
wrong-key valid-value swaps increase `1546->1822` and become `89.40%` of all
errors. The local path is materially active and affordable, so this is not a
dead adapter or cost failure. A parallel local correlation path does not
create symmetric occurrence ownership before the native global scan. The
paper boundary should close fixed-block local-attention hybrids and motivate a
live recurrent representation that can preserve multiple owner-specific
bindings simultaneously.

### Address Deblurring Falsifies The Simple Front-End Alias Hypothesis

P-GDN3-069 removes only the width-four temporal convolution from Momentum
Q/K, preserving V convolution, the complete successful `[S,M]` recurrence and
native FutureSeed. The strict contract proves that this is a clean, active,
parameter-reducing intervention rather than a fallback.

Balanced accuracy collapses `.94425->.36275`, joint exact `.824->0`, and
adjacent-owner swaps rise `151->1459`. The result is mechanistically useful:
the local Q/K convolution is not merely smearing nearby addresses; it is part
of the representation that makes the sharp P059 learning transition possible.
Future work must preserve that co-adapted address front end. Ownership should
enter as additive receiver-native evidence or a scalable state organization,
not by deleting, splitting or post-hoc orthogonalizing the learned key.

### Terminal Conv Prefill Is Redundant And Geometrically Misaligned

P-FS2-015 tests whether native `[S,M]` FutureSeed is missing the Q/K/V
short-convolution boundary. It preserves exact P059 and gives layer 1 a
receiver-native conv cache built from layer 0's final four hidden tokens. The
strict contract proves identity at zero gate, complete gradients, ordered
evidence dependence and exact external Momentum provenance. The route then
activates to mean gate `.051609` with nonzero board-varying cache.

The result rejects the hypothesis: balanced accuracy falls
`.94425->.35600`, joint exact `.824->.001`, and adjacent-owner swaps rise
`151->1424`. The ordinary stack already supplies layer 1 with the full aligned
layer-0 hidden sequence, which layer 1 projects and convolves itself. Seeding
its token-zero convolution with producer terminal tokens instead creates a
tail-to-head local adjacency. The paper should distinguish recurrent future
state, which is genuinely absent from the receiver's causal scan, from local
hidden context already present in the aligned residual stream. Future FS2
should change the transported state's receiver-readable semantics rather than
duplicate the residual path.
