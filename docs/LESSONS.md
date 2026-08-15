# Lessons

## 2026-08-15: Preserving erase response is not preserving address closure

- P-GDN3-054 uses a stable causal block-RLS statistic, one official DPLR scan
  per layer, exactly eight new parameters and no persistent state increase.
- All eight mix paths activate and the 4x4 precision blocks stay positive and
  nondiagonal, but the candidate ends at `.018` balanced versus `.494` control.
- Errors nearly double (`2024->3928`). Swaps fall `1546->190` only because
  retrieval broadly fails: 1,849 correct queries break and only 24 swaps repair.
- Constraining `z^T a=z^T k` preserves one erase-response scalar, not the full
  co-adapted query, erase, write and read geometry learned by GDN2.
- Tiny parameter count does not imply low cost. The custom causal statistic
  reaches `2.007x` warmed-step time and fails its systems gate.
- Close block size/grouping/mix/prior and all nearby decoupled-key rescue. A
  successor must make pair ownership intrinsic to learned state formation or
  carry new receiver-native ownership evidence.

## 2026-08-15: Accelerating a lossy FutureSeed trajectory does not recover binding

- P-FS2-013 adds only 88 zero-init coefficients and passes exact parent,
  official-kernel, gradient, bounded-state and exact-resume checks.
- The mechanism activates: loop2 secant delta/residual relative RMS is
  `.408061/.002687`, with nonzero board variation and learned scale.
- Hard macro and mixed exact do not move, while 61-64 blank falls `.004182`
  and its loop3-to-loop5 wrong-cell correction weakens.
- The residual contracts to `.00004161` of state RMS by loop5. The model is not
  missing scalar momentum; it is missing information that the state trajectory
  no longer represents.
- Close secant cap/init/edge/training variants. A useful FS2 must carry distinct
  ownership evidence or improve its credit path, while the GDN3 route must
  alter scalable live interference handling.
- Measure systems cost even for tiny parameter deltas: retaining cross-pass
  histories raised fresh-process elapsed time `18.36%` despite only `4.66%`
  more peak allocation.

## 2026-08-15: Oracle head diversity is not a usable confidence signal

- P-DIAG-OWN-003 isolates all eight heads at the official output projection
  without changing weights, recurrent states or logits elsewhere.
- A final-layer label oracle repairs `33.38%` of wrong-key swaps and reaches
  `.64825` balanced, so the information is not completely absent globally.
- The preregistered label-free max-margin proxy falls `.4945->.42325` balanced
  even while swaps fall `1546->1300`. Lower swap count again reflects broader
  breakage, not ownership closure.
- `28.53%` of swaps keep the same wrong owner across every final-only variant.
  The dominant error already exists within individual head states.
- Do not turn diagnostic oracle diversity into a selector, router, voting rule
  or temperature sweep. The next GDN3 must change the live within-head commit.
- Exact hashes do not guarantee bit-identical fresh-process BF16/Triton output:
  this replay differed on 10 of 4,000 predictions. Register and freeze the
  reload tolerance before reading interventions, then compare in one process.

## 2026-08-15: A second address space does not create ownership by itself

- P-GDN3-049 stores full native V in an independent H4/K16/V32 official GDN2
  state under a shared learned canonical address, while leaving native GDN2 and
  FutureSeed unchanged.
- The mechanism is genuinely live: both states, all eight gates and the rank-16
  projection activate; address, output and state vary across tokens and boards.
- Quality still falls: balanced accuracy is `.03700->.03075`, errors are
  `3852->3877`, and wrong-key fraction changes only `.058930->.056229`.
- This distinguishes address capacity from ownership organization. Re-encoding
  the same values in another dense matrix still superposes multiple bindings;
  it does not make a value belong uniquely to its key.
- Preserve contemporaneous controls. The same native initialization/data can
  land on very different absolute endpoints under BF16/Triton, so only the
  within-run control/candidate comparison is a valid decision.
- Preserve operator mistakes as evidence. Here an abort was written after
  natural completion, then retracted because training RNG is reset after
  warmup and before `Trainer`; the score is valid and was not rerun.
- Close companion K/projection/gate/sharing/transport variants. The next test
  must change live edit organization, not append another dense state bank.

## 2026-08-15: Lower wrong-key fraction can mean the value code stopped working

- P-GDN3-048 encodes V with native `k` and decodes with native `q`, preserving
  one coherent key/state, one official scan and exact zero-point behavior.
- All eight codes activate, Q/K mismatch changes the payload, and the state is
  finite, yet balanced accuracy falls `.17475->.07500` and errors rise
  `3301->3700`.
- Wrong-key swaps fall `770->337` and their error fraction falls
  `.233263->.091081`; paired transitions reveal the cost: 627 correct queries
  break while only 228 wrong queries become correct.
- Reciprocal algebra at FP32 is not enough for a BF16 learned memory. Endpoint
  same-address error is `.004211`, and small address-conditioned factors still
  perturb payload optimization enough to lose retrieval.
- Close payload-code radius/map/source/precision/sharing variants. Together
  with decoupled-key and Q/K gauge failures, this says ownership needs a
  learnable state organization or credit mechanism, not another reversible
  coordinate wrapper.

## 2026-08-15: Value-set recovery can make binding errors nearly pure

- P-GDN3-047 stores a token-identity certificate under the exact native GDN2
  address and gates, with only eight learned read scalars. It is parent-exact
  at zero and all companion states/gates activate.
- Balanced accuracy improves `.17475->.48300` and total errors fall
  `3301->2068`; this is a real retrieval gain, not dead code or broad collapse.
- Wrong-key valid-value swaps nevertheless rise `770->1973` and become
  `95.41%` of all remaining errors. Lower CE (`2.91->0.92`) mostly means the
  model learned the candidate value set.
- Paired transitions matter: 1,580 wrong predictions become correct, while 347
  correct predictions break. Aggregate gains can hide destructive binding
  churn.
- Do not rescue certificate payload, gate, normalization, transport or width.
  A next GDN3 must encode ownership in a scalable recurrent memory/update
  organization rather than append another semantic side channel.

## 2026-08-15: Stable dynamic state frames can still erase binding

- P-GDN3-044 is a genuine live-transition intervention: the token-dependent
  frame difference enters native row decay, the one official scan transports
  state between frames, and terminal state returns to a canonical FutureSeed
  basis.
- The implementation is exact at identity and remains numerically controlled
  after training. Frame transport telescopes below `1e-7`, factors stay inside
  `.7101..1.4027`, canonicalization error is below `2.4e-7`, and both layers
  have variable nonzero frame, Q/K and decay changes.
- Balanced accuracy nevertheless falls `.36625->.0765` and errors rise
  `2535->3694`. A bounded stable coordinate system does not imply a learnable
  address system.
- Wrong-key swap fraction falls `.62091->.09556` because broad retrieval
  replaces binding errors. Require lower absolute errors alongside any
  conditional error-composition gain.
- Static and dynamic Q/K coordinate interventions are now jointly closed.
  Do not rescue frame radius, rank, controller or sharing. The next experiment
  should target exact receiver credit or a genuinely different memory edit,
  not another address reparameterization.

## 2026-08-15: A strong producer-native readout can erase learnability

- P-FS2-010 keeps native FutureSeed state transfer and both official GDN2
  scans, changing only one producer-native state decode plus bounded rank-32
  hidden fusion.
- The path is not dead: read/feature/residual relative RMS is
  `.3601/2.7084/.3853`, output-projection RMS is `.03519`, and the residual
  reaches `.499865` of hidden RMS against its fixed `.5` cap.
- Balanced accuracy collapses `.17475->.0280`, errors rise `3301->3888`, and
  turning the trained edge off leaves balanced accuracy at `.0280`. The
  backbone co-adapted to a harmful path rather than gaining usable evidence.
- The transferred-state RMS board std is exactly zero because native
  FutureSeed normalizes every board to fixed RMS. Do not preregister RMS
  variation after a normalization that removes it; measure content variation
  with pairwise state distance or centered cosine instead.
- The lower conditional swap fraction `.23326->.07433` is another broad-error
  artifact. Close producer decoder, rank, cap, injection and gate variants.

## 2026-08-15: Current-query feedback is not free binding credit

- P-GDN3-041 is the narrow test left open by decoupled-key failures: write,
  erase ownership and read all remain on native `k`; only the erased content
  estimate includes a bounded signed current-query term.
- The mechanism is strongly active and stable. Lambda RMS is `.3273/.3194`,
  all eight heads move, mixed alignment remains `.6924..1.3120`, and sampled
  transition spectral norm is at most `1.0562`.
- Balanced accuracy nevertheless collapses `.36625->.0145` and errors rise
  `2535->3942`. This rules out dead gradients, state explosion and cost as the
  explanation.
- The swap fraction `.62091->.03754` is meaningless without absolute
  retrieval: 3,942 of 4,000 candidate queries are wrong.
- Preserve native GDN2's learned erase-content equation in future work. A
  next FutureSeed mechanism may decode completed producer memory through a
  coherent interface, but should not inject another query/address term into
  the live ownership correction.

## 2026-08-15: More expressive fixed addresses can block learning

- P-GDN3-040 is the clean complement test missing from P024: it keeps the
  complete native K32 map and adds an exact K8xK8 product block, with zero new
  parameters and one official scan.
- Product Q/K, state and read paths all activate; product/native state RMS is
  `.2700/.2652` and native FutureSeed remains active. This is not dead code.
- Balanced accuracy collapses `.17475->.01225`, errors rise `3301->3951`, and
  the product validation curve stays near chance through epoch8.
- Swap fraction `.233263->.031891` is not success when usable retrieval is
  destroyed. Always pair conditional error composition with absolute errors.
- Close factor size, direct-sum weight/scale and lifted-gate variants. Added
  feature expressivity is not a substitute for a learnable credit path; the
  next state organization must preserve native address learning rather than
  impose another analytic basis.

## 2026-08-14: Strong recurrent address context is not binding closure

- P-GDN3-039 is an exact Raven/GDN composition rather than a replacement
  carrier: Raven recurrent retrieval changes only the coherent native Q/K
  input, while GDN2 retains V, gates, state, committed edit and FutureSeed.
- Every mechanism gate passes. Address residual RMS is `1.738/2.991`, Q/K
  changes reach `1.908-6.299x`, slot entropy is `.766/.726`, and Raven state
  varies by board. Activation is not the problem.
- Balanced accuracy collapses `.36625->.06275` and errors rise `2535->3749`.
  The swap fraction improvement `.62091->.09149` again reflects replacing
  binding errors with broader retrieval failure.
- Costs are acceptable (`1.179x` endpoint wall, `1.361x` warmed step,
  `1.419x` peak allocation), so reject the quality hypothesis directly.
- P017, P019 and P039 now close Raven allocation, Raven-to-V control and
  Raven-to-Q/K context at their fixed settings. Do not rescue slots, top-k,
  width, adapter scale or transport. The next useful mechanism must alter
  scalable state organization while preserving a learnable native read/write
  map.

## 2026-08-14: Better state readability can worsen key binding

- P-FS2-009's analytic `C_r^-1 C_p` pullback is a clean causal intervention:
  zero new parameters/state/scans, exact parent identity and all cost gates.
- It improves balanced accuracy `.1385->.17825` and lowers errors
  `3446->3287`, proving that private metric mismatch matters modestly.
- The same candidate raises wrong-key valid-value swaps `602->748` and their
  error fraction `.17470->.22756`; joint exact remains `.002`.
- Do not equate a lower average retrieval error with better bindings. Require
  absolute joint closure and an improved wrong-key denominator together.
- Close transpose, inverse, interpolation, scale and metric rescues. A next
  FutureSeed/GDN3 mechanism must partition or update live memory so colliding
  bindings remain distinguishable, not merely rotate inherited state.

## 2026-08-14: Relative carrier replication is not cross-task transfer

- P033's contemporaneous control validates P020's per-layer Log-SPD relative
  MQAR gain, but P034 shows that a credible carrier effect can still fail on a
  different closure task.
- All 12 Sudoku metrics activate with bounded eigenvalues, and hardest-range
  blank accuracy improves `+0.005769`; hard exact remains unchanged and mixed
  exact regresses. Activation plus partial-cell improvement is not board-level
  closure.
- Same-board loop3-to5 correction improves on 51-55 and 61-64 but weakens on
  56-60. Require the preregistered all-range criterion rather than selecting
  the favorable difficulty band after the fact.
- Small parameter/state overhead does not imply cheap execution. Repeated
  per-layer matrix exponentials reduce throughput `15.497->9.337` boards/s
  despite only +50,592 parameters and +3.99% allocated memory.
- Close static metric scale, rank and sharing variants on this Sudoku parent.
  The next intervention must address a different causal boundary rather than
  tuning an active mechanism that already missed exact closure.

## 2026-08-14: More coherent address rows can reduce swaps while worsening retrieval

- Native K64/V32 passes the unmodified official-GDN2 contract and halves no
  payload dimension in the recurrent state: address state doubles from 4,096
  to 8,192 values/layer while one coherent erase/write/read key is preserved.
- Wrong-key valid-value swaps fall `1271->294`, but total errors rise
  `2775->3818` and balanced accuracy falls `0.30625->0.0455`. Conditional
  error composition is not a capacity metric.
- The K64 learning curve stays near chance through epoch4 and reaches only
  `0.02725` at epoch6, while historical K32 is already `0.71575`. Added rows
  receive finite gradients, so this is delayed/diluted learnability rather
  than an inactive implementation.
- Close raw K-axis expansion. Severe key-Gram anisotropy does not by itself
  justify more address dimensions; the next mechanism must improve binding or
  credit in the coherent namespace.

## 2026-08-14: Geometry collapse is not enough to admit an address wrapper

- P-DIAG-ADDR-001 measures very low effective rank and extreme anisotropy on
  L1024, yet receiver-native reprojection, exact write survival, and surprise
  all fail to predict errors.
- Respect the frozen causal branch: close cache/reprojection/metric
  combinations and test a coherent live-state transition instead of fitting
  the visible geometry symptom.

## 2026-08-13: A solved write proxy can still miss the read problem

- P-FS2-008 fits the receiver-native weighted write residual to `0.001132x`
  native MSE, yet balanced accuracy reaches only `0.2245` versus historical
  native FutureSeed `0.7475`.
- Use a same-trained-weight edge-off counterfactual for state additions. Here
  the residual improves pooled query CE by only `1.15%` and raises wrong-key
  swaps `906->934`; most of the apparent `0.0400->0.2245` paired gain is not
  attributable to the ridge edge itself.
- Numerical conditioning and bounded state are necessary, not sufficient.
  Cholesky, solve residual, state bounds and write fit all pass while real
  binding fails.
- Measure independently warmed steps. Fit and wall ratios barely pass at
  about `1.58x`, but warmed-step cost is `2.1845x` and closes the mechanism.
- Close lambda, jitter, solver, weighting, projection, residual-scale, Top-K,
  replay and training rescues. The next run must target actual live
  query-address binding, not another write-reconstruction surrogate.

## 2026-08-13: Stable memory can still be unusable memory

- P-GDN3-030 passes every registered activation and sampled-stability check.
  Its transition spectral norm is at most `0.999988`, terminal-state RMS stays
  `0.2476/0.4699`, and learned beta std is `0.1983/0.3145`.
- Balanced accuracy nevertheless collapses `0.30625->0.0205`, with total
  errors increasing `2775->3918`. Contractivity prevents explosive observed
  state geometry; it does not preserve a useful address/value map by itself.
- Wrong-key swap fraction improves `0.458018->0.041858`. Together with
  P028/P029, this makes conditional error composition unsafe as a standalone
  target. Require absolute accuracy and lower total errors.
- All cost gates pass, so this rejection is not explained by an impractical
  kernel. Close separate-erase/write DPLR rather than sweep beta, angle,
  normalization, rank, or training.
- Preserve P-FS2-007's causal surprise signal, but replace its sparse ordered
  replay with all-token receiver-native state formation.

## 2026-08-13: More native delta transforms are not automatically better memory

- P-GDN3-029 uses a genuine official two-transform recurrent operator, and
  both projection branches, gradients, incoming-state dependency and order
  sensitivity activate. Balanced accuracy still falls `0.30625->0.1370` and
  total errors rise `2775->3452`.
- Its wrong-key swap fraction improves `0.458018->0.169177`. As with P028, a
  lower swap fraction can result from replacing binding errors with broader
  retrieval failure. Require absolute accuracy and total errors alongside
  error composition.
- Endpoint projection divergence proves available capacity, not that both
  sequential committed edits are independently useful. Future contracts
  should inspect the actual committed state contributions when that claim is
  scientifically necessary.
- The warmed timing comparison used an earlier A100 reference and permanent
  inactive diagnostic hooks. Record its `1.963x` miss as supporting evidence,
  but close this family on the large quality failure rather than timing alone.

## 2026-08-13: Low swap rate can hide a failed memory

- P-GDN3-028 proves that a low wrong-key-swap fraction is not by itself a
  successful binding mechanism. Atomic erase-orthogonal shared-payload pairs
  reduce swap fraction `0.458018->0.074980`, yet balanced accuracy falls
  `0.30625->0.04975` and errors increase `2775->3801`. Track error count and
  value/read learnability alongside swap composition; a method can improve the
  latter simply by replacing swap errors with other failures.
- The trained weighted-polar geometry did not remain in its registered stable
  regime: mean condition reached `8.29/10.50`, maximum condition
  `170.62/100.15`, and diagonal error about `0.02`. Contract-time algebraic
  correctness does not guarantee that a learned address pair stays numerically
  useful. Close auxiliary projection, whitening, payload scaling, and ordering
  rescues rather than treating this as a tuning problem.

## 2026-08-13: Surprise preserves values but does not bind them

- P-FS2-007 cleanly separates admission from capacity: exact committed-edit
  surprise K16 raises directional MQAR balanced accuracy
  `0.30625 -> 0.48825`, while matched recency K16 falls to `0.01525`. The
  signal is therefore causal and not a generic extra-scan benefit.
- The candidate still misses the frozen absolute gates, and `2,044/2,047`
  remaining errors are correct-value/wrong-key swaps. Sparse replay converts
  many retrieval failures into almost pure binding failures instead of closing
  them.
- Receiver-native reprojection avoids producer-basis transport, but a bounded
  event tape cannot repair an address collision already encoded by the live
  recurrent transition. Close K, admission score, cache width, positional and
  receiver-projection rescue.
- Fit wall and allocation can look cheap while the independently warmed step is
  not: the replay path is `1.438x` baseline per step, above the registered
  `1.25x` ceiling. Future successors must report both endpoint wall time and a
  warm compile-independent step measure.
- The next high-information test should be an order-independent live block
  update or another genuinely new scalable address-binding recurrence trained
  from scratch. It should not be another value cache, separate correction bank,
  sequential second write, static metric wrapper or Sudoku graft.

## 2026-08-13: A diagnostic proxy needs tighter parity than its effect budget

- P-FS2-006 tried to use an FP32 replay of official BF16/Triton GDN2 to form a
  receiver-native inherited-minus-live FutureSeed signal. The tensor equations
  are correct, but the first production comparison gives output relative RMS
  `0.002523`, above the frozen `0.002` parity ceiling; terminal parity is
  `0.001143`.
- The planned holdout intervention is only 1% of committed-edit RMS. A proxy
  mismatch of this order can determine the direction being tested, so it is
  not defensible to relax the parity threshold after seeing the result.
- No cotangent aggregation or quality scoring occurred. This is an integrity
  boundary for hand-replayed recurrence diagnostics, not evidence against the
  inherited-vs-live mechanism. Prefer signals exposed by the exact official
  graph or a from-scratch benchmark for the next test.
- A shuffled-state sham must reuse one donor permutation through all receiver
  calls in a microbatch; changing donors across layers/loops destroys the
  trajectory coherence it is intended to control.

## 2026-08-13: Algebraic validity does not imply production viability

- P-GDN3-027's compiled Online Inverse-Gram Preconditioned Delta recurrence
  passes the strict parent-identity, state, gradient, equivariance and
  no-fallback contract, but post-compile training is conservatively `40.95x`
  the matched official control versus a frozen `<2.5x` ceiling. Stop before
  quality scoring: a dense KxK online geometry state implemented outside a
  fused linear-memory kernel is not a viable GDN3 path at L1024. This does not
  say that address geometry is irrelevant; P020 remains the positive signal.
  It closes same-family chunk/block/compiler rescue and requires the next
  mechanism to expose a genuinely scalable live transition.

## 2026-08-13: Receiver-local conflict must be broad to justify a new gate

- Aggregate opposition can be mathematically real and still be too localized
  to support a mechanism. On 51-55, P-LOOP-003 measures cosine `-0.9711` and
  conflict `0.7909`, but only two of eleven receiver edges are active, one
  board carries `79.11%` of the phase energy, and leave-one-board cosine is
  only `0.4060`.
- The harder ranges contradict a universal phase story: opening and
  continuation are aligned at `+0.9580/+0.8898`, not opposed. A scalar phase
  gate would therefore encode a narrow batch/depth accident.
- Exact receiver-boundary reconstruction matters. The diagnostic matches
  direct gate gradients to `3.71e-5` relative error while leaving every model
  parameter unchanged, so rejection is not attributable to a weak proxy.
- Do not rescue phase credit with per-edge selectors, phase strength or loss
  weights. The next useful intervention must change live recurrent address
  organization and target the observed wrong-key binding failure directly.

## 2026-08-12: Aggregate FutureSeed gradient conflict is not closure

- P-LOOP-002 cleanly separates optimizer credit from the forward mechanism.
  It changes only 88 native FutureSeed gate gradients, preserves the canonical
  gradient norm and has zero inference delta.
- Mechanical activation is decisive: 57/100 steps project, the active removed
  opening fraction averages `0.831642`, and norm errors stay below `1e-7`.
  The failed quality gate is not explained by a dead intervention.
- Removing aggregate opening-versus-continuation opposition does not improve
  closure. Hard macro exact falls `0.001302 -> 0`, mixed exact is unchanged,
  and 56-64 late correction weakens.
- A scalar gate sees only the radial projection of a receiver's state demand.
  Do not rescue this result with projection strength, per-head selection,
  Adam-moment edits or loop-loss weights. Those variants refine the same
  insufficient control surface.
- The next diagnostic must localize receiver-state cotangent demand or
  macro-loop convergence and must pre-register the distinct architecture it
  opens. GPU occupancy alone is not a reason to run another wrapper.

## 2026-08-12: Dynamic address metrics need a coordinate contract

- Applying a different full matrix to Q and K in each block silently puts old
  state and new queries in inconsistent frames. Keep key/write/state canonical
  unless a fused recurrence explicitly transports the live state.
- Do not split L1024 into sixteen external official-GDN2 calls merely to gain
  block boundaries; retain one official chunk graph and batch the exclusive
  prefix-Gram computation before it.
- Before training a data-dependent conditioner, test its frozen geometric
  prediction without changing logits. Failure there is a mechanism decision,
  not an invitation to sweep block size or spectral cap.
- P-GDN3-026 is the concrete boundary. Its query-only canonical-state variant
  passes exact identity, causality, equivariance, official-kernel and bounded
  activation contracts, yet at full strength its median effective-rank gain is
  `-0.000020`, anisotropy ratio is `1.000088`, and binding-margin gain is `0`.
  Only `36.57%` of margins improve. Completed-block Gram is not predictive
  enough to repair the next block's binding geometry, so do not train or sweep
  this family.
- A strict pre-training diagnostic saved the entire 10-epoch budget. This is
  the desired outcome of a kill gate, not missing GPU utilization.
- The broader P020-P026 sequence says address interference is real but not
  solved by more wrappers: static Log-SPD helps, while extra edits, state banks,
  compact hashes, correction memory and online Gram all fail to close. Shift
  the next mechanism question to loop dynamics/training signal and use a
  contemporaneous runtime control because the historical MQAR trajectory is
  not reproducible under the current Triton/runtime stack.

## 2026-08-05 research scope lock

- The fixed research program is: use hard Sudoku scaling as the scaffold for
  jointly iterating native FutureSeed and the next official-FLA linear recurrent
  mechanism, GDN3. A generic idea should plausibly transfer to language or other
  sequence tasks, but those tasks must not silently replace the benchmark.
- P-CAUSAL-024 is the concrete scope-drift lesson. Its ModernBERT audit was
  careful, but it answered a different question. It was discarded before assets,
  model execution, or GPU use and must not enter the evidence chain.
- FutureSeed/GDN co-design should start from state semantics, not another small
  gate. A transported KxV state is only useful if the receiving layer interprets
  its K axis consistently. Cross-layer address compatibility is therefore a
  higher-information GDN3 hypothesis than scalar seed-selection variants.
- Scaling remains mandatory after a mechanism passes a cheap falsifier: expand
  independent Sudoku data, model/state capacity, and training compute. The
  falsifier prevents hours of blind scaling; it is not a substitute for the
  full-budget result.
- A single shared address namespace across all GDN2 layers is the strongest
  generic FutureSeed/GDN2 quick mechanism result so far. At matched step9100 it
  raises mean official 51-64 blank accuracy `0.2037 -> 0.4479`, lowers CE
  `1.9370 -> 1.1634`, and costs only `+6.9%` elapsed time. This is not a
  blank-only illusion: 61-64-blank mean wrong cells fall `39.91 -> 36.99` from
  loop1 to loop5, with 186/256 cases improving. Exact is still zero, so the
  right next move is one from-scratch full-diversity scale where the shared
  basis and content maps co-adapt, not an address-strength/rank/loss sweep.
- The from-scratch D256/L12 scale does not inherit the small probe's opening by
  accident: at step500 it reaches 50-blank loop5 exact `0.7778` with train CE
  `0.0143`. Loop1 exact is only `0.2121`, so loops contribute real correction
  at this early gate. This clears stability and optimization as immediate
  blockers, but it says nothing yet about the 51-64 blank cliff. Do not use the
  easy gate to claim GDN3 success; let the preregistered hard gates decide.
- The step6000 gate confirms that the step3000 opening was not a one-checkpoint
  accident. Representative h58 exact stays zero at loops1/2 and opens to
  `0.0059` at loops3-5 while wrong cells fall `31.68 -> 30.46`; h53 loop5
  exact/blank improves from `0.0156/0.5907` at step3000 to `0.0293/0.6946`.
  The mechanism is doing harder-range recurrent correction.
- Local correction is still not global closure. The h64 probe removes `5.06`
  wrong cells across loops but remains at zero full-board exact. Continue only
  the registered trajectory to step9000/12000 and require full official-range
  endpoint metrics; do not convert this pass into an address/LR/loss/seed or
  nearby-scale sweep.
- The step9000 readout changes the hardest-tail diagnosis: h64 loop5 exact opens
  from `0` to `0.0293`, blank accuracy rises `0.6093 -> 0.7313`, and loops
  remove `11.58` wrong cells per board (`28.78 -> 17.20`). Exact is zero through
  loop2 and appears only at loop3, so the gain is recurrent global correction,
  not just a better one-pass operating point.
- Fixed buckets can move unevenly even when the mechanism scales: h53 exact
  jumps `+0.1855`, h64 opens `+0.0293`, but h58 gains only `+0.0059`. Do not
  select the favorable buckets as the endpoint claim. Freeze step12000 and
  require full official 51-55/56-60/61-64 ranges plus same-board loop exports
  before authorizing a matched D256 normal-GDN2 control.
- The frozen endpoint reverses the favorable fixed-probe impression. Full
  official 51-55/56-60/61-64 loop5 exact is
  `0.3008/0.0996/0.0762`; macro exact `0.1589` and mixed exact `0.2168` miss
  the registered `0.3191/0.40` routes. Every hard range is also below the
  sealed D192/L10 canonical model. Endpoint ranges, not selected fixed buckets,
  decide whether a mechanism scales.
- Genuine loop correction is necessary but not sufficient. Official hard
  exact is zero at loop1 and opens in later loops; a same-board 64-blank case
  removes `16` wrong cells across loops but stalls at six errors. Recurrent
  computation is active, yet the representation still fails to close enough
  global constraints.
- P-GDN3-001 and P-GDN3-002 together localize the lesson: cross-layer state
  compatibility matters, but forcing every layer into one shared Q/K namespace
  sacrifices useful layer-private address dynamics at scale. Preserve the
  compatibility hypothesis and reject this implementation.
- A matched D256 normal-GDN2 control is not authorized after the registered
  endpoint miss; it would quantify a failed candidate rather than change the
  next decision.
- Frozen checkpoint geometry rules out an easy post-hoc fix. Adjacent mature
  Q/K bases have identity residual near `1.42`; even optimal joint-Q/K
  Procrustes leaves about `1.11` residual, while fitted rotations are about
  `1.41` from identity and contain nearly null directions. Do not retry a free
  K/V transport matrix under a different name.
- The next falsifier is coordinate coherence at birth, not permanent sharing:
  clone Q/K/V projections and short convolutions at initialization, keep
  distinct parameters, and require measurable specialization after one step.
  This preserves zero inference overhead and layer-private capacity while
  testing whether FutureSeed compatibility can emerge through co-adaptation.
- Coordinate coherence at birth is mechanically viable at full size. The CUDA
  contract gives exact initial equality with zero parameter-count increase,
  retains official FLA/Triton backward, and produces `1.809e-5` cross-layer
  divergence after one optimizer step. This is evidence that the intervention
  is a temporary optimization scaffold rather than hidden weight tying; it is
  not yet evidence of better Sudoku quality, which remains gated at step500.
- Mechanical viability was misleading in P-GDN3-003. Despite exact equality,
  distinct storage, finite gradients, and measurable one-step divergence, the
  full model remains near chance at step500: holes50 loop5 exact/blank
  `0/0.1321` and CE `2.0089`. Deep layers initialized as identical Q/K/V and
  convolution operators form a harmful optimization symmetry; tiny early
  divergence is not enough for functional specialization. Do not rescue with
  partial copying or initialization strength.
- The next scale candidate must preserve private layer diversity from step0.
  Position-address/payload separation already has a strong matched D192 signal
  (`+0.2296` mean hard blank), so its one D256/L12 full-budget test has higher
  decision value than another cross-layer-sharing variant.
- P-GDN3-004 confirms that the combination matters: canonical position Q/K
  keeps a readable address system while private per-layer V and state-edit
  dynamics preserve specialization. At step500 it reaches h50 loop5 exact
  `0.8990`, versus `0.7778` for the permanently shared namespace and `0` for
  coherent copied initialization, with CE `0.002967`. The useful abstraction
  is stable address semantics, not parameter equality across depth.
- Faster easy-stage optimization is necessary but still not the research
  endpoint. At the same P-GDN3-004 gate, h53/h58/h64 exact remains zero;
  h64 removes `2.12` wrong cells across loops, but h58 is flat to slightly
  worse. Preserve the trajectory to its registered hard gates and do not turn
  the h50 win into a nearby address, width, loss, or seed table.
- The orthogonal-innovation FutureSeed successor should remain independent of
  this result. Position Q/K changes how state is addressed; P-FS3-001 changes
  which newly written component of the producer terminal state is transported.
  That separation creates a clean next causal test if P-GDN3-004 later closes
  or completes, instead of stacking two unmeasured mechanisms in one run.

## 2026-06-03 GPU1 bootstrap

- AIStation development work for this repository is GPU1-only. GPU2 can be visible
  in the platform table, but experiment commands should bind to the GPU1 container
  and use `CUDA_VISIBLE_DEVICES=0` inside that container.
- Persistent experiment state belongs under `/huyang2/double-loop`; avoid `/root`,
  `/root/.cache`, and other reset-prone locations for environments, caches,
  artifacts, models, and run logs.
- Keep GitHub as the source of truth: run from a detached commit SHA on the GPU
  node, archive run metadata with that SHA, then commit and push tracking changes
  deliberately.
- The bitter lesson applies here: prioritize scalable search/training feedback and
  measured experiment loops over hand-built solver shortcuts.

## 2026-06-03 uv bootstrap

- GPU1's system Python can lack `ensurepip`/`python3-venv`, so `setup.sh` must
  bootstrap uv without requiring `python3 -m venv`. Prefer a local wheelhouse,
  then `pip --target` if pip exists, then the standalone uv installer into
  `/huyang2/double-loop/.cache/uv-bootstrap/bin`.
- The current GPU1 image already includes `/opt/conda/bin/python` with
  `torch 2.7.0+cu126` and CUDA on the A100. Reuse that configured GPU stack
  before downloading large torch wheels again.
- Avoid broad `pkill -f` process patterns. They can match the shell command that
  is trying to clean up the process and terminate the SSH session itself.
- Keep the experiment script syntactically valid on the reusable GPU Python
  stack. The current image uses Python 3.10, so avoid newer nested f-string
  syntax even if local tools can parse it.
- After remote patch transfer, inspect script tails and run the wrapper end to
  end. A syntactically valid shell script can still be semantically truncated
  before the metadata-recording step.
- Scale-up controls should be environment-driven in `run.sh`. Keeping model
  width/depth, curriculum, rollout, and optimizer knobs configurable lets GPU1
  runs increase useful compute without creating one-off wrapper scripts or
  broad low-signal sweep tables.
- Generated run/cache/model directories should not make a subsequent experiment
  look source-dirty. Dirty provenance should track source edits, while run
  metadata is committed after the experiment completes.
- A full-run smoke preflight must not inherit the parent `RUN_NAME`; otherwise
  the smoke and full jobs can write into the same tracking directory and blur
  config, logs, and scores.
- Experiment recorders should trust the run-local `config.json` for the run's
  source SHA and dirty flag. Recomputing dirty state after outputs are written
  can make a clean run look dirty just because tracking artifacts now exist.

## 2026-06-03 RWKV7 CUDA pivot

- The first 9x9 cliff run (`9x9-cliff-20260603T090939Z-d8ce276`) and the smaller
  rescue probe (`9x9-cliff-small-20260603T092602Z-d8ce276`) both hit the wrong
  bottleneck: the pure PyTorch recurrent scan had not emitted `step=0100` before
  the kill window, even while the A100 was doing work. Treat this as an
  implementation throughput failure, not as evidence about 9x9 reasoning quality.
- Do not respond to that signal with more batch/curriculum table filling. The
  next high-ROI question is whether a real CUDA RWKV7 WKV kernel can make the
  same FutureSeed hypothesis cheap enough to evaluate.
- RWKV7 `wind` CUDA keeps the useful `s0 -> sT` state interface needed by
  FutureSeed, but it imposes hard shape constraints: CUDA bf16, `T % 16 == 0`,
  and `head_dim` divisible by 16. For 9x9 Sudoku, pad 81 tokens to 96 and use a
  compatible shape such as `D_MODEL=128 HEADS=8 HEAD_DIM=16`.
- `torch.utils.cpp_extension.load` requires a `ninja` executable even when the
  container already has a working CUDA PyTorch. When reusing `/opt/conda/bin/python`,
  `setup.sh` still needs to install/link `ninja` under the repo-local `.cache/bin`
  and `run.sh` must prepend that directory to `PATH`.
- When installing PyPI packages with `pip --target`, console scripts are written
  under the target's `bin/` directory, not necessarily under the imported
  package path. For `ninja`, link `.cache/ninja-pylib/bin/ninja` into
  `.cache/bin/ninja` before trying to compile CUDA extensions.
- The `modded-nanogpt-rwkv` wind kernel is not the right A100 default on the
  current GPU1 image: CUDA 12.6 reaches `ptxas`, then fails because `movmatrix`
  is not recognized while assembling for sm_80. Use the official
  `BlinkDL/RWKV-CUDA` state-passing clampw kernel for GPU1 scale-up; keep wind
  as an explicit future option for a toolchain/GPU where that asm is supported.
- The CUDA state-passing 9x9 cliff run (`9x9-cliff-cuda-20260603T103352Z-92ee7b9`)
  reverses the earlier 9x9 kill signal: with `D_MODEL=128`, `LAYERS=8`,
  `HEAD_DIM=16`, `MAX_LOOPS=5`, and a 4-8 then 8-12 hole curriculum, train CE
  fell below 1.0 by step100 and final eval reached holes8 exact 1.0000, holes12
  exact 0.9629, and holes16 exact 0.8535. This supports continuing 9x9 CUDA
  scale-up; the K8 oracle gap is effectively zero, so rollout selector work is
  low priority for this branch.

## 2026-06-03 9x9 mechanism ablation

- The one-hour ablation budget was enough for three targeted GPU1 runs at
  source SHA `5e2253f`: no outer loop, no FutureSeed, and no training
  feature-diff noise. This was a mechanism test, not a sweep; each run answered
  whether a specific part of the successful 9x9 CUDA cliff run was structural.
- Removing the outer loop (`9x9-ablate-no-loop-20260603T111606Z-5e2253f`) did
  not break easy-distribution 9x9: full-board exact was 0.9922, holes12 exact
  was 0.9551, and train CE reached 0.0135 by step800. It did reduce hard-hole
  transfer: holes16 exact was 0.7715 versus the baseline loop5 0.8535. Treat
  loop compute as hard-constraint refinement, not as the only reason the model
  can solve 9x9.
- Removing FutureSeed (`9x9-ablate-no-future-seed-20260603T111747Z-5e2253f`)
  collapsed the run: train CE was still 0.4498 at step800, full-board loop5
  exact was 0.0664, holes12 exact was 0.0195, and holes16 exact was 0.0000.
  K8 oracle exact rose to only 0.2031 while the best selector was 0.0762, so
  the problem is not just selector choice. FutureSeed is carrying necessary
  cross-layer state for 9x9 scaling.
- Removing training feature noise (`9x9-ablate-no-feature-noise-20260603T112449Z-5e2253f`)
  improved this setup: train CE reached 0.0007 at step800, holes12 exact was
  0.9766, holes16 exact was 0.9004, and K8 selector gap was zero. For the next
  9x9 scale run, default to `NOISE_SCALE=0` or a much smaller value; do not
  spend budget on rollout selector work while FutureSeed is enabled and the
  oracle gap remains tiny.

## 2026-06-21 FutureSeed as cheap bidirectional context

- The paired RWKV9 GPU1 probe at source SHA `955e266` directly tested the
  current paper story: FutureSeed should let a causal/recurrent backbone cheaply
  access future/noncausal information. The protocol held model, CUDA kernel,
  curriculum, seed, batch, loops, and step budget fixed, changing only
  `FUTURE_SEED_SCALE=0` versus `FUTURE_SEED_SCALE=1`.
- Without FutureSeed, the model was trainable but position-biased: h12 loop5
  exact was `0.0156`, blank accuracy was `0.6688`, and early blank cells were
  far worse than late blank cells (`0.4697` versus `0.8382`). This is the
  failure mode expected from a causal scan that sees future constraints too
  late.
- With FutureSeed, h12 loop5 exact reached `0.9492`, blank accuracy reached
  `0.9940`, and the early/late blank accuracy gap nearly vanished (`0.9947`
  versus `0.9955`). Step800 train CE also fell from `0.4779` without FutureSeed
  to `0.0097` with FutureSeed at essentially the same wall time.
- Treat this as a strong mechanism win for FutureSeed-as-future-boundary-condition,
  not as an official EqR/Maze superiority claim. The next high-ROI validation is
  the same diagnostic in the official EqR code path or a causal Maze backbone;
  Sudoku seed sweeps are low value.

## 2026-06-24 Off-mainline bidirectional scan correction

- Correction: the official EqR "FutureSeed mixer replacement" experiment was
  misnamed. It used a forward+reverse token scan as a mixer replacement. That is
  not the FutureSeed algorithm. FutureSeed in this project means cross-layer
  terminal-state seeding: one recurrent layer processes the full sequence, then
  its terminal state seeds the next recurrent layer's initial state.
- Do not call a right-to-left scan FutureSeed. Do not use the bidirectional scan
  result as evidence for or against the FutureSeed mainline.
- The archived scan mixer had an early sample-efficiency signal, but it failed
  the e1024 gate: official mixer-base reached `accuracy=0.6644`, `exact=0.0249`,
  `lm_loss=0.7664`, while the scan replacement stayed at `accuracy=0.4231`,
  `exact=0`, `lm_loss=1.4029`. Its lower residual16 (`3.065` versus base
  `4.979`) means stable wrong convergence, not better reasoning.
- The patch script is now disabled by default and requires
  `ALLOW_OFF_MAINLINE_BIDIR_SCAN=1` to reproduce the archived side probe.
- Next FutureSeed experiments must preserve the actual mechanism: terminal
  recurrent state -> normalized/gated seed -> next layer or next loop initial
  state. No hidden reverse scan should be introduced under the FutureSeed name.

## 2026-06-21 Official EqR Maze baseline

- The fair Maze comparison must be anchored in the official EqR codebase, not the
  no-Hydra proxy runner. The current official clone is upstream SHA
  `aba94e9cde0f273ce644db5261cd6915ba6561f0`; the clean baseline uses that
  code path, and the FutureSeed condition applies only the FutureSeed patch plus
  a runtime SDPA fallback for the local CUDA environment.
- Official scalar token accuracy is not a valid Maze success signal by itself.
  In the matched e256 and e512 official runs, both clean EqR and FutureSeed reach
  about `0.87` token accuracy with exact `0`, while path-aware visualization on
  official test cases shows loop16 path F1 near zero. The models mostly predict
  non-PATH classes; this looks accurate only because PATH is a minority token.
- The e512 pair keeps the same conclusion: base final step3500 has token acc
  `0.868423`, residual16 `8.001`, and path F1 `0.001273`; FutureSeed has token
  acc `0.868430`, residual16 `7.920`, and path F1 `0.0`. FutureSeed still shows
  an early step500 optimization signal, but it does not solve official Maze or
  beat the path-aware baseline under this budget.
- Adding a generic path-token weight (`path_token_id=5`, `path_token_weight=8`)
  changes the official Maze failure mode from "predict almost no PATH" to "broad
  high-recall mask". Clean EqR reaches loop16 path F1 `0.4682`, precision
  `0.3064`, recall `0.9998`, and predicted PATH fraction `0.4325` against true
  `0.1326`. This makes Maze usable as a path-aware diagnostic but not a solved
  task.
- Under that same path-aware objective, FutureSeed is neutral on official EqR:
  loop16 path F1 is `0.4684`, only `+0.0001` over clean EqR, and the hard-case
  false positives remain about `270` per case. This supports the mechanistic
  boundary that EqR's mixer already supplies noncausal interaction, so FutureSeed
  should be argued as a cheap future-context module for causal/recurrent
  backbones, not as a small patch that always improves an already noncausal
  mixer.
- Testing a causal RWKV7 state-passing backbone on the same official
  path-weighted Maze objective does not rescue the benchmark. No-FutureSeed
  reaches loop8 path F1 `0.4690`; FutureSeed reaches `0.4666`; both have
  near-zero or negative loop gain and hundreds of false positives per case. The
  lesson is sharper: the current official Maze objective permits a broad-mask
  shortcut even for a causal recurrent model, so it is not a clean benchmark for
  the cheap-bidirectional mechanism.
- A generic PATH/non-PATH boundary objective answers the next obvious question:
  broad masks can be made costly without maze rules, search, repair, or a
  selector, but the model then overprunes true path cells. In the causal RWKV
  pair, no-FutureSeed loop8 path F1 is `0.4089` with precision/recall
  `0.3256/0.5828`; FutureSeed reaches `0.4278` with `0.3270/0.6280`. The
  FutureSeed gain is a weak recall-preservation signal under pruning pressure,
  not a Maze success claim, and loop gain remains near zero. Do not continue
  with boundary-weight sweeps; change the proxy/objective or the generic
  recurrent decision/state mechanism.
- The next Maze work should either align training/evaluation with path recovery
  using a generic objective, or test FutureSeed on a causal/recurrent Maze
  backbone where bidirectional information is actually the bottleneck. Do not
  claim Maze progress from token accuracy, and do not add selector, repair,
  search, or maze-specific postprocessing to patch this result.

## 2026-06-22 RWKV Maze budget decoder

- The learned path-budget decoder answers a sharper calibration question. A
  generic count head can learn the true PATH fraction on official Maze30: final
  count absolute error is about `0.0095-0.0098`, and predicted path fraction is
  close to the true `0.1337`.
- This does not solve path recovery. Decoding the top model-ranked cells under
  the learned budget cuts false positives from about `246-255` per case to about
  `78-80`, but creates about `78-80` false negatives. The model knows roughly how
  many PATH cells to choose, but it does not rank true-path cells above false
  positives reliably.
- FutureSeed is neutral-to-negative on this probe: budget loop8 path F1 is
  `0.3289` versus no-FutureSeed `0.3387`, and raw loop gain remains near zero in
  both conditions. Do not claim Maze support for FutureSeed from this result.
- Do not sweep budget/count weights. The missing piece is a generic ranking or
  self-correction signal that makes later loops move true-vs-false PATH ordering,
  not another mass-calibration head.
- After AIStation/GPU restart, a stale project-local torch extension cache can
  make `StatePassingRWKV7.apply` hang before the first step with low GPU
  utilization. Rebuild only the specific project cache directory such as
  `/huyang2/double-loop/.cache/torch_extensions/rwkv7_statepassing_clampw_n16`
  before changing modeling code.

## 2026-06-22 RWKV Maze denoising-attractor feedback

- The generic denoising-attractor training probe is negative. Both no-FutureSeed
  and FutureSeed DAT were killed at the predeclared step300 gate on official
  Maze30 because loop16 did not reduce false positives relative to loop1.
  no-FutureSeed: F1 `0.4683 -> 0.4680`, FP `268.3 -> 269.0`; FutureSeed:
  F1 `0.4691 -> 0.4688`, FP `265.9 -> 266.4`.
- DAT is mechanically learnable: DAT loss falls to about `0.21-0.22`, and the
  model reaches the familiar high-recall path-mask operating point. The problem
  is that denoising corrupted token distributions back to labels still teaches a
  fixed point for the same broad mask, not a comparison that separates true PATH
  cells from false positives.
- FutureSeed only shifts the operating point slightly on this probe
  (`+0.0008` loop16 F1 over no-FS at step300) and does not change loop dynamics:
  both arms have loop gain `-0.0003`. Do not sweep corruption mix, DAT weight,
  logit temperature, batch size, or seed for this mechanism.
- Abortable Maze probes must dump hard-case visuals before termination. This run
  produced a metric dashboard from logs, but no input/target/loop1/4/8/16 case
  grids because the final visual writer is only called at normal completion.
  Fix instrumentation before the next abortable Maze probe.

## 2026-06-15 D320 effective-batch h120 scaling

- D320 width was not fairly judged by the earlier batch48 run. With microbatch
  `24` and gradient accumulation `3`, the effective-batch D320 route preserves
  the h96/h108 foundation and opens h120. The useful scaling move was general
  training infrastructure: checkpoint/resume, effective batch, bf16, and the
  CUDA statepassing RWKV kernel.
- The step12800 continuation confirms real but slow scaling. h120 loop6 exact
  moved from the old D320 step9800 checkpoint `0.0566` through
  `0.0820 -> 0.1016 -> 0.1191`, and final full eval reached `0.1270`. h108
  remained strong at `0.9355`; h132 stayed closed at `0.0`.
- Width is useful but not a shortcut. The D320 final score is still below the
  D256 late-continuation best `0.1777`, and the run used about `27GB` allocated
  / `32GB` reserved on an 80GB GPU with moderate utilization. Before jumping to
  larger width, improve throughput/effective compute for the current D320 path.
- The packed D320 continuation confirms that this was a real systems bottleneck.
  Microbatch `48` with accumulation `2` reached about `60-64GB` memory use and
  near-100% GPU utilization, while h120 checkpoint loop6 exact moved
  `0.1289 -> 0.1777` from step13200 to step13600. Final full eval reached
  h96/h108/h120/h132 loop6 exact `0.9941`/`0.9492`/`0.1738`/`0.0`. This nearly
  matches the D256 best regime, but does not beat it decisively or open h132.
- Continuing the packed D320 path to step16600 does beat the old platform:
  checkpoint h120 loop6 exact moves `0.1445 -> 0.1738 -> 0.1953` at
  steps 14600/15600/16600, and final full eval reaches h120 exact `0.2363`.
  This is enough evidence that h120 remains compute-limited under the clean
  FutureSeed+loop recipe.
- The step19600 continuation confirms the same direction: h120 checkpoint loop6
  exact moves `0.1934 -> 0.2266 -> 0.2520`, and final full eval reaches h120
  exact `0.2891`. This is still a clean scaling win, not a reason to add
  selector or Sudoku repair.
- The step22600 continuation is the first useful plateau signal for packed D320.
  h120 checkpoint loop6 exact moves only `0.2637 -> 0.2676 -> 0.2461`, and final
  full eval reaches h120 exact `0.2754`, below the step19600 final `0.2891`.
  h96/h108 remain strong at `0.9961`/`0.9492`, so this is not foundation
  collapse; it is low marginal ROI for the same h120 hard-stage continuation.
  Do not spend the next budget on another identical 3k-step extension. Change an
  effective scaling axis or test a simple FutureSeed/loop state update that lets
  late loops keep revising boards.
- The delayed loop-credit probe answers the simplest objective-shaping escape
  hatch. Resuming step22600 with `LOOP_LOSS=delayed` and `LOOP_LOSS_START=4`
  keeps h96/h108 alive, but h120 loop6 exact falls to `0.1602` at step23100 and
  h120 loop1/3/4/5/6 exact is only `0.0000`/`0.0781`/`0.1406`/`0.1563`/`0.1602`.
  Moving supervision credit to late loops does not create late correction; it
  erases useful h120 structure. Stop delayed loop-credit variants unless a
  different state dynamic gives a reason to revisit them.
- Loop remains essential at h120: final h120 loop1 exact was `0.0`, loop3 was
  `0.0664`, and loop6 was `0.1270`. This supports recurrence as the active
  mechanism, not one-pass prediction. It does not justify selector or repair
  work because the clean path still has positive compute slope and K-oracle has
  not shown selector headroom.
- For packed D320, the loop evidence is even clearer: final h120 loop1/3/4/5/6
  exact is `0.0000`/`0.0820`/`0.1562`/`0.1699`/`0.1738`. The remaining limit is
  not one-pass recognition; it is turning late-loop local fill into more valid
  full-board solves.
- At step16600, packed D320 h120 loop1/3/4/5/6 exact is
  `0.0000`/`0.1621`/`0.2188`/`0.2363`/`0.2363`. Loop remains essential, but
  loop6 no longer adds exact solves beyond loop5, so do not respond with a
  loop-count sweep. Buy more clean training or change the state update only if
  exact stalls.
- At step19600, h120 loop1/3/4/5/6 exact is
  `0.0000`/`0.2070`/`0.2773`/`0.2852`/`0.2891`. Loop is still the mechanism,
  but the marginal loop5-to-loop6 gain is small. Deeper loop counts remain low
  ROI.
- At step22600, h120 loop1/3/4/5/6 exact is
  `0.0000`/`0.1914`/`0.2656`/`0.2715`/`0.2754`. Loop remains necessary, but the
  last two loops are mostly polishing. The next mechanism question is not "more
  loops"; it is whether the state update can keep correcting wrong partial
  boards after loop4 without adding Sudoku-specific repair.
- Delayed loop credit made that loop curve worse, not better. At step23100,
  h120 loop3/4/5/6 exact is `0.0781`/`0.1406`/`0.1563`/`0.1602`; this rejects
  the theory that all-loop loss alone is the late-loop bottleneck.
- The learned loop update gate is the first simple state-dynamics change with a
  positive h120 signal after the packed D320 plateau. Resuming the step22600
  checkpoint with `LOOP_UPDATE_MODE=learned_gate` and init `0.95` reaches final
  h120 loop6 exact `0.3340`, with checkpoint step23600 h120 loop6 exact
  `0.3496`. This beats the clean step19600/22600 finals `0.2891`/`0.2754` and
  avoids the delayed-loss collapse `0.1602`. The learned gate is not just a
  no-op: final eval uses lower update gates in early loops and near-full update
  in loops 4-6. This supports simple learned state dynamics as a real mechanism.
- The same learned-gate run still has h132 exact `0.0` and h132 blank accuracy
  only about `0.17`. Do not overclaim it as opening the next frontier. It makes
  h120 late-loop correction better; it does not solve the larger-scale
  interaction problem. Next work should either continue this checkpoint with
  meaningful compute, or test one slightly richer learned update rule such as
  per-channel/state-conditioned gating. It should not pivot back to selector,
  repair, Sudoku priors, feature-noise tables, or loop-count sweeps.
- Continuing the learned-gate checkpoint to step24600 did not restore a useful
  slope. The checkpoint h120 loop6 exact was `0.3281`, below the prior
  learned-gate checkpoint `0.3496` and not above the prior full-eval `0.3340`.
  The run was stopped deliberately. Treat this as evidence that a single scalar
  learned update gate helped the plateau once, but same-direction extra steps
  are now low ROI unless paired with a real state-dynamics or scaling change.
- A new detached worktree can fail before GPU training if `.cache/bin/ninja` is
  not linked or on `PATH`. The failed
  `d320-mb48eff96-h120-s13600-20260615T205559Z-785f3cd` launch is an environment
  abort, not a model result. Future remote launchers should always link the base
  repo-local ninja or prepend `/huyang2/double-loop/.cache/bin` and
  `/opt/conda/bin` before loading the RWKV CUDA extension.
- Treat GPU halts as platform events when a periodic checkpoint exists. The
  interrupted `d320-effb72-h120-resume9800-s12800-20260615T1703Z-785f3cd`
  segment resumed cleanly into
  `d320-effb72-h120-resume10000-s12800-20260615T1801Z-785f3cd`; do not count the
  interrupted segment as a model result.

## 2026-06-17 Maze proxy

- The no-Hydra EqR maze runner works on GPU1 and gives a cheap proxy for
  recurrence on grid path propagation. It runs online generated mazes, archives
  config/logs/source metadata, and does not require Hydra or flash-attn.
- Visualization is part of the experiment evidence, not a presentation layer.
  Every archived run should have a run-level dashboard with training curves,
  loop readouts, difficulty-transfer plots when available, and links to
  case-level artifacts. For Maze runs, save case trajectories by default so the
  loop story can be judged from concrete false-positive/false-negative changes,
  not only scalar F1. If an old run lacks per-case predictions, explicitly show
  that limitation in the dashboard instead of pretending it can be reconstructed.
- The first 15x15 perfect-maze probe is neutral for FutureSeed transfer:
  FutureSeed loop8 path F1 was `0.5912`, base loop8 path F1 was `0.5905`, exact
  was `0.0` for both, and loop gain was only `+0.0008` for FutureSeed versus
  `+0.0000` for base.
- The harder Maze21 pressure setting is the first useful non-Sudoku signal.
  With 21x21 perfect mazes and path length 80-140, the 1200-step FutureSeed run
  reached loop10 path F1 `0.8051`, while the matched base reached `0.7614`.
  More importantly, FutureSeed showed a larger loop gain (`+0.1177` versus
  `+0.0518`) and reduced over-predicted path mass more aggressively. This
  suggests the loop is doing real path refinement, not just producing a static
  one-pass mask.
- The 2400-step Maze21 FutureSeed long-viz run opens exact path solving:
  loop1 exact/path F1 is `0.0000`/`0.7817`, while loop10 exact/path F1 is
  `0.9961`/`0.9997`. Train exact stayed zero through step1000, appeared at
  step1400, jumped to `0.6250` by step1800, and reached `0.9531` at step2400.
  The earlier 1200-step high-F1/no-exact result was therefore not a hard
  mechanism ceiling; it was under-computed for exact global cleanup.
- Maze21 visualizations show the actual loop behavior. Loop1 tends to mark a
  broad connected path region with many false-positive corridors. Loops 2-4
  remove most wrong branches while preserving the true path, and later loops
  polish rare misses. In the largest-gain case 191, loop1 has F1 `0.5972` with
  69 false positives and 16 misses; loop10 has F1 `1.0000` with zero false
  positives and zero misses. This is useful evidence that recurrence is doing
  iterative correction rather than a cosmetic confidence pass.
- On harder Maze31, state dynamics rather than candidate availability is now the
  main bottleneck. Dense ranking learned an average true-path vs false-positive
  score gap but did not prune the mask; the follow-up confidence-aware state
  competition (`maze31-cross-confstate-d256l4-s800-20260617T215136Z-91b7671`)
  kept non-collapsed candidate weights at loop12
  keep/proposed/context `0.3486`/`0.4412`/`0.2102` and high
  context-vs-proposed RMS `0.9635`, yet loop1-to-loop12 path F1 moved
  `0.5852 -> 0.5849`, precision `0.4158 -> 0.4154`, and predicted path fraction
  `0.4643 -> 0.4651`. Ten hard visualized cases stayed at zero false negatives
  but about 288 false-positive path cells. Do not spend the next budget on
  ranking-loss variants, gate-bias/temperature sweeps, or extra candidate
  diversity. The next high-ROI mechanism is a stronger generic recurrent
  decision-boundary update, such as learned threshold or normalization state,
  that can actually turn score gaps into mask sparsity without maze repair or
  rule priors.
- The direct learned decision-boundary probe
  (`maze31-boundary-d256l4-s800-20260617T224859Z-d538baa`) shows that the
  boundary can move, but current training pressure moves it in the wrong
  direction. Loop12 path F1 was `0.5854`, loop gain `-0.0003`, precision
  `0.4177 -> 0.4173`, and predicted path fraction `0.4587 -> 0.4593`. The
  learned threshold was not tiny: abs mean was about `0.30`. But raw path
  fraction was only `0.4339` at loop12 and calibration expanded it to `0.4606`;
  prune-flip fraction was `0.0` while add-flip fraction was `0.0267`. Ten hard
  visualized cases got slightly wider, with false positives `283.5 -> 285.1`.
  This reframes the bottleneck: a learned threshold is trainable and can affect
  predictions, but CE plus path-weight pressure rewards high-recall expansion
  once the true path is covered. Do not sweep threshold scale, gate bias, seed,
  or ranking variants. The next useful mechanism should alter generic recurrent
  training pressure so later loops are rewarded for reducing excess predicted
  mass without maze repair or topology rules.
- The late-loop path-mass pressure probe
  (`maze31-boundary-mass-w1-d256l4-s800-20260617T234853Z-3e1bfc3`) is the first
  clear sign that training pressure can reverse the expansion tendency. Loop12
  path F1 moved `0.5849 -> 0.5879`, precision `0.4154 -> 0.4235`, and predicted
  path fraction `0.4651 -> 0.4437`. Soft mass diagnostics moved in the intended
  direction too: non-PATH PATH probability `0.2630 -> 0.2137`, excess PATH
  probability fraction `0.1717 -> 0.1051`. But recall fell `1.0000 -> 0.9727`,
  true-path PATH probability fell `0.7880 -> 0.6500`, and hard visualized cases
  traded false positives `285.5 -> 276.5` for false negatives `0.0 -> 11.7`.
  This is a weak positive mechanism result, not a solved method. Do not sweep
  mass weight. The next high-ROI direction is a stronger generic
  recall-preserving pruning objective, focused on reducing excess predicted mass
  while explicitly preventing true-path probability collapse.
- The current visualization selector mostly captured solved largest-gain cases.
  That is good for explaining what the loop fixes, but not enough for studying
  the rare remaining failures after loop10. The next visualization upgrade
  should reserve final-failure cases first, then fill remaining slots with
  largest-gain cases.
- The Maze proxy reinforces the clean scaling principle: continue with more
  compute, harder tasks, and simple recurrent state dynamics. Do not pivot to
  selector, path repair, or handcrafted maze priors while this scalable route is
  still producing clear gains.
- Maze31 is the first clear failure frontier for the current small maze model.
  With 31x31 perfect mazes, path length 160-260, hidden 192, 2 layers, train
  loops 6 and eval loops 12, the run finishes at loop12 path F1 `0.5094` and
  exact `0.0000`; loop1 path F1 is `0.5203`, so the loop gain is negative
  (`-0.0109`). Training F1 oscillates around `0.55` from step200 through
  step1800 and CE stays near `0.35`, unlike Maze21 where exact opens late.
- The Maze31 failure visualizations are qualitatively different from Maze21.
  Maze21 loop1 gave a broad but useful path guess that later loops pruned. In
  Maze31 D192, loop1 already misses too much of the true path and later loops
  mostly preserve a wrong mask. Case 71 drops from loop1 F1 `0.3478` to loop12
  F1 `0.1763`, with false negatives rising from 87 to 127. This rejects "just
  add eval loops" as the next answer for Maze31.
- The immediate high-ROI Maze31 question is effective capacity, not base
  comparison. A D320 short capacity probe can decide whether the frontier is
  representation capacity or whether the next useful axis is curriculum/state
  dynamics. Do not extend the exact D192 Maze31 configuration.
- The D320 Maze31 capacity probe was interrupted after step600 by the GPU1
  lease, before final eval, but the early training curve is still informative:
  path F1 stays around `0.56` with exact `0.0`, matching the D192 plateau rather
  than showing a capacity breakout. Do not rerun the same D320 hard-from-step-1
  setup just for a final number. The next high-ROI Maze31 test is data/curriculum
  scaling, not another same-distribution width repeat.
- Maze31 path-length curriculum is a negative result with a useful cause. The
  `80-140` warmup reached only path F1 `0.1634` at step600, and the tighter
  `120-200` bridge collapsed to path F1 `0.0164` at step400. Shorter paths make
  PATH labels too sparse and encourage conservative non-PATH predictions. This
  is worse than hard-from-start, which at least learns a broad path mask.
- Maze31 simple depth scaling also does not open the frontier. Hidden-192 with
  4 layers reaches loop12 path F1 `0.5364`, exact `0.0`, and loop gain `-0.0008`.
  Loop1 and loop12 have almost identical precision/recall, so the loop is not
  revising the state. Width-alone, depth-alone, and naive curriculum are all low
  ROI now; the next useful step is a simple state-dynamics change or genuinely
  larger effective compute, not another one-axis table entry.
- Joint width+depth scale gives only a local mask improvement on Maze31, not
  global solving. D256/L4 reaches loop12 path F1 `0.5649`, exact `0.0`, and loop
  gain `-0.0013`. Recall rises to `0.8061`, but precision is only `0.4376` and
  predicted PATH fraction is `0.3559` versus true `0.1932`. The failure is now
  sharper: the model can cover much of the true path, but it cannot prune wrong
  branches, and the recurrent loop is nearly inert.
- Fixed linear state dynamics changes the coverage bias but does not solve
  Maze31. Positive delta-carry (`scale=0.35`, decay `0.95`) raises loop12 path
  F1 to `0.5721` and makes loop gain slightly positive (`+0.0012`), but it
  increases predicted PATH fraction to `0.3697` and does not improve precision.
  Negative delta-carry (`scale=-0.35`) collapses predicted PATH fraction to
  `0.1217` and recall to `0.2846`, yielding path F1 `0.3481`. This supports the
  state-dynamics thesis but rejects fixed sign/scale carry transforms; next work
  should be a small learned/gated update that can adaptively keep or prune.
- The first learned-gate implementation was diagnostic: applying the gate after
  logits left it untrained because loop carry is detached between calls. The
  corrected readout-gated run proves the gate can learn (`H` gate mean moves
  `0.896 -> 0.876`, std nonzero), but it still does not reopen recurrence:
  loop12 path F1 is `0.5851`, loop gain is only `+0.0001`, and predicted PATH
  fraction rises to `0.4646` versus true `0.1932`. A simple per-token keep gate
  is not enough; do not sweep gate bias. The next generic state update needs
  richer comparison/competition between candidate path hypotheses, not another
  scalar coverage knob.
- State competition is a small move toward pruning but not yet a loop mechanism.
  The `state_compete` run lets H state choose between previous, proposed, and
  context-competed candidates. It improves the broad-mask operating point versus
  readout-gate (`pred_frac 0.4556` vs `0.4646`, precision `0.4198` vs `0.4157`,
  loop12 F1 `0.5874` vs `0.5851`), but loop gain is still `-0.0001`. The
  softmax remains dominated by the proposed candidate (`~0.961` at loop12), so
  it does not yet use later loops to revise. Next work should increase real
  recurrence pressure or make the competing candidate stronger; do not turn this
  into a seed/gate-bias table.
- Final-only recurrence pressure does not fix `state_compete` collapse. The
  Maze31 D256/L4 final-loop-only run with `train_loops=8` reaches loop12 path F1
  `0.5852`, exact `0.0`, and loop gain `-0.0001`. Loop1 and loop12 are nearly
  identical: precision `0.4163 -> 0.4162`, predicted path fraction
  `0.4626 -> 0.4629`, and candidate weights stay collapsed around proposed
  `0.961`. Casebook failures show broad false-positive path branches copied
  from loop1 to loop12. This rejects simple loss-pressure as the next path; the
  next high-ROI mechanism is a stronger generic context/alternative candidate
  that can create real comparison before the softmax competition.
- Stronger generic candidate generation fixes weight collapse but not loop
  correction. `state_compete_cross` builds context from previous/proposed/delta
  and channel interaction, then applies noncausal attention. On Maze31 D256/L4
  it reaches loop12 path F1 `0.5847`, exact `0.0`, and loop gain only `+0.0002`.
  The weights are no longer collapsed (`keep/proposed/context =
  0.370/0.389/0.241` at loop12) and the context is genuinely different from
  proposed (`context_minus_proposed_rms ~1.03`), but precision and pred_frac
  barely move from loop1 to loop12 (`0.4156 -> 0.4160`, `0.4632 -> 0.4622`).
  This localizes the bottleneck: candidate diversity alone is insufficient; the
  alternative state must learn to carry error-correcting information across
  loops, likely via a simple temporal/predictive state objective rather than
  more gate, bias, temperature, or seed sweeps.
- Naive temporal prediction is learnable but teaches self-copying, not
  correction. The `state_compete_cross` predictive probe trained context logits
  at loop `t` to match stop-gradient logits at loop `t+1` with weight `0.1`.
  The predictive loss fell from `0.0094` to `0.0018`, and eval next-logit MSE
  fell to about `0.001`, so the auxiliary task worked mechanically. But loop12
  path F1 was only `0.5837`, exact `0.0`, and loop gain was effectively `0`.
  Precision/predicted fraction stayed frozen at `0.4142/0.4664`, recall was
  `1.0`, and casebook failures copied `290` false positives from loop1 to
  loop12. The lesson is sharp: predicting the next recurrent output just
  distills the broad mask. The next high-ROI objective must be improvement-aware
  or residual/contrastive, predicting how a later loop is better than loop1
  rather than predicting the next logits themselves.
- A simple CE-improvement target is also too weak for loop correction. The
  `state_compete_cross` context-improvement probe asked context logits to beat
  detached current logits on currently wrong tokens (`weight=0.1`,
  `margin=0.01`). The target was satisfied: at loop12 the diagnostic loss was
  `0.0`, context CE beat current CE by about `0.590` on current errors, and
  candidate weights stayed diverse (`keep/proposed/context =
  0.366/0.323/0.311`, `context_minus_proposed_rms = 0.950`). But loop12 path F1
  was `0.5830`, exact `0.0`, loop gain was `-0.0006`, and precision/predicted
  fraction worsened slightly from `0.4144/0.4654` to `0.4135/0.4672`. The hard
  cases still copy broad false-positive masks, e.g. `288 -> 290` false
  positives with no misses. This rejects token-level CE advantage as the next
  mainline; the correction signal must be tied to pruning or uncertainty
  sharpening, not just making context more confident on already-wrong cells.
- Ranking inside the current PATH mask is better aligned but still too weak in
  the hard min/max form. The `state_compete_cross` context-ranking probe trained
  context PATH scores so true-path cells inside the current predicted PATH set
  outrank false-positive PATH cells (`weight=0.05`, `margin=0.25`). This moved
  the operating point in the right direction but barely: loop1 to loop12
  precision was `0.4167 -> 0.4171`, predicted path fraction was
  `0.4608 -> 0.4596`, recall fell `0.9938 -> 0.9923`, and loop gain was only
  `+0.0001`. Candidate competition stayed healthy
  (`keep/proposed/context = 0.380/0.380/0.240`,
  `context_minus_proposed_rms = 1.004`), but the ranking loss stayed around
  `0.83` and the loop12 hard margin stayed negative (`-0.011`). Mean positive
  and negative PATH scores were almost identical. The lesson: ranking is the
  right family of signal for pruning, but the current hard min/max objective is
  too sparse or too hard. Do not seed-sweep it; make the pruning signal smoother
  or denser, or expose uncertainty/candidate contrast directly in the recurrent
  state update.
- Dense pairwise ranking fixes the average score separation but still does not
  produce loop-time pruning. The dense context-ranking probe used all true-path
  versus false-positive PATH pairs inside the current predicted PATH set
  (`weight=0.05`, `margin=0.25`, mode `dense`). It reaches loop12 path F1
  `0.5857`, exact `0.0`, and loop gain `+0.0003`. Precision and predicted path
  fraction move in the desired direction but only slightly (`0.4162 -> 0.4166`,
  `0.4634 -> 0.4629`), with recall nearly unchanged (`0.9984 -> 0.9982`).
  Unlike hard ranking, dense ranking learns a real average score preference:
  loop12 positive PATH score `4.0795` versus false-positive score `4.0246`,
  mean margin `+0.0548`. But the hard margin remains very negative (`-1.234`),
  ranking loss stays around `0.815`, and hard-case false positives barely move
  (`286.4 -> 286.2` average). The lesson: average pairwise preference is not
  enough to change the recurrent operating point. The next state-dynamics work
  should expose uncertainty/candidate contrast to the state or learn a simple
  threshold/normalization mechanism; do not keep adding ranking-loss variants.
- The useful signal is diagnostic, not positive: the model is mostly learning a
  broad path mask. In the FutureSeed run, the true path fraction was `0.1785`
  while the predicted PATH fraction was `0.4209`; recall was almost `1.0` but
  precision was only `0.4236`. This explains why exact is zero and why loops do
  not matter yet.
- Do not use this 15x15 run to claim FutureSeed works on maze. The next maze
  experiment should make recurrence pressure real, for example by increasing
  path length/grid size or by tracking whether later loops reduce over-predicted
  PATH cells. A flat seed table at the same 15x15 setting would be low ROI.
- The 21x21 recurrence-pressure run does create a meaningful loop signal.
  With path range `80-140`, path-loss weight `1.5`, hidden `192`, layers `2`,
  and train/eval loops `6/10`, FutureSeed reaches loop10 path F1 `0.8051`
  versus base `0.7614`. FutureSeed's loop gain is `+0.1177`; base loop gain is
  `+0.0518`. This supports recurrence as a transferable mechanism beyond
  Sudoku.
- The Maze21 FutureSeed gain is not just more PATH recall. At eval, the true
  path fraction is `0.2070`; FutureSeed moves predicted path fraction from
  `0.3270` at loop1 to `0.2797` at loop10, while precision rises
  `0.5640 -> 0.7031`. Base improves too, but remains broader at loop10
  predicted path fraction `0.3198` and precision `0.6308`. This is the first
  clean non-Sudoku evidence that FutureSeed plus loop can refine an over-broad
  global hypothesis.
- Exact path solving is still not opened on Maze21 held-out eval: both arms have
  loop10 exact `0.0`. Do not overclaim. The next high-ROI maze question is
  whether more compute or a simple state-dynamics change can sharpen from
  high-F1 masks to exact single paths. A seed table at the same budget is lower
  ROI than increasing recurrence pressure or examining exact-failure cases.
- Guarded path-mass pressure preserves recall but kills pruning. The
  `maze31-boundary-massguard` probe changed the late-loop mass objective so that
  when true-path probability fell below the loop1 floor, the model optimized
  only the recall guard and stopped applying non-path mass pressure. It did
  preserve recall (`1.0000 -> 1.0000`), but loop12 exactly copied loop1 at the
  hard mask level: path F1 `0.5847 -> 0.5847`, precision
  `0.4152 -> 0.4152`, predicted PATH fraction `0.4653 -> 0.4653`, hard-case
  false positives `288.2 -> 288.2`, and false negatives stayed `0.0`. Soft
  non-path PATH probability moved only slightly (`0.2627 -> 0.2561`), while the
  guard was active on every eval sample at loop12
  (`path_mass_guard_violation_frac = 1.0`). The lesson: recall-preserving
  pruning cannot be a stop-pressure guard. If the guard disables false-positive
  gradients, loops learn to copy the broad mask. The next objective must keep
  positive-cell margins and false-positive mass pressure active at the same
  time, for example through a normalized margin or constrained/Lagrangian form,
  not another sweep of mass weight, threshold, margin, or seed.
- Simultaneous path-mass pressure with a true-cell probability floor avoids the
  two obvious failures but still does not create useful loop correction. The
  `maze31-boundary-massconstr` retry kept non-path PATH-mass pressure active and
  added a cell-level loop1 probability floor on true-path cells. It no longer
  froze perfectly like guarded pressure, and it did not collapse recall like the
  original soft mass run. But the effect was tiny: loop1 to loop12 path F1
  `0.5836 -> 0.5841`, precision `0.4144 -> 0.4151`, recall
  `0.9984 -> 0.9971`, predicted PATH fraction `0.4655 -> 0.4641`, and hard-case
  false positives/false negatives `288.1/0.5 -> 287.5/1.3`. The positive floor
  was active on `30.7%` of true-path cells at loop12, but non-path PATH
  probability slightly worsened (`0.2556 -> 0.2581`). Lesson: probability-floor
  regularization mostly creates small calibration tradeoffs, not a robust
  prune/keep decision. Do not sweep this floor mode. The next high-ROI direction
  should change the recurrent decision variable itself, for example a learned
  per-loop budget/normalization state or a contrastive boundary objective that
  directly separates true-path from false-positive PATH candidates while
  preserving true-path margins.
- Learned recurrent budget state is not enough when the training pressure still
  points at broad masks. The `maze31-budgetstate-massconstr` probe added a
  generic per-sample per-loop budget shift to the decision boundary, using only
  PATH margin, soft PATH mass, entropy, candidate disagreement, candidate
  weights, and loop index. The module was active and candidates stayed diverse
  (`keep/proposed/context = 0.376/0.444/0.181`, context-proposed RMS `1.204`),
  but the learned boundary moved in the wrong direction: loop12 threshold mean
  was negative (`-0.4865`), prune flips were exactly `0.0`, and add flips were
  about `0.0200`. Loop1 to loop12 improved only slightly in F1
  (`0.5852 -> 0.5861`) and precision (`0.4168 -> 0.4184`), while recall dropped
  (`0.9938 -> 0.9897`). Hard visualized cases confirm the failure mode:
  false positives barely changed (`283.4 -> 282.2`) while false negatives rose
  (`3.3 -> 5.6`). Lesson: boundary capacity is no longer the clean bottleneck.
  If the objective rewards coverage more than correction, a learned budget
  variable also becomes an expander. Do not sweep budget scale, threshold bias,
  seed, or margin; the next work should change the generic self-correction
  pressure itself.
- Loop-pair self-correction is learnable but still turns into calibration, not
  clean repair. The `maze31-selfcorr-w1m002` probe used loop1 as the broad mask
  reference and directly trained later loops to lower PATH probability on loop1
  false-positive cells while preserving true-path probability. The loss did
  move (`0.0500` at step100 to `0.0412` at step800), and hard predicted PATH
  fraction fell slightly (`0.4644 -> 0.4617`). But the intended probability
  signal did not happen: loop12 false-positive candidate PATH probability was
  slightly higher than loop1 (`+0.0007`), while true-path probability rose
  (`+0.0027`). Hard counts show a tradeoff rather than correction: FP count
  improved by `-2.11`, but FN count worsened by `+0.54`; hard visualized cases
  moved from `286.5/1.0` FP/FN to `285.0/2.3`. The boundary still had
  `prune_flip=0.0` and `add_flip=0.0432`. Lesson: a loop1-relative soft
  probability objective is more direct than CE/mass/floor, but it is still too
  easy to satisfy through calibration and recall tradeoff. Do not sweep
  self-correction weight, margin, seed, or start loop; the next direction needs
  a stronger generic sparse-correctness pressure rather than another small
  probability regularizer.
- PATH decision-margin pressure is also insufficient by itself. The
  `maze31-pathmargin-w1-p025n025` probe trained later loops to put true PATH
  cells above the strongest non-PATH logit and non-PATH cells below it. The
  margin objective did move soft quantities: false-positive PATH margin mean
  fell `1.1403 -> 0.1463`, non-PATH PATH probability fell
  `0.2551 -> 0.1813`, and calibrated soft PATH fraction fell
  `0.3533 -> 0.2507`. But hard outputs did not change at all: loop1 and loop12
  path F1, precision, recall, and predicted PATH fraction were identical
  (`0.5834`, `0.4139`, `1.0000`, `0.4667`), with `prune_flip=0.0` and
  `add_flip=0.0`. The ten visual hard cases were literally unchanged
  (`290.0/0.0 -> 290.0/0.0` FP/FN). Lesson: loss-only pressure can make the
  model less confident on false positives without moving the discrete decision
  boundary. Do not sweep path-margin weight, margin, seed, start loop, or
  temperature. The remaining bottleneck is the generic recurrent decision
  mechanism that converts soft uncertainty into pruning, not another scalar
  regularizer.
- Clean cross-candidate scale can fail before the loop question even starts.
  The `maze31-crossscale-d320l6-loop8x16` probe deliberately removed the scalar
  pruning/correction losses and spent more generic capacity and recurrence:
  hidden `320`, layers `6`, train/eval loops `8/16`, `state_compete_cross`,
  path-loss weight `1.5`, and batch `12`. GPU use was healthy
  (`~51GB`, `98-100%`), but PATH prediction never opened: steps
  `100/200/300/400` all had path F1 `0.0`, with CE stuck around
  `1.06`. The run was killed by exact PID rather than burning the lease. Lesson:
  naive scale-up of the more complex generic state update can introduce an
  optimization attractor where the model predicts no path. This is not a reason
  to add maze-specific repair or resume scalar-loss sweeps. It says the next
  bitter-lesson-compliant scaling move must preserve the ability to open PATH
  prediction, either through a data/compute schedule or a simpler optimizable
  state update, before asking whether later loops can prune.
- The matched clean D320/L6 fallback changes that interpretation in an important
  way. Removing `state_compete_cross` and keeping only FutureSeed + loop at
  hidden `320`, layers `6`, train/eval loops `8/16`, path-loss `1.5`, and batch
  `16` still looked dead at steps `100/200/300` with path F1 `0.0`, but then
  opened at step400: CE `0.3691`, path F1 `0.5861`, exact `0.0`. The process
  was killed before final eval, so there is no loop16 held-out score. Lesson:
  the clean scale path is slow-starting but not dead; the complex
  cross-candidate state update made optimization worse. The next high-ROI run
  is not another scalar loss or a bigger `state_compete_cross`; it is a clean
  D320/L6 run long enough to finish eval and answer whether loop16 can prune
  after PATH prediction opens.
- The full clean D320/L6 Maze31 readout shows that clean scaling opens PATH but
  still does not create strong late-loop correction. The
  `maze31-cleanscale-d320l6-loop8x16-s1200` run used only FutureSeed + loop,
  hidden `320`, layers `6`, train/eval loops `8/16`, path-loss `1.5`, batch
  `16`, and no state competition, scalar pruning loss, repair, search, selector,
  feature noise, or maze rule. It reproduced the slow opening pattern:
  steps `100/200/300` had path F1 `0.0`, step400 opened at CE `0.3691` and
  train path F1 `0.5862`, then the curve fluctuated and ended with train path
  F1 `0.4226`. Held-out loop1 to loop16 improved only
  `0.5301 -> 0.5344`, with precision essentially flat
  (`0.4431 -> 0.4434`), recall up (`0.6664 -> 0.6794`), and predicted PATH
  fraction up (`0.2907 -> 0.2961`). The 16-case visualization aggregate shows
  the same failure mode: average false positives rose `202.9 -> 205.1` while
  false negatives fell `88.1 -> 86.6`. Lesson: clean scale is more optimizable
  than the complex cross-candidate update, but loop depth is still mostly adding
  coverage rather than pruning wrong branches. Do not repeat this exact setup as
  a loop-depth or same-budget scale sweep. The next bitter-lesson-compliant move
  should be a simple generic recurrent decision/state dynamic that can revise a
  boundary, not another scalar loss, selector, repair, or maze-specific prior.
- Equal-compute EqR baseline now gives positive evidence for FutureSeed as an
  opening mechanism on hard Maze31. The `maze31-eqrbase-d320l6-loop8x16-s1200`
  run matched the clean FutureSeed D320/L6 setup except
  `EQR_FUTURE_SEED_SCALE=0`: same path range `160-260`, batch `16`, train/eval
  loops `8/16`, hidden `320`, layers `6`, no state competition, no feature
  noise, no pruning loss, no selector, no repair, and no maze rule. It stayed
  completely unopened through step600: path F1 was `0.0` at steps
  `100/200/300/400/500/600`, with CE stuck around `1.06`, so it was stopped by
  exact PID. The matched FutureSeed run opened at step400 with train path F1
  `0.5862` and final loop16 eval path F1 `0.5344`. Lesson: FutureSeed is not
  merely a Sudoku-specific trick; it materially changes optimization/opening on
  Maze31 against an equal-compute EqR baseline. But this only proves the first
  criterion. The second criterion is still open: FutureSeed+loop must make later
  loops reduce false positives without adding false negatives before we claim it
  is a stronger full paradigm than EqR.
- Generic loss-only FP/FN pressure can harm the very opening behavior that makes
  FutureSeed valuable. The `maze31-fs-tversky-w075-a075b065-d320l6-loop8x16`
  probe added a late-loop soft TP/FP/FN Tversky objective to the clean D320/L6
  FutureSeed setup (`weight=0.75`, `alpha=0.75`, `beta=0.65`), with no maze
  rules, repair, search, selector, feature noise, state competition, or sweep.
  It stayed in the no-PATH attractor through step600: path F1 was `0.0` at
  `100/200/300/400/500/600`, CE stayed around `1.06-1.08`, and Tversky loss
  stayed around `0.82`. This is worse than the matched clean FutureSeed run,
  which opened at step400. Lesson: the second criterion cannot be solved by
  simply adding a direct pruning loss after the fact. The next bitter-lesson
  compliant direction must preserve the FutureSeed opening dynamics while
  changing generic recurrent state/decision dynamics, or use a curriculum that
  applies correction pressure only after the model has learned to open.
- Delaying generic FP/FN pressure until after FutureSeed opening is useful for
  the training trajectory, but it still does not prove loop self-correction. The
  `maze31-fs-delayedtversky-a450-w075-a075b065-d320l6-loop8x16` run kept
  Tversky weight `0` through the step400 opening point, then enabled the same
  `weight=0.75`, `alpha=0.75`, `beta=0.65` pressure at step450. Unlike the
  always-on run, it opened cleanly (`step400 path_f1=0.5455`) and stayed stable
  after pressure turned on; final loop16 path F1 rose to `0.5824`, better than
  the clean D320/L6 final `0.5344`. But loop dynamics remained almost static:
  loop1 to loop16 was only `0.5822 -> 0.5824`, precision
  `0.4305 -> 0.4308`, recall `0.9096 -> 0.9091`, pred PATH fraction
  `0.4082 -> 0.4077`, and visual FP/FN `260.5/30.8 -> 260.2/31.3`. Lesson:
  delayed pressure can improve the overall operating point, but it is not the
  missing recurrent repair mechanism. Do not sweep Tversky timing/weight next;
  use this as evidence that the main bottleneck is generic recurrent
  state/decision dynamics that make later loops perform real revision.
- Source snapshots must stay lean on the remote GPU path. The first
  `maze31-fs-feedback-probembed` launch at SHA `c3065f3` was aborted before
  Python training or GPU compute started because `run.sh` spent more than
  5 minutes building a full `git archive` over tracked historical runs. The
  follow-up commit `7021726` added `SOURCE_SNAPSHOT_MODE=lean`, reducing the
  source snapshot to about `199KB` and allowing the same experiment to start
  normally. Lesson: archival should capture the exact SHA, patch, key scripts,
  and run metadata without letting old run artifacts become a startup
  bottleneck.
- Generic previous-belief feedback is active but does not create late-loop
  self-correction on Maze31. The successful
  `maze31-fs-feedback-probembed-s1-delayedtv-a450-d320l6-loop8x16-s1200` run
  fed detached previous-loop logits back as token-embedding expectations, with
  no maze rules, search, repair, selector, or sweep. It slightly improved the
  final operating point over delayed Tversky alone (`0.5836` vs `0.5824`
  loop16 path F1), and the feedback module was mechanically active
  (`gate ~= 0.126`, `rms ~= 0.107`). But loop dynamics stayed flat:
  loop1 to loop16 path F1 was `0.583669 -> 0.583586`, precision
  `0.428444 -> 0.428409`, recall `0.926020 -> 0.925730`, and predicted PATH
  fraction `0.417560 -> 0.417465`. Lesson: the missing ingredient is not
  simply giving later loops access to earlier predictions. Do not sweep
  feedback scale, gate bias, or seed. The next high-ROI work needs a generic
  state transition or training signal that rewards a measurable correction
  trajectory, while preserving FutureSeed's opening advantage.
- A delayed hard-decision loop-pair correction target also does not solve
  Maze31 self-correction. The
  `maze31-hardcorr-w025-delayedtv-a450-d320l6-loop8x16-s1200` run preserved the
  useful FutureSeed opening trajectory: path F1 was `0.0` through step300,
  opened at step400 (`0.5851`), and completed step1200. Final loop16 path F1
  was `0.5844`, slightly above delayed Tversky and feedback, but loop dynamics
  again stayed flat: loop1 to loop16 was `0.584663 -> 0.584378`, precision
  `0.416904 -> 0.416740`, recall `0.990164 -> 0.989434`, and predicted PATH
  fraction `0.458850 -> 0.458673`. The new hard-correction diagnostics are the
  key result: loop16 candidate false-positive margin stayed positive
  (`0.3010`), prune success was exactly `0.0`, and the tiny false-positive
  count reduction (`-0.029`) was offset by new false negatives (`+0.141`).
  Lesson: simply supervising loop1 false positives after opening is still not
  enough to make recurrent compute move the hard PATH boundary. Do not sweep
  hard-correction weights, margins, or seeds. The next useful direction must
  change the recurrent state dynamics or target schedule so later loops have a
  robust way to revise discrete decisions without trading away recall.
- A minimal learned recurrent gate is not enough to recover Maze31 opening on
  the hard D320/L6 setup. The
  `maze31-learnedgate-delayedtv-a450-d320l6-loop8x16-s1200` probe tested
  `EQR_STATE_UPDATE_MODE=learned_gate` with the same delayed Tversky pressure
  that previously preserved clean FutureSeed opening. It was a single
  mechanism test, not a gate-bias or Tversky-weight sweep. The run reached
  step600 with path F1 still exactly `0.0` at every logged point
  (`100/200/300/400/500/600`), CE stuck around `1.06-1.11`, and Tversky active
  from step500 onward. It was killed by exact PID and archived with
  `abort.json`. Lesson: a scalar keep/update gate adds cost but does not solve
  the no-PATH attractor or late-loop repair. Do not sweep learned-gate bias,
  width, seed, or Tversky weight. The bitter-lesson compliant path is now either
  clean FutureSeed scaling with more data/time/capacity, or a more substantial
  generic state/update objective that first preserves opening and then rewards
  loop-time correction.
- Plain longer clean FutureSeed training is not enough to make Maze31 loops
  self-correct. The
  `maze31-cleanlong-d320l6-loop8x16-s3000` run removed all extra mechanisms:
  no learned gate, Tversky, hard correction, feedback, selector, search, repair,
  maze rules, feature noise, or sweep. It did answer the opening question:
  training path F1 became nonzero at step500 (`0.5304`) and occasionally reached
  `0.59` on train batches. But held-out eval degraded badly versus the earlier
  1200-step clean baseline. Loop1 path F1 was `0.3998`; loop16 fell to
  `0.2995`, for loop gain `-0.1003`. Later loops did reduce predicted path mass
  (`pred_frac 0.1504 -> 0.0928`) and false positives (`78.23 -> 47.76`), but
  recall collapsed (`0.3573 -> 0.2227`) and false negatives increased
  (`119.35 -> 144.28`). Lesson: more steps on the current clean D320/L6 setup
  improve neither generalization nor correction. The model learns a narrower
  mask, not a better path. Do not run 5000-step clean repeats on this exact
  setup. The next bitter-lesson compliant move should either stabilize opening
  with a generic curriculum/data schedule or change FutureSeed/loop state
  dynamics so later loops preserve true-path mass while pruning false-positive
  mass.
- The previous Maze31 `EqR baseline` should be described as a patched
  no-Hydra online-proxy baseline, not a faithful official EqR reproduction.
  Official EqR comparison requires the upstream `locuslab/eqr` pipeline at
  `aba94e9cde0f273ce644db5261cd6915ba6561f0`, official `eqr_maze_unique`
  data/config, `scripts/train.sh eqr_maze_unique`, FlashAttention, and a
  working `adam_atan2_backend`. The `official-eqr-compare-gate-20260621`
  preparation created isolated `eqr-clean` and `eqr-futureseed` clones, with
  the FutureSeed patch restricted to 2 files and 45 insertions, but did not run
  training: `adam_atan2_backend` was missing after install attempts, and
  `huggingface_hub` timed out on `locuslab/EqR-data`. Do not claim official EqR
  reproduction until those two gates pass; do not silently replace AdamATan2
  with AdamW and call it official.
- Official EqR data and Python wheels can be staged offline, but the current
  GPU1 container still needs a clean restart before training evidence can be
  trusted. On 2026-06-21, `locuslab/EqR-data` was downloaded locally and
  uploaded as `eqr-data-full.tgz` (`1.3G` extracted under
  `/huyang2/double-loop/official_eqr_compare/eqr-clean/data`). The dependency
  wheelhouse was also downloaded locally and uploaded. A Python-only
  `adam-atan2` wheel did not contain `adam_atan2_backend`; NVIDIA's
  `nvidia-cuda-nvcc-cu12` wheels (`12.6.85` and probed `12.9.86`) exposed
  `ptxas`/headers but not `bin/nvcc`, so they could not build a PyTorch CUDA
  extension by themselves. A dependency-only fallback compiled the official
  `adam-atan2` source for A100 `sm80` with system CUDA 11.7 while bypassing
  PyTorch's CUDA-version guard, and the backend import passed. The wrapper also
  needed `PYTHONPATH` for venv dependencies, official Hydra fields
  (`epochs`/`train_epochs_per_iter` rather than a bogus YAML `max_steps`),
  `+` syntax for absent optional fields, W&B disabled, and direct single-GPU
  Python launching to avoid `torchrun` local distributed waits. Even then, the
  base sanity reached step 0 and stalled in unkillable `D` state at
  `cxiWaitEventWait` with 503MiB GPU memory and 0% util. Next official EqR
  attempt should restart GPU1 first, then rerun the direct-Python base/FutureSeed
  pair from the already staged offline data and wheels.
- The official EqR comparison is now a completed short-budget gate, and it
  changes the evidence boundary. On 2026-06-21, the official upstream EqR SHA
  `aba94e9cde0f273ce644db5261cd6915ba6561f0` was run on the official
  `maze-30x30-unique-1k` data for 64 epochs / 448 steps with the official
  `AdamATan2` optimizer backend. The dependency path was kept honest by staging
  wheels and Hugging Face data locally before uploading to GPU1; `adam-atan2`
  still needed a dependency-only A100 `sm80` source build because available
  CUDA-nvcc wheels did not contain `bin/nvcc`. The installed FlashAttention was
  ABI-incompatible with PyTorch 2.7.0+cu126, so the launcher applies the same
  runtime PyTorch SDPA fallback to clean EqR and FutureSeed. This is a kernel
  compatibility change, not a task/model/loss/repair/search trick. The
  official final eval at step448 gives FutureSeed a real token-level signal
  over clean EqR (`all/accuracy 0.5577 vs 0.5158`, `total_loss 1.4536 vs
  1.4784`), but exact accuracy is still `0.0` for both and loop16 residual is
  not better (`437.060` vs `436.811`). Lesson: the honest claim is now
  "FutureSeed improves official-code short-budget optimization", not
  "FutureSeed+loop solves Maze reasoning" and not "later loops perform
  correction". Next work should stay on the official EqR path and scale budget
  or loop-specific readout; do not fall back to local maze-probe tables,
  selector, repair, or human-rule postprocessing.
- Full-diversity data scaling is the strongest clean Sudoku scaling result so
  far. On 2026-07-12, P-SCALE-029 held D224/L12 native FutureSeed GDN, loop5,
  every-loop CE, effective batch128, curriculum, and 6000-step compute fixed,
  but replaced 1,000 source boards repeated through 1,001 augmentations with
  3,831,994 independent Sudoku-Extreme source rows. Final mixed loop5 exact
  rises from `0.2070` to `0.2500`; official `51-55` rises `0.2832 -> 0.3926`,
  and `56-64` rises `0.0801 -> 0.1270`. Holes64 checkpoint exact moves
  `0.0254 -> 0.2500` from loop1 to loop5. The key operational lesson is to
  avoid killing larger-data runs at step1000: this run trailed early, reached
  parity around step3000, then crossed strongly after step4500. The scientific
  lesson is that independent relational diversity lets later loops convert
  local token quality into globally valid boards. Prefer more independent data
  and useful compute on the efficient D224 frontier over blind width, task
  rules, repair, selector, or loss tables.
- Full-diversity hard-token scaling remains useful from step6000 to8000, but the
  marginal return is starting to shrink. P-SCALE-030 resumes the exact P-SCALE-029
  D224/L12 FutureSeed-GDN state and adds only 2,000 steps of independent 51-64
  blank boards. Official `51-55` exact rises `0.3926 -> 0.4473` and `56-64`
  rises `0.1270 -> 0.1621`; fixed holes53/60/64 loop5 exact reaches
  `0.2559/0.3066/0.2852`. The important mechanism signal is that loop1 stays
  fixed from step7000 to8000 while loop4/5 improve, and a 64-blank board goes
  `26 wrong -> 4 -> 0` across loops `1/3/5`. Thus extra hard compute improves
  late recurrent closure, not merely first-pass coverage. The primary gate
  passes, but the strong `56-64 >=0.20` or mixed `>=0.32` gate does not. Do not
  run the identical D224 state to 10000/12000 just to chase a threshold. The
  next scaling point should pair the proven 3.83M-board diversity with one
  larger generic backbone and enough tokens for delayed crossover; keep matched
  no-FS/EqR as a separate paper gate rather than mixing it into this result.

- 2026-07-15: The matched D224 full-diversity no-FutureSeed control gives a
  strong early causal result. At step1000, no-FS CE is `1.6580` versus `0.9971`
  with FutureSeed. On holes53/60/64, no-FS exact is zero and blank accuracy is
  only `0.2657/0.2677/0.2671`; matched FutureSeed reaches exact
  `0.0176/0.0215/0.0215` and blank accuracy `0.5145/0.5254/0.5250`. The no-FS
  model's loop1-5 outputs are nearly unchanged, and visual hard cases retain
  roughly 27 duplicate conflicts. This supports FutureSeed as an optimization
  and sample-efficiency mechanism, but it is not yet the final long-training
  claim because full-diversity models can accelerate after step3000. Continue
  only the predeclared matched arm; do not rescue no-FS with a hyperparameter or
  objective sweep.

  The stronger mechanism-specific readout is sequence position. Averaged over
  holes53/60/64 at step1000 loop5, no-FS blank accuracy is
  `0.2032/0.2633/0.3340` for early/middle/late sequence thirds, whereas
  FutureSeed is `0.5199/0.5223/0.5228`. FutureSeed does not merely raise the
  mean; it nearly removes the left-to-right context imbalance. Prefer this
  diagnostic when claiming cheap future context.

  The same difference persists after clean scaling to step3000. Matched no-FS
  remains at CE `1.6379`, fixed holes53/60/64 exact all zero, and loop5 blank
  `0.2737/0.2737/0.2755`; FutureSeed is CE `0.7939`, exact
  `0.0273/0.0332/0.0430`, and blank `0.5788/0.5885/0.5844`. No-FS has used
  three times the optimizer steps and still has not reached the FutureSeed
  step1000 regime. Its early/middle/late blank accuracy remains
  `0.2057/0.2685/0.3488`, while FutureSeed is position-flat at
  `0.5840/0.5815/0.5863`. This is a more direct cheap-future-context result than
  an aggregate score alone: without terminal-state seeding, extra training and
  extra loops do not remove the directional information deficit.

  The predeclared long-training gate closes the causal test at step4500. no-FS
  remains at CE `1.6372`, fixed holes53/60/64 exact all0, and blank
  `0.2755/0.2777/0.2766`; FutureSeed reaches CE `0.6415`, exact
  `0.1055/0.1172/0.1230`, and blank `0.6166/0.6332/0.6133`. Mixed FutureSeed
  exact rises `0.0234 -> 0.1113` across loop1-5, while no-FS stays zero. The
  positional gap also persists at `0.2085/0.2696/0.3518` without FutureSeed
  versus a nearly flat `0.6216/0.6193/0.6223` with it. Core model/training
  files were verified byte-identical across the two provenance SHAs. Stop the
  no-FS arm at4500 and do not rescue it with hyperparameter sweeps. The honest
  claim is cheap future-context initialization plus enabled recurrent
  refinement under matched causal compute; the next required paper gate is an
  explicit bidirectional/noncausal baseline measured on quality, throughput,
  VRAM, and parameter count.

- 2026-07-15: In the staged Sudoku runner, the global training endpoint is the
  sum of `HOLE_STAGES`; `FULL_STEPS` does not truncate a staged resume. For an
  exact resume to global step N, stage counts must sum to N. A resume that has
  already reached N can be evaluated without more optimizer updates by using
  stage counts that sum to N. Preserve the extra completed checkpoint and write
  `abort.json` when a legacy launcher overshoots; never silently discard valid
  training state.

## 2026-07-16 Native FutureSeed data-compute upper bound

- P-SCALE-034 supersedes the earlier recommendation to stop an unchanged
  D224/L12 run at step10000/12000. That recommendation was based on the old,
  repeatedly augmented 1,000-board regime and short hard-tail continuations.
  Under the 3,831,994-board independent-data regime, the same model is still on
  a strong positive slope: mixed loop5 exact rises `0.2715 -> 0.3066 -> 0.3730`
  from step8000 to10000 to12000, while official 56-64 exact rises
  `0.1621 -> 0.1855 -> 0.2285`.
- This is a data-and-compute result, not an architecture tweak. Width, depth,
  FutureSeed update, loop count, every-loop CE, optimizer, effective batch,
  and hard distribution stayed fixed. D256 had already failed to beat D224, so
  ordinary width is still a rejected scaling axis for this frontier.
- The extra compute is teaching recurrent closure rather than a stronger first
  guess. At step12000, mixed exact across loops1-5 is
  `0.0234 / 0.0664 / 0.2617 / 0.3496 / 0.3730`; loop1 remains fixed while later
  loops improve. Hard boards change from `31 -> 4 -> 0`, `30 -> 9 -> 0`,
  `15 -> 7 -> 1`, and `29 -> 12 -> 5` wrong cells across loops1/3/5.
- Do not turn this into unlimited training by faith. Continue the identical
  run to step16000 because every tracked hard metric set a new best. After
  step12000, two successive gates without at least a `+0.01` new best on mixed
  or official-hard exact terminate pure scaling for this state formulation.
- The identical step16000-to20000 continuation remains decisively positive:
  mixed loop5 exact rises `0.3945 -> 0.4375`, official 51-55 rises
  `0.5605 -> 0.5840`, and official 56-64 rises `0.2598 -> 0.3125`. Fixed
  holes53/60/64 also rise to `0.4238/0.4395/0.3945`. This passes the later
  `+0.01` gate and reaches the strong upper-bound criterion without changing
  model, data, objective, FutureSeed, or loop count.
- More compute still helps the recurrent passes rather than the first guess.
  At step20000, mixed loop1-5 exact is
  `0.0234/0.0840/0.3242/0.4160/0.4375`, and selected 56-64-blank boards change
  `33->5->0`, `30->12->0`, `24->11->1`, or `27->8->5` wrong cells across
  loops1/3/5. Continue unchanged to step24000; do not dilute this result with
  simultaneous width, loss, noise, or state-update changes.
- Step24000 is positive but no longer uniformly positive. Mixed loop5 exact
  rises `0.4375 -> 0.4648`, holes60/64 rise `0.4395/0.3945 ->
  0.4629/0.4414`, and official 51-55 rises `0.5840 -> 0.6387`; however,
  formal official 56-64 falls `0.3125 -> 0.2949`. The nearby case-bank value
  `0.3008` is not a substitute for the formal aggregate.
- Recurrent correction remains strong at step24000: mixed loop1-5 exact is
  `0.0234/0.1055/0.3477/0.4414/0.4648`, and selected hard boards change
  `32->7->0`, `32->11->0`, `31->3->0`, or `29->7->5` wrong cells. The final
  step30000 gate should distinguish continued broad scaling from a hardest-tail
  plateau. Do not rescue a flat endpoint with loss, noise, width, or seed
  sweeps.

## 2026-07-21 Official FLA backbone double-check

- Exact KDA continuation closes the remaining early-ranking ambiguity. At
  step1000, GDN/KDA/GDN2 CE is `0.9413/0.9491/0.9539`; mixed loop5 exact is
  tied at `0.02344`, fixed holes53 exact is tied at `0.01758`, and official
  46-50 exact is `0.99805/0.99609/0.99219`. All three remain zero exact at
  51-64 blanks. The step500 ranking was mostly a finite-budget artifact.
- More expressive state updates are not free. Under the same D32/expand-v2
  recipe, continuation cost is `5.15/5.65/5.70` seconds per step and peak
  allocation is `7281/7809/9382` MiB for GDN/KDA/GDN2. GDN is the pragmatic
  efficiency choice here, while KDA/GDN2 provide no hard-closure gain. Do not
  convert this into native-geometry, seed, LR, or loss tables.
- Do not infer an architecture ceiling from a short shared-recipe gate. At
  step500, official FLA GDN/GDN2 46-50-blank exact was `0.8594/0.7539`; after
  exact matched continuation to step1000 it was `0.99805/0.99219`. The large
  opening gap mostly disappeared, while mixed loop5 and fixed holes53 exact
  tied at `0.02344` and `0.01758`. The honest conclusion is that GDN is more
  efficient under this D32/expand-v2 recipe, not universally better.
- A more expressive recurrent update is not automatically a better FutureSeed
  carrier. Label-free state diagnostics show token81 seed retention of
  `0.2945/0.3925/0.5298` for GDN/KDA/GDN2, so KDA/GDN2 do not lose because they
  erase FutureSeed. More persistent future context can still be less useful to
  the task. Measure downstream closure, not only state survival.
- Interface matching and native-recipe matching are different experiments.
  KDA/GDN2 official defaults use a different head/state geometry from the
  shared D32/expand-v2 comparison. Keep the current result scoped to the shared
  recipe; do not launch a native-geometry table without a new mechanism-level
  prediction.
- A comparison where every arm has FutureSeed ranks FutureSeed-enabled
  backbones; it does not estimate FutureSeed's causal gain. Preserve the
  separate matched no-FutureSeed control for that paper claim.
- AIStation `remainTime` is seconds. Save optimizer and all RNG state every 100
  steps, and treat lease expiry as infrastructure interruption rather than
  model failure. Avoid whole-monorepo `git status` and snapshots on remote NFS
  after a detached worktree has already been verified; use a relevant-source
  snapshot and preserve `abort.json` plus exact checkpoint provenance.
- One dataset-equivalent nominal draw budget was still useful for the clean
  native FutureSeed GDN. From step8000 to step30000, mixed loop5 exact rose
  `0.2715 -> 0.4805`, while formal official56-64 rose `0.1621 -> 0.3848`.
  The step24000 dip to `0.2949` was evaluation variation, not a durable ceiling.
  Do not infer saturation from one finite hard-bucket checkpoint.
- Recurrent correction is now direct rather than inferred from blank accuracy.
  At step30000, mixed exact is `0.0234/0.0898/0.3379/0.4570/0.4805` across
  loops1-5, and official56-64 is `0/0.0391/0.2129/0.3496/0.3848`. Concrete
  boards change `37->8->0` or `31->14->0` wrong cells across loops1/3/5.
  Failures such as `29->14->8` remain; five loops help substantially but are
  not an unlimited solver.
- Prefer formal bucket aggregates over fixed-hole or visualization subsets.
  At step30000, formal official56-64 improves `+0.0898` from step24000 while
  fixed holes64 falls `-0.0195`; official51-55 also falls `-0.0234`. The honest
  result is strong hardest-tail progress with bucket-wise variance, not uniform
  monotonic improvement.
- Stop a successful scaling ladder at its preregistered endpoint. P-SCALE-034
  validates clean data/compute scaling but mixed gains are diminishing by 30k.
  The next run should change one generic axis such as independent hard-data
  coverage or learnable state capacity. Appending step36000 to the unchanged
  recipe would answer little and violate the decision-driven experiment rule.

## 2026-07-22 Progressive recurrent-state scaling

- Algebraic function preservation is not enough under BF16. Concatenating the
  old and new value channels into one wider GEMM was exact in FP32 but changed
  16 token predictions in the real BF16 graph. Keeping the learned 64-channel
  bank in its original GEMMs and adding a separate zero-readout bank preserves
  loops1-5 bit-for-bit while leaving the new branch trainable.
- State capacity can be added without forgetting, but 500 steps do not yet
  show a distribution-level gain. At step30500, fixed holes53/64 improve by
  `+0.0234/+0.0215`, while mixed and formal official56-64 fall by
  `-0.0254/-0.0312`; official51-55 is flat. Treat this as a noisy diagnostic,
  not a positive result or a reason to sweep state widths.
- Preserve the binding decision gate. The expanded model is numerically stable,
  reaches train CE `0.4485`, and still performs real loop correction such as
  `33->14->0` wrong cells. Continue only to step31500, then reject ordinary
  state-width scaling unless a preregistered hard metric beats step30000 by at
  least `+0.02`.
- The step31500 binding gate is a real but nonuniform positive. Relative to the
  exact step30000 parent, mixed exact rises `+0.0352` and fixed holes53/60/64
  rise `+0.0410/+0.0273/+0.0449`, while formal official56-64 rises only
  `+0.0059`. This supports one final unchanged strong endpoint, not an expand-v
  table or a broad state-capacity claim.
- Expanded state capacity primarily changes recurrent refinement, not opening.
  Mixed loop1 remains `0.0234`, while loop3/4/5 reach
  `0.3730/0.4941/0.5156`. That is the desired qualitative signature for added
  learned memory, but the formal hardest bucket must still validate it.
- Treat AIStation lease expiry as execution metadata, not a scientific event.
  The step31500 run preserved a complete step31475 model/optimizer/scheduler/RNG
  checkpoint, reopened GPU1, and finished from that exact state. Never shorten
  the 512-board evaluation or switch to GPU2 to fit a lease.
- A checkpoint peak is not a scaling law. The expanded state reached mixed
  exact `0.5156` at step31500 but fell to `0.4668` at the unchanged step33000
  endpoint; formal official56-64 similarly fell `0.3906 -> 0.3633`. Training
  CE stayed finite and official51-55 remained `0.6270`, so this is distribution
  instability rather than a crash. Do not select step31500 after the fact and
  claim monotonic state scaling.
- Separate loop value from state-width value. At step33000, mixed loop1-to5
  exact still rises `0.0234 -> 0.4668`, and individual hardest boards change
  `32 -> 3 -> 0` wrong cells. Loops are doing useful work. What failed is the
  claim that a wider instance of the same GDN state update makes that work
  uniformly reliable. Close expand-v/LR/seed/loss/noise rescue tables and move
  only to a genuinely different generic data-coverage or state-formulation
  question.

## 2026-07-24 Strict official-carrier rerun

- Same seed is not enough for a one-seed architecture comparison. Different
  constructors consumed different random streams and changed 11 of 77 shared
  tensors. Reset the backbone-independent shell from a separate fixed stream
  and require all shared tensor hashes to match before training.
- Do not diagnose an implementation bug from an unexpected ranking alone.
  Official FLA source hashes, exact layer classes, Triton convolution,
  recurrent-state layout, naive-reference output/state/backward, initial-state
  gradients, and post-run metrics all passed for GDN/GDN2/KDA. The clean result
  can still differ because the prior experiment was confounded.
- The corrected state-matched step500 ordering is GDN2 opening first, RWKV7
  second, KDA third, and GDN last: official 46-50 exact is
  `0.7969/0.7285/0.6465/0.3633`. This withdraws the old “GDN beats RWKV”
  statement, but all four are still zero exact at 51-64 blanks, so it does not
  establish a hard-task or asymptotic winner.
- State matching and native-architecture matching answer different questions.
  The strict table uses `H6 x K32 x V32` for every carrier, while official GDN
  recommends `H x K = 0.75D` and `expand_v=2`. One preregistered
  `D192/H6/K24/V48` GDN run is a valid configuration diagnosis because it stays
  close in parameters and state size. A geometry/seed/LR table is not.
- At this short gate, recurrent improvement mostly happens by loop2 and then
  stalls. Same-puzzle wrong cells are RWKV7 `24->23`, GDN `23->17`, GDN2
  `21->20`, and KDA `28->24` across loops1-5. Easy-bucket opening must not be
  presented as evidence that later loops perform sustained global correction.

## 2026-07-25 Native GDN geometry diagnostic

- Resolve a plausible implementation/configuration objection once, with a
  preregistered intervention, instead of defending a surprising ranking by
  argument. The state-matched GDN used K32/V32; the one allowed native probe
  used the official-style K24/V48 geometry and changed no other substantive
  training argument.
- Separate implementation validity from hypothesis success. Native GDN passed
  official source/class, Triton, chunk backward, Torch output/state/gradient,
  full-stack gradient, shared initialization, data, metric, and no-fallback
  checks. That makes the negative scientific result stronger; it does not turn
  the result positive.
- More recurrent state is not automatically more useful state. K24/V48 adds
  `12.5%` state elements and `7.6%` parameters, yet CE changes
  `1.0509 -> 1.0598`, official 46-50 exact only `0.3633 -> 0.3887`, and
  51-64 exact remains zero.
- Check the behavior loops are supposed to provide. On the identical puzzle,
  state-matched GDN changes wrong cells `23->21->17->17->17`; native GDN
  changes `23->21->21->22->22`. The larger state does not sustain
  self-correction and is `24%` slower per optimizer step.
- Close the question after a decisive negative gate. Do not convert K/V ratio,
  expand-v, seed, LR, or loss into a table. The remaining paper question is
  FutureSeed's causal value under a chosen carrier and enough clean compute,
  not whether one more GDN geometry rescues a 500-step ranking.

## 2026-07-26 FutureSeed causal four-carrier gate

- A table where every carrier has FutureSeed cannot establish FutureSeed's
  contribution. The minimum causal experiment is a paired on/off control for
  every audited carrier, with only the state-injection scale and artifact paths
  allowed to differ.
- Native FutureSeed generalizes beyond RWKV7 at this short-budget gate.
  Official46-50 loop5 exact changes from zero without FutureSeed to
  `0.7285/0.3633/0.7969/0.6465` with FutureSeed for
  RWKV7/GDN/GDN2/KDA. The preregistered rule required two carriers above
  `+0.10`; all four pass.
- The effect is not merely a small per-cell calibration gain. b46-50 blank
  accuracy rises from about `0.38-0.39` to `0.97-0.99`, and full-board exact
  moves from zero to hundreds of solved boards. GDN/GDN2/KDA noFS are initially
  competitive at step100 but stall near CE `1.62-1.63` after the hard-stage
  switch, while FS continues to about `1.01-1.05`.
- A strict on/off contract must distinguish initialization from functional
  activation. Official RWKV7 zero-initializes its output projection, so gate
  gradients may be mathematically zero at construction time. Verify
  byte-identical constructor hashes separately, then load the same trained
  checkpoint into FS-on/off models and require only the intervention to change
  output and gradients.
- Fail closed on infrastructure too. Omitting the project-local `ninja` path
  caused RWKV CUDA loading to stop rather than silently choose another
  implementation. Archive these failed audit attempts; they explain why the
  final contract is trustworthy.
- Do not overclaim the positive result. Every carrier remains zero exact at
  51-55 and 56-64 blanks. FutureSeed is now solid as a generic finite-budget
  optimization/opening mechanism for causal recurrent carriers, but global
  hard closure and asymptotic efficiency remain open.
- The next experiment is not another carrier, seed, gate, or geometry table.
  Use one efficient official carrier for a quality-matched scale test and ask
  whether FS reaches a common quality target materially earlier or opens
  51-55 exact. That decision directly tests the cheap-future-context paper
  claim.

## 2026-07-28 FutureSeed2 static state-entry selection

- Preserve the accepted baseline when testing a new FutureSeed mechanism.
  The selective gate was zero-initialized, bit-exact to FutureSeed1, and
  changed only how the already-computed GDN2 state crosses a layer boundary.
- A mechanism being alive is not evidence that it is useful. After 100 steps,
  gate delta RMS was `0.00950` and the imported seed moved by `0.53%`, but hard
  exact mean fell `0.2318 -> 0.2090`; the 61-64 range fell
  `0.1973 -> 0.1387`.
- The failure is concentrated in late recurrent correction. On the same
  64-blank board `b0048`, FutureSeed1 changes wrong cells
  `32 -> 12 -> 3 -> 0 -> 0`; the static selective gate changes
  `36 -> 13 -> 6 -> 6 -> 5`. Another shared board remains solved by both, so
  this is sample-dependent damage rather than global model collapse.
- A globally shared `K x V` mask cannot reliably classify a state entry as
  always useful or harmful. Any later selection hypothesis must be explicitly
  content-dependent. Do not turn that observation into a gate-rank, scale,
  seed, LR, loss, or continuation sweep.
- The strict official-FLA contract remained intact:
  `fla.layers.gdn2.GatedDeltaNet2`, `ChunkGDN2FunctionBackward`, Triton q/k/v
  convolution, GPU1, clean detached SHA, exact checkpoint migration, and no
  fallback. Keep the original GDN2+FutureSeed1 checkpoint as the strong line.
- 2026-07-29: A zero-init 270-parameter content-adaptive FutureSeed head-trust
  gate is mathematically and operationally valid but does not beat the frozen
  GDN2+FutureSeed1 control. Hard-range mean exact moves `0.2318 -> 0.2253`;
  61-64 exact moves `0.1973 -> 0.1699`. Although recurrent-state summary
  features vary across samples (`std=0.0714`), the resulting gate barely does
  (`std=0.00065`). Along with the rejected static KxV mask, this shows that
  simple seed-strength modulation is not FutureSeed2. Do not sweep gate
  features, scale, seed, LR, loss, or length. A future mechanism must improve
  seed content itself while preserving the state basis and exact baseline
  initialization.

## 2026-07-29 FutureSeed2 same-layer block memory

- Do not assume that a recurrent terminal state is a reusable memory merely
  because it returns to the same layer. The strict block-memory contract is
  exact, uses the official FLA GDN2/Triton path, and adds no parameters, but
  official 51-55/56-60/61-64 loop5 exact all fall to zero from
  `0.3672/0.1309/0.1973`.
- Separate first-pass competence from iterative compatibility. Mixed loop1
  exact barely changes (`0.0234 -> 0.0215`), while loop5 exact collapses
  (`0.2520 -> 0.0195`). The pretrained backbone still opens the puzzle; raw
  state carry specifically destroys the computation that later loops perform.
- FutureSeed1's cross-layer state transfer is a directional initialization,
  not ordinary persistent memory. A same-layer terminal state contains
  position- and pass-specific residue that the next macro step cannot consume
  unchanged, even after the existing normalization and head gate.
- Visualize matched trajectories before interpreting aggregate exact. On the
  same 51-55-blank board, FutureSeed1 changes wrong cells
  `17 -> 4 -> 1 -> 1 -> 1`, whereas block memory changes
  `18 -> 13 -> 13 -> 13 -> 14`. This localizes the failure to refinement
  rather than data identity or initial opening.
- Close simple carry after this result. Do not sweep blend, strength, decay,
  seed, LR, loss, or continuation length. A viable FutureSeed2 must learn a
  generic transformation or compression of future evidence before reuse while
  remaining exactly FutureSeed1 at initialization.

## 2026-07-29 FutureSeed2 compatible two-hop readout

- Coordinate compatibility is necessary but not sufficient. Querying an old
  state with its producer layer's own q projection, state read, gated RMSNorm,
  and output projection passes an exact direct-formula contract and is much
  less destructive than raw state carry, but it still does not beat
  FutureSeed1.
- Separate activation from usefulness. The zero-initialized eight-scalar path
  learns mean absolute scale `0.01697`; its residual RMS grows from `0.157` at
  loop1 to `0.560` at loop3. Nevertheless mixed loop5 exact falls
  `0.2520 -> 0.2363`, and 61-64 exact falls `0.1973 -> 0.1719`.
- The interference begins after opening. Aggregate loops1 and2 are exactly
  tied, then the candidate trails at loops3-5. On matched batch 17,
  FutureSeed1 changes wrong cells `22 -> 9 -> 2 -> 0 -> 0`, while two-hop
  readout changes `22 -> 13 -> 8 -> 6 -> 5`.
- A mechanism can help one board and still be the wrong inductive bias. On
  matched batch 175 the two-hop path solves at loop3 while FutureSeed1 retains
  one error, but hard-range mean falls `0.2318 -> 0.2227`. Use distributions
  and matched counterexamples, not a selected success story.
- Efficiency is part of the mechanism claim. Reusing pretrained projections
  adds only eight parameters but many extra small kernels; train time rises
  `659.9s -> 821.5s` (`+24.5%`). Cheap parameter count is not cheap compute.
- Close radius extension. Raw block carry and producer-compatible two-hop
  readout answer the time- and depth-radius variants. Do not sweep hop, scale,
  seed, LR, loss, or length. A future FutureSeed2 must improve how generic
  future evidence is formed or compressed, not merely send terminal state
  farther or keep it longer.

## 2026-07-29 Gain-budget numerical contract

- A mathematical closed form is not automatically a production certificate.
  The projection passed small property tests and official CUDA parity, but a
  real training batch near `tau=1` exposed a few-ulps FP32 boundary violation
  before the first candidate step.
- Treat a certificate abort separately from a mechanism result. The matched
  control completed, but the candidate produced no score; it is invalid to
  call that a negative quality result.
- Do not silently add tolerance or relax the requested budget. The projection
  family already contains a strict fail-closed endpoint: `lambda=0` preserves
  effective erase strength and minimizes anisotropic shear. Reserve a small
  interior margin before the closed-form projection; when the margin consumes
  all feasible slack, use that endpoint and log its frequency.
- A strict fix still has to be cheap. Recomputing the full certificate after
  the endpoint passed every correctness check but added `23.33%` over the
  matched external identity, so ABCCBA rejected it before training. Prefer one
  actual certificate pass with a preregistered numerical reserve over a second
  dense pass across every token and head.
- One pass alone was not quite enough: `ac566f96` measured `+20.98%`, just
  above the fixed `20%` gate. Do not rerun the same noisy benchmark until it
  happens to pass. Multiple logically independent global assertions can share
  one boolean certificate and one device reduction without weakening any
  condition.
- A benchmark can be the bug. The same layer-size test reported projection
  overhead from `+0.17%` to `+26.46%` because it emptied the allocator and
  timed one forward/backward per sample. Keep the fixed `20%` decision gate,
  but estimate steady-state time from multiple warmed calls and measure
  cold-cache memory separately. Always persist raw samples even when the gate
  fails.
- The stable instrument put the projection at `+14.56%` time and `+18.46%`
  memory, so systems cost is acceptable. It also confirmed that a small
  interior margin is not a complete numerical fallback: a real formal batch
  still crossed the FP32 certificate before step1.
- Do not keep enlarging a guessed safety margin. The rank-one family has an
  analytic zero-shear endpoint. Select it only after the actual one-pass
  certificate detects a residual violation, then independently audit the
  BF16 gate that enters the official recurrence. This is stricter and cheaper
  than a second dense FP32 certificate pass.
- A scalar certificate should stay scalar until the final gate construction.
  SHA `4715a5a8` passed every mathematical and CUDA correctness test, but
  materializing several K-dimensional endpoint selections raised stable time
  overhead to `+28.10%` even though memory stayed at `+18.49%`. Decide
  token/head endpoint rows in scalar space, then construct the projected gate
  once and independently audit the actual FP32/BF16 values.
- Strict and training lanes have different jobs. Official FP32
  fused-recurrent forward carries the hard certificate; official BF16 chunk
  carries real training and is audited under an explicit low-precision
  tolerance. Neither mixed-dtype failure nor full-FP32 chunk failure is hidden
  behind a fallback.
- Keep the causal comparison fair. Untouched `mode=none` is the upstream
  bitwise regression gate; `external_identity` and `decay_funded` share the
  same external normalization, cast, kernel, and audit overhead. Only the erase
  projection differs between the formal arms.
- Strict non-expansion was the wrong abstraction for this checkpoint. The
  final scalar-certificate SHA passed 24/24 math tests and all official CUDA
  correctness gates, but still cost `+32.14%` time over matched identity.
  More importantly, the same-semantics smoke clipped `99.996%` of token/head
  rows, forced `56.40%` to the isotropic endpoint, and retained mean
  anisotropy scale `0.019`. It changed almost every memory edit rather than
  removing rare pathological gain.
- Do not assume non-normality is wasted instability. The paired smoke moved
  51-55-blank loop5 exact `0.625 -> 0.125` and CE `0.749 -> 1.088` under the
  strict budget. The pretrained solver uses mild transient expansion as
  useful computation. Any future generic stability mechanism must preserve
  the identity path and suppress only learned outliers, not globally force
  every step under `c=1`.

## 2026-07-30 GDN2 Fast-Slow decay

- Smooth forgetting is mechanically valid but not a hard-Sudoku solution.
  A positive causal K=4 FIR lowers forgetting-hazard TV to `0.9839x`, keeps
  erase/write token-fast, and preserves the exact official GDN2 recurrent
  kernel and backward path.
- Exact identity controls matter. The shared wrapper is bit-exact to untouched
  GDN2 in output, terminal state, and every checked gradient, both with and
  without a FutureSeed initial state. The quality result is therefore not an
  implementation fallback.
- Lower gate variation is not the same as better recurrent convergence.
  Mixed loop5 exact rises `0.2402 -> 0.2500`, but official
  51-55/56-60/61-64 changes
  `0.3594/0.1387/0.1641 -> 0.3535/0.1465/0.1621`; hard mean delta is exactly
  zero.
- Selected examples can lie in either direction. One paired board changes
  from 19 wrong cells to solved, while another changes from 3 wrong cells to
  26 wrong cells. The intervention changes which attractor wins rather than
  making all trajectories more stable.
- Cheap layer math can still retain expensive activations. The microbenchmark
  reports `+6.97%` time and `+8.18%` memory, while the full D192/L10
  five-loop training graph reports `+7.19%` time and `+20.65%` peak memory.
  Measure the complete model before calling a gate filter cheap.
- Close decay-only smoothing on this line. Do not sweep FIR length, rho, seed,
  LR, loss, width, or duration. Clean data/compute scaling has stronger
  evidence and better matches the bitter lesson.

## 2026-07-31 GDN2 address-payload separation

- Absolute position metadata can be present without becoming a stable recurrent
  address. The frozen counterfactual preserves each token's canonical position
  embedding but collapses under non-row traversals, so additive metadata alone
  does not make GDN2 order-independent.
- Decouple where memory is accessed from what memory carries. Driving only Q/K
  from canonical position while keeping hidden content responsible for
  V/decay/erase/write/output raises equal-compute step9100 mean official 51-64
  blank accuracy from `0.2037` to `0.4333`; train CE falls
  `1.9370 -> 1.2196`. Parameters, data, optimizer, FutureSeed, loops, and the
  official FLA recurrence are unchanged.
- Keep the evidence boundary precise. Both equal-compute arms still have zero
  exact on every hard range. The mechanism fixes a large optimization problem;
  it does not by itself establish global Sudoku closure.
- The advantage persists at matched total compute. At identical step9300,
  parameters, data, case hashes, loop budget, and official kernel, mean hard
  blank accuracy is `0.2753 -> 0.5437` and train CE is
  `1.5707 -> 0.9661`. The earlier result was not an extra-200-step artifact.
- Later loops become useful after address learning improves. On matched
  64-blank batch69, normal GDN2 changes
  `53 -> 50 -> 50 -> 51 -> 51` wrong cells, while position-Q/K changes
  `32 -> 21 -> 18 -> 15 -> 15`. Address factorization improves both the
  initial representation and recurrent correction.
- Keep the evidence boundary precise. Every separately sampled official
  51-64 range still has zero exact at step9300. The next bottleneck is global
  consistency after partial correction, not address binding alone.
- This direction remains compatible with the bitter lesson: it is a generic,
  parameter-neutral factorization of recurrent address and payload, not a
  Sudoku rule. Continue only with matched from-scratch data/model/compute
  scaling, not address-mode, loss-weight, or seed tables.
- More compute helps, but short continuation stacking is already showing
  diminishing and heterogeneous returns. Step9300->9600 raises mean hard blank
  `0.5437 -> 0.5800` and opens exact `0.0059` in two ranges, entirely through
  later loops. It misses the predeclared `0.02` exact and `+0.04` blank gates.
- Average scaling can hide unstable trajectories. One 56-blank board goes from
  four final errors to solved, and one 64-blank board improves `29 -> 10`, but
  another 64-blank board regresses `23 -> 34`. The next scaling axis should
  increase generic capacity or independent data coverage, not append another
  300 steps to the same state.

## 2026-07-31 GDN2 stable address binding

- GDN2 does not store a separate symbolic key beside each value. Its recurrent
  matrix is written in K coordinates and later read in Q coordinates. A stable
  token-plus-position address therefore has to define a usable coordinate
  system, not merely appear as extra metadata in the hidden state.
- Parameterization matters. Fixed and learned rotations were active but
  harmful (`0.1757` and `0.1696` mean hard blank versus normal `0.2037`). A
  direct Euclidean residual shared by Q and K is the first clean positive:
  CE `1.9370 -> 1.6845`, mean hard blank `0.2037 -> 0.2844`, and mean
  loop1-to-loop5 gain changes from `-0.0135` to `+0.0089`.
- More freedom is not automatically better. Fully independent read/write
  address maps add twice the address parameters but reach only `0.2313`, trail
  the shared map by `-0.0531`, and cost `+54.7%` runtime over normal. On a
  mechanically selected board, shared corrects `34 -> 30` wrong cells while
  split maps regress `41 -> 44`.
- The retained interpretation is a shared memory namespace. K writes and Q
  reads benefit from the same stable address basis; forcing them to learn two
  bases and their alignment from task loss makes optimization harder. This is
  generic to recurrent linear attention and does not encode a Sudoku rule.
- Keep the evidence boundary precise. Shared address binding is a real partial
  improvement, but every 51-64 blank range still has zero exact. Do not sweep
  phase, scale, rank, seed, LR, loss, or duration. Test transfer next on matched
  associative retrieval or language modeling before naming a general GDN3.
- CUDA equivalence has two contracts. Zero-init output and terminal state are
  bit-exact, while the pinned BF16/Triton backward can vary across clean
  processes because of GPU accumulation (`0.0` versus `0.0625` max base-gradient
  difference in observed runs). Report per-tensor tolerance and reference scale
  instead of claiming impossible cross-process bit identity.

## 2026-07-31 KDA occurrence rotary

- Fine-grained decay represents multiple ages but does not automatically
  create queryable versions of repeated writes to the same semantic key.
- Standard occurrence rotary changes recency preference. It raises the
  in-distribution final-occurrence accuracy `0.2222 -> 0.6481`, while other
  occurrence indices remain near chance.
- The primary length512/repeat16 result is negative: content addressing reaches
  `0.0371` and occurrence rotary reaches `0.0313`. Both are near 32-class
  chance, so the preregistered old-version retrieval hypothesis fails.
- Do not tune rotary base, frequency, beta, seed, width, or duration, and do not
  build a learned counter on this evidence. A future revisit requires
  qualitatively different versioned/orthogonal state capacity rather than
  better occurrence labels on an overwrite-like state.

## 2026-08-01 Address-conditioned GDN2 write carrier

- A mechanism can be algebraically exact and CUDA-correct but still be useless
  because its easiest optimization path is not the intended behavior. The
  carrier fold reproduces the direct recurrence within BF16 tolerance, keeps
  the official `ChunkGDN2FunctionBackward`, and is exactly the shared-address
  baseline when the carrier is one.
- The near-one sigmoid carrier did not learn selective memory writes. After the
  matched continuation its mean is `0.997525`, token std is only `1.05e-5`, no
  value is below `0.95`, and effective write norm is uniformly reduced by
  about `0.25%`. This is global shrinkage, not address-specific routing.
- Mechanical capacity is not evidence of useful computation. The candidate
  adds only 3,840 parameters and remains numerically stable, yet mean official
  51-64 loop5 blank changes only `0.501809 -> 0.501669`, CE is slightly worse,
  and loop correction is unchanged while time/VRAM rise `11.42%/31.72%`.
- Close this exact `sigmoid(6 + bias + scale * address)` design. Do not sweep
  its seed, bias, scale, LR, loss, rank, width, or duration. A future generic
  carrier must expose address variation through a nonsaturated exact-identity
  parameterization and demonstrate selective writes on an interference or
  retrieval task before returning to downstream benchmarks.
- Keep the causal boundary clear: this negative result rejects one optimization
  geometry, not the broader idea that online memory learning rates can depend
  on address. The shared Euclidean address residual remains the retained GDN2
  modification.

## 2026-08-01 Raven as a FutureSeed carrier

- A newer architecture is not automatically a better carrier. Under the same
  FutureSeed shell and exactly 1024 recurrent-state values per head, official
  Raven reaches train CE `1.1388` versus GDN2 `1.0186` and is `27.5%` slower
  per optimizer step.
- Sparse routing does reduce model and memory cost: Raven uses 4.650M
  parameters and 7262 MiB peak allocation versus GDN2's 5.462M and 8025 MiB.
  That systems saving is real, but it is not a quality win.
- Dense global constraints are a poor match for this small-slot regime. On
  official 46-50 blanks, GDN2 reaches `0.7969` exact while Raven remains at
  zero despite `0.7771` blank accuracy. On 51-55 blanks, Raven trails blank
  accuracy by `0.0782`; both remain exact zero.
- Raven's loops are not completely inert. The shared case changes from 34 to
  30 wrong cells, but aggregate exact remains zero and loop5 mostly preserves
  the loop2 operating point. GDN2 starts much closer and has nonzero exact.
- Preserve the evidence boundary. This rejects matched-state Raven as the
  current Sudoku/FutureSeed backbone, not Raven's long-context recall claim.
  Test Raven again only when the question is sparse long-range retrieval; do
  not rescue this Sudoku result with slot, top-k, seed, LR, or loss tables.
- Infrastructure failures are not scientific evidence. The first launch was
  halted by an expired GPU1 lease before step100 and has a separate
  `abort.json`; only the exact clean relaunch is used in the comparison.

## 2026-08-02 FutureSeed-tied causal write preconditioning

- Separating stable address from online write geometry is mechanically sound.
  A parameter-free causal curvature state can be folded into official GDN2 as
  `k'=m*k,b'=b/m`, preserving the erase address while changing the rank-one
  write direction. CUDA forward, terminal state, backward, full-layer
  gradients, and `ChunkGDN2FunctionBackward` all pass against a direct CUDA
  Torch recurrence.
- The quality signal is real but too small. Against the exact frozen
  position-Q/K step9100 control, hard 51-64 loop5 blank improves
  `0.4333 -> 0.4563` and CE improves `1.2196 -> 1.1434`, but mean delta
  `+0.0230` misses the `+0.03` gate and official hard exact remains zero.
- Better local optimization is not the same as better recurrent reasoning.
  Most candidate advantage is already visible at loop1; its additional
  loop1-to-loop5 gain over control is only `+0.0023/+0.0053/+0.0014` across
  the three hard ranges. Some cases improve sharply, while other predictions
  freeze or regress.
- Generic prefix curvature is currently too expensive: formal training time
  rises `27.5%` and peak allocated VRAM rises `60.9%`, despite unchanged
  parameters and official GDN2 recurrence. Do not hide this systems failure
  behind the blank-accuracy gain.
- Close this exact implementation. Do not sweep multiplier bounds, center,
  strength, seed, LR, loss, width, or duration. Preserve position-Q/K as the
  baseline. Revisit write preconditioning only with a fused kernel or on a
  direct online-memory interference/retrieval task where its mechanism can be
  isolated.

## 2026-08-02 GDN2 clean scaling at step 10000

- Clean data and optimizer compute remain productive on the canonical
  D192/L10 GDN2 FutureSeed trajectory. Mixed loop5 full-board exact improves
  `0.2520 -> 0.2852` from step9100 to10000, while loop1 stays near `0.0234`;
  the extra solves are created by recurrent refinement rather than a stronger
  one-pass predictor.
- Do not mistake a positive aggregate scaling slope for monotonic behavior in
  every difficulty bucket. Official 51-55 improves to `0.4062`, but 56-60 and
  61-64 remain volatile; mean 51-64 exact moves only `0.2318 -> 0.2415`.
- Hard-case trajectories are essential evidence. A 64-blank case reaches
  `31 -> 18 -> 3 -> 1 -> 0` wrong cells across loops 1-5, while failures stall
  on a few high-confidence wrong digits. Report both, rather than reducing the
  run to average blank accuracy.
- The preregistered mixed-exact gate passes, so the high-ROI decision is one
  unchanged continuation to step12000. Do not branch into loss, noise, state,
  width, depth, seed, or curriculum variants before that endpoint.

## 2026-08-03 GDN2 clean scaling endpoint at step 12000

- The step10000 improvement was not a transient checkpoint. With every
  scientific variable unchanged, mixed loop5 full-board exact rises again from
  `0.2852` to `0.3379`; official 51-55/56-60/61-64 exact reaches
  `0.4492/0.1543/0.2637`.
- Scale acts mainly through recurrent computation. Loop1 stays at `0.0234`,
  while loop1-to-loop5 exact gain widens from `+0.2617` at step10000 to
  `+0.3145` at step12000. This rejects the explanation that longer training
  merely improves a one-pass predictor.
- The remaining ceiling is now localized. Most improvement happens by loop4:
  mixed exact is `0.3320` at loop4 and `0.3379` at loop5. Successful 64-blank
  boards can move `35 -> 22 -> 4 -> 0`, while failures stall
  `22 -> 17 -> 11 -> 5 -> 5`. More compute teaches a strong attractor, but not
  universal closure of the last few coupled errors.
- Respect the preregistered boundary. Combined 56-64 exact reaches `0.2090`,
  but 51-55 exact misses `0.50` and the strong gates are not met. Do not call
  this a strong success, do not tag it, and do not append an unregistered
  training tail.
- The next causal question is not another GDN2 modification. Run one matched
  long no-FutureSeed control. Only that comparison can distinguish a genuine
  long-scale FutureSeed advantage from ordinary GDN2, data, and loop scaling.

## 2026-08-03 Matched no-FutureSeed step500 gate

- Turning off only native FutureSeed creates a large early learning gap even
  in the strongest clean official-FLA GDN2 setup. At step500, holes53/58/64
  blank accuracy is `0.2727/0.2539/0.2639` without FutureSeed versus
  `0.5133/0.3830/0.4356` with it.
- The no-FutureSeed model has not opened full-board exact even at holes50, and
  loop5 is slightly worse than loop1 on each fixed hard set. At this budget,
  recurrent computation alone does not recover the future information that
  FutureSeed supplies to deeper layers.
- Do not overclaim this as an endpoint advantage. It establishes strong
  short-budget optimization/information value; only the unchanged 12k control
  can show whether the advantage persists or merely shifts the learning curve.
- Lease boundaries are not experiments. Preserve the exact model, optimizer,
  data RNG, Python/Torch/CUDA RNG, and fixed evaluation state through a hashed
  checkpoint, then resume under semantic-contract validation.
- The step1000 gate rules out a first-few-step transient. No-FutureSeed hard
  blank accuracy remains around `0.26-0.28`, while FutureSeed reaches
  `0.44-0.56`; the FS-minus-no-FS gap grows on holes53/58/64 compared with
  step500.
- More loops do not compensate for missing future initialization at this
  budget. No-FutureSeed loop5-minus-loop1 hard blank changes by less than
  `0.0025` in magnitude and all fixed exact scores remain zero. Continue long
  training to test delayed catch-up rather than adding a loop-specific hack.
- Through step2500, no-FutureSeed CE remains near `1.58-1.61`; this is a strong
  delayed-opening warning but not the preregistered endpoint. Do not turn a
  discouraging intermediate curve into an early scientific stop. Preserve the
  exact state across GPU leases and answer the 12k asymptotic question.
- The step2500 lease rollover is verified infrastructure, not another
  experiment: atomic checkpoint, SHA256, byte count, exact process-group stop,
  empty GPU process list, and `scientific_failure=false` are all required before
  the next detached-SHA resume.
- The matched step3000 result rules out a small calibration gap. No-FutureSeed
  versus FutureSeed train CE is `1.5946` versus `0.8270`; mean holes53/58/64
  loop5 blank accuracy is `0.2681` versus `0.5595`. Even holes50 is `0` exact
  and `0.3956` blank without FutureSeed versus fully solved with FutureSeed.
- More recurrent loops do not synthesize missing future context at step3000.
  No-FutureSeed loop1-to-loop5 hard blank changes stay within `0.0016`, while
  FutureSeed improves materially across loops. The current evidence supports
  FutureSeed as an optimization/information opener, not merely a readout tweak.
- Keep the claim bounded: step3000 proves a large finite-compute gap, not a
  final frontier gap. Continue the same no-FutureSeed trajectory to the
  preregistered endpoint instead of converting the positive intermediate result
  into a premature stop.
- The no-FutureSeed plateau survives the harder-data transition: CE stays near
  `1.61` from steps4100-4500 after the curriculum expands to 51-60 blanks. This
  strengthens the delayed-opening diagnosis but still is not a substitute for
  the registered step6000/9000/12000 fixed evaluations.
- A strict matched control can be compute-inefficient on an A800 without being
  a fallback: this run used official CUDA kernels and 11.6GB VRAM while a
  10-second sample showed `16-27%` SM utilization. Do not change microbatch or
  accumulation mid-trajectory just to fill the card; preserve causal parity,
  then optimize systems throughput in a separate experiment.
- Step4500 is another verified lease boundary: atomic model/optimizer/data-RNG
  state, SHA256 and byte count, exact process-group stop, empty compute-process
  list, and `scientific_failure=false` are required before exact continuation.
- Step6000 turns the early gap into strong finite-compute causal evidence.
  No-FutureSeed/FutureSeed hard loop5 exact mean is `0/0.0651`, blank accuracy
  is `0.2762/0.6451`, and train CE is `1.6035/0.7974` under matched optimizer
  steps, data, architecture, loop supervision, and official FLA GDN2 kernels.
- FutureSeed is not only improving loop1. On fixed holes53/58/64, mean
  loop1-to-loop5 blank gain is `+0.1132` with FutureSeed and `+0.00135` without
  it. A paired 384-board visualization confirms that no-FutureSeed removes
  about one eighth of one wrong cell per range while FutureSeed removes several
  and creates exact boards. FutureSeed opens a state in which recurrent compute
  becomes useful; loop count alone does not synthesize missing future context.
- Keep the conclusion bounded. This proves a large same-step optimization and
  information-flow advantage, not yet an asymptotic frontier or language-task
  claim. Continue the preregistered control to step9000/12000 and do not compare
  wall times across different physical GPUs and lease segments.
- Step9000 strengthens the causal claim rather than merely repeating step6000.
  No-FutureSeed still has zero fixed hard exact, hard blank accuracy 0.2837,
  and essentially zero loop gain; FutureSeed reaches 0.2142 exact, 0.7422
  blank accuracy, and +0.1844 loop1-to-loop5 blank gain.
- The same-puzzle evidence separates initialization from recurrent computation.
  FutureSeed turns loop computation into large correction trajectories, while
  the matched causal control spends loops around the same wrong board.
- Compute compression is now lower-bounded, not guessed. No-FutureSeed at
  step9000 still fails to reach FutureSeed at step6000, so reaching that hard
  level requires more than 1.5x the optimizer steps without FutureSeed.
- Do not convert this intermediate win into an endpoint or universal claim.
  The preregistered same-step frontier decision remains step12000, and Maze or
  language evidence is still required for a broad paper claim.
- Report systems measurements honestly. Current no-FutureSeed throughput is
  about 4.583 seconds/step with 11,597 MiB steady VRAM, but raw elapsed fields
  across the two arms are not speed evidence because physical GPUs and lease
  histories differ.
- The preregistered step12000 endpoint makes the causal result persistent, not
  merely early. No-FutureSeed still has train CE `1.5812`, zero fixed hard
  exact, hard blank accuracy `0.2897`, and only `+0.00083` loop1-to-loop5 blank
  gain. FutureSeed has CE `0.5969`, hard exact `0.2930`, blank accuracy
  `0.7741`, and `+0.1991` loop gain under the same architecture, data, seed,
  supervision, and optimizer-step budget.
- Same-board evidence confirms real correction rather than calibration. On the
  paired 384-board hard pool, FutureSeed adds `+0.2786` exact and `+0.4892`
  blank accuracy; a 64-blank board goes `39 -> 14 -> 2 -> 0 -> 0` wrong while
  no-FutureSeed stays near 51 wrong. FutureSeed supplies information that makes
  loop computation useful; loops alone do not synthesize it.
- This closes the no-FutureSeed duration question on Sudoku. Do not run it
  longer, repeat another seed, or tune a rescue. The next decision-changing
  test is cross-task generalization of the same causal mechanism. Maze or
  language/retrieval evidence is required before making a universal claim.
- The validated Zoology directionality gate isolates the mechanism outside
  Sudoku. A matched causal GDN2 reaches `0.9305` past-query accuracy but only
  `0.0110` future-query accuracy; native FutureSeed reaches `0.9955/0.9930`.
  Future-query exact changes from `0` to `0.9860` with the same parameters,
  initialization, data, optimizer, epochs, and official FLA kernel.
- FutureSeed is terminal-state transfer, not a reverse scan. Perturbing later
  values changes no-FS early-query logits by exactly zero and FS logits by mean
  `0.07178`; scale 0 is output-identical to the validated causal GDN2 path and
  the FutureSeed gate receives a nonzero gradient.
- Do not claim raw speed from the sequential pair. The first arm paid Triton
  compilation, so `703.7` versus `935.7` examples/s is an archived systems
  observation, not a randomized warmed efficiency comparison.
- This result is strong enough to stop synthetic seed repetition. The next
  decision-changing evidence must test transfer or OOD scaling; one synthetic
  directionality task cannot establish language-model quality.
- P-CAUSAL-008 is an invalid language carrier, not a FutureSeed failure. On the
  fixed WikiText byte-MLM gate, causal GDN2 reached `0.4247` masked accuracy,
  FutureSeed reached `0.4218`, and the intended full-bidirectional ceiling
  reached only `0.1879`. Because the ceiling did not open, no causal-to-future
  quality gap existed for FutureSeed to close.
- An active mechanism is not automatically useful evidence. P-CAUSAL-008
  verified scale-0 identity, zero causal future dependency, nonzero FutureSeed
  dependency and gate gradient, yet FutureSeed still had 3,999 repairs versus
  4,220 regressions. Correct plumbing does not rescue an invalid benchmark
  carrier.
- A tiny custom byte-level attention model is not an established masked-language
  baseline merely because it uses noncausal attention. As with Zoology MQAR,
  first reproduce a validated upstream bidirectional MLM recipe; only then
  replace its mixer with matched GDN2 and FutureSeed.
- Compare systems only after independent warmup. Sequential wall times include
  Triton compilation and validation order, so P-CAUSAL-008 retains warmed
  throughput and peak memory but makes no raw wall-time claim.
- An established architecture is still not a valid carrier when the registered
  budget does not open it. Exact Transformers BERT-mini on WordPiece
  WikiText-103 reached only `0.0716` masked accuracy after 20.48M input tokens
  and was nearly flat after step500, missing the preregistered `0.10` endpoint.
  Stop before causal/GDN2/FutureSeed rather than rescuing the baseline after
  seeing the curve. This result is a carrier-budget boundary, not FutureSeed
  evidence.
- A noncausal dependency check proves only that information can flow; it does
  not prove the attention carrier has learned the association. In P-CAUSAL-010
  the near-parameter-matched SDPA arm had nonzero future dependency but
  plateaued near `0.50` accuracy and `0.81` CE on both directions, while native
  FutureSeed reached `0.992` future accuracy under the same ten epochs. Because
  upstream Zoology MHA is known to open sharply only around epoch25, validate
  the exact official MHA with only its mask removed before using attention as a
  ceiling. Do not call an under-opened custom attention arm a Transformer loss.
- P-CAUSAL-011 showed that even the exact official Zoology MHA does not become
  a valid directional-MQAR ceiling merely by deleting its causal mask. Source,
  parameters and initialization were exact; future dependency was active; yet
  30 epochs ended at past/future accuracy `0.4850/0.4845` and joint exact
  `0.048`. The causal order is part of the original carrier's useful binding
  bias, not just a restriction. Full visibility without a suitable binding
  bias can remain ambiguous. Do not rescue this mask-removal baseline and do
  not use it to claim a Transformer loss; test FutureSeed length scaling
  directly, and calibrate attention later on an independently opened task.
- P-CAUSAL-012 shows that FutureSeed remains highly consequential after a 16x
  context increase, but it is not length invariant. At L1024, causal GDN2 is
  near chance on both past and future queries, while the matched FutureSeed arm
  reaches `0.7535/0.7415`; its `+0.733` future delta is real, but its `0.7475`
  retention from L64 misses the registered scaling gate.
- Long-context FutureSeed errors are predominantly binding errors, not missing
  values. `82.4%` of wrong future predictions and `78.9%` of wrong past
  predictions select another key's correct value from the same sample. This
  supports one state/address-capacity test and rejects blind epoch, seed, LR,
  loss, or middle-length sweeps.
- A failed absolute gate and a large matched delta can both be true. Archive
  the endpoint as a scaling boundary while retaining the mechanism evidence;
  do not turn either fact into a universal success or failure statement.
- A nominally warmed sequential benchmark can still inherit Triton autotuning
  order effects. The implausibly faster FutureSeed L1024 timing is diagnostic,
  not a paper efficiency result. Cost claims require fresh processes and
  alternating order.
- P-CAUSAL-013 shows that model width and recurrent-state capacity are not the
  same scaling axis. D256/H8/D32 doubled state values but increased parameters
  `3.39x`; both causal and FutureSeed arms stayed near 1% accuracy while the
  frozen D128 FutureSeed arm reached `0.7475` balanced accuracy under the same
  data and ten epochs. A larger model can miss a sharp algorithmic opening even
  when its training loss falls.
- Never treat a conditional error subtype as improved when total accuracy has
  collapsed. D256's same-case value-swap rate fell only because near-random
  predictions almost never selected any of the four values in the sequence.
  Binding diagnostics are meaningful only at a comparable quality level.
- When whole-model scaling changes optimization behavior, isolate state
  capacity before tuning. Keep the proven D128 width/head geometry and enlarge
  only official GDN2 value state; do not rescue D256 with epochs, LR, seed or a
  geometry table.
- P-CAUSAL-014 shows that state scalar count is not the same as address
  capacity. Official GDN2 `expand_v=2` kept D128 and K32 fixed while changing
  V32 to V64, doubling state values with only `+22.49%` parameters. FutureSeed
  still beat its matched causal arm, but balanced accuracy fell from `0.7475`
  to `0.27675` and joint exact from `0.339` to zero. Do not rescue this axis.
- Interpret a `K x V` fast-weight matrix by axis. Increasing V widens the
  payload stored per address but leaves key-address rank bounded by the same K.
  The dominant L1024 error was choosing another key's valid value, so the
  observed failure points toward address separability or state compression,
  not more value width. Future state scaling must say which mathematical
  capacity it increases; total state elements alone are not a mechanism.
- P-CAUSAL-015 rules out missing relative position as the simple explanation
  for the failed bidirectional MQAR ceilings. Parameter-free RoPE preserved the
  exact tensors and added future-aware relative phases, but finished at
  `0.4955/0.4820` past/future accuracy and `0.046` joint exact after 30 epochs.
  Do not sweep RoPE theta, scale, epochs or optimizer settings on this proxy.
- Visibility, relative position and binding are separate capabilities. Every
  wrong prediction from plain SDPA, mask-removed official MHA and RoPE SDPA is
  another valid value from the same sample. These models recover the candidate
  set but cannot associate each random key with its own value in the fixed
  two-layer shell. FutureSeed makes only `16/3` future/past errors versus
  RoPE's `1036/1009` on the same 2,000 queries per direction.
- Three failed attention controls do not establish that FutureSeed beats
  Transformers in general. They close this particular carrier for a quality
  ceiling. Use the validated causal-GDN2/FutureSeed pair for the length and
  memory scaling mechanism figure, and require a separately opened published
  bidirectional task before making a Transformer cost-quality comparison.
- P-CAUSAL-016 gives the first complete FutureSeed context-length curve under
  fixed memory load. Future-query accuracy is `0.992/0.981/0.978/0.985/0.7415`
  at L64/128/256/512/1024, while matched causal GDN2 remains near `0.01` at
  every length. This is a strong directionality-scaling result, not a marginal
  ablation.
- Do not infer smooth information dilution from two endpoints. The intermediate
  curve stays essentially flat through L512 and then drops sharply at L1024.
  The mechanism boundary is localized to that transition under the fixed
  training recipe; extra L192/L384 points would be table filling, not insight.
- FutureSeed can rescue both directions after long irrelevant context. The
  causal stack retains past retrieval at L64 but loses it by L128, whereas the
  matched FutureSeed stack keeps past accuracy above `0.984` through L512.
  Terminal-state transfer is therefore doing more than exposing formally
  inaccessible future tokens; it also improves optimization of long-range
  bindings already available to a causal scan.
- Constant extra state does not mean zero compute overhead. Peak allocated
  memory differs by only about `1.50 MiB` between matched arms, but FutureSeed
  adds state normalization and cross-layer transfer. Measure that cost with
  long repeated trials before making a speed claim.
- A fresh process is necessary but not sufficient for trustworthy timing.
  P-CAUSAL-016 used only 20 measured steps, producing 0.30--1.01 second samples
  and inconsistent arm ratios. Preserve these as diagnostics; do not place
  them in a paper cost frontier until longer repeated timing is available.
- P-CAUSAL-017 shows that an explicit reverse scan is not automatically a
  useful quality ceiling. Two independent forward/reverse official-GDN2
  streams with learned fusion had nonzero future dependency and healthy
  gradients, yet stayed near chance at L512. Visibility and long-range
  association retention are separate requirements.
- Do not claim that FutureSeed beats bidirectional recurrence from an invalid
  carrier. FutureSeed exceeded this baseline by about `0.9623` balanced
  accuracy, but the baseline missed its own `0.95/0.95` past/future gate.
  Archive the boundary and require an independently opened baseline/task pair.
- FutureSeed is not accurately described as only a cheap reverse scan. At
  L512, plain causal GDN2 also loses past retrieval, while FutureSeed restores
  both past and future accuracy to about `0.986`. Terminal-state transfer acts
  as a cross-layer long-range memory route in addition to changing direction.
- Apply quality gates before systems gates. Once the bidirectional carrier
  finished at chance, repeated throughput measurements became irrelevant;
  stop their exact process groups and report cost as not measured rather than
  manufacturing a cheapness result against a broken model.
- P-CAUSAL-018 separates task validity from model comparison. Under one official
  pretrained BERT-Tiny checkpoint, changing only full attention to strict causal
  drops masked accuracy from `0.355546` to `0.215732` and raises CE from
  `4.033269` to `5.453952`; causal future dependency remains exactly zero. The
  fixed WikiText carrier therefore has a strong right-context signal.
- Never present an inference-time mask intervention on a bidirectionally trained
  checkpoint as a fair causal baseline. P-CAUSAL-018 authorizes the carrier, not
  a FutureSeed win. The actual paper comparison must train matched GDN2 scale 0
  and scale 1 with the same initialization, data, optimizer, tokens, and kernel.
- Asset provenance is part of experiment validity. The official checkpoint was
  downloaded locally through Kimi WebBridge, hashed before upload, and matched
  remotely at `dd152f84...6d9d0`; the formal launcher stayed blocked until that
  hash was committed and pushed.
- P-CAUSAL-019 is the first valid matched real-text FutureSeed comparison. The
  causal GDN2 carrier opened to `0.2765` masked accuracy, so the result is not
  hidden behind another failed baseline. FutureSeed improves CE by `0.08735`
  with a positive paired interval but changes accuracy by only `+0.00253`.
  Report it as a weak mechanism signal, not a language-quality win.
- Future dependency and task benefit are different quantities. Suffix removal
  has exactly zero effect on causal GDN2 but raises FutureSeed CE by `0.4392`
  with a `0.3246` lower confidence bound. The route carries useful right
  context even though the current readout converts little of it into top-1
  gains.
- Always count repairs and regressions together. FutureSeed repairs 119 masked
  tokens and regresses 107, leaving only 12 net. A few clean visual examples
  would badly overstate the aggregate result without regression coverage.
- A fixed token budget still needs an exact sample ledger. The corpus has
  10,003 grouped windows, not 20,000. Using the first 10,000 under 16 fixed
  corruptions gives exactly 160,000 unique pairs; batches must cross corruption
  boundaries instead of wrapping partial batches and silently repeating data.
- Fail-closed auditors need their own tests. Two preflight attempts stopped on
  a cross-device equality check and Python object-ID reuse during autograd graph
  traversal. Both were implementation-only aborts archived before training;
  neither can be counted as a scientific failure.
- P-CAUSAL-020 falsifies the simplest downstream-depth explanation for P019.
  Moving from L2/one seed route to L4/three routes raises suffix-removal CE cost
  `0.4392->0.5865`, so the extra routes carry more future information, but the
  matched CE advantage grows only `0.08735->0.09948`. One hop was not the main
  bottleneck.
- Statistical significance is not practical significance. L4 FutureSeed's
  accuracy delta `+0.00759` has a positive paired interval and yields 147
  repairs versus 111 regressions, but it remains far below the registered
  `+0.03` quality gate. Report the positive interval without promoting it to a
  headline win.
- More model depth is not automatically useful scaling under a fixed small data
  budget. L4 nearly doubles trainable contextual parameters and cuts diagnostic
  throughput relative to L2, while gaining only `+0.01213` additional CE
  advantage. Stop L6/L8 table filling; the next credible language test needs a
  larger training regime or a different established carrier.
- Training-token count is not independent-data count. P-CAUSAL-020 reported
  20.48M input tokens but obtained them by remasking the same roughly 1.28M
  independent WordPiece tokens 16 times. Future scaling ledgers must report
  both optimization tokens and unique source windows/tokens.
- P-CAUSAL-021 isolates data diversity at fixed model and compute. Replacing
  10,000 windows times 16 corruptions with 160,000 independent windows times
  one corruption raises the FutureSeed CE advantage `0.09948->0.16867` and
  the accuracy advantage `0.00759->0.01666`.
- Aggregate behavior moved with the metric: repairs/regressions change from
  `147/111` in P-CAUSAL-020 to `182/103` in P-CAUSAL-021, so net repairs more
  than double from 36 to 79. This is stronger evidence than selected examples,
  while still falling short of the registered language-quality threshold.
- The fixed-compute result identifies independent data as a genuine scaling
  axis for FutureSeed, not a complete solution. It authorizes one joint
  data-and-compute continuation because the endpoint slope remains positive;
  it does not authorize a width, depth, seed, learning-rate or loss table.
- P-CAUSAL-021 warmed throughput is effectively equal between arms and the
  FutureSeed peak-allocation increase is about 20 MB. Keep these as system
  diagnostics until a longer repeated benchmark supports a robust cost claim.
- P-CAUSAL-022 turns the real-text signal into a strong result by jointly
  scaling independent data and optimization at unchanged model size. At 81.92M
  tokens, causal/FutureSeed accuracy is `0.30936/0.37389` and CE is
  `4.61828/3.90513`; both preregistered quality routes pass with positive paired
  lower bounds.
- The FutureSeed advantage itself follows the training scale, not just the two
  arms' absolute quality. From step1000 to5000, accuracy delta grows
  `0.00569->0.06453` and CE advantage grows `0.15974->0.71315`. This is the
  strongest evidence that the route was data/compute limited rather than a
  fixed architectural curiosity.
- Suffix intervention remains the causal check at scale. Removing right
  context changes causal CE by exactly zero but costs FutureSeed `2.12152` CE,
  with lower confidence bound `1.88127`. Better top-line metrics and stronger
  measured future dependence move together.
- Selected examples must be backed by population counts. P-CAUSAL-022 has 440
  repairs and 134 regressions over 4,742 targets, for 306 net repairs; fixes
  occur on both halves of the sequence. This is qualitatively different from
  P019's nearly balanced 119/107 behavior.
- A successful scaling point does not authorize a dense scaling table. P022
  answers the registered saturation question. The next experiment must change
  the evidence level, through a materially larger established language regime
  or a valid bidirectional quality-cost comparison, rather than adding another
  nearby token budget.
- P-CAUSAL-023 separates mechanism quality from practical systems cost. Frozen
  FutureSeed beats BERT-Tiny by `+0.01856` masked accuracy and `0.12818` CE on
  the fixed L128 carrier, while preserving the matched causal advantage. This
  strengthens the future-context claim without proving equal training
  provenance or general model superiority.
- Linear asymptotic complexity is not the same as being faster on real
  hardware. At L128, the current official-FLA GDN2+FutureSeed path reaches only
  `0.113x` BERT batch-64 throughput and uses `2.13x` peak allocation; batch-1
  latency is about `9x` larger. Remove practical cheapness from the current
  claim even though the recurrent route is linear in sequence length.
- FutureSeed itself has measurable runtime overhead on this implementation.
  Relative to matched causal GDN2, batch-64 throughput falls to `0.838x` while
  peak allocation is effectively unchanged. State transfer is parameter-free,
  but it is not execution-free.
- A timing stability miss cannot be used selectively. BERT batch-64
  masked-recovery CV is `0.138`, above the registered `0.10`, so the formal
  speed claim is non-claimable. The robust medians still miss both cost routes
  by a wide margin, making the scientific decision unchanged.
- Fresh-process BF16 Triton inference can move a single argmax boundary token.
  Keep model/data/source/state hashes exact, but specify a bounded numerical
  replay tolerance far below the scientific gate. Never discard or promote an
  architecture based on an unregistered serialization-level equality check.
- Do not fabricate a long-context cost curve from a checkpoint whose learned
  absolute position table ends at 128. A future crossover experiment must
  train or adopt a carrier with valid long-context positional semantics and
  preserve task quality at every reported length.
- Mode-specific mechanisms require mode-specific activation evidence. In the
  P-GDN3-004 fit, generic residual-address fields such as `addr_scale` and
  `qchg/kchg` remain zero by construction because position-Q/K replaces the
  projection input rather than perturbing content Q/K. The decisive checks are
  `gdn2_address_enabled=1`, nonzero Q/K cosine diagnostics, strict execution-path
  provenance, and source inspection. Treating unrelated zero fields as a
  fallback would incorrectly discard an active mechanism.
- A mechanical fit is not a science result. P-GDN3-004's two finite optimizer
  steps establish only that the full D256/L12 CUDA, official-FLA/Triton,
  FutureSeed, checkpoint, and position-address paths compose. Easy-Sudoku
  opening at the frozen step500 gate still decides whether the trajectory is
  worth continuing.
- Easy-stage optimization and hard global closure are distinct gates.
  P-GDN3-004 reaches perfect h50 exact by loop2 at step3000, yet h53 loop5
  exact/blank is only `0.003906/0.582658`. Position-address/payload
  factorization is useful without being a scalable hard-Sudoku solution.
- A positive slope cannot rescue a frozen threshold. From step1000 to step3000,
  h53/h58/h64 loop5 blank rises by
  `+0.028486/+0.015255/+0.061829`, and h53 gains a late-loop solve, but
  both registered h53 alternatives `0.02/0.60` are still missed. Stop the
  trajectory and preserve the learning signal without changing the decision.
- Late-loop residual structure should determine the next mechanism, not a
  nearby hyperparameter. P-GDN3-004 creates h53 exact only from loop3 and
  reduces h64 wrong cells `31.74->26.28`. This makes the frozen checkpoint
  a valid parent for innovation-residual FutureSeed, whose hypothesis is about
  newly written state content rather than address strength or more training.
- Exact zero-init identity and nonzero gradients separate implementation
  validity from scientific value. P-FS3-001 activates to
  `abs(tanh(alpha))=0.010348` with innovation fraction `0.866`, yet hard
  macro and mixed exact both move by exactly zero. A live mechanism can still
  test the wrong decomposition.
- Orthogonality to the inherited state is not a useful enough proxy for
  producer-written evidence on this carrier. It slightly regresses all three
  official hard-range blank accuracies and does not improve same-board
  loop correction consistently. Do not tune the scale, floor, or projection.
- Small CE movement is not closure. Candidate CE improves by only `0.00155`
  while every hard exact count is unchanged. Require board-level decisions and
  same-board loop evidence before promoting a FutureSeed content mechanism.
- Keep orchestration failures separate from science failures. The first wrapper
  stopped after a successful control because it checked the wrong JSON
  filename; exact recovery preserved the preregistered candidate. Archive the
  incident, but do not contaminate the mechanism decision or rerun the control.
- Exact identity and nonzero gradients still do not guarantee that a learned
  content bottleneck preserves useful structure. P-FS3-002 reaches loop5
  payload/update/residual relative RMS `0.3362/0.8955/0.002382`, yet hard macro
  and mixed exact remain exactly unchanged.
- Compressing all 32 K address rows to one V payload is too lossy on this
  carrier. The learned row attention remains nearly uniform (`max=0.0339`
  versus `1/32=0.03125`), and official hard blank accuracy falls in every
  range. Preserve address-conditioned multi-part information before adding
  more decoder capacity.
- Same-board population evidence prevents a few attractive trajectories from
  becoming a false mechanism claim. The codec improves 98/90/94 boards at
  loop5 in the 51-55/56-60/61-64 banks, but worsens 86/78/98 and does not
  strengthen loop3-to5 correction across all ranges.
- Small parameter count is not a reliable cost proxy. Only 59 new shared
  parameters increase matched continuation time by `21.31%` and peak allocated
  memory by `14.55%` because they operate on every layer, head, token, loop,
  and KxV state. Measure execution, not parameter count.
- A failed one-payload codec does not authorize a payload-count table. The next
  valid experiment must make a structural prediction: preserve
  address-conditioned producer state or change the generic GDN recurrent
  memory/update, with a new zero-init contract and board-level gate.
- Preserve information before adding decoder capacity. P-FS3-003 keeps every
  K address row and its full V payload, changing only a bounded scalar on that
  row's actual producer update. This directly tests the address-structure
  diagnosis left by P-FS3-002 without creating a payload-count sweep.
- Dynamic routing needs nonuniformity evidence. A nonzero mean residual alone
  can hide a global rescale, so the P-FS3-003 activation gate separately
  requires within-state K-row gain variation and between-board gain variation,
  plus same-board late-loop correction.
- Structural bounds make a content intervention auditable. Zero initialization
  gives exact terminal identity, independent K/V permutation tests rule out a
  hidden fixed basis, and `abs(residual)<=abs(T-I)` prevents the router from
  manufacturing an unbounded update under a new name.
- Address-conditioned structure does matter locally. P-FS3-003 grows a live
  loop5 residual to `0.012122` relative RMS, improves official61-64 blank by
  `+0.005464`, and reduces mean loop5 wrong cells `26.43->26.06`; 121/256
  hardest-range boards improve versus 78 regress. This is materially better
  than pooling all K rows to one payload, even though it is not a solve result.
- Local wrong-cell gains still cannot substitute for full-board closure.
  P-FS3-003 leaves hard51-64 macro exact at `0.000651` and mixed exact at
  `0.025391`, while 51-55 and 56-60 blank accuracy regress. Report the
  61-64 signal as a structural diagnostic, not a promoted FutureSeed method.
- Three active transfer-side content variants now share the same failure:
  orthogonal innovation, one-payload learned compression, and address-local
  routing all leave hard exact unchanged. Stop adding nearby FutureSeed
  residuals or gates; the next falsifier must change the generic GDN recurrent
  memory/state update itself.
- Tiny modules can remain expensive when inserted at every recurrent state
  handoff. P-FS3-003 adds only 35 parameters and keeps peak allocation overhead
  to `+5.86%`, yet elapsed overhead is `+15.66%`. A low parameter count is not
  a systems argument.
- Pre-model orchestration errors are recoverable only when provenance proves
  that no model path ran. The missing FLA PYTHONPATH and nested-quote wrapper
  failures both occurred before model construction, were recorded as
  non-science aborts, and did not justify changing or rerunning the registered
  experiment.
- The global endpoint is the sum of explicit curriculum stages. The first
  P-FS3-002 full-stack probe used `--steps=3001` but inherited a longer
  `HOLE_STAGES` sum and therefore continued to3006. Stop it exactly, record a
  non-science abort, fix the launcher before formal work, and never use that
  output for model selection.
- After three active transfer-side mechanisms leave hard exact unchanged, the
  next experiment must intervene inside the generic recurrent update. More
  elaborate FutureSeed payload routing would be a nearby table, not a new
  causal question.
- GDN2 erase/write coherence is falsifiable without replacing the official
  kernel. A zero-init per-head coupling can preserve every parent output and
  state exactly, while a positive learned value must measurably contract the
  aggregate gate gap. Require both activation and contraction before reading
  Sudoku metrics as evidence for the mechanism.
- Learned sign is part of the hypothesis, not an implementation detail.
  P-GDN3-005 activates to loop5 mean absolute mix `0.012261`, but the train
  mean is negative and the observed gap ratio is `1.001982`. The model uses the
  new degree of freedom to differentiate erase and write strengths rather than
  enforce the hand-specified aggregate coherence. This directly falsifies the
  premise; constraining the sign would be a nearby rescue, not a discovery.
- A mechanism can improve CE and selected wrong-cell counts while adding no
  global closure. Coherent delta changes train CE `0.858617->0.855293` and
  lowers loop5 mean wrong cells on some 56-64 boards, but hard macro and mixed
  exact are unchanged and loop3-to5 correction is weaker in both ranges.
  Promote board-level exact only when the additional recurrent computation
  closes additional boards.
- Tiny parameter count is especially misleading inside a recurrent kernel
  path. The 96 scalar parameters require materializing and differentiating
  full gate tensors at every layer, token, head, and loop; matched elapsed time
  rises `11.18%` and peak allocation `18.35%`. Cost follows tensor operations
  and saved activations, not the trainable-parameter delta.
- Four live but non-closing interventions now rule out another nearby scalar or
  residual refinement: three FutureSeed content routes plus coherent-delta
  gate coupling. Following the bitter lesson, the next high-information test
  should add scalable learned recurrent memory/address/state capacity and let
  optimization discover its use, rather than encode another fragile aggregate
  prior.
- More recurrent state is not automatically more useful state interaction.
  P-GDN3-006 activates all 12 independent auxiliary experts, reaches loop5
  residual relative RMS `0.010408`, and carries board-varying terminal state,
  yet hard macro and mixed exact remain exactly unchanged.
- A parallel expert behind a zero-init readout can learn locally without
  changing the parent solver's global decisions. Train CE improves
  `0.858617->0.855923`, but every official hard-range blank score regresses and
  same-board loop3-to5 correction weakens on both 56-60 and 61-64.
- State capacity and update coupling are different hypotheses. The dual-state
  expert adds a private address/update subspace, but its residual output does
  not make current writes depend directly on what the main memory already
  contains. The next test should close that feedback loop rather than add a
  third state bank or tune expert size.
- Full recurrent experts are a costly way to buy generic capacity. A 21.6%
  parameter increase and 25% state-value increase more than double matched
  continuation time (`+106.57%`) and reduce throughput
  `15.497->7.502` boards/s, even though peak allocation rises only `46.79%`.
  Measure recurrent kernel count and saved activations, not only parameters or
  state values.
- Strong activation evidence protects the mechanism conclusion. Exact
  zero-readout identity, two-stage gradients, all-12 expert activity,
  auxiliary FutureSeed RMS, board variation, and finite address/cosine metrics
  rule out a dead-path explanation. The negative result is architectural, not
  an implementation ambiguity.
- Do not rescue dual-state expert width, count, address form, residual scale,
  or continuation duration. Those would turn one decisive capacity test into
  a nearby compute table. Following the bitter lesson now means a simpler,
  scalable state-conditioned controller around the existing official core.
- Reading inherited memory before proposing the next write is not enough to
  close hard boards. P-GDN3-007 activates all 11 receiving paths, with loop5
  read RMS `0.725926` and residual relative RMS `0.018716`, yet hard macro exact
  remains `0.000651` and mixed exact regresses by `0.001953`.
- Strong state dependency does not imply a better state transition. The
  controller changes K/V/erase/write by
  `0.038044/0.009582/0.005715/0.006785`, but official61-64 blank and
  same-board late correction both regress. The issue is not an inactive or
  overly weak feedback path.
- A pre-scan controller sees the inherited layer state, not the live state
  after each token update inside the linear recurrence. Calling it
  "closed-loop" should therefore be qualified: it closes the layer-level
  interface while leaving the within-scan transition itself unchanged.
- Lower CE again fails as a closure proxy. P-GDN3-007 improves train CE
  `0.858617->0.855740` while losing one mixed exact board and opening no new
  hard-range exact boards. Use full-board exact and same-board correction for
  architectural promotion.
- State feedback is affordable relative to a second recurrent expert but not
  free. Only 30,720 parameters reduce throughput `15.497->12.858` boards/s and
  add `20.52%` elapsed time because controller tensors are materialized across
  every layer, token, head, and loop.
- Do not rescue state-feedback controller hidden size, output scale, target
  subset, layer sharing, or duration. Six consecutive matched
  active-but-nonclosing content/update/state interventions now justify moving
  below readout-side residuals: the next high-information GDN3 test must change
  the scalable live recurrent transition itself.
- A post-scan live-state transition can improve token accuracy without closing
  boards. P-GDN3-008 improves official51-55/56-60/61-64 blank accuracy by
  `+0.001325/+0.001853/+0.004609`, but hard macro exact rises only
  `+0.000651` and mixed exact is unchanged. This is useful directional evidence,
  not a promotion result.
- Transition depth alone is not stable memory. Reusing unconstrained
  erase/write gates in a second official sweep drives terminal residual
  relative RMS from `1.1981e4` at loop1 to `4.7109e8` at loop5. Unit-normalized
  FutureSeed can mask enormous state magnitude while preserving finite output;
  always instrument the carried state before calling a recurrent path healthy.
- Strong activation plus lower CE still does not identify global closure.
  P008 activates all 11 correction paths, reaches correction-K relative RMS
  `0.120164`, and improves CE `0.858617->0.855832`, yet closes only one extra
  hard-range board-equivalent in the aggregate and misses both quality routes.
- Same-board late correction must be checked at the hardest range. P008
  strengthens loop3-to5 correction on 51-55 and 56-60 but weakens it on 61-64;
  easier-range refinement does not establish scalable hard-board reasoning.
- Extra scans carry systems cost even with few new parameters. Eleven
  head-shared projections add only 11,264 parameters, but the 11 extra official
  transitions reduce throughput `15.497->12.151` boards/s and add `27.54%`
  elapsed time and `22.95%` peak allocation.
- Do not rescue consolidation source, scale, decay/gates, scan count, producer
  subset, or duration. The next general recurrent update should control state
  geometry by construction, such as a learned normalized or contractive live
  transition, while preserving scalable capacity. Another fixed sweep,
  pre-scan residual, parallel expert, scalar prior, or transfer router would
  repeat a closed boundary rather than test a new mechanism.
- Forward cache composition does not guarantee a trainable recurrent
  composition. P-GDN3-009's zero-angle 64+17 official path is bit-exact to the
  unsplit parent for model output and all terminal states, including nonzero
  incoming state, yet each final state graph exposes only one rather than two
  `ChunkGDN2FunctionBackward` nodes.
- Prove the initial-state gradient contract before building a mechanism around
  an external recurrent boundary. Exact forward parity, official kernel
  provenance, and bounded state geometry are insufficient when the inserted
  transition cannot receive the registered cross-boundary learning signal.
- Contract failures can be high-information results. P009 consumed no
  continuation steps and no benchmark evaluation, but it closes external
  chunk splitting, graph-assertion relaxation, custom backward, alternate
  boundary, and nearby controller/scale rescue for this mechanism.
- The next stable live-transition candidate should remain inside one
  differentiable official GDN2 invocation, for example by changing a general
  address/update parameterization before the single scan. It must not hide a
  second scan, custom solver, or unproven cache-gradient bridge.
- Staying inside one official autograd function is necessary but not sufficient
  for exact migration. P-GDN3-010 doubles the physical sequence with
  mathematically zero auxiliary writes, yet changes full-model output by max
  absolute `0.04559326171875` relative to the 81-step parent.
- Chunked recurrent numerics depend on physical sequence geometry. An inserted
  step with `g=0`, `b=0`, and zero V may be an algebraic state identity while
  still changing chunk partitioning, reductions, normalization or read timing.
  Assert end-to-end output and every carried state before treating such a
  microstep as an exact-resume insertion.
- P009 and P010 jointly close the easy composition routes around the pinned
  operator: split calls keep forward parity but lose the required state graph;
  expanded single calls keep one graph but lose forward parity. The next
  high-information recurrent transition must preserve the original token
  geometry or be implemented as an explicitly audited kernel-level update.
- Contract-first evaluation again saves compute. P010 used no continuation,
  benchmark evaluation or control rerun, and its failure should not be rescued
  with gate/decay/order/count/rank/scale/tolerance or duration changes.
- Explicit `.float()` conversion does not create an FP32 island under CUDA
  autocast. P-GDN3-011's direct synthetic route preserves norm to `5.96e-8`,
  while the same route in the production autocast context reaches FP32 norm
  max error `5.1444e-4` after one optimizer step.
- Precision contracts must exercise the actual nested autocast context and an
  activated parameter regime. Zero-angle identity and an outside-autocast
  geometry test both passed, yet neither covered the numerical behavior used
  by the trained route.
- A stable algebraic parameterization is not automatically a stable deployed
  operator. Plane orthogonality remained excellent (`1.40e-6` dot,
  `2.38e-7` norm error), but lower-precision contractions broke the registered
  Frobenius invariant. Record both geometric and execution-precision checks.
- Fail-close before formal continuation. The P011 probe consumed one exact
  resumed training step, produced no NaN/OOM/fallback, and returned GPU1 to
  0 MiB. Fixing autocast after seeing the registered miss would be a precision
  rescue, so the orthogonal head-write family is closed without a Sudoku score.
- A signed recurrent spectrum is learnable but does not by itself close hard
  boards. P-GDN3-012 uses `b'>1` on `8.707%` of mixed loop5 channels with erase
  residual relative RMS `0.303028`, yet hard51-64 macro exact is unchanged and
  mixed exact loses one board.
- Activation strength is not a substitute for causal usefulness. The adapter
  changes erase by about 30%, spans `[0,1.84375]`, and remains numerically
  bounded, ruling out dead-path and instability explanations for the negative
  result.
- Aggregate blank movement can hide a worse recurrent trajectory. P012's
  loop5 blank deltas are nearly neutral, but same-board loop3-to5 correction
  weakens from `0.066->0.039` on 56-60 and `0.184->0.098` on 61-64.
- Lower-dimensional gate flexibility still materializes over every token,
  head, layer and loop. Only 12,288 parameters add `16.64%` continuation time
  while peak allocation rises `6.04%`; measure execution topology, not just
  parameter count.
- P005, P007, P008 and P012 now cover coherent nonnegative gates, inherited-
  state-conditioned gates, an extra post-scan transition and adaptive signed
  erase. The next experiment should target generic memory organization or
  address-state interaction, not erase source, bound, scale or training rescue.
- Exact multibank migration is feasible inside one pinned-official call.
  P-GDN3-013 doubles H8 to H16 while preserving full output, every base and
  companion state, and finite nonzero incoming-state behavior bit-exactly; it
  also keeps one official backward path per layer.
- Serial zero initialization can create a production optimization barrier.
  P013 zero-initializes both its companion Q/K projections and its read gate.
  At step3001 the gate opens to `0.001048`, but Q/K weights, Q/K residuals and
  paired-state residuals remain exactly zero because those projections had no
  first-step loss path.
- Synthetic two-stage gradients are necessary but not sufficient activation
  evidence. Opening the gate by hand yields finite nonzero Q/K gradients and a
  minimum state residual relative RMS of `0.026429`; the real one-step resume
  still does not reach that stage.
- Bind production activation to a specific checkpoint before launch. When that
  checkpoint misses, extending one extra step or changing initialization is a
  rescue even if the mechanism is mathematically viable. P013 therefore has no
  formal Sudoku score and does not falsify multibank memory capacity.
- New recurrent-capacity candidates should expose their differentiating state
  parameters to first-order gradients while preserving exact parent behavior.
  Do not repackage P013 through a different read-gate map or initialization;
  that nearby family is closed for this experiment sequence.
- Zero rows do not guarantee an exact lower-dimensional embedding of a pinned
  recurrent operator. P-GDN3-014 appends a zero K32 state bank and a zero extra
  write projection, yet the K64 call changes full model output at initialization.
- Audit normalization in the actual physical operator geometry. Reproducing
  parent K32 normalization externally and fixing the nominal `1/sqrt(32)` scale
  was not enough to preserve the parent function after the row-axis expansion.
- Parent identity is a binding systems gate, not a tolerance to tune after the
  fact. P014 stops before step3001, consumes no benchmark score, and closes
  normalization placement, address map, row count, scale and init rescue for
  this exact coupled-row construction.
- Keep orchestration failures separate from mechanism evidence. P014 R1-R3
  exited before CUDA/model execution on interpreter or environment assertions;
  only R4 reached and falsified the registered model identity claim.
- Algebraic equivalence does not guarantee trainable numerical conditioning.
  P-GDN3-015 preserves zero-init full output and all terminal states bit-exact,
  matches a direct Bi-Axis recurrence and keeps one official backward per layer,
  yet one optimizer step drives its global inverse moving frame above `3505x`.
- Instrument both sides of a moving-frame transform. P015's grouped decay is
  finite and structured (`0.032549` magnitude with nonzero group/board/token
  variation), but cumulative scale falls to `0.000285`, write-frame relative RMS
  reaches `198.57`, and restored-state relative RMS reaches `0.998929`. Looking
  only at decay logits would incorrectly label this path healthy.
- A production stability probe can be the final scientific decision. P015
  returns status0 and writes a valid checkpoint, metrics and visualization, but
  the preregistered scale bounds fail before quality evaluation. Do not spend a
  100-step continuation to rediscover an already-observed conditioning failure.
- Retain the abstraction, reject the implementation. Persistent V-axis lifetime
  is not covered by prior K-gate, state-feedback or address-capacity failures,
  but revisiting it requires an intrinsically bounded chunk-local transition.
  Group count, softplus map, epsilon, clipping, initialization, precision and
  scale changes are rescues of this exact global-frame experiment.
- Cross-layer state reuse has a coordinate problem that scalar gates cannot
  express. Producer and receiver GDN2 layers learn independent K/V bases, so
  direct terminal-state transfer may preserve magnitude while misaligning
  address and payload semantics.
- Orthogonal transport is a high-information FutureSeed test because it changes
  only the cross-layer coordinate map: no new memory, scan, recurrent core or
  task logic is needed, and Frobenius geometry has an explicit invariant.
- Zero-init algebra is not enough. P-FS3-004 must prove bit-exact full-model
  identity, direct gradients on every edge/head, and bounded opened geometry in
  the production autocast context before an exact-resume step is authorized.
- Migration wiring is part of the contract. P-FS3-004's first probe stopped
  before step3001 because two new angle tensors were absent from the explicit
  upgrade allowlist; adding only those exact names and repeating the full
  pushed-SHA CUDA contract produced a clean, auditable resume.
- Orthogonal cross-layer basis transport is production-feasible. On final SHA
  `0335534`, all edge minima activate after one step: K/V rotation RMS
  `0.005694/0.005998` and transported-state residual RMS `0.014258`, with
  nonzero board/head variation. Norm and orthogonality errors stay below
  `2.21e-6`, substantially inside the fixed bounds.
- Keep mechanism and quality claims separate. An eight-board one-step probe can
  prove exact migration, first-order access and stable geometry, but cannot say
  whether cross-layer basis mismatch causes the 51-64 exact cliff. The matched
  100-step quality/cost test was not run, so P-FS3-004 ends as activation
  evidence rather than a Sudoku win or loss.
- Intrinsic gauge bounds can convert a numerical failure into a clean
  scientific test. P-GDN3-016 keeps every frame and inverse inside `[0.25,4]`,
  retains one official backward per layer and activates all 12 V-lifetime
  paths, resolving P015's `3505x` inverse-frame ambiguity.
- A bounded coordinate system can still be driven to a bad operating point.
  P016 reaches frame extrema `0.250029/3.998026` and inverse `3.999530`; bounds
  prevent explosion but do not prevent saturation or preserve useful memory.
- Funding nonexpansive V-axis lifetime through common K contraction changes the
  parent's forgetting policy too aggressively. Loop5 V-decay/common-K magnitude
  reaches `0.351812/0.351764`, while hard macro and mixed exact collapse to zero
  and CE rises `0.858617->1.498811`.
- Same-board evidence makes the failure causal and decision-relevant. On all
  256 official61-64 boards the candidate has more loop5 errors, and loop1-to5
  correction falls from `5.426` to `1.441` cells. Aggregate instability or one
  unlucky exact board cannot explain the result.
- Small parameter count does not imply cheap recurrent execution. The 196,608
  coordinate parameters add `28.60%` elapsed time and `21.55%` peak allocation
  because frame transforms are materialized across layer, token, head and loop.
- P015 and P016 jointly close the registered grouped V-lifetime family at this
  parent: P015 rejects the global cumulative frame numerically; P016 rejects the
  bounded gauge scientifically. Do not tune cap, group count, common-shift map,
  max operator, initialization or training settings under a new run name.
- Continuous automation is not a research objective. Once an experiment has a
  bounded hypothesis, falsifiable probe and fixed kill gate, stop at the next
  decision boundary instead of automatically launching another candidate.
- Raven can be isolated as an allocation control plane without replacing GDN.
  P-GDN3-017 keeps one pinned-official position-QK transition per layer and
  adds no state, token or scan, while exact zero initialization preserves the
  parent and gives every router a direct first-order gradient.
- A one-step activation probe does not guarantee endpoint routing geometry.
  P017 moves from loop5 allocation range `0.706..1.368` and entropy `0.997790`
  at step3001 to `0.000145..7.983757` and entropy `0.765180` at step3100. Bind
  non-collapse limits at the formal endpoint, not only at migration.
- Strong routing activation is not evidence of useful memory specialization.
  Endpoint K/g relative change reaches `0.508998/1.127739`, yet hard macro
  exact is unchanged and mixed exact regresses. The controller learns a large
  intervention without learning a closure-producing partition.
- Same-board evidence separates a local benefit from a general mechanism.
  P017 improves loop3-to5 correction on 51-55, but weakens it on 56-60 and
  61-64; the hardest range has 124 worse versus 87 better loop5 boards.
- Cross-hardware quality and systems claims need different contracts. Exact
  predictions can be compared with the frozen A10080 control, while P017's
  A10040 systems evidence uses preregistered absolute gates. Throughput and
  memory pass at `12.622` boards/s and `15720.7/16898.0 MiB`; no same-hardware
  overhead claim is made.
- Do not benchmark a gate that cannot change the decision. Once endpoint
  allocation stability and both exact-quality routes fail, a separate timing-
  CV GPU run has zero decision value. Record it as not measured rather than
  burning another continuation.
- P017 closes fixed-bank softmax allocation, not all Raven/GDN hybrids. A
  successor would need qualitatively different stable recurrent state
  organization or state interaction. Slot count, temperature, top-k, route
  scale and longer training are rescues of this experiment, not new science.
- Kernel feasibility is part of mechanism design. P018's S8 persistent Raven
  controller never produced a model score because pinned official chunk GSA
  requires every Triton dot axis to be at least 16. Moving once to S16 was a
  preregistered production-minimum correction, not a slot-count sweep.
- Raven and GDN can be combined without ambiguity. P019 preserves the dense
  position-QK GDN transition while an independent official Raven state carries
  sparse retrieved content across layers and writes only into V. Zero adapters
  retain bit-exact parent output/main-state identity, while every Raven and
  adapter path becomes trainable after opening.
- Stable sparse retrieval still need not solve dense global closure. At the
  endpoint P019 has V residual RMS `0.066201`, minimum entropy `0.670635`,
  maximum slot mass `0.370819`, and finite 11-edge state transport, yet hard
  macro exact improves by only `0.000651` and mixed exact regresses.
- Check the hardest same-board correction before crediting a local exact gain.
  P019 improves average loop3-to5 correction on 51-55 and 56-60, but weakens
  61-64 from `0.1836` to `0.1172` cells; one extra solved 51-55 board does not
  establish a scalable recurrent-memory improvement.
- Recurrent controller cost follows repeated stateful execution, not parameter
  count. P019 adds only `5.60%` parameters and `17.54%` peak allocation, but
  throughput halves from `15.497` to `7.451` boards/s and elapsed rises
  `107.99%` because 12 Raven scans run inside each of five reasoning loops.
- P017 and P019 jointly close two nearby Raven/GDN hypotheses at this parent:
  stateless allocation over GDN rows collapses, while a genuine persistent
  Raven memory remains stable but is too costly and does not improve exact
  closure. Slot/top-k/width/injection/scale or duration changes are rescues, not
  evidence-driven next experiments.
- Stop autonomous iteration when the registered mechanism family closes. GPU
  occupancy is not a research metric; release the task after hashes, matched
  case evidence and GitHub provenance are complete, and require a new explicit
  mechanism hypothesis before spending more compute.
- Runtime reproduction is a science gate, not bookkeeping. The current A100
  replay of the historical L1024 FutureSeed carrier reaches balanced `0.1735`
  rather than `0.7475`; P020 can only be called an exploratory mechanism signal.
- A bounded learned Q/K metric is the only positive address intervention in
  P020--P023: it raises balanced accuracy by `0.30875` with unchanged state and
  scan count. Its failure tail is `94.16%` wrong-key valid-value swaps, showing
  that learning the value set and binding values to keys are distinct problems.
- More writes are not more memory. P021's second live edit is fully active but
  overwrites the shared state and collapses balanced accuracy to `0.00975`.
- Same-byte state repartition is not free capacity. H8/K16/V32 preserves 4,096
  values, yet P022 is slower (`2.55x` fit) and weaker than P020. State bytes
  alone do not capture optimization or addressing cost.
- Individually positive deltas need an explicit combination gate. P023 reaches
  only `0.3090`, below P020 by `0.17325`; lowering its swap fraction while
  increasing total errors is not binding closure.
- P020--P023 close static metric conditioning, extra rank-one writes, same-byte
  bank factorization and their fixed combination. A successor must change
  stable address-binding state organization on a reproducible carrier; do not
  rescue this family with metric/head/K/V/epoch/LR/seed tuning.
- P024 is the zero-capacity collision test left by that boundary: a fixed
  compact bilinear binding of two normalized K16 factors inside each existing
  K32 head requires joint address agreement while preserving parameters, state bytes and one
  official scan. Judge it only by endpoint binding quality and cost; do not
  turn hash count, partition or weighting into a sweep.
- Lower wrong-key swap fraction is not sufficient when total retrieval errors
  explode. P024 cuts the fraction from `0.94157` to `0.03035`, yet balanced
  accuracy collapses to `0.0115`; report both the error class and denominator.
- A fixed product address can erase collisions and learnability together. Two
  active, norm-balanced K16 factors do not preserve the parent's useful linear
  K32 geometry merely because they occupy the same state bytes.
- Close P024 without interpolation or factorization tuning. The evidence calls
  for a preserved linear base state plus an independently bounded correction
  memory, not another replacement hash topology.
- Compare address organizations at identical state bytes before crediting
  capacity. P025's learned semi-orthogonal basis reaches `0.36175` balanced
  versus fixed compression `0.01875`, proving that address geometry itself is
  causal rather than the extra 2,048 state values.
- Error taxonomy needs its denominator. P025 fixed compression reports only
  `0.04025` wrong-key swaps among errors because it makes 3,925 mostly arbitrary
  errors; the lower fraction is not a better binding system.
- A bounded active companion state can still be the wrong ownership boundary.
  P025 passes activation, geometry and cost gates but remains `0.1205` balanced
  below P020. Exact committed evidence should not automatically live in a
  separately indexed correction bank.
- Do not stack individually interesting failed modules. P020 shows a main-state
  address metric signal and P025 shows learned-basis causality, but P025's
  independent bank misses every absolute gate. Combining them would be an
  unregistered rescue, not a justified next architecture.
- Diagnose gradient ownership before adding another mechanism. P-LOOP-001
  finds strong opening-versus-continuation conflict only in the 96 native
  FutureSeed gate parameters on 51-55 and 61-64; GDN2 address/edit and shared
  parameters have zero negative pairs across all three hard ranges.
- A conflict claim needs both direction and useful trajectory evidence. The two
  admitted ranges combine cosine `-0.904763/-0.718710`, pair conflict `0.40`,
  cancellation below `0.70`, and loop1-to5 wrong-cell reduction. The aligned
  56-60 counterexample prevents overgeneralizing this into a global optimizer
  diagnosis.
- Inspect the full task cosine matrix, not only one aggregate. Here loops3-5
  are almost collinear; the actionable conflict is loop1 against continuation.
  A first intervention should therefore be a deterministic FS-only opening
  projection with unchanged forward and gradient norm, not delayed loss or a
  learned loop selector.
- Python autograd node proxies must remain alive during large graph provenance
  walks. Tracking only `id(fn)` can suffer object-ID reuse and falsely report a
  missing official backward path; archive this as an engineering abort and
  rerun from a new pushed clean SHA without changing science gates.
- Function-preserving initialization does not make a new degree of freedom
  useful. P-GDN3-031 begins at exact native GDN2, yet erase/write key cosine
  falls to `0.0483/0.0600` and balanced accuracy collapses to `0.011`.
- Erase and write addresses need a shared ownership anchor. When erase becomes
  nearly orthogonal to the rows populated by write/read, the state can remain
  spectrally bounded while associative retrieval disappears.
- Never credit a lower conditional swap fraction without total errors and the
  prediction histogram. P031 moves `0.458018->0.035642` swaps among errors only
  because errors rise `2775->3956` and 3,657/4,000 predictions collapse onto
  two value tokens.
- A failed direct-decoupling result cannot be rescued by adding an angle/tie
  penalty and still answer the same hypothesis. Such a constraint defines a
  new coupled-address mechanism and needs independent motivation and gates.
- Unlicensed idea repositories can inform a clean-room equation, but code must
  not be copied and weak unmatched demonstrations must not be treated as
  evidence. P031 used only the mathematical idea from `GDN_decouple_k`.
- Exact source and process-order restoration did not recover the historical
  directional-MQAR L1024 score. P-DIAG-CARRIER-002 reaches only `0.10825`
  balanced after the exact causal-to-FS order, while the same process reaches
  `0.99525` at L64. The long carrier is optimization-trajectory sensitive, not
  generally broken.
- A historical absolute score cannot gate a new architecture once exact
  reproduction fails. Use a contemporaneous control in the same source,
  process, GPU task and fixed arm order, and judge a candidate by preregistered
  relative quality plus absolute usability.
- Cross-layer parameter sharing is not automatically namespace alignment.
  P033's two layers use the identical bounded metric object and both train it,
  yet balanced accuracy drops `0.17325->0.1250`; each layer still needs its own
  adaptation around independently learned content projections.
- A newly stable control can change the interpretation of old evidence without
  changing the old result. P033 reproduces P020's control (`0.17325` versus
  `0.1735`), so P020's per-layer metric gain to `0.48225` is a credible relative
  mechanism signal after retiring the non-reproducible `.7475` absolute gate.
- Chained pinned-official chunks do carry gradients through physical state.
  P035 measures nonzero chunk2-to-chunk1 V, K-decay and V-decay gradients; a
  terminal grad-fn count alone is not a valid cross-chunk credit diagnostic.
- Boundedness does not make a value-lifetime axis useful. P035 keeps every
  local frame in `[0.25,4]` with active group/board/token variation, yet balanced
  accuracy falls `0.09975->0.01375` and errors rise `3601->3945`.
- A lower wrong-key-swap fraction still needs an accuracy denominator. P035
  lowers conditional swaps `0.131630->0.038530` by preventing useful retrieval,
  independently repeating the collapse pattern from P024/P028/P030/P031/P032.
- Do not fuse a correctness-first multi-chunk prototype after its quality gate
  fails. P035's `7.356x` warmed-step ratio is an optimization target only if the
  recurrence improves quality; here fusion would preserve the wrong mechanism.
- A high erase/write cosine is not enough to preserve GDN2 address ownership.
  P036 keeps mean cosine near `0.91` and minimum near `0.893`, yet balanced
  accuracy collapses `0.36025->0.01375`; read, write and erase need the same
  coherent binding, not only a shared neighborhood.
- Strong activation plus bounded state makes P036 a quality falsification, not
  an optimization excuse. Correction relative RMS reaches about `0.46`, all
  cost gates pass, and terminal state remains finite while errors rise
  `2559->3945`.
- P031 and P036 close the complete decoupled-key neighborhood. Do not rescue it
  with angle, tangent radius, initialization, regularization or duration; move
  to a different scalable recurrent state organization or committed update.
- A mathematically coherent committed residual can still destroy learnability.
  P037 keeps one exact key, one state and a contractive transition, yet balanced
  accuracy collapses `0.36625->0.01525` and errors rise `2535->3939`.
- Do not average away GDN2's coordinate-wise control surfaces without causal
  evidence. Replacing K-wise erase and V-wise write gates by one beta is active,
  stable and cheap, but removes degrees of freedom needed to learn retrieval.
- A residual RMS near one is not useful credit by itself. P037's two layers
  reach `1.07581/1.06857` committed-residual relative RMS while future/past CE
  rises to `4.6391/5.0233`.
- Conditional swap reduction again needs an absolute denominator. P037 moves
  `.62091->.03783` swaps among errors only because almost every query fails.
- Close beta reducer, target, residual-scale, decay and training rescues. A next
  recurrence must preserve native erase/write expressivity while changing a
  genuinely different scalable state-organization boundary.
- A learned content partition can reduce binding swaps without improving the
  task. P038 lowers wrong-key swap fraction `.76383->.59393`, yet balanced
  accuracy falls `.4940->.36525` and total errors rise `2024->2539`.
- Whole-state routing is too coarse for associative memory. Assigning each
  token's read, erase and write to one shared soft slot makes two full states
  specialize, but it also delays optimization and loses useful cross-binding
  retrieval.
- Check both mean and worst-case state separation. P038 mean slot cosine falls
  to `.8655/.7237`, while max cosine remains `.99977/.99699`; aggregate
  specialization can hide nearly identical trajectories on some examples.
- A lower wrong-key fraction is causal evidence only when total errors and
  joint exact also improve. Here joint exact drops `.041->0`, so the isolated
  swap gain is not an architecture win.
- Close slot count, router form, temperature, gate map and duration rescue. A
  successor must keep coherent native address ownership without using one
  global content hash to choose an entire recurrent trajectory.
- A cross-layer state can be useful yet unreadable without changing the live
  recurrence. The clean test is producer-native decoding plus an attributable
  bounded receiver fusion, not another K/V transport or erase-key rewrite.
- New FutureSeed readouts need an edge-off counterfactual on the same trained
  model. Activation alone cannot distinguish useful transferred evidence from
  capacity that the rest of the network ignores.

## 2026-08-15: Native Q/K disagreement is useful structure, not drift to remove

- P-GDN3-042 is an exact test of bounded midpoint-preserving Q/K coherence:
  it keeps the coherent erase/write/read state, one official scan and native
  FutureSeed while adding only 256 zero-initialized parameters.
- All eight adapter heads activate, alpha stays in `.9570-1.0547`, and Q/K
  changes are finite, yet balanced accuracy falls `.17475->.05725`, past
  accuracy nearly collapses `.1885->.0145`, and errors rise `3301->3771`.
- Reducing wrong-key swaps `770->331` is not success when arbitrary retrieval
  errors increase. Always couple conditional error composition to total errors,
  directional accuracy and joint exact.
- P020's large positive Log-SPD signal should be read as within-map address
  geometry, not pressure to make native Q and K similar. Preserve their full
  learned differential and exact read/write/erase ownership.
- Close alpha cap, coordinate/head/layer granularity, initialization,
  regularization, normalization and duration rescues. A successor must change
  a different scalable address geometry or state organization boundary.

## 2026-08-15: Pairing-preserving Q/K gauges do not fix binding

- P-GDN3-043 preserves one coherent read/write/erase key and the complete
  native Q/K differential while applying inverse-transpose dual coordinates.
  The raw bilinear pairing is preserved to about `.00205` relative RMS at BF16,
  production replay is exact, and all eight learned gauge heads activate.
- This is still harmful: matched balanced accuracy falls `.04850->.01975`,
  future/past fall `.05450/.04250->.01800/.02150`, and errors rise
  `3806->3921`. A mathematically valid coordinate gauge is not a useful memory
  intervention merely because it preserves pairwise dot products.
- Conditional swap improvements remain unsafe evidence. Wrong-key valid-value
  swaps fall `302->163`, but overall retrieval worsens and the registered
  fraction improvement is missed. Binding quality depends on the learned
  normalized state trajectory, not only one raw Q/K algebraic invariant.
- Matrix exponentials also impose a hidden step-time cost: end-to-end elapsed
  is only `1.0898x`, while an isolated warmed step is `1.9149x`. Always retain
  the independent warm benchmark even when compilation or evaluation dilutes
  total wall time.
- P031/P036/P042/P043 close the nearby decoupled-key, tied-neighborhood,
  forced-coherence and dual-gauge families. Further Q/K caps, ranks, sharing or
  normalization variants are table filling; move to a scalable live-state
  organization or transition mechanism.

## 2026-08-15: Terminal-read self-distillation suppresses binding rather than repairing it

- P-FS2-011 leaves native FutureSeed inference exactly unchanged and sends a
  real all-token receiver-native read loss only into the producer state path
  and receiving FutureSeed gate. Its strict contract and all cost gates pass.
- The auxiliary target is easy to satisfy without learning retrieval:
  inherited/live read cosine reaches `.9370`, yet balanced accuracy collapses
  `.12375->.01775` and total errors rise `3505->3929`.
- A large drop in valid-value/wrong-key swaps (`543->185`) is again misleading
  when correct retrieval collapses. Always report absolute errors and accuracy
  with conditional error composition.
- Matching a detached receiver terminal state is not a credit signal for
  binding correctness. It rewards agreement with the receiver's own mistakes
  and can flatten useful producer evidence even with no labels or inference
  overhead.
- Close auxiliary weight, teacher, detach, normalization, token/layer mask and
  duration variants. Future FS2 work needs a causal, receiver-native content
  mechanism or a task-independent learning signal that cannot be minimized by
  destroying retrieval.

## 2026-08-15: Key-spectrum whitening is not coherent binding credit

- P-GDN3-045 preserves GDN2's exact native read/write/erase ownership and
  inference graph, while sending a scale-free covariance credit only into the
  native K projection and K convolution. The strict identity, official-kernel
  and selective-gradient contract passes.
- The credit does not make the learned address space healthier. Endpoint
  effective rank is `25.423/12.352` and anisotropy is `3.006/4.779`; the
  second layer becomes markedly more concentrated despite the active loss.
- Retrieval collapses from `.4830` to `.0130` balanced accuracy and total
  errors rise `2068->3948`. Wrong-key valid-value swaps fall `1959->144`
  only because almost every query fails. Conditional error composition is not
  evidence without absolute retrieval quality.
- A covariance target competes with the task's useful address geometry and can
  be defeated by the coupled CE trajectory. P020's positive Log-SPD result
  must not be generalized into isotropic keys.
- Close coefficient, covariance scope/normalization, mask, detach, seed, data,
  loss and duration variants. The direct decoupled-key request and its coherent
  spectrum refinement are both bounded; a successor must change scalable live
  state organization or transition, not another Q/K loss.

## 2026-08-15: Stable token identity improves retrieval but not binding closure

- P-GDN3-046 gives both native Q and coherent K the same cross-layer token-only
  residual. Repeated semantic keys have exact residual identity while native
  payload, gates, state, official scan and FutureSeed remain unchanged.
- The signal is real: balanced accuracy rises `.36625->.42450`, future accuracy
  rises `.35150->.43350`, and total errors fall `2535->2302`.
- It does not solve binding. Joint exact reaches only `.012`; wrong-key
  valid-value swaps increase `1574->1717` and their fraction rises
  `.620907->.745873`. Better value-set retrieval can expose more address
  ambiguity rather than eliminate it.
- Use native-vs-native replay to calibrate pinned BF16/Triton backward checks.
  Here native replay and candidate-parent gradient differences are essentially
  identical, while forward and recurrent-state identity stay bit exact.
- Independent warmed-step cost is `1.37958x`, even though arm-order elapsed is
  misleadingly lower. Keep a warm benchmark separate from compile/eval wall.
- Close scale, normalization, position, sharing, rank and duration variants.
  The decoupled-key line is bounded; move to scalable state organization or a
  different committed recurrent transition that can preserve binding identity.

## 2026-08-15: Same-seed native replay is exact; binding is the dominant error

- P-REPRO-001 runs two native GDN2 plus FutureSeed arms from one serialized
  initialization with identical data, warmup and post-warm RNG reset. The
  trained parameter hashes and all `4000` query predictions are exactly equal.
- Balanced/future/past/joint accuracy is `.494/.454/.534/.041` in both arms,
  and both make exactly `2024` errors. The current protocol is reproducible;
  do not attribute cross-experiment control spread to unavoidable Triton noise.
- Of those errors, `1546` are valid values attached to the wrong key. This is
  `76.38%` of all failures and directly diagnoses ownership interference rather
  than missing value content or insufficient future context.
- Keep contemporaneous controls because different mechanisms change compile
  and optimization trajectories, but stop spending architecture budget on
  dense side states, Q/K wrappers or receiver residuals. The next useful test
  must isolate committed key-value pairs inside a scalable live memory update.

## 2026-08-15: Pair isolation without learnable ownership destroys retrieval

- P-GDN3-050 gives every exact official committed edit a factorized key/value
  slot, and its strict contract proves the GDN2 and GSA paths, gradients,
  bounded transition, parent identity and exact cost accounting are real.
- The wrong-key swap fraction falls `.763834->.066702`, but balanced accuracy
  simultaneously collapses `.494->.04425` and errors rise `2024->3823`.
  Conditional swap composition is again meaningless without absolute retrieval.
- Hard top-1 straight-through allocation develops dead capacity: the two layers
  use only `15/10` of 16 slots. Explicit pair storage does not itself solve who
  owns an update; it moves the problem into discrete routing credit assignment.
- A second recurrent scan is also too expensive here: elapsed/post-warm wall is
  `3.277/3.251x`, despite a `1.586x` independently warmed step and `1.387x`
  allocation. Compile and validation do not explain the full optimization hit.
- Close slot count, temperature, anchors, gate scale, normalization, scan
  placement and duration rescue. The next mechanism must make ownership a
  differentiable part of the native live transition while preserving baseline
  retrieval, not add another sparse or dense side memory.

## 2026-08-15: Protecting FutureSeed state does not protect binding ownership

- P-FS2-012 cleanly separates the inherited producer state from the receiver's
  live writes. The inherited KxV matrix is immutable, the receiver uses its own
  zero-start official GDN2 state, and receiver-native queries read both planes.
- The read-only path is not dead. Removing it only at evaluation drops
  balanced/future accuracy from `.4445/.4510` to `.2110/.0110`, while its read
  RMS and relative RMS are `.1570/.01889` with nonzero board/token variation.
- Causal use is not causal benefit. Relative to the reproducible native
  control, balanced accuracy falls `.4940->.4445`, joint exact falls
  `.041->.009`, errors rise `2024->2222`, and wrong-key swaps rise
  `1546->1749`.
- Overwrite is therefore not the dominant source of binding loss. A protected
  plane can faithfully preserve and repeatedly expose the wrong association.
  FS2 needs ownership-preserving content at commit time, not another retained
  copy or receiver readout.
- Separate contractions can also hide substantial cost despite zero new model
  parameters: elapsed/post-warm wall reaches `1.989/1.977x`. Close read fusion,
  side-plane normalization/gating and all training-setting rescue.

## 2026-08-15: Fixed temporal lag is not instance ownership

- P-GDN3-051 keeps erase and write on one coherent official address and adds
  only eight causal lag scalars. Its exact identity, gradient, causality,
  equivariance and official-kernel contract passes.
- The learned response is inconsistent with a missing one-token phase. Layer 0
  learns three positive lags, while all four layer-1 lags become negative;
  global signed mix is only `.008715` and mean value-position cosine gain is
  `.006799`.
- Balanced accuracy collapses `.494->.202` and errors rise `2024->3192`.
  Wrong-key swaps fall `1546->865` only because useful retrieval fails. Turning
  lag off after training remains at `.1995`, so co-adaptation does not hide a
  beneficial lagged path.
- A topology audit makes the failure concrete: `99.55%` of native wrong-key
  swaps stay within the same future/past direction class, `99.61%` choose the
  adjacent owner by write rank, and `346` events participate in reciprocal
  two-cycles. The model knows the coarse direction but confuses instance
  ownership inside that class.
- Close lag radius/form/convolution/sharing and all training rescue. A successor
  must preserve native retrieval and improve generic occurrence ownership; it
  must not encode another fixed phase prior.

## 2026-08-15: Exact committed-edit collision credit suppresses learning

- P-GDN3-052 keeps native inference bit exact and routes a target-free causal
  collision loss only into the two native K projections and short convolutions.
  Exact committed edits, surprise variation and above-random collision tails
  are all active, so this is not a dead auxiliary objective.
- Balanced/future/past/joint accuracy collapses from
  `.494/.454/.534/.041` to `.013/.011/.015/0`; errors rise `2024->3948`.
  Swaps fall `1546->148` only because useful retrieval disappears.
- The validation curve stays near one percent for all ten epochs. Penalizing
  local normalized key overlap from the start interferes with the base model's
  own phase transition into binding, even though the loss uses exact kernel
  edits and no labels.
- Training-time recomputation is also noncompetitive: elapsed, post-warm,
  warmed-step and allocation ratios are `3.543/3.521/3.405/4.539x`.
- Close coefficient, window, floor, normalization, detach and all nearby loss
  rescue. Future ownership mechanisms must preserve the native CE trajectory
  and change a bounded live representation or transition, not impose another
  global address-geometry objective.

## 2026-08-15: The dominant swap is competing write superposition

- A fixed frozen causal ablation identifies the component and layer without
  training or changing the official recurrence. Removing the wrong owner's
  layer-0 write repairs `1519/1546` swaps; layer1 alone repairs only 48.
- This is not a usable suppression rule. The same intervention preserves only
  `5/1141` of the competing owner's own correct queries. One binding is repaired
  by deleting the other, which is exactly the ownership problem.
- Erase is secondary: both-layer erase-off repairs 402 swaps but retains only
  `.7309` competing-owner queries and just 78 repairs retain both answers. Do
  not reopen decoupled erase keys, erase scale or gate controllers.
- The first layer needs a representation that can retain two nearby bindings
  simultaneously. Train the next foundational state topology from scratch on
  the validated MQAR regime; a zero-init mature-checkpoint graft would test
  migration speed rather than architecture learnability.
- Keep the oracle boundary explicit. The `.87425` balanced selected score uses
  the baseline wrong value to choose which write to delete; it is evidence,
  not a deployable selector, repair or quality claim.

## 2026-08-15: Independent address redundancy can destroy the learning transition

- P-GDN3-053 gives every native head two independently learned full K32xV32
  address banks, sends the same committed edit to both, and averages both
  official reads from the first training step. The strict contract proves
  independent Q/K geometry, strong state/output disagreement and active H8
  FutureSeed; symmetry did break.
- Despite doubled state, balanced accuracy collapses `.494->.00875` and the
  ten-epoch validation curve never leaves chance. The fall in wrong-key swaps
  `1546->144` is broad retrieval failure, not ownership repair.
- The negative result is stronger than “decoupled keys did not help.” A global
  second coordinate system changes every read and removes the native model's
  late optimization transition. Future work must retain an exact native path
  and make ownership an additive, learnable live-state capability whose
  utility can emerge without averaging incompatible coordinate systems.
- Time ratios above `2.03x` also reject this topology as a scalable GDN3 core.
  Close bank count, mean-read weighting, Q/K coupling and initialization
  rescue rather than tuning around the collapse.

## 2026-08-15: Exact dual keys do not preserve a co-adapted binding system

- P-DIAG-DUAL-001 freezes the reproducible L1024 model and gives the four
  known owner writes an almost exact dual basis. Off-diagonal query/write
  pairing is only `3.62e-6`; this is a strong representational ceiling, not an
  optimization miss.
- Balanced accuracy still falls `.49425->.23025`, errors rise `2023->3079`,
  and native-correct retention is only `25.90%`. The intervention repairs just
  `21.15%` of swaps while breaking `1465` formerly correct queries.
- The model's query, erase, write and read directions form one co-adapted
  coordinate system. Making one component mathematically cleaner after
  training invalidates the other components.
- This closes direct `decouple_k`, dual-basis, ridge and bank rescue. A useful
  GDN3 must acquire ownership end to end while retaining the native learning
  path, or improve generic loop convergence; it cannot bolt an orthogonal
  address correction onto a mature state.
