# P-GDN3-051: Causal Lagged-Address Commit

## 1. Metainfo

- Status: registered; not yet launched
- Decision field: validated directional MQAR L1024 wrong-key binding regime
- Parent: frozen deterministic P-REPRO-001 initialization and replay-B score
- Candidate: D128/L2/H4/K32/V32 pinned-official GDN2, native FutureSeed,
  10 epochs, batch32, seed123
- Control: frozen P-REPRO-001 replay B; do not rerun it

## 2. Mechanism Hypothesis

P-REPRO-001 proves that `76.38%` of native endpoint errors are valid values
assigned to the wrong key. P-FS2-012 then shows that keeping the inherited
FutureSeed matrix read-only does not reduce those swaps. The missing operation
is therefore more likely at commit time than during later state preservation.

Directional MQAR exposes a precise temporal binding boundary. Every
association is serialized as `key_t, value_(t+1)`. Native GDN2 commits the
current payload with the current token's K address. Its short convolution can
learn to copy the previous key into the value token, but the recurrence does
not encode that alignment structurally. A value can consequently be correct
while being committed under a nearby token's address.

The falsifiable hypothesis is that a small causal lag in the coherent
erase/write address will align the value token with the key token that owns it.
This is not P-GDN3-031/P036 key decoupling: erase and write continue to use one
identical address, and no independent erase projection exists. It is not a
Raven composer, cache, selector, side state, value-lifetime gate or static Q/K
metric. It changes only which time-indexed instance of the native learned K
sequence owns the official rank-one edit.

## 3. Exact Intervention

For each layer and head, add one zero-initialized scalar `a`; let
`m=tanh(a)`. Compute the exact native K sequence after its unchanged projection
and Triton short convolution, then form a causal one-token shift:

```text
k_prev[0] = k[0]
k_prev[t] = k[t-1], t > 0
k_commit[t] = (1-m) k[t] + m k_prev[t]
```

Call the unchanged pinned official kernel once:

```text
(o, S) = chunk_gdn2(q, k_commit, v, g, b, w, S_in)
```

Kernel-side Q/K normalization remains enabled, so `a=0` is bit-exact native
GDN2. Erase and write share `k_commit`; Q remains the native receiver query.
The fixed D128/L2/H4 model adds exactly eight scalars, no recurrent state,
scan, token, second core or task rule. Native terminal FutureSeed is unchanged.

## 4. Distinction And Expected Signal

Direct decoupled-key P031 failed because erase and write/read ownership split
into almost orthogonal learned maps. Anchored P036 failed even with a cosine
floor. P051 never gives erase a private map: one coherent commit key performs
both erase and write, and its alternative is the same native key sequence at a
causally adjacent time step. P039 recurrently rewrote the whole Q/K input and
collapsed retrieval; P051 adds no recurrent controller or hidden residual.

If temporal address/payload phase is causal, the learned mix should move
positive, value-position commit keys should become more aligned with the
preceding key-position address, total errors and wrong-key swaps should fall,
and turning the mix off in the same trained model should remove the gain. If
the mix activates but quality does not improve, this closes explicit one-token
address lag; no lag radius, convolution, mix cap, per-channel map or training
rescue follows.

## 5. Fixed Data And Provenance Contract

- train/test examples: `10,000/1,000`;
- sequence length/pairs: `1,024/4`, mixed future and past directions;
- epochs/batch/seed: `10/32/123`;
- matched initialization SHA256:
  `7402e46c48cbd47070d65262d1a1b62a32ac55a4d644a6fb9644b96b0914850f`;
- frozen control score SHA256:
  `df8ae1c212590666440f12029d60c74a6d09cbc059b23e58bcf864c00ae98854`.

Formal source must be an exact pushed/read-back SHA in a clean detached
worktree. Only CUDA index0 on the registered AIStation A100 may be visible.
Require pinned FLA SHA `9c8e42e762fce087c27b673af4922795d9edb85e`,
clean Zoology SHA `1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb`, Triton
short convolution, two official GDN2 layers, exactly two
`ChunkGDN2FunctionBackward` nodes and no fallback.

## 6. Strict CUDA Contract

Before science, prove all of:

1. candidate parent tensors exactly match the frozen initialization;
2. exact parameter delta is eight, state/token/scan deltas are zero;
3. zero logits give bit-exact full output, both terminal states and native
   FutureSeed, including finite nonzero incoming states;
4. all eight lag logits receive finite nonzero gradient at zero;
5. an opened mix changes K, output and terminal state while erase/write retain
   exactly one common address;
6. the shift is strictly causal, leaves token0 unchanged and has no wraparound;
7. simultaneous head permutation of Q/K/g/b/w/mix commutes within fixed BF16
   tolerance;
8. shifting the K history changes the result, every tensor remains finite and
   there is no alternate CPU/eager model path.

Any contract miss writes an integrity abort and closes the implementation.

## 7. Activation, Quality And Cost Gates

Endpoint activation is conjunctive:

- both layers and all eight logits are finite;
- at least four heads have positive `m >= .02`, global mean `m >= .01`, and
  committed/native K relative RMS is at least `.01`;
- at true value positions, commit-to-preceding-key cosine exceeds
  native-current-to-preceding-key cosine by at least `.02`;
- terminal states are finite with nonzero board variation and native
  FutureSeed has exactly one active route.

All quality checks are conjunctive against frozen replay B:

- balanced accuracy `>=.60` and gain `>=+.10`;
- future accuracy `>=.60` and gain `>=+.14`;
- past accuracy does not regress by more than `.03`;
- joint exact `>=.10`;
- total query errors fall by at least `15%`;
- wrong-key valid-value swap count falls by at least `20%`;
- the same trained model with all lag logits disabled loses at least `.05`
  balanced accuracy and has more total errors.

Candidate/control elapsed, post-warm wall and independently warmed-step ratios
must each be `<1.20x`; peak allocation must be `<1.10x`. These thresholds are
fixed before launch.

## 8. Kill And Next Decision

Any integrity, activation, quality or cost miss closes causal lagged commit.
Do not rescue lag radius, interpolation form, normalization, convolution,
per-head/per-channel sharing, seed, LR, loss, batch, width, depth, data or
duration. A complete pass authorizes exactly one hard-Sudoku transfer with the
same mechanism and no setting changes. A miss redirects the project away from
temporal commit alignment toward a genuinely different live memory
organization.

## 9. Submission Record

Not applicable.
