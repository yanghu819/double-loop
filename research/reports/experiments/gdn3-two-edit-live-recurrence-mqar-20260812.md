# P-GDN3-021: Two-Edit Live GDN2 Recurrence

## 1. Metainfo

- Status: discarded after one direct A100 L1024 run
- Decision field: directional MQAR L1024 from scratch
- Fixed setting: D128/L2/H4/K32, native FutureSeed, 10 epochs, batch32, seed123
- Added parameters: exactly 131,072; added recurrent state: zero

## 2. Evidence And Hypothesis

On the current A100 runtime, the exact historical GDN2 source reaches only
`0.1735` balanced accuracy. P-GDN3-020's bounded Log-SPD address metric raises
this to `0.48225`, but joint exact remains `0.044`. Address geometry matters,
yet one rank-one erase/write edit per token still does not close binding.

The next falsifiable hypothesis is that a single edit forces independent
associations into one local update direction. Two independent edits on the
same bounded K32xV32 state can separate address/payload evidence without adding
slots, a second state, a selector, or a task rule.

## 3. Mechanism

For each logical token, one pinned official `chunk_gdn2` call receives two
adjacent microsteps. The first is the unchanged parent decay/erase/write. The
second has independently learned K, V, erase and write projections, zero extra
decay, and reuses Q only to expose the post-edit state. The model retains the
second output. Both edits are trained from scratch; this is not a zero-init
checkpoint graft. Fixed local initialization streams keep every parent tensor
identical to the frozen control while adding no sweep variable.

## 4. Novel Boundary

P-GDN3-010 only asked whether a zero-initialized doubled-token wrapper could be
bit-identical during Sudoku checkpoint migration; it failed before training.
P-GDN3-021 instead tests the foundational recurrence from scratch on the
directional binding-error regime. It is not Log-SPD capacity rescue, a Raven
router, a V-lifetime wrapper, a parallel expert, or a FutureSeed codec.

## 5. Prediction And Gate

The single run passes only if balanced accuracy is at least `0.70`, joint exact
at least `0.25`, both directions at least `0.68`, balanced gain over P020 is at
least `0.10`, and wrong-key/valid-value fraction among errors falls by at least
`0.10`. Both layers must use two edits with independent addresses and finite
nonzero auxiliary payload. Fit and warmed-step time must be at most `2.5x` the
same-runtime parent and peak allocation at most `1.5x`. `0.85` balanced and
`0.60` joint is the strong-result threshold. Any miss closes this exact
mechanism without projection, order, gate, seed, LR, loss, width, depth,
duration, or initialization rescue.

## 6. Decision

Discard. Exact pushed source `18ad10f` completed all 10 epochs on one A100.
Balanced accuracy was `0.00975`, future/past accuracy `0.0110/0.0085`, and
joint exact `0`. The validation curve remained in `[0.0085,0.0115]` for every
epoch, so this is not a late-convergence miss.

Both auxiliary paths were strongly active. Address separation was
`0.7282/0.8390`, while auxiliary payload RMS reached `1.7118x/3.6221x` the
parent payload and erase/write means remained near `0.5`. The mechanism
therefore produced independent addresses but continuously overwrote useful
state at filler tokens. Wrong-key swaps among errors fell only because almost
all predictions became invalid values; it is not a binding improvement.

Fit time was `124.61s`, warmed-step time `0.57785s`, and peak allocation
`1.675GB`. The candidate score/cases/checkpoint/decision SHA256 values are
`49c26f...9f68`, `cec4c0...ec22`, `d71525...e41eb`, and `4de1de...3037`.
Close without write scale, gate, order, initialization, seed, LR, loss, width,
depth or duration rescue. The next test isolates writes into same-byte banks
instead of applying two destructive edits to one state.
