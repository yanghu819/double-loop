# P-GDN3-025: Committed-Delta Correction Memory

## 1. Metainfo

- Status: pre-registered, implementation complete, awaiting strict GPU contract
- Benchmark: directional MQAR L1024 wrong-key regime
- Fixed setting: D128/L2/H4/K32/V32, native FutureSeed, 10 epochs,
  batch32, seed123
- Arms: fixed shared K16 correction basis; learned semi-orthogonal K16
  correction basis
- Resource: one task-mode GPU, no concurrent model process

## 2. Evidence And Hypothesis

P020's bounded Log-SPD address metric raises current-runtime balanced accuracy
from `0.1735` to `0.48225`, but `94.16%` of its remaining errors are
correct-value/wrong-key swaps. P021 shows that a second write into the parent
state overwrites useful payload. P022/P023 show that ordinary same-byte bank
splitting does not close binding. P024's product hash removes wrong-key swaps
by also destroying the useful linear address channel. Earlier companion-state
work did not give the companion address path a demonstrated production
gradient.

The untested hypothesis is therefore a correction state whose payload is not
another learned V stream. It receives the exact `v_new` residual already
committed by pinned official GDN2 after accounting for the decayed live state.
If wrong-key swaps come from interference among committed edits, a smaller
receiver-native address state should preserve that evidence without replacing
or rewriting the parent state. A learned stable K32-to-K16 basis should beat a
fixed same-state-byte compression if address organization, rather than raw
extra capacity, is the causal lever.

## 3. Mechanism

The parent H4/K32/V32 official GDN2 scan and native FutureSeed path remain
unchanged. During each parent scan, capture the exact official `v_new` tensor
at the recurrence boundary. Detach this tensor from the parent backward so the
correction loss cannot distort the parent transition. A second pinned official
GDN2 scan writes it into an independent H4/K16/V32 correction state. Its
readout is added through a zero-initialized bounded `tanh` gate, preserving
bit-exact parent output and nonzero-incoming-state behavior at initialization.

The fixed arm uses pairwise orthonormal K32-to-K16 rows. The learned arm starts
from exactly those rows and applies a per-head Cayley rotation, so its rows
remain semi-orthogonal. Both arms add exactly 2,048 state values/layer and use
two official scans/layer. The fixed arm adds 8 scalar read parameters; the
learned arm adds 3,976 parameters. No selector, search, Sudoku rule, extra V
producer, product hash, cross-loop cache, custom recurrence kernel or fallback
is present.

## 4. Falsifiable Prediction And Gate

The endpoint passes only if all hold:

- both parent hashes and both dataset hashes match the exact P020 provenance;
- all four official scan backwards are present, zero-read parent output and
  nonzero-incoming main terminal state are bit exact, and read/basis gradients
  are finite and nonzero in the registered two-stage contract;
- both arms activate in both layers; committed edit, correction read, board and
  token variation are finite/nonzero; correction/main state RMS is in
  `[1e-4, 10]`; learned basis movement is at least `1e-4` and row orthogonality
  error at most `1e-4`;
- learned balanced accuracy is at least `0.70`, joint exact at least `0.25`,
  both directions at least `0.68`, and balanced gain over P020 at least `0.10`;
- learned beats the fixed same-state-byte arm by at least `0.05` balanced or
  joint exact, and lowers wrong-key swap fraction by at least `0.10` from P020;
- both arm fit ratios versus P020 are at most `1.65`; learned warmed-step and
  allocation ratios are at most `1.60`; learned/fixed system delta is at most
  `1.10`.

Any miss discards committed-delta correction memory. There is no K, basis
rank, gate, erase/write, detach policy, sharing, seed, LR, loss, batch, width,
depth, epoch or duration rescue. A pass authorizes one fused Sudoku-scale
transfer; a failure closes this correction-memory organization.

## 5. Configuration

- train/test examples: `10,000/1,000`
- sequence length / KV pairs: `1024/4`
- train tokens: `102.4M` per arm
- P020 score/cases SHA256:
  `a9c750e84fc918910c008b679016c3f24a5e821e3c20b84b1f0b07c1c64b9b3e` /
  `127fb40e32a14a54f6f90c784dfe728c85dcf200535e0fb300e3e28a55c8751b`
- pinned FLA source SHA:
  `9c8e42e762fce087c27b673af4922795d9edb85e`

## 6. Artifacts

- R1 source SHA: `fc42fa1e3710d01b5762efefc3c79e928e100603`
- R1 contract log SHA256:
  `38e8c9708dcf43632e96fb94708a637a57ece28d452a8364d76f11c565578d92`
- R1 `abort.json` SHA256:
  `b05bbf0ed25963b259192535d7bc403312186156bf9821d97b55c124f515d8f1`

R1 stopped before training on a non-science contract error. The checker
compared a separately constructed plain model with the wrapped model. Zoology
recursively reinitializes nested `out_proj` parameters, so changing the module
tree changes those parent parameters despite resetting the seed. R2 instead
checks each wrapper directly against its own pinned-official base with the same
weights, for both zero and nonzero incoming state, and separately checks exact
fixed/learned full-model parity while the read gates are zero. The mechanism,
data, optimizer, budget and science gates are unchanged.

- R2 source SHA: `6faa6ff1f7bf196bba2f11a5ec1ca2be8e3d2e49`
- R2 contract log SHA256:
  `8e88dda18dc934ad8346c1230af8ef43210fffe52aaffa31f9f2db50fc83c1a9`
- R2 `abort.json` SHA256:
  `fa0aa807cc2f7abeb3ea50109a1e7e26e14bbc32cfa81215b7f6b1ffd2fb6a97`

R2 also stopped before training. Parent parameter hashes were exact, but the
fixed and learned full models were compared sequentially in training mode, so
Zoology dropout consumed different random masks. R3 performs identity checks
in evaluation mode, then explicitly restores training mode for the official
backward and two-stage gradient checks. No model or gate changed.

- R3 source SHA: `e48f7ef123628e8e7e2686621914f4bad3ff7799`
- R3 contract JSON/log SHA256:
  `b5720618781da106168d84d4a9b6dffdd32dca20b750ed8cd19feede5f741042` /
  `2922fc62819aeb309d2e17bc5ddfbcb94f796448967ae06f263618ecd705824a`
- R3 formal pre-training log/abort SHA256:
  `cbe3b14e6cc0a8a89848fb10a174581f837186cc678cef895a4d5d7613c5452a` /
  `516b870696a39870659b28ce1493a543026041f589d7a7594fec54ab6a50899c`

R3 passed the full CUDA contract, including same-weight zero/nonzero-state
identity, four official backward functions and both gradient stages. The
endpoint then exited before constructing either training arm because the
hash-verified P020 balanced value is `0.48224999999999996`, while a redundant
guard compared it exactly with the decimal literal `0.48225`. R4 changes only
that already hash-protected check to absolute tolerance `1e-12`.

## 7. Results

Pending.

## 8. Decision

Pending the sole registered endpoint.

## 9. Submission

Not applicable.
