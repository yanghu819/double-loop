# P-DIAG-DUAL-001: Frozen Dual-Address Ownership Ceiling

## 1. Metainfo

- Status: complete; direct dual-address route closed
- Date: 2026-08-15
- Benchmark: frozen P-REPRO-001 directional MQAR L1024 replay B
- Model: trained native D128/L2/H4/K32/V32 official GDN2 plus FutureSeed
- Resource: sole AIStation A800-SXM4-80GB, CUDA index0, exact UUID

## 2. Evidence And Question

P-DIAG-EDIT-001 finds that suppressing the competing layer-0 write repairs
`1519/1546` wrong-key swaps, but retains only `5/1141` queries owned by that
write. The state contains useful payload for both bindings, while one native
address trajectory cannot keep their ownership separated. P-GDN3-053 then
shows that globally duplicating address banks and averaging every read destroys
the native learning transition.

The sole question here is an upper bound: if the four owner-query directions
are made exactly biorthogonal to four write directions in the already-trained
model, can the model repair swaps while preserving its correct owners? A strong
yes opens one scalable causal dual-state transition. A no closes direct address
orthogonalization and redirects work to loop/training dynamics.

## 3. Diagnostic Intervention

The native frozen arm runs unchanged. A tied-DPLR control rewrites only layer
0's algebra into one official DPLR scan with identical normalized erase/write
keys; it must preserve predictions. The oracle diagnostic keeps native query
and erase keys, but at the four known value-write positions replaces only the
write direction with the normalized dual basis of the four owner-query
directions. Thus `q_i^T a_j=0` for `i!=j`, up to floating-point error.

The diagnostic uses frozen case owner positions and the whole sequence, so it
is explicitly acausal and non-deployable. It has zero training steps, zero new
parameters and zero persistent state. It is not a selector, solver, repair or
quality claim; it is a representational causal ceiling.

## 4. Boundary

This is not P031/P036 decoupled-key training: those mechanisms ask SGD to
learn erase/write divergence and collapse the read/write/erase closure. Here
erase remains native and only an oracle write basis is tested in a frozen
model. It is not P027 OIG: no KxK online state or trainable recurrence is
launched. It is not P053: the exact native read path remains and no second bank
is averaged globally.

## 5. Fixed Contract

The launcher requires exact GPU index/UUID, pinned FLA and Zoology SHAs, clean
detached pushed source, frozen checkpoint/case hashes, no other compute process
and no backend fallback. It verifies the trained parameter hash and fresh
native replay. The tied-DPLR control must have at least `.98` prediction
agreement with native and balanced drift at most `.02`. Owner-query Gram
condition is capped at `1e6`; dual pairing off-diagonal error must be at most
`2e-4`, with minimum absolute diagonal at least `1e-3`.

## 6. Decision Gate

Open exactly one causal bounded dual-state successor only if all hold:

- balanced accuracy gain over frozen native is at least `.15`;
- total errors fall at least `20%`;
- at least `50%` of native wrong-key swaps are repaired; and
- at least `80%` of native-correct queries remain correct.

Any integrity or tied-control miss invalidates the diagnostic and writes
`abort.json`. A completed ceiling miss closes direct dual-address
orthogonalization without a ridge, owner-count, layer, scale, key, seed,
training or kernel rescue. No Sudoku run is authorized by this diagnostic.

## 7. Artifacts

Archive diagnostic JSON/log, exact source/config, GPU samples, checkpoint/case
hashes, GitHub readback and an artifact manifest under the formal run directory.

## 8. Decision

Close direct dual-address orthogonalization. Do not launch a causal dual-key,
dual-bank, ridge, owner-count, layer, scale, key or training rescue from this
diagnostic.

R1 source `2bd7e66` completed native and tied-DPLR execution, then exited
before the oracle intervention because the surrounding production autocast
downcast a 4x4 `eigvalsh` input to BF16. CUDA does not implement that operation
for BF16. The launcher wrote a non-science `abort.json`; no mechanism or gate
assertion failed. R2 changes only the registered Gram eigensolve/solve region
to explicit FP32 with autocast disabled, matching the preregistered numerical
definition.

R2 source `25743fc` completes with status0 on the sole A800 GPU. The native arm
reproduces balanced/future/past/joint `.49425/.454/.5345/.041`, `2023` errors
and `1546` wrong-key swaps. Rewriting layer0 as a tied DPLR control preserves
`.99025` of predictions and balanced `.4945`, so the diagnostic algebra itself
is faithful.

The oracle dual geometry is numerically strong: maximum owner-query Gram
condition `4409.71`, minimum eigenvalue `.0009057`, maximum off-diagonal dual
pairing error `3.62e-6` and minimum diagonal pairing `.036923`. Nevertheless,
balanced/future/past/joint falls to `.23025/.146/.3145/0`; errors rise
`2023->3079`. It repairs only `327/1546` native swaps (`21.15%`) and retains
only `512/1977` native-correct queries (`25.90%`). The registered quality
ceiling therefore fails every threshold despite passing geometry.

The causal conclusion is stronger than "keys are not orthogonal enough."
Native query, erase, write and read geometry are co-adapted. Replacing only the
write coordinate system, even with an exact owner oracle, destroys more correct
bindings than it repairs. Future work must preserve the native trajectory and
learn ownership as part of the end-to-end transition or loop dynamics; it must
not retry post-hoc key separation.

Formal run:
`/huyang2/double-loop/runs/p-diag-dual-001-owner-ceiling-r2-20260815T1326Z-25743fc`.
Diagnostic JSON/log SHA256 are
`dbab5254a96a0a00a94b121ba73218e3494fbd1a8e89c34458f84b0b1d05de4d` /
`7a94062f59da04a9d0852c87e00e09bcff819814485fb15fd8944bdbcc546c78`;
source snapshot SHA256 is
`fdd4f1d125eab02f582fef967fa112298fde553f469f02684e506cb2b1767af3`.
Active GPU samples average `59.21%` SM, peak at `90%`, and average `1188.6`
MiB observed memory; PyTorch peak allocated memory is `599,437,824` bytes.
