# Milestone: Address/Payload-Factorized GDN2

Status: **important algorithm candidate; retained for clean scaling**

- Recorded: 2026-07-31 17:31 CST
- Canonical tag: `milestone/gdn2-address-payload-v1-20260731`
- Evidence parent: `03a5010c401ff3b99f35f9e8a449c65f3361d487`
- Branch: `codex/gdn2-address-binding-20260730`
- Runtime: AIStation GPU1, one NVIDIA A800 80GB
- Upstream carrier: official FLA GDN2 SHA `9c8e42e7`

## 1. What Changed

Ordinary GDN2 derives every recurrent operation from one content-entangled
hidden stream `x_t`:

```python
q_t = Q(x_t)
k_t = K(x_t)
v_t = V(x_t)
decay_t = Decay(x_t)
erase_t = Erase(x_t)
write_t = Write(x_t)
output_gate_t = OutputGate(x_t)
```

The recurrent matrix therefore has to learn two different jobs through the
same representation:

1. **Address:** which memory location should this token read or modify?
2. **Payload:** what information should it write, erase, preserve, or emit?

The new carrier separates those jobs:

```python
address_i = position_embedding(i)

q_i = Q(address_i)
k_i = K(address_i)

v_i = V(x_i)
decay_i = Decay(x_i)
erase_i = Erase(x_i)
write_i = Write(x_i)
output_gate_i = OutputGate(x_i)
```

In plain language: position says **where**; content says **what** and **how**.

For a randomized cell traversal `pi`, Q/K are first computed in canonical cell
order and then gathered as `q[pi(t)]`, `k[pi(t)]`. V and every edit/output gate
follow the actual hidden token stream. The output is restored to canonical cell
order before evaluation.

## 2. What Did Not Change

This is not a replacement recurrent kernel and it does not add a Sudoku
solver. The following remain unchanged between control and candidate:

- official FLA GDN2 chunk/Triton recurrence;
- recurrent state shape and rank-1 update structure;
- parameter count: `5,461,688` in both matched arms;
- D192/L10/H6/D32 model and five-loop contract;
- native FutureSeed mechanism and all-loop CE supervision;
- official Sudoku boards, randomized cell order, curriculum, optimizer, seed,
  batch, learning rate, and total optimizer steps;
- no search, repair, selector, oracle rollout, task rule, CPU model smoke,
  GPU2, or kernel fallback.

The only model switch is:

```text
control:      GDN2_ADDRESS_MODE=none
candidate:    GDN2_ADDRESS_MODE=position_qk
```

## 3. Why This Modification Exists

The frozen traversal probe supplied the causal clue. A normal GDN2 checkpoint
performed well in row-major order but collapsed under reverse, column, box, and
fixed-random traversals even though every token retained its canonical absolute
position embedding and outputs were restored before scoring.

That result ruled out the simple explanation that position information was
missing. The stronger explanation was that additive position metadata remained
mixed with changing token content, so the recurrent Q/K geometry never became a
stable address system. Address/payload factorization directly tests that
explanation without changing the memory update rule or adding capacity.

## 4. Matched-Compute Evidence

Both arms finish at optimizer step `9300`, use identical parameter counts and
official case-bank hashes, and execute the strict official FLA path.

| Official blank range | Normal GDN2 | Position-Q/K | Delta |
|---|---:|---:|---:|
| 51-55 | 0.3035 | 0.5660 | +0.2625 |
| 56-60 | 0.2781 | 0.5057 | +0.2275 |
| 61-64 | 0.2444 | 0.5594 | +0.3150 |
| Mean | 0.2753 | 0.5437 | **+0.2683** |

Additional matched diagnostics:

- train CE: `1.5707 -> 0.9661`;
- 200-step continuation wall time: `1311.3s -> 1272.8s`;
- peak CUDA allocation: `8094.9 -> 8239.4 MiB`;
- matched 64-blank case, wrong cells across loops 1-5:
  normal GDN2 `53 -> 50 -> 50 -> 51 -> 51`;
  position-Q/K `32 -> 21 -> 18 -> 15 -> 15`.

The gain is therefore not explained by extra parameters, extra optimizer
steps, a different recurrent kernel, or a larger prediction mask. The candidate
both starts from a better representation and gives later loops useful errors to
remove.

## 5. Scaling Boundary

The exact step9300 candidate was continued for 300 unchanged hard-stage steps:

| Official blank range | Step 9300 blank | Step 9600 blank | Step 9600 exact |
|---|---:|---:|---:|
| 51-55 | 0.5660 | 0.5845 | 0.0059 |
| 56-60 | 0.5057 | 0.5171 | 0.0059 |
| 61-64 | 0.5594 | 0.6386 | 0.0000 |
| Mean | 0.5437 | 0.5800 | 0.0039 |

This is a real but weak opening. Official loop1 exact remains zero, so the few
complete solutions are created by later loops. However, the run misses every
predeclared continuation gate: exact `<0.02`, mean blank `<0.60`, and mean
blank improvement `<+0.04`.

Short same-checkpoint continuation stacking is therefore stopped. The next
valid expensive experiment must scale generic model/state capacity or unseen
data coverage from a matched initialization.

## 6. Allowed And Forbidden Claims

Supported now:

- Stable recurrent addressing is a major optimization bottleneck for hard,
  randomly ordered Sudoku in this GDN2+FutureSeed backbone.
- Separating address from payload is parameter-neutral and substantially
  improves CE, hard-cell accuracy, and conditional later-loop correction.
- The gain survives matched total optimizer compute.

Not supported yet:

- Sudoku is solved beyond 50 blanks.
- This carrier universally outperforms GDN2 on language, retrieval, or Maze.
- The variant has better asymptotic exact accuracy rather than only much better
  finite-budget optimization.
- The current result alone is sufficient to rename the carrier `GDN3`.

## 7. Next Decisive Gate

Run exactly one clean from-scratch comparison:

```text
normal GDN2 vs position-Q/K GDN2
same shared initialization
same post-initialization RNG and data order
same independent-board coverage
same model, state, loops, optimizer, and compute budget
GPU1 only; strict official FLA; no fallback
```

The candidate becomes the primary GDN3 algorithm only if it preserves a large
hard-blank/CE advantage and opens full-board exact earlier or reaches a matched
exact target with materially less compute. If the gap disappears, retain the
current result as a checkpoint-local optimization insight rather than a new
general architecture.

## 8. Canonical Evidence

- Matched experiment:
  `research/reports/experiments/gdn2-address-matched-total-compute-20260731.md`
- Scaling boundary:
  `research/reports/experiments/gdn2-address-hard-stage-scale-20260731.md`
- Algorithm implementation:
  `experiments/rwkv_fs_sudoku/study_rwkv_futureseed_loop.py`
- Strict launcher:
  `scripts/run_gdn2_address_arm.sh`
- Matched visualization:
  `runs/gdn2-address-matched-s9300-20260731/index.html`
- Scaling visualization:
  `runs/gdn2-address-scale-s9300-s9600-20260731/index.html`
