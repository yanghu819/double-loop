# RWKV FutureSeed Causal-Backbone Cliff - 2026-06-04

Source SHA: `c9c212a9e99aa41bd1ada01c3eb9e3d7a9449324`  
GPU: AIStation GPU1 only, `CUDA_VISIBLE_DEVICES=0`; no CPU smoke; RWKV CUDA kernel.  
Noise: `NOISE_SCALE=0`, `ROLLOUT_NOISE_SCALE=0`; feature-diff is only the mode label in the script, not active noise.

## Question

Does FutureSeed as part of the RWKV/stateful backbone change optimization and loop refinement on 9x9 hard random-hole Sudoku, without EqR noncausal mixers?

## Runs

- nofs: train_ce=0.6316, train_sec=829.7, loop1/h16=0.0078, loop3/h16=0.0039, loop5/h16=0.0039, h24 loop5=0.0000, h28 loop5=0.0000, h32 loop5=0.0000
- fs: train_ce=0.0057, train_sec=840.9, loop1/h16=0.9102, loop3/h16=0.9961, loop5/h16=0.9961, h24 loop5=0.8828, h28 loop5=0.6855, h32 loop5=0.4043

## Loop Exact Deltas: FS - noFS

- h16: loop1 +0.9023, loop3 +0.9922, loop5 +0.9922
- h20: loop1 +0.7090, loop3 +0.9746, loop5 +0.9805
- h24: loop1 +0.4238, loop3 +0.8789, loop5 +0.8828
- h28: loop1 +0.1309, loop3 +0.6660, loop5 +0.6855
- h32: loop1 +0.0137, loop3 +0.3809, loop5 +0.4043

## Rollout Readout

- nofs: K1 oracle=0.0039, confidence=0.0039; K4 oracle=0.0039, confidence=0.0039; K8 oracle=0.0039, confidence=0.0039
- fs: K1 oracle=0.9961, confidence=0.9961; K4 oracle=0.9961, confidence=0.9961; K8 oracle=0.9961, confidence=0.9961

## Decision

- Strong positive signal for the actual mainline: FutureSeed as RWKV/stateful backbone, not EqR mixer adapter.
- Optimization cliff: launcher logs show step600 CE `0.0057` for FS versus `0.6316` for no-FS.
- Exact cliff: FS h16/h24/h28/h32 loop5 = `0.9961/0.8828/0.6855/0.4043`; no-FS = `0.0039/0/0/0`.
- Loop compression is also real: FS loop1 h16 `0.9102` already beats no-FS loop5 h16 `0.0039`; FS loop3 h16 reaches `0.9961`.
- Rollout selector work is not the next bottleneck in this deterministic no-noise readout: K1/K4/K8 oracle and selectors are identical.
- First launch aborted before training because structured linked eval patterns currently support at most 9 holes on 9x9; hard h16+ eval was relaunched without structured patterns.

## Next Step

- Rerun the same A/B with one more seed or a smaller batch if lease is tight. If effect repeats, stop debating EqR adapters and invest in RWKV7/CUDA FutureSeed backbone and structured eval at holes<=9 as a separate probe.

## Files

- `runs.csv`: per-run clean and rollout metrics.
- `summary.json`: machine-readable aggregate plus abort metadata.
