# Directional MQAR Length Scaling

## 1. Metainfo

- Plan: `P-CAUSAL-010`
- Status: in progress; exact-SHA CUDA preflight starting
- Date: 2026-08-04 CST
- Machine: AIStation task-mode GPU1 only
- Branch: `codex/p-causal-010-mqar-length-scaling`

## 2. Hypothesis

P-CAUSAL-007 established that native FutureSeed lets a causal GDN2 retrieve a
random value written after its query at sequence length 64. The remaining
paper question is whether this route scales with context or merely exploits a
short fixed layout.

With the number of associations held at four, increasing sequence length adds
only irrelevant filler and increases query-to-write distance. A scalable
FutureSeed route should keep high future-query accuracy, while strict causal
GDN2 must remain at chance for information-theoretic reasons. Full
bidirectional attention should also solve the task, but its sequence mixing
cost grows quadratically rather than linearly.

## 3. Configuration

The first gate uses only sequence lengths 64 and 1024. Intermediate lengths
128/256/512 are conditional on a valid endpoint, so this is not an automatic
five-point table.

- data: deterministic balanced directional MQAR, vocabulary 256;
- four unique associations per sequence: two past and two future;
- 10,000 train and 1,000 fixed validation examples per length and arm;
- future queries, past writes, past queries and future writes occupy four
  normalized quarters of the sequence;
- filler, key and value vocabularies are disjoint;
- model shell: upstream Zoology D128/L2, learned positions, tied embedding,
  standard MLP, batch32;
- optimization: exactly 10 epochs, AdamW LR1e-3/WD0.1/cosine, seed123;
- recurrent arms: pinned official FLA GDN2 H4/D32, expand-v1, short-conv4,
  strict chunk/Triton path and BF16 layer autocast;
- no-FutureSeed and FutureSeed arms contain the same parameters and identical
  initialization; only the fixed scale is 0 or 1;
- attention ceiling: full noncausal SDPA, four heads of dimension57. Its inner
  attention width228 keeps total parameters within0.5% of GDN2 while retaining
  the same model width, depth, positions and MLP.

FutureSeed remains the P-CAUSAL-007 mechanism: layer 0 scans left-to-right,
its per-example/head RMS-normalized terminal recurrent state is gated and used
as layer 1's initial state. There is no reverse scan, bidirectional
concatenation, extra layer, outer loop, oracle, selector, search or repair.

## 4. Environment

- SSH alias: `aistation-task-gpu1`
- remote root: `/huyang2/double-loop`
- Python: `/opt/conda/bin/python`
- GPU: exactly one visible `NVIDIA A100-SXM4-80GB`, UUID
  `GPU-53e9f3b4-2966-65d3-6614-09c540921519`
- Zoology: `1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb`
- official FLA: `9c8e42e762fce087c27b673af4922795d9edb85e`

## 5. Commands

```bash
CUDA_VISIBLE_DEVICES=0 RUN_NAME=<registered-name> ./run.sh mqar_length_scaling
```

The launcher runs a full L1024 CUDA forward/backward preflight for all three
arms, proves scale-0 identity and directional dependencies, checks official
source/kernel provenance, verifies parameter fairness, and reproduces the
exact P-CAUSAL-007 L64 train/test hashes before formal training.

## 6. Artifacts

Planned run contents:

- `config.json`, `launch.env`, source SHA and lean source snapshot;
- `preflight.json` and `preflight.log`;
- per-length/per-arm configs, metrics curves, scores and same-case events;
- root `comparison.json` and `score.json`;
- `visualizations/index.html`, summary and hardest-case JSON;
- exact exit status, GPU before/after, and `abort.json` on failure.

Checkpoints are not needed for this gate and will not enter Git.

## 7. Registered Readouts

For every length and arm report:

- past and future query accuracy, exact and CE;
- joint sample exact;
- parameter count and fixed train-token count;
- independently warmed training tokens/s and peak allocated VRAM;
- full validation curve and same-sequence prediction events.

The L64 carrier gate requires no-FutureSeed past accuracy at least0.90 and
future accuracy at most0.10, plus both directions at least0.90 for FutureSeed
and attention. L1024 strong support requires FutureSeed future accuracy at
least0.80, delta over causal at least+0.70, past accuracy at least0.90, and at
least80% closure of the causal-to-attention future-accuracy gap. FutureSeed
future accuracy at least0.50 is partial support only.

## 8. Budget, Kill Criteria, And Claim

Budget is one full-size CUDA preflight plus six endpoint arms, with a strict
two-hour wall limit. Stop immediately on wrong GPU, SHA, data hash, official
class/source, chunk/Triton kernel, parameter/init mismatch, scale-0 mismatch,
causal future leakage, dead FutureSeed/attention dependency, non-finite
forward/backward, OOM or L64 carrier failure.

Do not rescue with another seed, learning rate, width, depth, head dimension,
epoch count, loss or easier vocabulary. Do not mix the association-count axis
into this run. If L1024 passes, then and only then fill L128/256/512 with this
frozen protocol and run a separate fixed-length memory-load experiment.

A strong pass supports the paper claim that terminal-state FutureSeed gives a
causal recurrent stack a length-scalable future-information route while
retaining linear recurrent sequence mixing. It does not yet prove language
quality or end-to-end wall-time superiority.

## 9. Result And Submission

Pending. No tag is authorized before a valid endpoint and archived systems
comparison.
