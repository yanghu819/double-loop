# Official Zoology Bidirectional MHA Carrier Gate

## 1. Metainfo

- Plan: `P-CAUSAL-011`
- Status: discarded by the registered directional carrier gate
- Date: 2026-08-04 CST
- Machine: AIStation task-mode GPU1 only
- Branch: `codex/p-causal-011-official-bidir-mqar`
- Source SHA: `40002497ed5e56f1879b996d09608065c963b27d`
- Run: `zoology-official-bidir-mqar-20260804T093000Z-4000249`
- Exact launcher PID/PGID: `46796/46796`

## 2. Hypothesis

P-CAUSAL-010 proved that its custom noncausal attention could see later values,
but it did not learn the key-value association in ten epochs. P-CAUSAL-005
already established that the exact upstream Zoology MHA remains near chance
until epoch22 and then opens sharply around epochs23-25. If P-CAUSAL-010 failed
because its custom carrier and budget did not preserve the validated upstream
optimization path, then exact upstream MHA with only the triangular causal mask
removed should solve both directional query types within the pre-existing
30-epoch opening budget.

This is a carrier calibration, not a FutureSeed experiment. It is necessary
because an attention model that never opens cannot serve as a paper ceiling.

## 3. Configuration

- Data: exact P-CAUSAL-007 mixed-direction MQAR, vocab256, sequence64, four
  associations, 10,000 train and 1,000 fixed validation examples.
- Model: upstream Zoology D128, two layers, one-head MHA, learned position
  embeddings, upstream MLP, residual/norm stack and tied embedding/readout.
- Training: upstream Trainer, batch32, AdamW LR `1e-3`, weight decay `0.1`,
  cosine schedule, seed123, maximum30 epochs, upstream valid-accuracy early
  stop above `0.99`.
- Semantic change: `OfficialBidirectionalMHA` directly inherits upstream
  `MHA`. Its `Wqkv`, `out_proj`, state dict, initialization and attention
  equation are unchanged; the upper-triangular causal mask is omitted.
- Not present: GDN2, FutureSeed, reverse scan, extra layer/step, selector,
  search, repair, task rule, second seed, or tuning table.

## 4. Environment

- Remote alias: `aistation-task-gpu1`.
- Work root: `/huyang2/double-loop`.
- GPU contract: exactly one visible A100-SXM4-80GB, UUID
  `GPU-53e9f3b4-2966-65d3-6614-09c540921519`, exposed as CUDA index0.
- Python: `/opt/conda/bin/python`; no CPU model smoke.
- Zoology: exact clean commit
  `1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb` with pinned hashes for
  `attention.py`, `model.py` and `train.py`.

## 5. Commands

```bash
CUDA_VISIBLE_DEVICES=0 PERSIST_ROOT=/huyang2/double-loop \
  ./run.sh official_bidir_mqar
```

The launcher runs a full-size CUDA preflight before the only training arm.

## 6. Artifacts

- Remote/local run: `runs/zoology-official-bidir-mqar-20260804T093000Z-4000249`.
- Resolved protocol: `config.json` and `output/config.json`.
- Integrity: `preflight.json`, `preflight.log`, `git_sha.txt`, GPU snapshots,
  and `source_snapshot.sha256`.
- Result: `score.json`, `output/metrics.jsonl`, `formal.log`, and
  `abort.json` for the scientific gate failure.
- Visualization: `visualizations/index.html`, summary and 12 hardest cases.

## 7. Registered Readouts

- past/future accuracy, exact and CE;
- balanced accuracy and joint exact;
- validation opening curve and first epoch at `0.90`/`0.99`;
- parameter count, actual training tokens, warmed tokens/s and peak CUDA memory;
- hardest fixed examples with query/write positions, direction, target and
  prediction.

The carrier passes only when both past and future accuracy are at least
`0.90`. The desired stronger endpoint is joint exact at least `0.80`.

## 8. Kill Criteria And Decision

Stop before interpretation on wrong GPU/SHA/source/data hash, non-identical
official versus bidirectional parameters or initialization, causal future
leakage, dead bidirectional future dependency, non-finite backward, OOM/NaN or
the 30-minute wall limit. If either direction remains below `0.90` at epoch30,
archive and close this carrier; do not rescue via seed, LR, width, depth, head,
dropout, loss or extra epochs.

A pass authorizes a separate formal length-scaling experiment with causal
GDN2, native FutureSeed GDN2 and this opened official-attention ceiling. It
does not itself compare FutureSeed against attention.

## 9. Result And Submission

The implementation and provenance checks passed, but the carrier did not open.

Preflight proved:

- exactly one GPU1 with the registered UUID;
- exact clean Zoology commit and source hashes;
- exact P-CAUSAL-007 train/test hashes;
- official causal and bidirectional state keys, parameter count (`437,760`)
  and every initialized tensor identical (`max diff=0`);
- official causal future perturbation dependency exactly `0`, versus
  bidirectional mean dependency `0.007393`;
- finite full-size CUDA forward/backward for both classes.

Training then consumed the registered 30 epochs, or 19.2M input tokens:

| Metric | Official MHA, mask removed |
|---|---:|
| past accuracy / exact / CE | `0.4850 / 0.216 / 0.8314` |
| future accuracy / exact / CE | `0.4845 / 0.218 / 0.8635` |
| balanced accuracy | `0.48475` |
| joint exact | `0.048` |
| best validation accuracy | `0.5015` at epoch13 |
| final validation accuracy | `0.48475` at epoch29 |
| warmed training throughput | `321,951` tokens/s |
| warmed peak allocated memory | `69.6` MiB |
| training plus validation time | `79.98` seconds |

The model immediately rises above the random-vocabulary regime, but settles
near a two-way ambiguity instead of undergoing P-CAUSAL-005's sharp retrieval
transition. Seeing both sides is therefore not sufficient for this exact
architecture to bind a repeated key to its adjacent value. Removing the mask
also removes the useful order-aligned bias that the official causal baseline
exploits. That interpretation is consistent with the nearly symmetric
past/future metrics and CE near a binary ambiguity, but it remains a mechanism
inference rather than a direct proof.

The registered gate required both directional accuracies at least `0.90`, so
the experiment is discarded without a second seed, extra epoch, LR, width,
depth, head, dropout or loss rescue. This cannot be cited as evidence that
FutureSeed beats Transformers. It says only that this exact mask-removal
carrier is not a valid quality ceiling for directional MQAR.

Artifacts:

- `runs/zoology-official-bidir-mqar-20260804T093000Z-4000249`;
- `preflight.json`, `score.json`, logs and `abort.json`;
- validation curve and 12 hardest fixed cases in
  `visualizations/index.html`;
- source snapshot SHA256 recorded; the 66 MiB archive remains outside Git.

Next decision: do not spend another run repairing attention. Directly test the
already-validated strict official-FLA GDN2 no-FutureSeed/FutureSeed pair at
length1024 and combine it with the frozen L64 endpoint. That isolates the
paper's core mechanism scaling question. Attention quality/cost calibration
must later use a task and established recipe where a bidirectional carrier is
independently known to open.

No tag is authorized.
