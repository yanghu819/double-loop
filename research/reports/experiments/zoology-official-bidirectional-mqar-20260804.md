# Official Zoology Bidirectional MHA Carrier Gate

## 1. Metainfo

- Plan: `P-CAUSAL-011`
- Status: approved; not yet launched
- Date: 2026-08-04 CST
- Machine: AIStation task-mode GPU1 only
- Branch: `codex/p-causal-011-official-bidir-mqar`

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

Pending. The formal run will archive resolved config, preflight, score, logs,
source snapshot hash, exact source/data hashes, PID/PGID, GPU state and HTML
hardest-case visualization under `/huyang2/double-loop/runs`.

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

Pending. No tag is authorized for a carrier-only calibration.
