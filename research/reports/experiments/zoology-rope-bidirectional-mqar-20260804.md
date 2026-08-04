# P-CAUSAL-015: RoPE Bidirectional MQAR Carrier

## 1. Metainfo

- Plan: `P-CAUSAL-015`
- Run: assigned by `scripts/run_zoology_rope_bidirectional_mqar.sh`
- Date: 2026-08-04 CST
- Machine: AIStation task-mode GPU1, one A100-SXM4-80GB
- Status: approved; implementation and strict preflight pending

## 2. Hypothesis

The two failed bidirectional ceilings had access to the entire sequence but no
translation-invariant address for the task's local key/value relation. A
parameter-free standard RoPE transform on Q/K should let the same two-layer
full-attention model learn that a value is adjacent to its key, independent of
the absolute write position.

This is a carrier calibration, not a FutureSeed modification. It asks whether
we have a valid bidirectional quality ceiling before spending compute on the
requested length/cost paper figure.

## 3. Configuration

- Data: exact P-CAUSAL-007 mixed-direction MQAR hashes, 10,000 train and 1,000
  validation samples, sequence length 64, four unique associations.
- Model: Zoology `LanguageModel`, D128, two layers, four attention heads,
  head dimension 57, learned absolute positions and the unchanged upstream
  MLP/residual/readout stack.
- Intervention: standard RoPE on the first 56 Q/K dimensions; theta 10,000;
  no trainable parameter; full noncausal CUDA SDPA unchanged.
- Frozen protocol: batch 32, 30 epochs, AdamW LR 1e-3, weight decay 0.1,
  cosine schedule, seed 123 and query-only validation metric.
- References: frozen P-CAUSAL-010 causal GDN2, native FutureSeed and plain
  full SDPA; frozen P-CAUSAL-011 official MHA with only its mask removed.

Registered pass: past accuracy at least 0.90, future accuracy at least 0.90,
and joint exact at least 0.80. A miss is final for this proxy: no RoPE
base/scale, LR, epoch, width, depth, loss or seed rescue.

## 4. Environment

- SSH alias: `aistation-task-gpu1`
- Persistent root: `/huyang2/double-loop`
- CUDA visibility: only container index 0, physical GPU1 UUID
  `GPU-53e9f3b4-2966-65d3-6614-09c540921519`
- Python: `/opt/conda/bin/python`
- Zoology: pinned SHA `1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb`
- CPU model smoke and GPU2 are forbidden.

## 5. Commands

```bash
CUDA_VISIBLE_DEVICES=0 RUN_NAME=<generated> ./run.sh rope_bidirectional_mqar
```

The launcher first runs
`scripts/check_zoology_rope_bidirectional_mqar.py`; formal training is not
entered unless source, data, parameter/init identity, `rope_scale=0` exactness,
future dependency and finite CUDA backward all pass.

## 6. Artifacts

Pending. The run will archive `config.json`, `preflight.json`, logs,
`score.json`, source SHA/snapshot, all per-arm reference/candidate scores and
cases, selected same-sequence cases, and HTML/screenshots.

## 7. Results

Pending.

## 8. Conclusions

Pending. Success validates a relative-position Transformer ceiling and
authorizes the formal 64-to-1024 quality/cost experiment. Failure closes this
MQAR proxy for a Transformer-quality claim; it must not be reinterpreted as
evidence that FutureSeed beats Transformers.

## 9. Publication Record

No tag or publication artifact is permitted from this carrier-only gate.
