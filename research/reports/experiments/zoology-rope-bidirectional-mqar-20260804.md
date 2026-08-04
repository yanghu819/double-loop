# P-CAUSAL-015: RoPE Bidirectional MQAR Carrier

## 1. Metainfo

- Plan: `P-CAUSAL-015`
- Run: `zoology-rope-bidir-mqar-20260804T121135Z-cc4c80c`
- Date: 2026-08-04 CST
- Machine: AIStation task-mode GPU1, one A100-SXM4-80GB
- Status: discarded by the registered carrier gate

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

The completed run used exact PID/PGID `58288/58288` and detached source
`cc4c80c5285d92b8e9db3ffb04d8f191ab9eedf0`. It archives `config.json`,
`preflight.json`, full logs, `score.json`, source SHA/snapshot hash, all per-arm
reference/candidate scores and cases, selected same-sequence cases, HTML, and
desktop/mobile screenshots. `visualizations/binding_diagnostics.json` records
the exact wrong-key counts used by the failure analysis. The remote source
snapshot SHA256 is
`9d4f94ebb50597213deba3f6c256f7904c8968f205964c8cba205b6ba1549aa6`.
The scientific stop is recorded in `abort.json`; exit status 2 is deliberate,
not an infrastructure failure. GPU memory returned to zero.

## 7. Results

The strict CUDA preflight passed before formal training. It verified one A100
GPU1, exact Zoology source and fixed data hashes, identical 539,136 parameters
and initialized tensors across plain SDPA and both RoPE scale settings,
`rope_scale=0` output max difference exactly zero, nonzero future dependency,
and finite backward gradients for scale 0 and scale 1.

| Arm | Past acc | Future acc | Balanced acc | Joint exact |
|---|---:|---:|---:|---:|
| Causal GDN2, frozen | 0.9965 | 0.0100 | 0.50325 | 0.000 |
| GDN2 + FutureSeed, frozen | 0.9985 | 0.9920 | 0.99525 | 0.981 |
| Plain full SDPA, frozen | 0.4980 | 0.4995 | 0.49875 | 0.040 |
| Official MHA, mask removed, frozen | 0.4850 | 0.4845 | 0.48475 | 0.048 |
| Full SDPA + RoPE | 0.4955 | 0.4820 | 0.48875 | 0.046 |

RoPE misses all three registered thresholds. Its balanced accuracy is `-0.0100`
below plain SDPA and `-0.5065` below FutureSeed. Best aggregate validation
accuracy is only `0.5030` at epoch 2; it ends at `0.48875`. The sampled train
loss falls to `0.3158`, while validation CE rises from `1.1805` to `1.6443`,
so the extra relative feature is overfit rather than converted into a
general key/value algorithm.

The failure shape is exact. For plain SDPA, official mask-removed MHA, and
RoPE SDPA, 100% of wrong query predictions are another association's valid
value from the same sample. RoPE future/past wrong-key selections are
`1036/1009` out of `2000/2000` queries. The model sees the value set but does
not bind each key to its own value. By comparison, FutureSeed has only `16/3`
wrong future/past predictions, all remaining within-sample swaps.

The independently warmed RoPE diagnostic is `200.7k` tokens/s and `66.8 MiB`
peak allocated memory. It is not a final efficiency claim because this direct
implementation recomputes RoPE trigonometry and no quality gate was passed.

## 8. Conclusions

Discard the RoPE carrier and stop attention rescue on this proxy. Relative
addressing alone did not break the same two-candidate binding ambiguity seen
in both previous bidirectional controls. Do not sweep RoPE theta/scale, epochs,
LR, width, depth, loss, or seed.

This result strengthens the task-level mechanism diagnosis but still does not
show that FutureSeed beats Transformers in general. It says that, in this
fixed two-layer Zoology shell, recurrent state updates provide the useful
key/value binding bias while plain full attention, mask-removed upstream MHA,
and RoPE full attention do not. The requested paper scaling figure should now
measure the already validated causal-GDN2/FutureSeed pair across sequence
length and memory load. A Transformer quality ceiling must come from a
different established task where its bidirectional carrier independently
opens.

## 9. Publication Record

No tag or publication artifact is permitted from this carrier-only gate.
