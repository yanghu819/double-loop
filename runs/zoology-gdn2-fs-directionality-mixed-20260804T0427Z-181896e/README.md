# P-CAUSAL-007 Native FutureSeed Directionality

Completed 2026-08-04 on AIStation task-mode GPU1 only.

## Result

| Metric | No FutureSeed | FutureSeed |
|---|---:|---:|
| past accuracy | 0.9305 | 0.9955 |
| future accuracy | 0.0110 | 0.9930 |
| past exact | 0.8770 | 0.9910 |
| future exact | 0.0000 | 0.9860 |
| balanced accuracy | 0.47075 | 0.99425 |

The two arms use the same 538,704 parameters, initialization, data hashes,
optimizer, ten epochs, and pinned official FLA GDN2 kernel. The only behavioral
difference is native FutureSeed scale 0 versus 1. FutureSeed passes layer 0's
normalized terminal recurrent state into layer 1; it does not scan backward.

Raw throughput is archived but not used as an efficiency claim because the
first arm paid one-time Triton compilation. No outer loop is used in this
mechanism-isolation experiment.

Open `visualizations/index.html` for the learning curves, causal-integrity
checks, and ten paired same-sequence failure cases.

## Provenance

- experiment source: `181896e`
- Zoology: `1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb`
- FLA: `9c8e42e762fce087c27b673af4922795d9edb85e`
- source snapshot SHA256:
  `a23059a9cebd969df20913d0128e650ec19aecc0f86bb173fdf52f47ebc93157`
