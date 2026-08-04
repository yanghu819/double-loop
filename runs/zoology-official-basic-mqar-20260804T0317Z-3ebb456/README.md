# P-CAUSAL-005: exact upstream Zoology MQAR baseline

- Decision: passed
- Final validation accuracy: `0.99775`
- Final validation loss: `0.0153`
- Early stop: epoch 25, official threshold `>0.99`
- Project SHA: `3ebb45656e7f9667526bb068e27174f383600f1d`
- Zoology SHA: `1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb`
- Hardware: one visible AIStation GPU1 A100-SXM4-80GB

The exact documented two-layer Transformer baseline stayed near 25-29%
through epoch 21, then opened sharply: epoch22 `0.368`, epoch23 `0.965`,
epoch24 `0.990`, and epoch25 `0.99775`. This validates the upstream generator,
trainer, model, and metric pipeline. It is not evidence for GDN2 or FutureSeed.

The full source snapshot remains in persistent remote storage and is referenced
by `source_snapshot.ref`; the Git archive contains its hash and all compact run
metadata, logs, metrics, and visualization.
