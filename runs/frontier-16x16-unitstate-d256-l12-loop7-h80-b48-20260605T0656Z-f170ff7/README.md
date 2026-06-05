# 16x16 D256 Harder-Hole Batch48 Abort

Run: `frontier-16x16-unitstate-d256-l12-loop7-h80-b48-20260605T0656Z-f170ff7`

Source SHA on GPU1: `f170ff74bff34392a3bb32dbe8a3a1eb16c7a4ae`

Recorded UTC: `2026-06-05T06:58:19.113450+00:00`

## Abort Reason

The same D256/L12/loop7 configuration also exceeded the 80GB memory budget at `FULL_BATCH=48`, even with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 48.00 MiB.
```

## Decision

Try the same scale hypothesis once at `FULL_BATCH=32`. If batch32 failed too, the next move would be D224 or activation checkpointing rather than a blind batch sweep.

No CPU smoke was used.

## Lesson

The scaled unit-state run is memory-bound before it is compute-bound. Any future move beyond D256/L12/loop7 needs checkpointing, lower loop count, or a more efficient structural path.
