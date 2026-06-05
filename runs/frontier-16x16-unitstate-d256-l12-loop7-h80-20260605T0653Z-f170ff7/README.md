# 16x16 D256 Harder-Hole Batch64 Abort

Run: `frontier-16x16-unitstate-d256-l12-loop7-h80-20260605T0653Z-f170ff7`

Source SHA on GPU1: `f170ff74bff34392a3bb32dbe8a3a1eb16c7a4ae`

Recorded UTC: `2026-06-05T06:56:05.807804+00:00`

## Abort Reason

The scaled unit-state model launched on CUDA but exceeded the A800 80GB memory budget with `FULL_BATCH=64`:

```text
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 16.00 MiB.
```

## Decision

This is a resource boundary, not a negative mechanism result. Relaunch the same D256/L12/loop7/h80 curriculum at smaller batch with allocator fragmentation mitigation.

No CPU smoke was used.

## Lesson

Under current no-checkpointing training, D256/L12/loop7 is already near the practical 80GB activation wall. Further scaling should not assume that batch64 is feasible.
