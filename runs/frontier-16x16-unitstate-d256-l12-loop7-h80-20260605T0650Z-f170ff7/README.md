# 16x16 D256 Harder-Hole Launch Abort

Run: `frontier-16x16-unitstate-d256-l12-loop7-h80-20260605T0650Z-f170ff7`

Source SHA on GPU1: `f170ff74bff34392a3bb32dbe8a3a1eb16c7a4ae`

Recorded UTC: `2026-06-05T06:53:01.086346+00:00`

## Abort Reason

This launch failed before model training because it used `/huyang2/double-loop/.venv/bin/python`, and that reused venv did not contain `torch` after the AIStation restart:

```text
ModuleNotFoundError: No module named 'torch'
```

## Decision

Treat this as launch hygiene, not experiment evidence. Relaunch the same 16x16 harder-hole hypothesis with `/opt/conda/bin/python`, while keeping `/huyang2/double-loop/.cache/bin` first in `PATH` for the working repo-local `ninja`.

No CPU smoke was used.

## Lesson

After AIStation restarts, an existing `.venv` path is not enough. Check that the selected Python has CUDA PyTorch before launch, and prefer the container CUDA Python when the project venv is incomplete.
