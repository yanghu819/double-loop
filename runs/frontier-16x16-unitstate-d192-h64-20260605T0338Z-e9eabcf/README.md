# 16x16 Unit-State Launch Abort

Run: `frontier-16x16-unitstate-d192-h64-20260605T0338Z-e9eabcf`

Source SHA on GPU1: `e9eabcff07c207335ec8930c479ee841996e3b90`

Recorded UTC: `2026-06-05T03:37:26.322536+00:00`

## Abort Reason

This launch failed before useful training because PyTorch's CUDA extension loader found a broken `/opt/conda/bin/ninja` first in `PATH`.

The failure is infrastructure-level, not a negative model result:

```text
RuntimeError: Ninja is required to load C++ extensions
```

## Decision

Abort this exact PID and relaunch the same unit-state experiment with `/huyang2/double-loop/.cache/bin` before `/opt/conda/bin` in `PATH`, so RWKV7 CUDA statepassing builds with the repo-local working `ninja`.

No CPU fallback was used.

## Lesson

For RWKV7 CUDA runs on AIStation, validate the effective `ninja` binary before launch. A CUDA-capable PyTorch install is not enough if the JIT extension toolchain resolves to a broken system binary.
