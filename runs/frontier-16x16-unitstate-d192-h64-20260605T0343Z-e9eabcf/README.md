# 16x16 Unit-State Launch Abort

Run: `frontier-16x16-unitstate-d192-h64-20260605T0343Z-e9eabcf`

Source SHA on GPU1: `e9eabcff07c207335ec8930c479ee841996e3b90`

Recorded UTC: `2026-06-05T03:37:26.402626+00:00`

## Abort Reason

This second launch reproduced the same pre-training CUDA extension build failure: PyTorch's loader still resolved `ninja` through the broken `/opt/conda/bin/ninja` path.

The failure is infrastructure-level, not a negative model result:

```text
RuntimeError: Ninja is required to load C++ extensions
```

## Decision

Stop retrying without changing the launch environment. Relaunch only after making the repo-local toolchain path explicit:

```bash
PATH=/huyang2/double-loop/.cache/bin:$PATH
```

No CPU fallback was used.

## Lesson

Repeated extension-loader failures should be treated as launch hygiene bugs, not as experiment evidence. The successful follow-up run used the same model idea after fixing `PATH`.
