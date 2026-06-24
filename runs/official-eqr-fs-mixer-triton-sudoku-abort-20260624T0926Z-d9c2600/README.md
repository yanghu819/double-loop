# official-eqr-fs-mixer-triton-sudoku-abort-20260624T0926Z-d9c2600

This was the first official EqR FutureSeed mixer replacement launch after the
Triton scan backend landed.

It was stopped intentionally, not because of a CUDA/backend failure. The
training path reached step250, then the official train-time eval expanded to
`3304` eval batches at roughly one second per batch under `DISABLE_COMPILE=1`.
That would have spent close to an hour on low-information eval for a backend
viability check.

Decision: kill the exact process tree, archive `abort.json`, and relaunch a
train-only GPU check with eval disabled. The train-only relaunch completed all
`448` steps and saved step250/step448 checkpoints.

