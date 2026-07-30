# GDN2 Fast-Slow Decay Decision

The mechanism works mechanically but fails the task and systems gates.

- Forgetting-hazard TV falls to `0.9839x`; learned `rho=0.0990`.
- Mixed loop5 exact changes `0.2402 -> 0.2500`.
- Official 51-55/56-60/61-64 exact changes
  `0.3594/0.1387/0.1641 -> 0.3535/0.1465/0.1621`.
- Hard-range mean delta is exactly `0.0000`.
- Full-model time/peak-memory overhead is `+7.19%/+20.65%`.
- Paired cases contain both a `19 -> 0` rescue and a `3 -> 26` regression.

Decision: discard decay-only smoothing. It changes which solution basin wins
but does not produce reliable recurrent convergence. Do not sweep FIR length,
rho, seed, LR, loss, width, or continuation length.
