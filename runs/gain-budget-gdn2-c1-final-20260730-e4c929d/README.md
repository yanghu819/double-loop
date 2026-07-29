# Gain-Budgeted GDN2 Final Decision

- Plan: `P-GAIN-001`
- Status: discarded at strict systems preflight
- Exact source: `e4c929dd16629a586afb0089bd69090f8722371f`
- GPU: GPU1 `GPU-608b6f50-c148-465f-02d7-7f1e3658c79a`
- Official FLA source: `9c8e42e762fce087c27b673af4922795d9edb85e`
- Projection time overhead: `32.14%`
- Projection memory overhead: `18.49%`
- Pre-registered time limit: `20%`
- Formal candidate training: not launched

All mathematical and official CUDA correctness checks passed before the
systems benchmark rejected the mechanism. The older two-step diagnostic is
included only to explain the failure mode; it is not a formal candidate score.
It shows that strict `c=1` clips almost every token/head and removes almost all
erase-gate anisotropy, degrading paired 51-55-blank loop5 exact from
`0.625` to
`0.125`.

Decision: close strict `c=1` Gain-Budgeted GDN2. Do not sweep cap, seed, loss,
learning rate, model size, or training length. The broader lesson is that the
pretrained GDN2 uses mild transient expansion as useful computation; forcing
every step to be non-expansive is neither cheap nor behavior-preserving.
