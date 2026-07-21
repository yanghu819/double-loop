# GDN versus GDN2 Crossover

## Decision

Inconclusive crossover: quality gaps narrowed but did not reverse cleanly.

At step1000, GDN/GDN2 CE is `0.9413/0.9539`;
mixed loop5 exact ties at `0.02344`, and fixed holes53
exact ties at `0.01758`. Official 46-50
exact is `0.99805` versus
`0.99219`; both remain zero exact
at 51-64 blanks. GDN2 narrows the early gap but does not win task quality or
efficiency under the shared small-state recipe.

The post-run strict gate confirms pinned official FLA source, Triton short
convolution, and official CUDA backward nodes for GDN, KDA, and GDN2. Open
`index.html` for learning curves, loop metrics, resource use, and same-seed
hard-case visualizations.
