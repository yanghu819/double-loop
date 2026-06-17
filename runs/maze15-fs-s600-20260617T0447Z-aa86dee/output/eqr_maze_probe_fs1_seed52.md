# EqR Maze Probe

Question: does recurrent FutureSeed-style state mixing transfer from Sudoku to path propagation?

- grid: 15x15
- path range: 32-56
- mode: `perfect`
- future_seed_scale: 1.0
- train loops: 4
- eval loops: 8
- train CE: 0.3574
- train seconds: 128.9

## Clean Eval

- loop1: exact=0.0000, path_f1=0.5905, path_exact=0.0000, token_acc=0.7563, zH=1.3530, zL=1.4119
- loop2: exact=0.0000, path_f1=0.5911, path_exact=0.0000, token_acc=0.7571, zH=0.3134, zL=0.3214
- loop3: exact=0.0000, path_f1=0.5910, path_exact=0.0000, token_acc=0.7568, zH=0.1195, zL=0.1078
- loop4: exact=0.0000, path_f1=0.5913, path_exact=0.0000, token_acc=0.7575, zH=0.0483, zL=0.0463
- loop5: exact=0.0000, path_f1=0.5913, path_exact=0.0000, token_acc=0.7574, zH=0.0248, zL=0.0268
- loop6: exact=0.0000, path_f1=0.5913, path_exact=0.0000, token_acc=0.7573, zH=0.0125, zL=0.0134
- loop7: exact=0.0000, path_f1=0.5912, path_exact=0.0000, token_acc=0.7572, zH=0.0074, zL=0.0085
- loop8: exact=0.0000, path_f1=0.5912, path_exact=0.0000, token_acc=0.7573, zH=0.0058, zL=0.0072

## Decision

Continue cautiously: maze path learning is alive, but loop gain is weak; inspect residuals before scaling.
