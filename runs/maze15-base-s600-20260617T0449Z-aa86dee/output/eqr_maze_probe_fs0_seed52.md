# EqR Maze Probe

Question: does recurrent FutureSeed-style state mixing transfer from Sudoku to path propagation?

- grid: 15x15
- path range: 32-56
- mode: `perfect`
- future_seed_scale: 0.0
- train loops: 4
- eval loops: 8
- train CE: 0.3574
- train seconds: 128.0

## Clean Eval

- loop1: exact=0.0000, path_f1=0.5904, path_exact=0.0000, token_acc=0.7562, zH=1.3339, zL=1.4118
- loop2: exact=0.0000, path_f1=0.5905, path_exact=0.0000, token_acc=0.7565, zH=0.3326, zL=0.2806
- loop3: exact=0.0000, path_f1=0.5907, path_exact=0.0000, token_acc=0.7572, zH=0.1321, zL=0.1198
- loop4: exact=0.0000, path_f1=0.5907, path_exact=0.0000, token_acc=0.7569, zH=0.0663, zL=0.0609
- loop5: exact=0.0000, path_f1=0.5907, path_exact=0.0000, token_acc=0.7566, zH=0.0343, zL=0.0309
- loop6: exact=0.0000, path_f1=0.5905, path_exact=0.0000, token_acc=0.7564, zH=0.0193, zL=0.0186
- loop7: exact=0.0000, path_f1=0.5905, path_exact=0.0000, token_acc=0.7564, zH=0.0121, zL=0.0131
- loop8: exact=0.0000, path_f1=0.5905, path_exact=0.0000, token_acc=0.7564, zH=0.0088, zL=0.0106

## Decision

Continue cautiously: maze path learning is alive, but loop gain is weak; inspect residuals before scaling.
