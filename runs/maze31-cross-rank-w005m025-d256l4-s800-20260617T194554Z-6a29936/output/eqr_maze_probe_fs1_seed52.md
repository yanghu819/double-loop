# EqR Maze Probe

Question: does recurrent FutureSeed-style state mixing transfer from Sudoku to path propagation?

- grid: 31x31
- path range: 160-260
- train path stages: `fixed hard range`
- mode: `perfect`
- future_seed_scale: 1.0
- state update: `state_compete_cross` scale=0.0 decay=1.0 gate_bias=2.0
- train loops: 6
- eval loops: 12
- train CE: 0.3867
- train seconds: 664.7

## Clean Eval

- loop1: exact=0.0000, path_f1=0.5851, path_exact=0.0000, token_acc=0.7298, zH=1.1919, zL=1.4117
- loop2: exact=0.0000, path_f1=0.5852, path_exact=0.0000, token_acc=0.7304, zH=0.1644, zL=0.0280
- loop3: exact=0.0000, path_f1=0.5852, path_exact=0.0000, token_acc=0.7305, zH=0.0586, zL=0.0106
- loop4: exact=0.0000, path_f1=0.5852, path_exact=0.0000, token_acc=0.7305, zH=0.0228, zL=0.0059
- loop5: exact=0.0000, path_f1=0.5852, path_exact=0.0000, token_acc=0.7305, zH=0.0095, zL=0.0052
- loop6: exact=0.0000, path_f1=0.5852, path_exact=0.0000, token_acc=0.7305, zH=0.0047, zL=0.0052
- loop7: exact=0.0000, path_f1=0.5852, path_exact=0.0000, token_acc=0.7305, zH=0.0031, zL=0.0051
- loop8: exact=0.0000, path_f1=0.5852, path_exact=0.0000, token_acc=0.7305, zH=0.0027, zL=0.0051
- loop9: exact=0.0000, path_f1=0.5852, path_exact=0.0000, token_acc=0.7305, zH=0.0026, zL=0.0051
- loop10: exact=0.0000, path_f1=0.5852, path_exact=0.0000, token_acc=0.7305, zH=0.0026, zL=0.0051
- loop11: exact=0.0000, path_f1=0.5852, path_exact=0.0000, token_acc=0.7305, zH=0.0026, zL=0.0051
- loop12: exact=0.0000, path_f1=0.5852, path_exact=0.0000, token_acc=0.7305, zH=0.0026, zL=0.0051

## Decision

Continue cautiously: maze path learning is alive, but loop gain is weak; inspect residuals before scaling.
