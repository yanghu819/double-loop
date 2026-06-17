# EqR Maze Probe

Question: does recurrent FutureSeed-style state mixing transfer from Sudoku to path propagation?

- grid: 31x31
- path range: 160-260
- train path stages: `fixed hard range`
- mode: `perfect`
- future_seed_scale: 1.0
- train loops: 6
- eval loops: 12
- train CE: 0.3535
- train seconds: 1614.4

## Clean Eval

- loop1: exact=0.0000, path_f1=0.5662, path_exact=0.0000, token_acc=0.7628, zH=1.4124, zL=1.4119
- loop2: exact=0.0000, path_f1=0.5646, path_exact=0.0000, token_acc=0.7622, zH=0.0641, zL=0.0504
- loop3: exact=0.0000, path_f1=0.5650, path_exact=0.0000, token_acc=0.7623, zH=0.0128, zL=0.0100
- loop4: exact=0.0000, path_f1=0.5650, path_exact=0.0000, token_acc=0.7624, zH=0.0079, zL=0.0064
- loop5: exact=0.0000, path_f1=0.5649, path_exact=0.0000, token_acc=0.7624, zH=0.0065, zL=0.0057
- loop6: exact=0.0000, path_f1=0.5647, path_exact=0.0000, token_acc=0.7623, zH=0.0062, zL=0.0055
- loop7: exact=0.0000, path_f1=0.5651, path_exact=0.0000, token_acc=0.7624, zH=0.0062, zL=0.0056
- loop8: exact=0.0000, path_f1=0.5647, path_exact=0.0000, token_acc=0.7623, zH=0.0062, zL=0.0055
- loop9: exact=0.0000, path_f1=0.5648, path_exact=0.0000, token_acc=0.7623, zH=0.0060, zL=0.0054
- loop10: exact=0.0000, path_f1=0.5650, path_exact=0.0000, token_acc=0.7623, zH=0.0060, zL=0.0054
- loop11: exact=0.0000, path_f1=0.5649, path_exact=0.0000, token_acc=0.7623, zH=0.0061, zL=0.0055
- loop12: exact=0.0000, path_f1=0.5649, path_exact=0.0000, token_acc=0.7623, zH=0.0061, zL=0.0055

## Decision

Continue cautiously: maze path learning is alive, but loop gain is weak; inspect residuals before scaling.
