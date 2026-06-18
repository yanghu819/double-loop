# EqR Maze Probe

Question: does recurrent FutureSeed-style state mixing transfer from Sudoku to path propagation?

- grid: 31x31
- path range: 160-260
- train path stages: `fixed hard range`
- mode: `perfect`
- future_seed_scale: 1.0
- state update: `state_compete_boundary` scale=0.0 decay=1.0 gate_bias=2.0
- train loops: 6
- eval loops: 12
- train CE: 0.3887
- train seconds: 718.5

## Clean Eval

- loop1: exact=0.0000, path_f1=0.5847, path_exact=0.0000, token_acc=0.7260, zH=1.1269, zL=1.4120
- loop2: exact=0.0000, path_f1=0.5847, path_exact=0.0000, token_acc=0.7260, zH=0.1904, zL=0.0047
- loop3: exact=0.0000, path_f1=0.5847, path_exact=0.0000, token_acc=0.7260, zH=0.0670, zL=0.0041
- loop4: exact=0.0000, path_f1=0.5847, path_exact=0.0000, token_acc=0.7260, zH=0.0253, zL=0.0040
- loop5: exact=0.0000, path_f1=0.5847, path_exact=0.0000, token_acc=0.7260, zH=0.0099, zL=0.0040
- loop6: exact=0.0000, path_f1=0.5847, path_exact=0.0000, token_acc=0.7260, zH=0.0045, zL=0.0040
- loop7: exact=0.0000, path_f1=0.5847, path_exact=0.0000, token_acc=0.7260, zH=0.0027, zL=0.0040
- loop8: exact=0.0000, path_f1=0.5847, path_exact=0.0000, token_acc=0.7260, zH=0.0022, zL=0.0040
- loop9: exact=0.0000, path_f1=0.5847, path_exact=0.0000, token_acc=0.7260, zH=0.0021, zL=0.0040
- loop10: exact=0.0000, path_f1=0.5847, path_exact=0.0000, token_acc=0.7260, zH=0.0021, zL=0.0040
- loop11: exact=0.0000, path_f1=0.5847, path_exact=0.0000, token_acc=0.7260, zH=0.0021, zL=0.0040
- loop12: exact=0.0000, path_f1=0.5847, path_exact=0.0000, token_acc=0.7260, zH=0.0020, zL=0.0040

## Decision

Continue cautiously: maze path learning is alive, but loop gain is weak; inspect residuals before scaling.
