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
- train CE: 0.4355
- train seconds: 667.2

## Clean Eval

- loop1: exact=0.0000, path_f1=0.5834, path_exact=0.0000, token_acc=0.7260, zH=1.0975, zL=1.4121
- loop2: exact=0.0000, path_f1=0.5834, path_exact=0.0000, token_acc=0.7260, zH=0.2351, zL=0.0089
- loop3: exact=0.0000, path_f1=0.5834, path_exact=0.0000, token_acc=0.7260, zH=0.1115, zL=0.0051
- loop4: exact=0.0000, path_f1=0.5834, path_exact=0.0000, token_acc=0.7260, zH=0.0611, zL=0.0047
- loop5: exact=0.0000, path_f1=0.5834, path_exact=0.0000, token_acc=0.7260, zH=0.0277, zL=0.0046
- loop6: exact=0.0000, path_f1=0.5834, path_exact=0.0000, token_acc=0.7260, zH=0.0124, zL=0.0045
- loop7: exact=0.0000, path_f1=0.5834, path_exact=0.0000, token_acc=0.7260, zH=0.0059, zL=0.0045
- loop8: exact=0.0000, path_f1=0.5834, path_exact=0.0000, token_acc=0.7260, zH=0.0033, zL=0.0045
- loop9: exact=0.0000, path_f1=0.5834, path_exact=0.0000, token_acc=0.7260, zH=0.0024, zL=0.0045
- loop10: exact=0.0000, path_f1=0.5834, path_exact=0.0000, token_acc=0.7260, zH=0.0022, zL=0.0045
- loop11: exact=0.0000, path_f1=0.5834, path_exact=0.0000, token_acc=0.7260, zH=0.0021, zL=0.0045
- loop12: exact=0.0000, path_f1=0.5834, path_exact=0.0000, token_acc=0.7260, zH=0.0020, zL=0.0045

## Decision

Continue cautiously: maze path learning is alive, but loop gain is weak; inspect residuals before scaling.
