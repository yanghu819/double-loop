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
- train seconds: 603.9

## Clean Eval

- loop1: exact=0.0000, path_f1=0.5836, path_exact=0.0000, token_acc=0.7271, zH=1.1716, zL=1.4118
- loop2: exact=0.0000, path_f1=0.5831, path_exact=0.0000, token_acc=0.7261, zH=0.2118, zL=0.0091
- loop3: exact=0.0000, path_f1=0.5830, path_exact=0.0000, token_acc=0.7260, zH=0.0682, zL=0.0050
- loop4: exact=0.0000, path_f1=0.5830, path_exact=0.0000, token_acc=0.7260, zH=0.0258, zL=0.0046
- loop5: exact=0.0000, path_f1=0.5830, path_exact=0.0000, token_acc=0.7260, zH=0.0105, zL=0.0045
- loop6: exact=0.0000, path_f1=0.5830, path_exact=0.0000, token_acc=0.7260, zH=0.0049, zL=0.0045
- loop7: exact=0.0000, path_f1=0.5830, path_exact=0.0000, token_acc=0.7260, zH=0.0030, zL=0.0045
- loop8: exact=0.0000, path_f1=0.5830, path_exact=0.0000, token_acc=0.7260, zH=0.0025, zL=0.0045
- loop9: exact=0.0000, path_f1=0.5830, path_exact=0.0000, token_acc=0.7260, zH=0.0023, zL=0.0045
- loop10: exact=0.0000, path_f1=0.5830, path_exact=0.0000, token_acc=0.7260, zH=0.0023, zL=0.0045
- loop11: exact=0.0000, path_f1=0.5830, path_exact=0.0000, token_acc=0.7260, zH=0.0023, zL=0.0045
- loop12: exact=0.0000, path_f1=0.5830, path_exact=0.0000, token_acc=0.7260, zH=0.0022, zL=0.0045

## Decision

Continue cautiously: maze path learning is alive, but loop gain is weak; inspect residuals before scaling.
