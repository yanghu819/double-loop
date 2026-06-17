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
- train seconds: 716.2

## Clean Eval

- loop1: exact=0.0000, path_f1=0.5857, path_exact=0.0000, token_acc=0.7307, zH=1.1487, zL=1.4119
- loop2: exact=0.0000, path_f1=0.5853, path_exact=0.0000, token_acc=0.7296, zH=0.1868, zL=0.0060
- loop3: exact=0.0000, path_f1=0.5852, path_exact=0.0000, token_acc=0.7294, zH=0.0713, zL=0.0048
- loop4: exact=0.0000, path_f1=0.5851, path_exact=0.0000, token_acc=0.7293, zH=0.0272, zL=0.0045
- loop5: exact=0.0000, path_f1=0.5852, path_exact=0.0000, token_acc=0.7295, zH=0.0107, zL=0.0045
- loop6: exact=0.0000, path_f1=0.5852, path_exact=0.0000, token_acc=0.7296, zH=0.0049, zL=0.0044
- loop7: exact=0.0000, path_f1=0.5854, path_exact=0.0000, token_acc=0.7300, zH=0.0031, zL=0.0044
- loop8: exact=0.0000, path_f1=0.5854, path_exact=0.0000, token_acc=0.7301, zH=0.0026, zL=0.0044
- loop9: exact=0.0000, path_f1=0.5853, path_exact=0.0000, token_acc=0.7301, zH=0.0025, zL=0.0044
- loop10: exact=0.0000, path_f1=0.5853, path_exact=0.0000, token_acc=0.7301, zH=0.0025, zL=0.0044
- loop11: exact=0.0000, path_f1=0.5854, path_exact=0.0000, token_acc=0.7302, zH=0.0025, zL=0.0044
- loop12: exact=0.0000, path_f1=0.5854, path_exact=0.0000, token_acc=0.7302, zH=0.0025, zL=0.0044

## Decision

Continue cautiously: maze path learning is alive, but loop gain is weak; inspect residuals before scaling.
