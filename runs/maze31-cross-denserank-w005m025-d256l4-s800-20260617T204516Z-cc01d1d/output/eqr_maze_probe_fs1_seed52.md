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
- train CE: 0.3750
- train seconds: 678.0

## Clean Eval

- loop1: exact=0.0000, path_f1=0.5854, path_exact=0.0000, token_acc=0.7292, zH=1.2178, zL=1.4117
- loop2: exact=0.0000, path_f1=0.5854, path_exact=0.0000, token_acc=0.7291, zH=0.1351, zL=0.0232
- loop3: exact=0.0000, path_f1=0.5855, path_exact=0.0000, token_acc=0.7293, zH=0.0554, zL=0.0063
- loop4: exact=0.0000, path_f1=0.5856, path_exact=0.0000, token_acc=0.7294, zH=0.0232, zL=0.0053
- loop5: exact=0.0000, path_f1=0.5856, path_exact=0.0000, token_acc=0.7294, zH=0.0100, zL=0.0052
- loop6: exact=0.0000, path_f1=0.5857, path_exact=0.0000, token_acc=0.7295, zH=0.0049, zL=0.0051
- loop7: exact=0.0000, path_f1=0.5857, path_exact=0.0000, token_acc=0.7295, zH=0.0032, zL=0.0051
- loop8: exact=0.0000, path_f1=0.5857, path_exact=0.0000, token_acc=0.7296, zH=0.0027, zL=0.0051
- loop9: exact=0.0000, path_f1=0.5857, path_exact=0.0000, token_acc=0.7296, zH=0.0026, zL=0.0051
- loop10: exact=0.0000, path_f1=0.5857, path_exact=0.0000, token_acc=0.7295, zH=0.0025, zL=0.0051
- loop11: exact=0.0000, path_f1=0.5857, path_exact=0.0000, token_acc=0.7295, zH=0.0025, zL=0.0051
- loop12: exact=0.0000, path_f1=0.5857, path_exact=0.0000, token_acc=0.7296, zH=0.0025, zL=0.0051

## Decision

Continue cautiously: maze path learning is alive, but loop gain is weak; inspect residuals before scaling.
