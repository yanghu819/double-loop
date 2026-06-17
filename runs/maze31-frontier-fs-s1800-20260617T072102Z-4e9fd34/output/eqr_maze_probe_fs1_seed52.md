# EqR Maze Probe

Question: does recurrent FutureSeed-style state mixing transfer from Sudoku to path propagation?

- grid: 31x31
- path range: 160-260
- mode: `perfect`
- future_seed_scale: 1.0
- train loops: 6
- eval loops: 12
- train CE: 0.3477
- train seconds: 1043.1

## Clean Eval

- loop1: exact=0.0000, path_f1=0.5203, path_exact=0.0000, token_acc=0.7800, zH=1.3980, zL=1.4117
- loop2: exact=0.0000, path_f1=0.5086, path_exact=0.0000, token_acc=0.7823, zH=0.1665, zL=0.1630
- loop3: exact=0.0000, path_f1=0.5088, path_exact=0.0000, token_acc=0.7824, zH=0.0284, zL=0.0342
- loop4: exact=0.0000, path_f1=0.5092, path_exact=0.0000, token_acc=0.7825, zH=0.0097, zL=0.0171
- loop5: exact=0.0000, path_f1=0.5098, path_exact=0.0000, token_acc=0.7826, zH=0.0075, zL=0.0143
- loop6: exact=0.0000, path_f1=0.5093, path_exact=0.0000, token_acc=0.7824, zH=0.0073, zL=0.0141
- loop7: exact=0.0000, path_f1=0.5093, path_exact=0.0000, token_acc=0.7824, zH=0.0073, zL=0.0142
- loop8: exact=0.0000, path_f1=0.5094, path_exact=0.0000, token_acc=0.7825, zH=0.0071, zL=0.0135
- loop9: exact=0.0000, path_f1=0.5089, path_exact=0.0000, token_acc=0.7823, zH=0.0072, zL=0.0139
- loop10: exact=0.0000, path_f1=0.5093, path_exact=0.0000, token_acc=0.7825, zH=0.0072, zL=0.0139
- loop11: exact=0.0000, path_f1=0.5093, path_exact=0.0000, token_acc=0.7825, zH=0.0070, zL=0.0132
- loop12: exact=0.0000, path_f1=0.5094, path_exact=0.0000, token_acc=0.7825, zH=0.0071, zL=0.0137

## Decision

Continue cautiously: maze path learning is alive, but loop gain is weak; inspect residuals before scaling.
