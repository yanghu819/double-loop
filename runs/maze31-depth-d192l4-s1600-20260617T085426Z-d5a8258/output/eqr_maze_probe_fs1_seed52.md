# EqR Maze Probe

Question: does recurrent FutureSeed-style state mixing transfer from Sudoku to path propagation?

- grid: 31x31
- path range: 160-260
- train path stages: `fixed hard range`
- mode: `perfect`
- future_seed_scale: 1.0
- train loops: 6
- eval loops: 12
- train CE: 0.3477
- train seconds: 1680.3

## Clean Eval

- loop1: exact=0.0000, path_f1=0.5372, path_exact=0.0000, token_acc=0.7761, zH=1.4121, zL=1.4121
- loop2: exact=0.0000, path_f1=0.5366, path_exact=0.0000, token_acc=0.7763, zH=0.0300, zL=0.0217
- loop3: exact=0.0000, path_f1=0.5366, path_exact=0.0000, token_acc=0.7763, zH=0.0070, zL=0.0061
- loop4: exact=0.0000, path_f1=0.5366, path_exact=0.0000, token_acc=0.7762, zH=0.0054, zL=0.0052
- loop5: exact=0.0000, path_f1=0.5366, path_exact=0.0000, token_acc=0.7763, zH=0.0052, zL=0.0051
- loop6: exact=0.0000, path_f1=0.5367, path_exact=0.0000, token_acc=0.7763, zH=0.0052, zL=0.0051
- loop7: exact=0.0000, path_f1=0.5365, path_exact=0.0000, token_acc=0.7762, zH=0.0053, zL=0.0051
- loop8: exact=0.0000, path_f1=0.5366, path_exact=0.0000, token_acc=0.7762, zH=0.0060, zL=0.0055
- loop9: exact=0.0000, path_f1=0.5364, path_exact=0.0000, token_acc=0.7762, zH=0.0055, zL=0.0054
- loop10: exact=0.0000, path_f1=0.5365, path_exact=0.0000, token_acc=0.7762, zH=0.0052, zL=0.0051
- loop11: exact=0.0000, path_f1=0.5363, path_exact=0.0000, token_acc=0.7762, zH=0.0052, zL=0.0051
- loop12: exact=0.0000, path_f1=0.5364, path_exact=0.0000, token_acc=0.7762, zH=0.0052, zL=0.0051

## Decision

Continue cautiously: maze path learning is alive, but loop gain is weak; inspect residuals before scaling.
