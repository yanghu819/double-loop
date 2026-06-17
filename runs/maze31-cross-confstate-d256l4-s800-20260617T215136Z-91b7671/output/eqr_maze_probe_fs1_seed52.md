# EqR Maze Probe

Question: does recurrent FutureSeed-style state mixing transfer from Sudoku to path propagation?

- grid: 31x31
- path range: 160-260
- train path stages: `fixed hard range`
- mode: `perfect`
- future_seed_scale: 1.0
- state update: `state_compete_conf` scale=0.0 decay=1.0 gate_bias=2.0
- train loops: 6
- eval loops: 12
- train CE: 0.3750
- train seconds: 702.8

## Clean Eval

- loop1: exact=0.0000, path_f1=0.5852, path_exact=0.0000, token_acc=0.7286, zH=1.1876, zL=1.4117
- loop2: exact=0.0000, path_f1=0.5849, path_exact=0.0000, token_acc=0.7282, zH=0.1927, zL=0.0197
- loop3: exact=0.0000, path_f1=0.5849, path_exact=0.0000, token_acc=0.7281, zH=0.0722, zL=0.0064
- loop4: exact=0.0000, path_f1=0.5849, path_exact=0.0000, token_acc=0.7281, zH=0.0273, zL=0.0053
- loop5: exact=0.0000, path_f1=0.5849, path_exact=0.0000, token_acc=0.7281, zH=0.0107, zL=0.0051
- loop6: exact=0.0000, path_f1=0.5849, path_exact=0.0000, token_acc=0.7281, zH=0.0050, zL=0.0051
- loop7: exact=0.0000, path_f1=0.5849, path_exact=0.0000, token_acc=0.7281, zH=0.0033, zL=0.0051
- loop8: exact=0.0000, path_f1=0.5849, path_exact=0.0000, token_acc=0.7281, zH=0.0029, zL=0.0051
- loop9: exact=0.0000, path_f1=0.5849, path_exact=0.0000, token_acc=0.7281, zH=0.0028, zL=0.0051
- loop10: exact=0.0000, path_f1=0.5849, path_exact=0.0000, token_acc=0.7281, zH=0.0027, zL=0.0051
- loop11: exact=0.0000, path_f1=0.5849, path_exact=0.0000, token_acc=0.7281, zH=0.0027, zL=0.0051
- loop12: exact=0.0000, path_f1=0.5849, path_exact=0.0000, token_acc=0.7281, zH=0.0027, zL=0.0051

## Decision

Continue cautiously: maze path learning is alive, but loop gain is weak; inspect residuals before scaling.
