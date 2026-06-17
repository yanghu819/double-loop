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
- train seconds: 630.8

## Clean Eval

- loop1: exact=0.0000, path_f1=0.5845, path_exact=0.0000, token_acc=0.7284, zH=1.1557, zL=1.4117
- loop2: exact=0.0000, path_f1=0.5846, path_exact=0.0000, token_acc=0.7287, zH=0.2050, zL=0.0104
- loop3: exact=0.0000, path_f1=0.5847, path_exact=0.0000, token_acc=0.7287, zH=0.0769, zL=0.0052
- loop4: exact=0.0000, path_f1=0.5847, path_exact=0.0000, token_acc=0.7288, zH=0.0303, zL=0.0048
- loop5: exact=0.0000, path_f1=0.5847, path_exact=0.0000, token_acc=0.7289, zH=0.0123, zL=0.0047
- loop6: exact=0.0000, path_f1=0.5847, path_exact=0.0000, token_acc=0.7289, zH=0.0056, zL=0.0046
- loop7: exact=0.0000, path_f1=0.5847, path_exact=0.0000, token_acc=0.7289, zH=0.0033, zL=0.0046
- loop8: exact=0.0000, path_f1=0.5847, path_exact=0.0000, token_acc=0.7289, zH=0.0027, zL=0.0046
- loop9: exact=0.0000, path_f1=0.5847, path_exact=0.0000, token_acc=0.7289, zH=0.0025, zL=0.0046
- loop10: exact=0.0000, path_f1=0.5847, path_exact=0.0000, token_acc=0.7289, zH=0.0024, zL=0.0046
- loop11: exact=0.0000, path_f1=0.5847, path_exact=0.0000, token_acc=0.7289, zH=0.0024, zL=0.0046
- loop12: exact=0.0000, path_f1=0.5847, path_exact=0.0000, token_acc=0.7289, zH=0.0024, zL=0.0046

## Decision

Continue cautiously: maze path learning is alive, but loop gain is weak; inspect residuals before scaling.
