# EqR Maze Probe

Question: does recurrent FutureSeed-style state mixing transfer from Sudoku to path propagation?

- grid: 31x31
- path range: 160-260
- train path stages: `fixed hard range`
- mode: `perfect`
- future_seed_scale: 1.0
- state update: `state_compete` scale=0.0 decay=1.0 gate_bias=2.0
- train loops: 8
- eval loops: 12
- train CE: 0.3906
- train seconds: 492.3

## Clean Eval

- loop1: exact=0.0000, path_f1=0.5853, path_exact=0.0000, token_acc=0.7276, zH=1.3839, zL=1.4117
- loop2: exact=0.0000, path_f1=0.5851, path_exact=0.0000, token_acc=0.7274, zH=0.0299, zL=0.0056
- loop3: exact=0.0000, path_f1=0.5851, path_exact=0.0000, token_acc=0.7274, zH=0.0045, zL=0.0043
- loop4: exact=0.0000, path_f1=0.5851, path_exact=0.0000, token_acc=0.7275, zH=0.0042, zL=0.0043
- loop5: exact=0.0000, path_f1=0.5851, path_exact=0.0000, token_acc=0.7275, zH=0.0042, zL=0.0043
- loop6: exact=0.0000, path_f1=0.5851, path_exact=0.0000, token_acc=0.7275, zH=0.0042, zL=0.0043
- loop7: exact=0.0000, path_f1=0.5851, path_exact=0.0000, token_acc=0.7274, zH=0.0042, zL=0.0043
- loop8: exact=0.0000, path_f1=0.5851, path_exact=0.0000, token_acc=0.7275, zH=0.0042, zL=0.0043
- loop9: exact=0.0000, path_f1=0.5852, path_exact=0.0000, token_acc=0.7275, zH=0.0042, zL=0.0043
- loop10: exact=0.0000, path_f1=0.5852, path_exact=0.0000, token_acc=0.7275, zH=0.0042, zL=0.0043
- loop11: exact=0.0000, path_f1=0.5852, path_exact=0.0000, token_acc=0.7275, zH=0.0042, zL=0.0043
- loop12: exact=0.0000, path_f1=0.5852, path_exact=0.0000, token_acc=0.7275, zH=0.0042, zL=0.0043

## Decision

Continue cautiously: maze path learning is alive, but loop gain is weak; inspect residuals before scaling.
