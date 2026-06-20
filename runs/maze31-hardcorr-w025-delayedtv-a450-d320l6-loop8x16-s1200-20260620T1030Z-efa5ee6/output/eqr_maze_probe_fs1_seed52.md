# EqR Maze Probe

Question: does recurrent FutureSeed-style state mixing transfer from Sudoku to path propagation?

- grid: 31x31
- path range: 160-260
- train path stages: `fixed hard range`
- mode: `perfect`
- future_seed_scale: 1.0
- feedback: `none` scale=0.0 gate_bias=-2.0
- state update: `none` scale=0.0 decay=1.0 gate_bias=2.0
- train loops: 8
- eval loops: 16
- train CE: 0.3750
- train seconds: 2082.5

## Clean Eval

- loop1: exact=0.0000, path_f1=0.5847, path_exact=0.0000, token_acc=0.7293, zH=1.4159, zL=1.4124
- loop2: exact=0.0000, path_f1=0.5844, path_exact=0.0000, token_acc=0.7291, zH=0.0056, zL=0.0036
- loop3: exact=0.0000, path_f1=0.5845, path_exact=0.0000, token_acc=0.7292, zH=0.0036, zL=0.0036
- loop4: exact=0.0000, path_f1=0.5844, path_exact=0.0000, token_acc=0.7291, zH=0.0035, zL=0.0036
- loop5: exact=0.0000, path_f1=0.5844, path_exact=0.0000, token_acc=0.7291, zH=0.0035, zL=0.0036
- loop6: exact=0.0000, path_f1=0.5844, path_exact=0.0000, token_acc=0.7292, zH=0.0035, zL=0.0036
- loop7: exact=0.0000, path_f1=0.5844, path_exact=0.0000, token_acc=0.7292, zH=0.0035, zL=0.0036
- loop8: exact=0.0000, path_f1=0.5844, path_exact=0.0000, token_acc=0.7292, zH=0.0035, zL=0.0036
- loop9: exact=0.0000, path_f1=0.5845, path_exact=0.0000, token_acc=0.7292, zH=0.0035, zL=0.0036
- loop10: exact=0.0000, path_f1=0.5845, path_exact=0.0000, token_acc=0.7292, zH=0.0035, zL=0.0036
- loop11: exact=0.0000, path_f1=0.5843, path_exact=0.0000, token_acc=0.7291, zH=0.0035, zL=0.0036
- loop12: exact=0.0000, path_f1=0.5844, path_exact=0.0000, token_acc=0.7292, zH=0.0035, zL=0.0036
- loop13: exact=0.0000, path_f1=0.5844, path_exact=0.0000, token_acc=0.7292, zH=0.0035, zL=0.0036
- loop14: exact=0.0000, path_f1=0.5844, path_exact=0.0000, token_acc=0.7291, zH=0.0035, zL=0.0036
- loop15: exact=0.0000, path_f1=0.5844, path_exact=0.0000, token_acc=0.7292, zH=0.0035, zL=0.0036
- loop16: exact=0.0000, path_f1=0.5844, path_exact=0.0000, token_acc=0.7291, zH=0.0035, zL=0.0036

## Decision

Continue cautiously: maze path learning is alive, but loop gain is weak; inspect residuals before scaling.
