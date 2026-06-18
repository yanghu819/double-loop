# EqR Maze Probe

Question: does recurrent FutureSeed-style state mixing transfer from Sudoku to path propagation?

- grid: 31x31
- path range: 160-260
- train path stages: `fixed hard range`
- mode: `perfect`
- future_seed_scale: 1.0
- state update: `none` scale=0.0 decay=1.0 gate_bias=2.0
- train loops: 8
- eval loops: 16
- train CE: 0.3711
- train seconds: 2072.0

## Clean Eval

- loop1: exact=0.0000, path_f1=0.5822, path_exact=0.0000, token_acc=0.7485, zH=1.4157, zL=1.4120
- loop2: exact=0.0000, path_f1=0.5823, path_exact=0.0000, token_acc=0.7488, zH=0.0059, zL=0.0041
- loop3: exact=0.0000, path_f1=0.5824, path_exact=0.0000, token_acc=0.7488, zH=0.0041, zL=0.0040
- loop4: exact=0.0000, path_f1=0.5823, path_exact=0.0000, token_acc=0.7488, zH=0.0041, zL=0.0040
- loop5: exact=0.0000, path_f1=0.5822, path_exact=0.0000, token_acc=0.7487, zH=0.0041, zL=0.0040
- loop6: exact=0.0000, path_f1=0.5823, path_exact=0.0000, token_acc=0.7487, zH=0.0041, zL=0.0040
- loop7: exact=0.0000, path_f1=0.5824, path_exact=0.0000, token_acc=0.7488, zH=0.0041, zL=0.0040
- loop8: exact=0.0000, path_f1=0.5823, path_exact=0.0000, token_acc=0.7487, zH=0.0041, zL=0.0040
- loop9: exact=0.0000, path_f1=0.5824, path_exact=0.0000, token_acc=0.7488, zH=0.0041, zL=0.0040
- loop10: exact=0.0000, path_f1=0.5823, path_exact=0.0000, token_acc=0.7488, zH=0.0041, zL=0.0040
- loop11: exact=0.0000, path_f1=0.5824, path_exact=0.0000, token_acc=0.7488, zH=0.0041, zL=0.0040
- loop12: exact=0.0000, path_f1=0.5823, path_exact=0.0000, token_acc=0.7487, zH=0.0041, zL=0.0040
- loop13: exact=0.0000, path_f1=0.5823, path_exact=0.0000, token_acc=0.7488, zH=0.0041, zL=0.0040
- loop14: exact=0.0000, path_f1=0.5823, path_exact=0.0000, token_acc=0.7488, zH=0.0041, zL=0.0040
- loop15: exact=0.0000, path_f1=0.5824, path_exact=0.0000, token_acc=0.7488, zH=0.0041, zL=0.0040
- loop16: exact=0.0000, path_f1=0.5824, path_exact=0.0000, token_acc=0.7488, zH=0.0041, zL=0.0040

## Decision

Continue cautiously: maze path learning is alive, but loop gain is weak; inspect residuals before scaling.
