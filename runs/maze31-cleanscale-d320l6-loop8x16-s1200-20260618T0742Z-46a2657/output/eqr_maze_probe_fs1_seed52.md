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
- train CE: 0.3652
- train seconds: 2066.2

## Clean Eval

- loop1: exact=0.0000, path_f1=0.5301, path_exact=0.0000, token_acc=0.7721, zH=1.4174, zL=1.4120
- loop2: exact=0.0000, path_f1=0.5342, path_exact=0.0000, token_acc=0.7717, zH=0.0057, zL=0.0039
- loop3: exact=0.0000, path_f1=0.5344, path_exact=0.0000, token_acc=0.7717, zH=0.0039, zL=0.0039
- loop4: exact=0.0000, path_f1=0.5344, path_exact=0.0000, token_acc=0.7717, zH=0.0039, zL=0.0039
- loop5: exact=0.0000, path_f1=0.5345, path_exact=0.0000, token_acc=0.7717, zH=0.0039, zL=0.0039
- loop6: exact=0.0000, path_f1=0.5344, path_exact=0.0000, token_acc=0.7716, zH=0.0039, zL=0.0039
- loop7: exact=0.0000, path_f1=0.5343, path_exact=0.0000, token_acc=0.7716, zH=0.0039, zL=0.0039
- loop8: exact=0.0000, path_f1=0.5344, path_exact=0.0000, token_acc=0.7717, zH=0.0039, zL=0.0039
- loop9: exact=0.0000, path_f1=0.5344, path_exact=0.0000, token_acc=0.7717, zH=0.0039, zL=0.0039
- loop10: exact=0.0000, path_f1=0.5345, path_exact=0.0000, token_acc=0.7716, zH=0.0039, zL=0.0039
- loop11: exact=0.0000, path_f1=0.5344, path_exact=0.0000, token_acc=0.7716, zH=0.0039, zL=0.0039
- loop12: exact=0.0000, path_f1=0.5343, path_exact=0.0000, token_acc=0.7716, zH=0.0039, zL=0.0039
- loop13: exact=0.0000, path_f1=0.5343, path_exact=0.0000, token_acc=0.7716, zH=0.0039, zL=0.0039
- loop14: exact=0.0000, path_f1=0.5346, path_exact=0.0000, token_acc=0.7717, zH=0.0039, zL=0.0039
- loop15: exact=0.0000, path_f1=0.5344, path_exact=0.0000, token_acc=0.7716, zH=0.0039, zL=0.0039
- loop16: exact=0.0000, path_f1=0.5344, path_exact=0.0000, token_acc=0.7716, zH=0.0039, zL=0.0039

## Decision

Continue cautiously: maze path learning is alive, but loop gain is weak; inspect residuals before scaling.
