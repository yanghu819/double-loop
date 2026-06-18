# EqR Maze Probe

Question: does recurrent FutureSeed-style state mixing transfer from Sudoku to path propagation?

- grid: 31x31
- path range: 160-260
- train path stages: `fixed hard range`
- mode: `perfect`
- future_seed_scale: 1.0
- state update: `state_compete_budget` scale=0.0 decay=1.0 gate_bias=2.0
- train loops: 6
- eval loops: 12
- train CE: 0.3867
- train seconds: 672.2

## Clean Eval

- loop1: exact=0.0000, path_f1=0.5852, path_exact=0.0000, token_acc=0.7295, zH=1.1256, zL=1.4120
- loop2: exact=0.0000, path_f1=0.5855, path_exact=0.0000, token_acc=0.7299, zH=0.1705, zL=0.0043
- loop3: exact=0.0000, path_f1=0.5856, path_exact=0.0000, token_acc=0.7304, zH=0.0643, zL=0.0040
- loop4: exact=0.0000, path_f1=0.5856, path_exact=0.0000, token_acc=0.7306, zH=0.0246, zL=0.0039
- loop5: exact=0.0000, path_f1=0.5857, path_exact=0.0000, token_acc=0.7308, zH=0.0097, zL=0.0039
- loop6: exact=0.0000, path_f1=0.5858, path_exact=0.0000, token_acc=0.7309, zH=0.0045, zL=0.0039
- loop7: exact=0.0000, path_f1=0.5858, path_exact=0.0000, token_acc=0.7310, zH=0.0027, zL=0.0039
- loop8: exact=0.0000, path_f1=0.5859, path_exact=0.0000, token_acc=0.7311, zH=0.0023, zL=0.0039
- loop9: exact=0.0000, path_f1=0.5861, path_exact=0.0000, token_acc=0.7315, zH=0.0021, zL=0.0039
- loop10: exact=0.0000, path_f1=0.5861, path_exact=0.0000, token_acc=0.7315, zH=0.0021, zL=0.0039
- loop11: exact=0.0000, path_f1=0.5861, path_exact=0.0000, token_acc=0.7315, zH=0.0021, zL=0.0039
- loop12: exact=0.0000, path_f1=0.5861, path_exact=0.0000, token_acc=0.7315, zH=0.0021, zL=0.0039

## Decision

Continue cautiously: maze path learning is alive, but loop gain is weak; inspect residuals before scaling.
