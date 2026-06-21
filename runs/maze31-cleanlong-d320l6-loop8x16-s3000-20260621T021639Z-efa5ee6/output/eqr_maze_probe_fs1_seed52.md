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
- train CE: 0.3691
- train seconds: 5073.9

## Clean Eval

- loop1: exact=0.0000, path_f1=0.3998, path_exact=0.0000, token_acc=0.7924, zH=1.4383, zL=1.4120
- loop2: exact=0.0000, path_f1=0.2983, path_exact=0.0000, token_acc=0.7982, zH=0.0062, zL=0.0045
- loop3: exact=0.0000, path_f1=0.2988, path_exact=0.0000, token_acc=0.7983, zH=0.0044, zL=0.0044
- loop4: exact=0.0000, path_f1=0.2992, path_exact=0.0000, token_acc=0.7983, zH=0.0044, zL=0.0044
- loop5: exact=0.0000, path_f1=0.2993, path_exact=0.0000, token_acc=0.7982, zH=0.0044, zL=0.0044
- loop6: exact=0.0000, path_f1=0.2998, path_exact=0.0000, token_acc=0.7984, zH=0.0044, zL=0.0044
- loop7: exact=0.0000, path_f1=0.2990, path_exact=0.0000, token_acc=0.7981, zH=0.0044, zL=0.0044
- loop8: exact=0.0000, path_f1=0.2994, path_exact=0.0000, token_acc=0.7982, zH=0.0044, zL=0.0044
- loop9: exact=0.0000, path_f1=0.2997, path_exact=0.0000, token_acc=0.7983, zH=0.0044, zL=0.0044
- loop10: exact=0.0000, path_f1=0.3000, path_exact=0.0000, token_acc=0.7983, zH=0.0044, zL=0.0044
- loop11: exact=0.0000, path_f1=0.3000, path_exact=0.0000, token_acc=0.7982, zH=0.0044, zL=0.0044
- loop12: exact=0.0000, path_f1=0.2994, path_exact=0.0000, token_acc=0.7982, zH=0.0044, zL=0.0044
- loop13: exact=0.0000, path_f1=0.2994, path_exact=0.0000, token_acc=0.7981, zH=0.0044, zL=0.0044
- loop14: exact=0.0000, path_f1=0.2994, path_exact=0.0000, token_acc=0.7981, zH=0.0044, zL=0.0044
- loop15: exact=0.0000, path_f1=0.2994, path_exact=0.0000, token_acc=0.7981, zH=0.0044, zL=0.0044
- loop16: exact=0.0000, path_f1=0.2995, path_exact=0.0000, token_acc=0.7981, zH=0.0044, zL=0.0044

## Decision

Pivot: path F1 remains low; shorten path curriculum or increase capacity before judging FutureSeed transfer.
