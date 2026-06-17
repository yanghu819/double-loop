# EqR Maze Probe

Question: does recurrent FutureSeed-style state mixing transfer from Sudoku to path propagation?

- grid: 21x21
- path range: 80-140
- mode: `perfect`
- future_seed_scale: 1.0
- train loops: 6
- eval loops: 10
- train CE: 0.2207
- train seconds: 611.5

## Clean Eval

- loop1: exact=0.0000, path_f1=0.6873, path_exact=0.0000, token_acc=0.8346, zH=1.3919, zL=1.4112
- loop2: exact=0.0000, path_f1=0.7532, path_exact=0.0000, token_acc=0.8731, zH=0.4013, zL=0.4298
- loop3: exact=0.0000, path_f1=0.7858, path_exact=0.0000, token_acc=0.8953, zH=0.2260, zL=0.2301
- loop4: exact=0.0000, path_f1=0.7999, path_exact=0.0000, token_acc=0.9035, zH=0.1196, zL=0.1170
- loop5: exact=0.0000, path_f1=0.8050, path_exact=0.0000, token_acc=0.9063, zH=0.0790, zL=0.0778
- loop6: exact=0.0000, path_f1=0.8059, path_exact=0.0000, token_acc=0.9068, zH=0.0616, zL=0.0626
- loop7: exact=0.0000, path_f1=0.8057, path_exact=0.0000, token_acc=0.9068, zH=0.0529, zL=0.0561
- loop8: exact=0.0000, path_f1=0.8049, path_exact=0.0000, token_acc=0.9064, zH=0.0468, zL=0.0512
- loop9: exact=0.0000, path_f1=0.8051, path_exact=0.0000, token_acc=0.9065, zH=0.0416, zL=0.0482
- loop10: exact=0.0000, path_f1=0.8051, path_exact=0.0000, token_acc=0.9065, zH=0.0392, zL=0.0465

## Decision

Continue: recurrence is buying real path refinement on maze; compare against base or scale path length next.
