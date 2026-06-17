# EqR Maze Probe

Question: does recurrent FutureSeed-style state mixing transfer from Sudoku to path propagation?

- grid: 21x21
- path range: 80-140
- mode: `perfect`
- future_seed_scale: 1.0
- train loops: 6
- eval loops: 10
- train CE: 0.0045
- train seconds: 1237.7

## Clean Eval

- loop1: exact=0.0000, path_f1=0.7817, path_exact=0.0000, token_acc=0.8922, zH=1.3966, zL=1.4102
- loop2: exact=0.1426, path_f1=0.9341, path_exact=0.1426, token_acc=0.9719, zH=0.5203, zL=0.6496
- loop3: exact=0.6934, path_f1=0.9842, path_exact=0.6934, token_acc=0.9936, zH=0.4297, zL=0.4279
- loop4: exact=0.9434, path_f1=0.9977, path_exact=0.9434, token_acc=0.9991, zH=0.2727, zL=0.2610
- loop5: exact=0.9902, path_f1=0.9990, path_exact=0.9902, token_acc=0.9996, zH=0.1278, zL=0.1206
- loop6: exact=0.9902, path_f1=0.9992, path_exact=0.9902, token_acc=0.9997, zH=0.0556, zL=0.0522
- loop7: exact=0.9941, path_f1=0.9996, path_exact=0.9941, token_acc=0.9998, zH=0.0353, zL=0.0342
- loop8: exact=0.9961, path_f1=0.9997, path_exact=0.9961, token_acc=0.9999, zH=0.0377, zL=0.0305
- loop9: exact=0.9961, path_f1=0.9997, path_exact=0.9961, token_acc=0.9999, zH=0.0123, zL=0.0159
- loop10: exact=0.9961, path_f1=0.9997, path_exact=0.9961, token_acc=0.9999, zH=0.0094, zL=0.0115

## Decision

Continue: recurrence is buying real path refinement on maze; compare against base or scale path length next.
