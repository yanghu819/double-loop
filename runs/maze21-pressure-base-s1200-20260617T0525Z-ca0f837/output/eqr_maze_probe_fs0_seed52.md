# EqR Maze Probe

Question: does recurrent FutureSeed-style state mixing transfer from Sudoku to path propagation?

- grid: 21x21
- path range: 80-140
- mode: `perfect`
- future_seed_scale: 0.0
- train loops: 6
- eval loops: 10
- train CE: 0.2275
- train seconds: 606.9

## Clean Eval

- loop1: exact=0.0000, path_f1=0.7096, path_exact=0.0000, token_acc=0.8402, zH=1.3937, zL=1.4101
- loop2: exact=0.0000, path_f1=0.7460, path_exact=0.0000, token_acc=0.8717, zH=0.2662, zL=0.4914
- loop3: exact=0.0000, path_f1=0.7542, path_exact=0.0000, token_acc=0.8719, zH=0.1638, zL=0.4003
- loop4: exact=0.0000, path_f1=0.7559, path_exact=0.0000, token_acc=0.8723, zH=0.1477, zL=0.3994
- loop5: exact=0.0000, path_f1=0.7590, path_exact=0.0000, token_acc=0.8740, zH=0.1404, zL=0.3946
- loop6: exact=0.0000, path_f1=0.7610, path_exact=0.0000, token_acc=0.8754, zH=0.1388, zL=0.3918
- loop7: exact=0.0000, path_f1=0.7600, path_exact=0.0000, token_acc=0.8745, zH=0.1383, zL=0.3904
- loop8: exact=0.0000, path_f1=0.7617, path_exact=0.0000, token_acc=0.8755, zH=0.1383, zL=0.3900
- loop9: exact=0.0000, path_f1=0.7602, path_exact=0.0000, token_acc=0.8744, zH=0.1383, zL=0.3899
- loop10: exact=0.0000, path_f1=0.7614, path_exact=0.0000, token_acc=0.8754, zH=0.1377, zL=0.3892

## Decision

Continue: recurrence is buying real path refinement on maze; compare against base or scale path length next.
