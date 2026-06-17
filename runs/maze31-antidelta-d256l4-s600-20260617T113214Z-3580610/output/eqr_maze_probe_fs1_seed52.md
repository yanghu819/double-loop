# EqR Maze Probe

Question: does recurrent FutureSeed-style state mixing transfer from Sudoku to path propagation?

- grid: 31x31
- path range: 160-260
- train path stages: `fixed hard range`
- mode: `perfect`
- future_seed_scale: 1.0
- state update: `delta_carry` scale=-0.35 decay=0.95
- train loops: 6
- eval loops: 12
- train CE: 0.3652
- train seconds: 823.0

## Clean Eval

- loop1: exact=0.0000, path_f1=0.3373, path_exact=0.0000, token_acc=0.7953, zH=0.9188, zL=0.9177
- loop2: exact=0.0000, path_f1=0.3528, path_exact=0.0000, token_acc=0.7933, zH=0.3322, zL=0.3299
- loop3: exact=0.0000, path_f1=0.3486, path_exact=0.0000, token_acc=0.7936, zH=0.1138, zL=0.1125
- loop4: exact=0.0000, path_f1=0.3486, path_exact=0.0000, token_acc=0.7940, zH=0.0371, zL=0.0365
- loop5: exact=0.0000, path_f1=0.3487, path_exact=0.0000, token_acc=0.7941, zH=0.0116, zL=0.0114
- loop6: exact=0.0000, path_f1=0.3485, path_exact=0.0000, token_acc=0.7940, zH=0.0052, zL=0.0052
- loop7: exact=0.0000, path_f1=0.3487, path_exact=0.0000, token_acc=0.7941, zH=0.0036, zL=0.0036
- loop8: exact=0.0000, path_f1=0.3484, path_exact=0.0000, token_acc=0.7940, zH=0.0040, zL=0.0040
- loop9: exact=0.0000, path_f1=0.3480, path_exact=0.0000, token_acc=0.7939, zH=0.0041, zL=0.0042
- loop10: exact=0.0000, path_f1=0.3486, path_exact=0.0000, token_acc=0.7941, zH=0.0041, zL=0.0042
- loop11: exact=0.0000, path_f1=0.3483, path_exact=0.0000, token_acc=0.7939, zH=0.0041, zL=0.0042
- loop12: exact=0.0000, path_f1=0.3481, path_exact=0.0000, token_acc=0.7940, zH=0.0041, zL=0.0042

## Decision

Pivot: path F1 remains low; shorten path curriculum or increase capacity before judging FutureSeed transfer.
