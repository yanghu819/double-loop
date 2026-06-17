# EqR Maze Probe

Question: does recurrent FutureSeed-style state mixing transfer from Sudoku to path propagation?

- grid: 31x31
- path range: 160-260
- train path stages: `fixed hard range`
- mode: `perfect`
- future_seed_scale: 1.0
- state update: `delta_carry` scale=0.35 decay=0.95
- train loops: 6
- eval loops: 12
- train CE: 0.3516
- train seconds: 1083.3

## Clean Eval

- loop1: exact=0.0000, path_f1=0.5709, path_exact=0.0000, token_acc=0.7611, zH=1.9057, zL=1.9061
- loop2: exact=0.0000, path_f1=0.5722, path_exact=0.0000, token_acc=0.7605, zH=0.6517, zL=0.6607
- loop3: exact=0.0000, path_f1=0.5720, path_exact=0.0000, token_acc=0.7603, zH=0.2119, zL=0.2174
- loop4: exact=0.0000, path_f1=0.5720, path_exact=0.0000, token_acc=0.7603, zH=0.0662, zL=0.0688
- loop5: exact=0.0000, path_f1=0.5721, path_exact=0.0000, token_acc=0.7604, zH=0.0214, zL=0.0225
- loop6: exact=0.0000, path_f1=0.5721, path_exact=0.0000, token_acc=0.7604, zH=0.0101, zL=0.0107
- loop7: exact=0.0000, path_f1=0.5721, path_exact=0.0000, token_acc=0.7604, zH=0.0084, zL=0.0088
- loop8: exact=0.0000, path_f1=0.5721, path_exact=0.0000, token_acc=0.7604, zH=0.0071, zL=0.0075
- loop9: exact=0.0000, path_f1=0.5721, path_exact=0.0000, token_acc=0.7604, zH=0.0063, zL=0.0066
- loop10: exact=0.0000, path_f1=0.5721, path_exact=0.0000, token_acc=0.7604, zH=0.0061, zL=0.0065
- loop11: exact=0.0000, path_f1=0.5722, path_exact=0.0000, token_acc=0.7604, zH=0.0060, zL=0.0064
- loop12: exact=0.0000, path_f1=0.5721, path_exact=0.0000, token_acc=0.7604, zH=0.0060, zL=0.0063

## Decision

Continue cautiously: maze path learning is alive, but loop gain is weak; inspect residuals before scaling.
