# EqR Maze Probe

Question: does recurrent FutureSeed-style state mixing transfer from Sudoku to path propagation?

- grid: 31x31
- path range: 160-260
- train path stages: `fixed hard range`
- mode: `perfect`
- future_seed_scale: 1.0
- state update: `state_compete` scale=0.0 decay=1.0 gate_bias=2.0
- train loops: 6
- eval loops: 12
- train CE: 0.3848
- train seconds: 1137.8

## Clean Eval

- loop1: exact=0.0000, path_f1=0.5875, path_exact=0.0000, token_acc=0.7339, zH=1.3901, zL=1.4118
- loop2: exact=0.0000, path_f1=0.5874, path_exact=0.0000, token_acc=0.7337, zH=0.0373, zL=0.0235
- loop3: exact=0.0000, path_f1=0.5874, path_exact=0.0000, token_acc=0.7337, zH=0.0060, zL=0.0057
- loop4: exact=0.0000, path_f1=0.5874, path_exact=0.0000, token_acc=0.7337, zH=0.0052, zL=0.0053
- loop5: exact=0.0000, path_f1=0.5874, path_exact=0.0000, token_acc=0.7337, zH=0.0052, zL=0.0053
- loop6: exact=0.0000, path_f1=0.5874, path_exact=0.0000, token_acc=0.7337, zH=0.0052, zL=0.0053
- loop7: exact=0.0000, path_f1=0.5874, path_exact=0.0000, token_acc=0.7337, zH=0.0052, zL=0.0053
- loop8: exact=0.0000, path_f1=0.5874, path_exact=0.0000, token_acc=0.7337, zH=0.0052, zL=0.0053
- loop9: exact=0.0000, path_f1=0.5874, path_exact=0.0000, token_acc=0.7337, zH=0.0052, zL=0.0053
- loop10: exact=0.0000, path_f1=0.5874, path_exact=0.0000, token_acc=0.7337, zH=0.0052, zL=0.0053
- loop11: exact=0.0000, path_f1=0.5874, path_exact=0.0000, token_acc=0.7337, zH=0.0052, zL=0.0053
- loop12: exact=0.0000, path_f1=0.5874, path_exact=0.0000, token_acc=0.7337, zH=0.0052, zL=0.0053

## Decision

Continue cautiously: maze path learning is alive, but loop gain is weak; inspect residuals before scaling.
