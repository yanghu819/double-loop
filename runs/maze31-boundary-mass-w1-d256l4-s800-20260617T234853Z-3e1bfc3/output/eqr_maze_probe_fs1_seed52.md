# EqR Maze Probe

Question: does recurrent FutureSeed-style state mixing transfer from Sudoku to path propagation?

- grid: 31x31
- path range: 160-260
- train path stages: `fixed hard range`
- mode: `perfect`
- future_seed_scale: 1.0
- state update: `state_compete_boundary` scale=0.0 decay=1.0 gate_bias=2.0
- train loops: 6
- eval loops: 12
- train CE: 0.3809
- train seconds: 716.5

## Clean Eval

- loop1: exact=0.0000, path_f1=0.5849, path_exact=0.0000, token_acc=0.7281, zH=1.1306, zL=1.4118
- loop2: exact=0.0000, path_f1=0.5849, path_exact=0.0000, token_acc=0.7281, zH=0.1966, zL=0.0130
- loop3: exact=0.0000, path_f1=0.5874, path_exact=0.0000, token_acc=0.7336, zH=0.0824, zL=0.0049
- loop4: exact=0.0000, path_f1=0.5878, path_exact=0.0000, token_acc=0.7358, zH=0.0367, zL=0.0046
- loop5: exact=0.0000, path_f1=0.5880, path_exact=0.0000, token_acc=0.7363, zH=0.0169, zL=0.0044
- loop6: exact=0.0000, path_f1=0.5882, path_exact=0.0000, token_acc=0.7372, zH=0.0083, zL=0.0044
- loop7: exact=0.0000, path_f1=0.5881, path_exact=0.0000, token_acc=0.7380, zH=0.0046, zL=0.0044
- loop8: exact=0.0000, path_f1=0.5880, path_exact=0.0000, token_acc=0.7382, zH=0.0032, zL=0.0044
- loop9: exact=0.0000, path_f1=0.5880, path_exact=0.0000, token_acc=0.7383, zH=0.0028, zL=0.0044
- loop10: exact=0.0000, path_f1=0.5880, path_exact=0.0000, token_acc=0.7384, zH=0.0027, zL=0.0044
- loop11: exact=0.0000, path_f1=0.5879, path_exact=0.0000, token_acc=0.7385, zH=0.0026, zL=0.0044
- loop12: exact=0.0000, path_f1=0.5879, path_exact=0.0000, token_acc=0.7389, zH=0.0026, zL=0.0044

## Decision

Continue cautiously: maze path learning is alive, but loop gain is weak; inspect residuals before scaling.
