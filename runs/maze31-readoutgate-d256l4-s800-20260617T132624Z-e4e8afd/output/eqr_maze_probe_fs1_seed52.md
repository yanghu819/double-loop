# EqR Maze Probe

Question: does recurrent FutureSeed-style state mixing transfer from Sudoku to path propagation?

- grid: 31x31
- path range: 160-260
- train path stages: `fixed hard range`
- mode: `perfect`
- future_seed_scale: 1.0
- state update: `learned_gate` scale=0.0 decay=1.0 gate_bias=2.0
- train loops: 6
- eval loops: 12
- train CE: 0.3828
- train seconds: 1093.1

## Clean Eval

- loop1: exact=0.0000, path_f1=0.5850, path_exact=0.0000, token_acc=0.7283, zH=1.2732, zL=1.2409
- loop2: exact=0.0000, path_f1=0.5850, path_exact=0.0000, token_acc=0.7284, zH=0.1416, zL=0.1516
- loop3: exact=0.0000, path_f1=0.5851, path_exact=0.0000, token_acc=0.7285, zH=0.0256, zL=0.0196
- loop4: exact=0.0000, path_f1=0.5851, path_exact=0.0000, token_acc=0.7285, zH=0.0068, zL=0.0054
- loop5: exact=0.0000, path_f1=0.5851, path_exact=0.0000, token_acc=0.7284, zH=0.0049, zL=0.0050
- loop6: exact=0.0000, path_f1=0.5851, path_exact=0.0000, token_acc=0.7284, zH=0.0049, zL=0.0050
- loop7: exact=0.0000, path_f1=0.5851, path_exact=0.0000, token_acc=0.7284, zH=0.0049, zL=0.0050
- loop8: exact=0.0000, path_f1=0.5851, path_exact=0.0000, token_acc=0.7284, zH=0.0049, zL=0.0050
- loop9: exact=0.0000, path_f1=0.5851, path_exact=0.0000, token_acc=0.7285, zH=0.0049, zL=0.0050
- loop10: exact=0.0000, path_f1=0.5851, path_exact=0.0000, token_acc=0.7285, zH=0.0049, zL=0.0050
- loop11: exact=0.0000, path_f1=0.5851, path_exact=0.0000, token_acc=0.7285, zH=0.0049, zL=0.0050
- loop12: exact=0.0000, path_f1=0.5851, path_exact=0.0000, token_acc=0.7284, zH=0.0049, zL=0.0050

## Decision

Continue cautiously: maze path learning is alive, but loop gain is weak; inspect residuals before scaling.
