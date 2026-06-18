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
- train CE: 0.3887
- train seconds: 661.1

## Clean Eval

- loop1: exact=0.0000, path_f1=0.5844, path_exact=0.0000, token_acc=0.7273, zH=1.0912, zL=1.4120
- loop2: exact=0.0000, path_f1=0.5849, path_exact=0.0000, token_acc=0.7283, zH=0.1957, zL=0.0046
- loop3: exact=0.0000, path_f1=0.5851, path_exact=0.0000, token_acc=0.7289, zH=0.0766, zL=0.0040
- loop4: exact=0.0000, path_f1=0.5851, path_exact=0.0000, token_acc=0.7289, zH=0.0305, zL=0.0039
- loop5: exact=0.0000, path_f1=0.5851, path_exact=0.0000, token_acc=0.7289, zH=0.0125, zL=0.0039
- loop6: exact=0.0000, path_f1=0.5851, path_exact=0.0000, token_acc=0.7289, zH=0.0055, zL=0.0039
- loop7: exact=0.0000, path_f1=0.5851, path_exact=0.0000, token_acc=0.7289, zH=0.0030, zL=0.0039
- loop8: exact=0.0000, path_f1=0.5851, path_exact=0.0000, token_acc=0.7289, zH=0.0023, zL=0.0039
- loop9: exact=0.0000, path_f1=0.5851, path_exact=0.0000, token_acc=0.7289, zH=0.0021, zL=0.0039
- loop10: exact=0.0000, path_f1=0.5851, path_exact=0.0000, token_acc=0.7289, zH=0.0020, zL=0.0039
- loop11: exact=0.0000, path_f1=0.5851, path_exact=0.0000, token_acc=0.7289, zH=0.0020, zL=0.0039
- loop12: exact=0.0000, path_f1=0.5851, path_exact=0.0000, token_acc=0.7289, zH=0.0020, zL=0.0039

## Decision

Continue cautiously: maze path learning is alive, but loop gain is weak; inspect residuals before scaling.
