# EqR Maze Probe

Question: does recurrent FutureSeed-style state mixing transfer from Sudoku to path propagation?

- grid: 31x31
- path range: 160-260
- train path stages: `fixed hard range`
- mode: `perfect`
- future_seed_scale: 1.0
- feedback: `prob_embed` scale=1.0 gate_bias=-2.0
- state update: `none` scale=0.0 decay=1.0 gate_bias=2.0
- train loops: 8
- eval loops: 16
- train CE: 0.3750
- train seconds: 2131.2

## Clean Eval

- loop1: exact=0.0000, path_f1=0.5837, path_exact=0.0000, token_acc=0.7458, zH=1.4158, zL=1.4121
- loop2: exact=0.0000, path_f1=0.5836, path_exact=0.0000, token_acc=0.7458, zH=0.0072, zL=0.0063
- loop3: exact=0.0000, path_f1=0.5837, path_exact=0.0000, token_acc=0.7459, zH=0.0050, zL=0.0050
- loop4: exact=0.0000, path_f1=0.5836, path_exact=0.0000, token_acc=0.7459, zH=0.0045, zL=0.0045
- loop5: exact=0.0000, path_f1=0.5836, path_exact=0.0000, token_acc=0.7459, zH=0.0045, zL=0.0045
- loop6: exact=0.0000, path_f1=0.5837, path_exact=0.0000, token_acc=0.7459, zH=0.0045, zL=0.0045
- loop7: exact=0.0000, path_f1=0.5836, path_exact=0.0000, token_acc=0.7459, zH=0.0045, zL=0.0045
- loop8: exact=0.0000, path_f1=0.5836, path_exact=0.0000, token_acc=0.7458, zH=0.0045, zL=0.0045
- loop9: exact=0.0000, path_f1=0.5836, path_exact=0.0000, token_acc=0.7458, zH=0.0045, zL=0.0045
- loop10: exact=0.0000, path_f1=0.5837, path_exact=0.0000, token_acc=0.7459, zH=0.0045, zL=0.0045
- loop11: exact=0.0000, path_f1=0.5837, path_exact=0.0000, token_acc=0.7459, zH=0.0045, zL=0.0045
- loop12: exact=0.0000, path_f1=0.5836, path_exact=0.0000, token_acc=0.7458, zH=0.0045, zL=0.0045
- loop13: exact=0.0000, path_f1=0.5836, path_exact=0.0000, token_acc=0.7458, zH=0.0045, zL=0.0045
- loop14: exact=0.0000, path_f1=0.5836, path_exact=0.0000, token_acc=0.7458, zH=0.0045, zL=0.0045
- loop15: exact=0.0000, path_f1=0.5836, path_exact=0.0000, token_acc=0.7458, zH=0.0045, zL=0.0045
- loop16: exact=0.0000, path_f1=0.5836, path_exact=0.0000, token_acc=0.7458, zH=0.0045, zL=0.0045

## Decision

Continue cautiously: maze path learning is alive, but loop gain is weak; inspect residuals before scaling.
