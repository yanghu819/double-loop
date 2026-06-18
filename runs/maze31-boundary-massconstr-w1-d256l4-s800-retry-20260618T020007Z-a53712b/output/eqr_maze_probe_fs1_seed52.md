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
- train CE: 0.3906
- train seconds: 698.0

## Clean Eval

- loop1: exact=0.0000, path_f1=0.5836, path_exact=0.0000, token_acc=0.7270, zH=1.1053, zL=1.4119
- loop2: exact=0.0000, path_f1=0.5839, path_exact=0.0000, token_acc=0.7276, zH=0.2131, zL=0.0050
- loop3: exact=0.0000, path_f1=0.5841, path_exact=0.0000, token_acc=0.7279, zH=0.0791, zL=0.0042
- loop4: exact=0.0000, path_f1=0.5841, path_exact=0.0000, token_acc=0.7279, zH=0.0300, zL=0.0041
- loop5: exact=0.0000, path_f1=0.5841, path_exact=0.0000, token_acc=0.7279, zH=0.0117, zL=0.0040
- loop6: exact=0.0000, path_f1=0.5841, path_exact=0.0000, token_acc=0.7279, zH=0.0052, zL=0.0040
- loop7: exact=0.0000, path_f1=0.5841, path_exact=0.0000, token_acc=0.7279, zH=0.0031, zL=0.0040
- loop8: exact=0.0000, path_f1=0.5841, path_exact=0.0000, token_acc=0.7279, zH=0.0025, zL=0.0040
- loop9: exact=0.0000, path_f1=0.5841, path_exact=0.0000, token_acc=0.7279, zH=0.0023, zL=0.0040
- loop10: exact=0.0000, path_f1=0.5841, path_exact=0.0000, token_acc=0.7279, zH=0.0023, zL=0.0040
- loop11: exact=0.0000, path_f1=0.5841, path_exact=0.0000, token_acc=0.7279, zH=0.0023, zL=0.0040
- loop12: exact=0.0000, path_f1=0.5841, path_exact=0.0000, token_acc=0.7279, zH=0.0023, zL=0.0040

## Decision

Continue cautiously: maze path learning is alive, but loop gain is weak; inspect residuals before scaling.
