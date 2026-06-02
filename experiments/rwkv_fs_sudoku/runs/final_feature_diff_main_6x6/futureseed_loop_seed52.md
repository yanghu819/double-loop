# 6x6 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 6x6, box: 2x3, hole pattern: `unit`.

## Loop Metrics

- train: ce=0.7915, total=0.7915, loop_loss=final, noise=feature_diff, buffer=8192, loop1_loss=0.7893, loop_last_loss=0.7915, sec=240.1
- loop 1: exact=0.3164, valid=0.3164, solved=0.3164, blank_acc=0.5820
  future_seed: fs_gate=0.493, fs_state_norm=3.947
- loop 2: exact=0.3359, valid=0.3359, solved=0.3359, blank_acc=0.5879
  future_seed: fs_gate=0.493, fs_state_norm=3.947
- loop 3: exact=0.3516, valid=0.3516, solved=0.3516, blank_acc=0.5879
  future_seed: fs_gate=0.493, fs_state_norm=3.947

## Eval Noise

- clean eval loop3: exact=0.3516, valid=0.3516, solved=0.3516, blank_acc=0.5879
- noisy eval loop3: exact=0.3516, valid=0.3516, solved=0.3516, blank_acc=0.5840

## Stochastic Rollouts

- K1: oracle_exact=0.3203, oracle_solved=0.3203, disagree=0.0000
  confidence: exact=0.3203, valid=0.3203, solved=0.3203, blank_acc=0.5840; gap=0.0000
  consistency: exact=0.3203, valid=0.3203, solved=0.3203, blank_acc=0.5840; gap=0.0000
  residual: exact=0.3203, valid=0.3203, solved=0.3203, blank_acc=0.5840; gap=0.0000
  majority: exact=0.3203, valid=0.3203, solved=0.3203, blank_acc=0.5840; gap=0.0000
- K4: oracle_exact=0.4297, oracle_solved=0.4297, disagree=0.1016
  confidence: exact=0.3125, valid=0.3125, solved=0.3125, blank_acc=0.5723; gap=0.1172
  consistency: exact=0.3281, valid=0.3281, solved=0.3281, blank_acc=0.5859; gap=0.1016
  residual: exact=0.3320, valid=0.3320, solved=0.3320, blank_acc=0.5840; gap=0.0977
  majority: exact=0.3281, valid=0.3281, solved=0.3281, blank_acc=0.5859; gap=0.1016
- K8: oracle_exact=0.4727, oracle_solved=0.4727, disagree=0.1147
  confidence: exact=0.3594, valid=0.3594, solved=0.3594, blank_acc=0.5918; gap=0.1133
  consistency: exact=0.3398, valid=0.3398, solved=0.3398, blank_acc=0.5840; gap=0.1328
  residual: exact=0.3633, valid=0.3633, solved=0.3633, blank_acc=0.5977; gap=0.1094
  majority: exact=0.3398, valid=0.3398, solved=0.3398, blank_acc=0.5840; gap=0.1328
- K16: oracle_exact=0.5039, oracle_solved=0.5039, disagree=0.1171
  confidence: exact=0.3477, valid=0.3477, solved=0.3477, blank_acc=0.5879; gap=0.1562
  consistency: exact=0.3477, valid=0.3477, solved=0.3477, blank_acc=0.5918; gap=0.1562
  residual: exact=0.3438, valid=0.3438, solved=0.3438, blank_acc=0.5977; gap=0.1602
  majority: exact=0.3477, valid=0.3477, solved=0.3477, blank_acc=0.5918; gap=0.1562

## Hole Transfer

- holes2: exact=0.3516, valid=0.3516, solved=0.3516, blank_acc=0.5879; indep=0.3456, exact/indep=1.02
- holes4: exact=0.0938, valid=0.0938, solved=0.0938, blank_acc=0.4248; indep=0.0326, exact/indep=2.88
- holes6: exact=0.0664, valid=0.0664, solved=0.0664, blank_acc=0.3333; indep=0.0014, exact/indep=48.41

## Pattern Transfer

### row
- holes2: exact=0.1797, valid=0.1797, solved=0.1797, blank_acc=0.5000; indep=0.2500, exact/indep=0.72
- holes4: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2891; indep=0.0070, exact/indep=0.00
- holes6: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2083; indep=0.0001, exact/indep=0.00

### col
- holes2: exact=0.4180, valid=0.4180, solved=0.4180, blank_acc=0.6523; indep=0.4256, exact/indep=0.98
- holes4: exact=0.1875, valid=0.1875, solved=0.1875, blank_acc=0.5674; indep=0.1036, exact/indep=1.81
- holes6: exact=0.1602, valid=0.1602, solved=0.1602, blank_acc=0.5026; indep=0.0161, exact/indep=9.94

### box
- holes2: exact=0.3164, valid=0.3164, solved=0.3164, blank_acc=0.5977; indep=0.3572, exact/indep=0.89
- holes4: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4102; indep=0.0283, exact/indep=0.00
- holes6: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3086; indep=0.0009, exact/indep=0.00


## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /Users/huyang/Documents/double loop/experiments/rwkv_fs_sudoku/runs/final_feature_diff_main_6x6/futureseed_loop_case_seed52.html
