# 9x9 RWKV FutureSeed Loop Study

Mainline mechanism: RWKV recurrent backbone, FutureSeed cross-layer terminal-state initialization, depth-loop iterative refinement, and latent noise.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=0.9901, total=0.9901, loop_loss=final, noise=feature_diff, buffer=8192, dtype=bfloat16, fs_update=loop_residual, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, loop1_loss=1.2194, loop_last_loss=0.9901, sec=917.8
- loop 1: exact=0.0078, valid=0.0078, solved=0.0078, blank_acc=0.4520, early=0.4487, late=0.4500, early_late_gap=-0.0013
  future_seed: fs_gate=0.485, fs_update=0.533, fs_state_norm=7.759, fs_decay=0.50, fs_mem=5.296, fs_mem_delta=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0293, valid=0.0293, solved=0.0293, blank_acc=0.5193, early=0.5163, late=0.5218, early_late_gap=-0.0055
  future_seed: fs_gate=0.485, fs_update=0.533, fs_state_norm=7.759, fs_decay=0.50, fs_mem=5.077, fs_mem_delta=1.548, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0293, valid=0.0293, solved=0.0293, blank_acc=0.5333, early=0.5283, late=0.5352, early_late_gap=-0.0070
  future_seed: fs_gate=0.485, fs_update=0.533, fs_state_norm=7.759, fs_decay=0.50, fs_mem=4.995, fs_mem_delta=0.954, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0293, valid=0.0293, solved=0.0293, blank_acc=0.5342, early=0.5298, late=0.5372, early_late_gap=-0.0074
  future_seed: fs_gate=0.485, fs_update=0.533, fs_state_norm=7.759, fs_decay=0.50, fs_mem=4.968, fs_mem_delta=0.526, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0293, valid=0.0293, solved=0.0293, blank_acc=0.5344, early=0.5301, late=0.5372, early_late_gap=-0.0071
  future_seed: fs_gate=0.485, fs_update=0.533, fs_state_norm=7.759, fs_decay=0.50, fs_mem=4.959, fs_mem_delta=0.280, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/.worktrees/gdn-realbwd-official-sudoku-d192-fs-loopresid-629b578-20260627/runs/gdn-d192-official-sudoku-fs-loopresid-s1500-20260627T0135Z-629b578/output/futureseed_loop_case_seed52.html
