# 9x9 RWKV7 TimeMix (official contract) FutureSeed Loop Study

Mainline mechanism: RWKV7 TimeMix (official contract) recurrent backbone, FutureSeed cross-layer terminal-state initialization, and depth-loop iterative refinement.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=1.6274, total=1.6274, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.6278, loop_last_loss=1.6274, sec=1849.7
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2598, early=0.1977, late=0.3221, early_late_gap=-0.1244
  future_seed: fs_gate=0.000, fs_update=0.000, fs_state_norm=0.000, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2618, early=0.1979, late=0.3234, early_late_gap=-0.1255
  future_seed: fs_gate=0.000, fs_update=0.000, fs_state_norm=0.000, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2627, early=0.1991, late=0.3234, early_late_gap=-0.1242
  future_seed: fs_gate=0.000, fs_update=0.000, fs_state_norm=0.000, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2631, early=0.1997, late=0.3240, early_late_gap=-0.1242
  future_seed: fs_gate=0.000, fs_update=0.000, fs_state_norm=0.000, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2629, early=0.1997, late=0.3232, early_late_gap=-0.1234
  future_seed: fs_gate=0.000, fs_update=0.000, fs_state_norm=0.000, fs_decay=0.00, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/runs/sudoku-backbone-rwkv-nofs-s500-sharedinit-20260726T022718Z-6e51f06/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/runs/sudoku-backbone-rwkv-nofs-s500-sharedinit-20260726T022718Z-6e51f06/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=0.0000, blank_acc=0.3816, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}
- official_b51_55: index=/huyang2/double-loop/runs/sudoku-backbone-rwkv-nofs-s500-sharedinit-20260726T022718Z-6e51f06/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/runs/sudoku-backbone-rwkv-nofs-s500-sharedinit-20260726T022718Z-6e51f06/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.0000, blank_acc=0.2676, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/runs/sudoku-backbone-rwkv-nofs-s500-sharedinit-20260726T022718Z-6e51f06/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/runs/sudoku-backbone-rwkv-nofs-s500-sharedinit-20260726T022718Z-6e51f06/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.0000, blank_acc=0.2545, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.3826, early=0.2580, late=0.5205, early_late_gap=-0.2625
- b51_55 (51-55, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2618, early=0.1994, late=0.3241, early_late_gap=-0.1247
- b56_64 (56-64, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.2561, early=0.2024, late=0.3118, early_late_gap=-0.1095

## Decision

Depth loop is neutral on full-board exact in this run.

## Artifacts

- case_html: /huyang2/double-loop/runs/sudoku-backbone-rwkv-nofs-s500-sharedinit-20260726T022718Z-6e51f06/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/runs/sudoku-backbone-rwkv-nofs-s500-sharedinit-20260726T022718Z-6e51f06/output/case_bank
