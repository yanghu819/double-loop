# 9x9 RWKV7 TimeMix (official contract) FutureSeed Loop Study

Mainline mechanism: RWKV7 TimeMix (official contract) recurrent backbone, FutureSeed cross-layer terminal-state initialization, and depth-loop iterative refinement.

Board: 9x9, box: 3x3, hole pattern: `random`.

## Loop Metrics

- train: ce=1.0190, total=1.0348, loop_loss=all, noise=none, buffer=8192, dtype=bfloat16, fs_update=fixed, loop_update=fixed, upd_h=0.950, loop_fb=0.00, loop_time=0.00, scratch=none, scratch_delta=0.000, scratch_gauss=0.0000, exact_margin=0.0000, exact_softmin=0.000, loop1_loss=1.0946, loop_last_loss=1.0190, sec=2022.4
- loop 1: exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4741, early=0.4709, late=0.4775, early_late_gap=-0.0066
  future_seed: fs_gate=0.518, fs_update=1.000, fs_state_norm=16.581, fs_decay=0.00, fs_raw_rms=0.808, fs_raw_rms_std=0.064, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 2: exact=0.0195, valid=0.0195, solved=0.0195, blank_acc=0.4967, early=0.4927, late=0.4947, early_late_gap=-0.0020
  future_seed: fs_gate=0.518, fs_update=1.000, fs_state_norm=16.581, fs_decay=0.00, fs_raw_rms=0.768, fs_raw_rms_std=0.065, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 3: exact=0.0195, valid=0.0195, solved=0.0195, blank_acc=0.4982, early=0.4936, late=0.4981, early_late_gap=-0.0044
  future_seed: fs_gate=0.518, fs_update=1.000, fs_state_norm=16.581, fs_decay=0.00, fs_raw_rms=0.762, fs_raw_rms_std=0.065, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 4: exact=0.0195, valid=0.0195, solved=0.0195, blank_acc=0.4977, early=0.4947, late=0.4974, early_late_gap=-0.0028
  future_seed: fs_gate=0.518, fs_update=1.000, fs_state_norm=16.581, fs_decay=0.00, fs_raw_rms=0.760, fs_raw_rms_std=0.065, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000
- loop 5: exact=0.0195, valid=0.0195, solved=0.0195, blank_acc=0.4980, early=0.4945, late=0.4976, early_late_gap=-0.0032
  future_seed: fs_gate=0.518, fs_update=1.000, fs_state_norm=16.581, fs_decay=0.00, fs_raw_rms=0.760, fs_raw_rms_std=0.065, fs_norm_gain=1.000, fs_norm_gain_std=0.000, fb_in=0.000, fb_next=0.000, loop_time=0.000, upd_l=0.950, upd_h=0.950, scratch_gate=0.000, scratch_decay=0.000, scratch_delta=0.000, scratch_resid=0.000

## Case Bank

- official_b46_50: index=/huyang2/double-loop/runs/sudoku-backbone-rwkv-s500-sharedinit-20260724T154020Z-6e51f06/output/case_bank/official_b46_50/index.html; cases=/huyang2/double-loop/runs/sudoku-backbone-rwkv-s500-sharedinit-20260724T154020Z-6e51f06/output/case_bank/official_b46_50/cases.json; final_loop=5, exact=0.7227, blank_acc=0.9901, selected={'solved_by_loop': 4, 'almost_solved': 4, 'hard_failure': 2}
- official_b51_55: index=/huyang2/double-loop/runs/sudoku-backbone-rwkv-s500-sharedinit-20260724T154020Z-6e51f06/output/case_bank/official_b51_55/index.html; cases=/huyang2/double-loop/runs/sudoku-backbone-rwkv-s500-sharedinit-20260724T154020Z-6e51f06/output/case_bank/official_b51_55/cases.json; final_loop=5, exact=0.0000, blank_acc=0.5154, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}
- official_b56_64: index=/huyang2/double-loop/runs/sudoku-backbone-rwkv-s500-sharedinit-20260724T154020Z-6e51f06/output/case_bank/official_b56_64/index.html; cases=/huyang2/double-loop/runs/sudoku-backbone-rwkv-s500-sharedinit-20260724T154020Z-6e51f06/output/case_bank/official_b56_64/cases.json; final_loop=5, exact=0.0000, blank_acc=0.4664, selected={'solved_by_loop': 0, 'almost_solved': 0, 'hard_failure': 4}

## Official Blank-Range Eval

- b46_50 (46-50, n=512): exact=0.7285, valid=0.7285, solved=0.7285, blank_acc=0.9903, early=0.9860, late=0.9922, early_late_gap=-0.0062
- b51_55 (51-55, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.5226, early=0.5251, late=0.5196, early_late_gap=+0.0055
- b56_64 (56-64, n=512): exact=0.0000, valid=0.0000, solved=0.0000, blank_acc=0.4674, early=0.4622, late=0.4616, early_late_gap=+0.0006

## Decision

Depth loop adds useful full-board refinement.

## Artifacts

- case_html: /huyang2/double-loop/runs/sudoku-backbone-rwkv-s500-sharedinit-20260724T154020Z-6e51f06/output/futureseed_loop_case_seed52.html
- case_bank_root: /huyang2/double-loop/runs/sudoku-backbone-rwkv-s500-sharedinit-20260724T154020Z-6e51f06/output/case_bank
