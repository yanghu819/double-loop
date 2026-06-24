# official-eqr-mixer-base-trainonly-e1024-20260624T1000Z-ce2e76b

Train-only official EqR Sudoku mixer baseline used for the matched e1024 scale gate.

- Machine: AIStation `GPU1` only
- Source SHA: `ce2e76bed92d2f271518f0bf23f251c8eec492e5`
- Remote root: `/huyang2/double-loop/official_eqr_compare`
- Config: official `train/eqr_sudoku`, hidden size 192, heads 6, batch 128, `epochs=1024`
- Steps: `7168/7168`
- Final train loss: `0.767679`
- Final checkpoint: `/huyang2/double-loop/official_eqr_compare/outputs/mixer-base/outputs/y9l15z0t/2026-6-24/9-53-24/checkpoints/step_7168_y9l15z0t.pth`

This run is paired with `official-eqr-futureseed-mixer-triton-trainonly-e1024-20260624T1010Z-ce2e76b` and evaluated in `official-eqr-mixer-replacement-triton-e1024-matched-eval-20260624T1015Z-ce2e76b`.
