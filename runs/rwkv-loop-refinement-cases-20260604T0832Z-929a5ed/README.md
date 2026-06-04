# 9x9 FutureSeed Loop-Refinement Cases

Purpose: concrete cases where loop depth matters, not just FutureSeed noncausal/global interaction.

Source SHA on GPU1: `929a5edc54ade4c30e60e6d1859d705028e4c48f`

## Training Trace

- step100: ce=0.8118, loop1_loss=0.8162, loop_last_loss=0.8118
- step200: ce=0.3641, loop1_loss=0.3892, loop_last_loss=0.3641
- step300: ce=0.0760, loop1_loss=0.1385, loop_last_loss=0.0760
- step400: ce=0.0192, loop1_loss=0.0868, loop_last_loss=0.0192
- step500: ce=0.0098, loop1_loss=0.0535, loop_last_loss=0.0098
- step600: ce=0.0081, loop1_loss=0.0456, loop_last_loss=0.0081

## Cases

- `h24_01_refined_idx149`: h24, refined, idx=149, wrong loop1/3/5=7/1/0; png=`h24_01_refined_idx149.png`, html=`h24_01_refined_idx149.html`
- `h24_02_refined_idx41`: h24, refined, idx=41, wrong loop1/3/5=4/0/0; png=`h24_02_refined_idx41.png`, html=`h24_02_refined_idx41.html`
- `h24_03_refined_idx177`: h24, refined, idx=177, wrong loop1/3/5=4/0/0; png=`h24_03_refined_idx177.png`, html=`h24_03_refined_idx177.html`
- `h28_01_refined_idx115`: h28, refined, idx=115, wrong loop1/3/5=6/0/0; png=`h28_01_refined_idx115.png`, html=`h28_01_refined_idx115.html`
- `h28_02_refined_idx343`: h28, refined, idx=343, wrong loop1/3/5=6/0/0; png=`h28_02_refined_idx343.png`, html=`h28_02_refined_idx343.html`
- `h28_03_refined_idx155`: h28, refined, idx=155, wrong loop1/3/5=5/0/0; png=`h28_03_refined_idx155.png`, html=`h28_03_refined_idx155.html`
- `h32_01_refined_idx37`: h32, refined, idx=37, wrong loop1/3/5=7/0/0; png=`h32_01_refined_idx37.png`, html=`h32_01_refined_idx37.html`
- `h32_02_refined_idx308`: h32, refined, idx=308, wrong loop1/3/5=7/1/0; png=`h32_02_refined_idx308.png`, html=`h32_02_refined_idx308.html`
- `h32_03_refined_idx5`: h32, refined, idx=5, wrong loop1/3/5=6/0/0; png=`h32_03_refined_idx5.png`, html=`h32_03_refined_idx5.html`

## Readout

- All 9 selected cases are `refined`: loop1 has hidden-cell errors, loop5 has zero hidden-cell errors.
- This directly visualizes loop value: FutureSeed creates a usable global state, and recurrent loop depth cleans up remaining board-level inconsistency.
- The exporter wrote HTML/PNG successfully, then failed while serializing `Path` in config JSON. The script has been fixed; this README and `loop_refinement_cases.json` were reconstructed from the generated artifacts.
