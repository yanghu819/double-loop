# RWKV7 CUDA Sources

This directory vendors the minimal CUDA source needed to run the RWKV7 `wind`
kernel in the Sudoku FutureSeed experiment.

Source:
- Repository: `https://github.com/BlinkDL/modded-nanogpt-rwkv`
- Upstream default branch at integration time: `master`
- Files copied from `rwkv_cuda_wind/`

License:
- Upstream repository license is included as `LICENSE.modded-nanogpt-rwkv`.

Local usage:
- Python wrapper: `wind.py`
- Torch extension cache must stay inside the project via `TORCH_EXTENSIONS_DIR`.
- The `wind` kernel requires CUDA, bf16 tensors, `seq_len % 16 == 0`, and
  `head_dim` divisible by 16.
