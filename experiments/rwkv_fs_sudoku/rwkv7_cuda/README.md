# RWKV7 CUDA Sources

This directory vendors the minimal CUDA source needed to run RWKV7 kernels in
the Sudoku FutureSeed experiment.

Sources:
- Repository: `https://github.com/BlinkDL/modded-nanogpt-rwkv`
- Upstream default branch at integration time: `master`
- Files copied from `rwkv_cuda_wind/`
- Repository: `https://github.com/BlinkDL/RWKV-CUDA`
- Upstream branch at integration time: `main`
- Files copied from `rwkv7_fast_fused/cuda/rwkv7_statepassing_clampw.*`

License:
- Upstream repository license is included as `LICENSE.modded-nanogpt-rwkv`.
- `BlinkDL/RWKV-CUDA` did not expose a root `LICENSE` file at integration time;
  keep this attribution with the vendored state-passing source.

Local usage:
- Python wrapper: `statepassing.py`
- Python wrapper: `wind.py`
- Torch extension cache must stay inside the project via `TORCH_EXTENSIONS_DIR`.
- The `statepassing` kernel is the A100-compatible default CUDA route. It
  requires CUDA, bf16 recurrent tensors, `seq_len % 16 == 0`, and `head_dim`
  divisible by 4.
- The `wind` kernel requires CUDA, bf16 tensors, `seq_len % 16 == 0`, and
  `head_dim` divisible by 16; on the current GPU1 A100/CUDA 12.6 stack it fails
  to assemble because `ptxas` does not recognize `movmatrix`.
