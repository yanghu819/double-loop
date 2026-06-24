# fs-triton-backward-check-20260624-d9c2600

Purpose: verify that the official EqR FutureSeed mixer replacement now has a
real CUDA-accelerated scan backend, not just a Python or PyTorch prefix-scan
implementation.

This is an implementation viability check, not a model-quality result.

## Result

- Source SHA: `d9c2600e47d0b5e0950bf884ba8a155de3e6e198`
- Device: NVIDIA A800-SXM4-80GB on GPU1 only
- Torch: `2.7.0+cu126`
- Backend: Triton custom autograd scan
- Baseline: previous PyTorch prefix-scan fallback

Triton matched the prefix fallback numerically:

- max output diff: `4.470348358154297e-08`
- max input-gradient diff: `2.8421709430404007e-13`
- max forward decay-gradient diff: `5.4569682106375694e-12`
- max reverse decay-gradient diff: `4.547473508864641e-12`

After kernel warmup, the Triton path was about `12.2x` faster on the microcheck:

- Triton elapsed: `0.00526881217956543` seconds
- Prefix elapsed: `0.06410026550292969` seconds

Lesson: FutureSeed-as-cheap-bidirectional-mixer cannot rely on a Python loop.
The current implementation now has a genuine CUDA path via Triton, with the
prefix scan kept as a fallback.

