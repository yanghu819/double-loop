import json
import sys

import torch


repo = "/huyang2/double-loop/.worktrees/pdiag027-adaptiverms-e6a3f24-20260711T025509Z"
sys.path.insert(0, f"{repo}/experiments/rwkv_fs_sudoku")

from study_rwkv_futureseed_loop import FutureSeedLoopSudoku  # noqa: E402


if not torch.cuda.is_available():
    raise RuntimeError("P-DIAG-027 equivalence test is CUDA-only")


def make_model(norm_mode: str) -> FutureSeedLoopSudoku:
    return FutureSeedLoopSudoku(
        d_model=64,
        layers=2,
        heads=4,
        head_dim=16,
        channel_mult=2,
        l_cycles=1,
        max_loops=2,
        lambda_=0.95,
        loop_update_mode="fixed",
        loop_update_gate_init=0.95,
        future_seed_scale=1.0,
        future_seed_decay=0.0,
        future_seed_update="fixed",
        future_seed_norm_mode=norm_mode,
        loop_feedback_scale=0.0,
        loop_feedback_detach=False,
        loop_feedback_corrupt_prob=0.0,
        loop_feedback_corrupt_mix=0.0,
        loop_feedback_corrupt_mode="random_token",
        loop_time_scale=0.0,
        scratch_mode="none",
        scratch_scale=0.0,
        scratch_noise_scale=0.0,
        scratch_gauss_projections=0,
        scratch_gate_bias=-2.0,
        scratch_decay_bias=2.0,
        hidden_agg_noise_scale=0.0,
        hidden_agg_noise_temp=1.0,
        hidden_agg_noise_detach=True,
        hidden_agg_noise_mode="gumbel",
        hidden_agg_noise_topk=8,
        hidden_agg_noise_max_norm=0.0,
        activation_checkpoint=False,
        rwkv_kernel="statepassing",
        backbone="gdn",
        gdn_mode="triton_recurrent",
        gdn_expand_v=1.0,
        gdn_use_short_conv=False,
        gdn_conv_size=4,
        gdn_allow_neg_eigval=False,
    )


torch.manual_seed(20260711)
unit = make_model("unit")
adaptive = make_model("adaptive_rms")
missing, unexpected = adaptive.load_state_dict(unit.state_dict(), strict=False)
expected_missing = {
    "reasoner.future_seed_norm_slope",
    "reasoner.future_seed_norm_bias",
}
if set(missing) != expected_missing or unexpected:
    raise RuntimeError(f"state load mismatch: missing={missing}, unexpected={unexpected}")

device = torch.device("cuda:0")
unit = unit.to(device).eval()
adaptive = adaptive.to(device).eval()
inputs = torch.randint(0, 10, (3, 81), device=device)

unit_logits, unit_trace = unit.forward_trace(inputs, loops=2, noise_scale=0.0)
adaptive_logits, adaptive_trace = adaptive.forward_trace(inputs, loops=2, noise_scale=0.0)
output_max_abs = max(
    float((left - right).abs().max().detach().cpu())
    for left, right in zip(unit_logits, adaptive_logits)
)

unit_loss = sum(logits.float().square().mean() for logits in unit_logits)
adaptive_loss = sum(logits.float().square().mean() for logits in adaptive_logits)
unit_loss.backward()
adaptive_loss.backward()
unit_grads = {name: param.grad for name, param in unit.named_parameters() if param.grad is not None}
adaptive_grads = {name: param.grad for name, param in adaptive.named_parameters() if param.grad is not None}
common_grad_max_abs = max(
    float((grad - adaptive_grads[name]).abs().max().detach().cpu())
    for name, grad in unit_grads.items()
)

gain_means = [float(trace["fs_norm_gain_mean"].detach().cpu()) for trace in adaptive_trace]
gain_stds = [float(trace["fs_norm_gain_std"].detach().cpu()) for trace in adaptive_trace]
result = {
    "cuda_device": torch.cuda.get_device_name(0),
    "output_max_abs": output_max_abs,
    "common_grad_max_abs": common_grad_max_abs,
    "adaptive_gain_means": gain_means,
    "adaptive_gain_stds": gain_stds,
    "missing_parameters": sorted(missing),
}
print(json.dumps(result, sort_keys=True))

if output_max_abs > 1e-6 or common_grad_max_abs > 1e-5:
    raise RuntimeError(f"adaptive RMS zero-init is not equivalent: {result}")
if any(abs(value - 1.0) > 1e-7 for value in gain_means) or any(value != 0.0 for value in gain_stds):
    raise RuntimeError(f"adaptive RMS gain did not start at identity: {result}")
