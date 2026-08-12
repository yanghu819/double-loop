from __future__ import annotations

import unittest

import torch

from experiments.gdn2_diagnostics.receiver_live_contrast import (
    aggregate_discovery_direction,
    deterministic_board_permutation,
    paired_bootstrap_comparison,
    paired_bootstrap_interval,
    per_board_virtual_alpha_gradient,
    replay_gdn2_fp32,
    rms_match_signal_to_committed_edit,
    scaled_holdout_delta_v,
)


def _independent_replay(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Direct recurrence used only as an independent test oracle."""

    state = initial_state.float().clone()
    outputs = []
    edits = []
    for token in range(q.shape[1]):
        decayed = state * torch.exp(g[:, token].float()).unsqueeze(-1)
        erase_address = b[:, token].float() * k[:, token].float()
        erase = (erase_address.unsqueeze(-1) * decayed).sum(dim=-2)
        edit = w[:, token].float() * v[:, token].float() - erase
        state = decayed + k[:, token].float().unsqueeze(-1) * edit.unsqueeze(-2)
        output = (q[:, token].float().unsqueeze(-1) * state).sum(dim=-2)
        outputs.append(output)
        edits.append(edit)
    return torch.stack(outputs, dim=1), torch.stack(edits, dim=1), state


class ReceiverLiveContrastTensorTest(unittest.TestCase):
    def test_replay_is_exact_against_independent_token_loop(self) -> None:
        # Dyadic values, unit query scale, no normalization, and zero log-decay
        # make every operation exactly representable in FP32.
        q = torch.tensor(
            [
                [[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]],
                [[[0.5, 1.0]], [[1.0, 0.5]], [[0.5, 0.5]]],
            ]
        )
        k = torch.tensor(
            [
                [[[1.0, 0.0]], [[0.0, 1.0]], [[0.5, 0.5]]],
                [[[0.0, 1.0]], [[1.0, 0.0]], [[0.5, 0.5]]],
            ]
        )
        v = torch.tensor(
            [
                [[[1.0, 0.5]], [[0.25, 1.0]], [[0.5, -0.5]]],
                [[[0.5, 0.25]], [[1.0, -0.5]], [[0.25, 0.5]]],
            ]
        )
        g = torch.zeros(2, 3, 1, 2)
        b = torch.tensor(
            [
                [[[0.5, 0.0]], [[0.0, 0.25]], [[0.5, 0.5]]],
                [[[0.0, 0.5]], [[0.25, 0.0]], [[0.5, 0.25]]],
            ]
        )
        w = torch.tensor(
            [
                [[[0.5, 1.0]], [[1.0, 0.5]], [[0.5, 0.5]]],
                [[[1.0, 0.5]], [[0.5, 1.0]], [[0.25, 0.5]]],
            ]
        )
        initial_state = torch.tensor(
            [
                [[[0.5, 0.25], [0.0, 0.5]]],
                [[[0.25, 0.0], [0.5, 0.25]]],
            ]
        )

        expected_output, expected_edit, expected_state = _independent_replay(
            q=q,
            k=k,
            v=v,
            g=g,
            b=b,
            w=w,
            initial_state=initial_state,
        )
        replay = replay_gdn2_fp32(
            q=q,
            k=k,
            v=v,
            g=g,
            b=b,
            w=w,
            initial_state=initial_state,
            scale=1.0,
            use_qk_l2norm_in_kernel=False,
            board_shuffle_seed=19,
        )

        self.assertTrue(torch.equal(replay.raw_output, expected_output))
        self.assertTrue(torch.equal(replay.committed_edit, expected_edit))
        self.assertTrue(torch.equal(replay.final_state, expected_state))
        expected_frozen = (q.unsqueeze(-1) * initial_state[:, None]).sum(dim=-2)
        first_decayed = initial_state * torch.exp(g[:, 0]).unsqueeze(-1)
        first_pre_write_output = (
            q[:, 0].unsqueeze(-1) * first_decayed
        ).sum(dim=-2)
        self.assertFalse(torch.equal(replay.raw_output[:, 0], first_pre_write_output))
        self.assertTrue(torch.equal(replay.frozen_only, expected_frozen))
        self.assertTrue(
            torch.equal(replay.full_contrast, expected_frozen - expected_output)
        )

    def test_board_permutation_is_deterministic_derangement(self) -> None:
        first = deterministic_board_permutation(17, seed=314159)
        second = deterministic_board_permutation(17, seed=314159)

        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(first.sort().values, torch.arange(17)))
        self.assertFalse(bool((first == torch.arange(17)).any()))
        with self.assertRaises(ValueError):
            deterministic_board_permutation(1, seed=7)

    def test_virtual_gradient_matches_autograd_and_finite_difference(self) -> None:
        generator = torch.Generator().manual_seed(23)
        shape = (3, 4, 2, 5)
        v_gradient = torch.randn(shape, generator=generator)
        signal = torch.randn(shape, generator=generator)
        committed_edit = torch.randn(shape, generator=generator)
        rho = 0.2
        matched = rms_match_signal_to_committed_edit(signal, committed_edit)
        expected = per_board_virtual_alpha_gradient(
            v_grad=v_gradient,
            signal=signal,
            committed_edit=committed_edit,
            rho=rho,
        )

        alpha = torch.zeros(3, 2, 5, dtype=torch.float64, requires_grad=True)
        objective = (
            rho
            * torch.tanh(alpha)[:, None]
            * matched.double()
            * v_gradient.double()
        ).sum()
        objective.backward()
        torch.testing.assert_close(
            alpha.grad,
            expected.double(),
            rtol=1e-6,
            atol=1e-7,
        )

        direction = torch.randn((3, 2, 5), generator=generator, dtype=torch.float64)

        def directional_objective(step: float) -> torch.Tensor:
            return (
                rho
                * torch.tanh(direction[:, None] * step)
                * matched.double()
                * v_gradient.double()
            ).sum()

        step = 1e-5
        finite_difference = (
            directional_objective(step) - directional_objective(-step)
        ) / (2.0 * step)
        expected_directional = (expected.double() * direction).sum()
        torch.testing.assert_close(
            finite_difference,
            expected_directional,
            rtol=1e-6,
            atol=1e-7,
        )

        aggregate = aggregate_discovery_direction(expected.float())
        directional_derivative = (
            aggregate.mean_gradient * aggregate.descent_direction
        ).sum()
        self.assertLess(float(directional_derivative), 0.0)

    def test_holdout_delta_has_exact_committed_rms_budget(self) -> None:
        generator = torch.Generator().manual_seed(41)
        shape = (4, 5, 2, 3)
        signal = torch.randn(shape, generator=generator)
        committed_edit = torch.randn(shape, generator=generator)
        w = torch.sigmoid(torch.randn(shape, generator=generator))
        direction = torch.randn((2, 3), generator=generator)
        epsilon = 0.01

        result = scaled_holdout_delta_v(
            signal=signal,
            committed_edit=committed_edit,
            w=w,
            direction=direction,
            epsilon=epsilon,
        )
        direct_committed = w * result.delta_v
        direct_local_rms = direct_committed.square().mean(
            dim=(-2, -1), keepdim=True
        ).sqrt()
        baseline_local_rms = committed_edit.square().mean(
            dim=(-2, -1), keepdim=True
        ).sqrt()
        direct_rms = direct_committed.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
        baseline_rms = committed_edit.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()

        self.assertTrue(torch.equal(result.committed_delta, direct_committed))
        torch.testing.assert_close(
            direct_local_rms,
            epsilon * baseline_local_rms,
            rtol=2e-6,
            atol=1e-8,
        )
        torch.testing.assert_close(
            direct_rms,
            epsilon * baseline_rms,
            rtol=2e-6,
            atol=1e-8,
        )
        torch.testing.assert_close(result.achieved_rms, result.target_rms)

        opposite = scaled_holdout_delta_v(
            signal=signal,
            committed_edit=committed_edit,
            w=w,
            direction=-direction,
            epsilon=epsilon,
        )
        self.assertTrue(torch.equal(opposite.delta_v, -result.delta_v))
        self.assertTrue(torch.equal(opposite.achieved_rms, result.achieved_rms))

        bf16_reference = torch.randn(shape, generator=generator).bfloat16()
        quantized = scaled_holdout_delta_v(
            signal=signal,
            committed_edit=committed_edit,
            w=w,
            direction=direction,
            epsilon=epsilon,
            storage_reference=bf16_reference,
        )
        self.assertEqual(quantized.modified_v.dtype, torch.bfloat16)
        self.assertTrue(
            torch.equal(
                quantized.delta_v,
                quantized.modified_v.float() - bf16_reference.float(),
            )
        )
        self.assertTrue(
            torch.equal(quantized.committed_delta, w * quantized.delta_v)
        )
        quantized_relative_error = (
            (quantized.achieved_rms - quantized.target_rms).abs()
            / quantized.target_rms.clamp_min(1e-12)
        )
        self.assertLessEqual(float(quantized_relative_error.max()), 0.10)

    def test_paired_bootstrap_is_seed_deterministic(self) -> None:
        differences = torch.tensor([0.01, 0.03, -0.01, 0.04, 0.02, 0.05])
        first = paired_bootstrap_interval(
            differences,
            seed=2718,
            samples=2_000,
            batch_size=73,
        )
        second = paired_bootstrap_interval(
            differences,
            seed=2718,
            samples=2_000,
            batch_size=73,
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first.observed_mean, float(differences.mean()), places=8)

        comparison = paired_bootstrap_comparison(
            control=torch.tensor([0.5, 0.4, 0.7, 0.6]),
            candidate=torch.tensor([0.4, 0.3, 0.65, 0.55]),
            objective="minimize",
            seed=9,
            samples=1_000,
        )
        self.assertGreater(comparison.observed_mean, 0.0)
        self.assertEqual(comparison.probability_positive, 1.0)


if __name__ == "__main__":
    unittest.main()
