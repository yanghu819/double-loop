#!/usr/bin/env python3
from __future__ import annotations

import math
import unittest

import torch

from gain_budget_gdn2 import (
    fla_l2norm_fp32,
    gain_budgeted_gdn2_reference,
    project_erase_gate,
    rank_one_transition_sigma,
)


def transition_matrix(key: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    key = key.double()
    gate = gate.double()
    return torch.eye(key.numel(), dtype=torch.float64) - torch.outer(
        key,
        gate * key,
    )


class GainBudgetProjectionTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260729)

    def test_fla_l2norm_formula_and_dtype(self) -> None:
        x = torch.randn(4, 7, dtype=torch.bfloat16)
        actual = fla_l2norm_fp32(x)
        expected = x.float() * torch.rsqrt(
            x.float().square().sum(-1, keepdim=True) + 1e-6
        )
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_none_preserves_gate_values(self) -> None:
        k = fla_l2norm_fp32(torch.randn(3, 11))
        b = torch.rand_like(k)
        g = -torch.rand_like(k)
        actual, stats = project_erase_gate(k, b, g, mode="none")
        torch.testing.assert_close(actual, b.float(), rtol=0, atol=0)
        self.assertFalse(bool(stats["clipped"].any()))

    def test_delta_is_preserved(self) -> None:
        k = fla_l2norm_fp32(torch.randn(64, 32))
        b = torch.rand_like(k)
        g = -torch.rand_like(k) * 0.2
        actual, stats = project_erase_gate(
            k,
            b,
            g,
            mode="fixed_sigma",
            sigma_cap=1.001,
        )
        delta = (b * k.square()).sum(-1)
        actual_delta = (actual * k.square()).sum(-1)
        torch.testing.assert_close(actual_delta, delta, rtol=2e-5, atol=2e-6)
        self.assertLess(float(stats["delta_abs_error"].max()), 2e-6)

    def test_centered_component_only_shrinks(self) -> None:
        k = fla_l2norm_fp32(torch.randn(16, 24))
        b = torch.rand_like(k)
        g = -torch.full_like(k, 1e-4)
        actual, stats = project_erase_gate(
            k,
            b,
            g,
            mode="decay_funded",
            step_gain_cap=1.0,
        )
        self.assertTrue(bool(stats["clipped"].any()))
        self.assertTrue(bool((stats["scale"] >= 0).all()))
        self.assertTrue(bool((stats["scale"] <= 1).all()))
        self.assertTrue(bool((stats["effective_shear2"] <= stats["shear2"] + 1e-6).all()))
        torch.testing.assert_close(
            stats["effective_shear2"],
            stats["scale"].square() * stats["shear2"],
            rtol=3e-5,
            atol=2e-6,
        )
        self.assertTrue(bool((actual >= 0).all()))
        self.assertTrue(bool((actual <= 1).all()))
        self.assertEqual(actual.shape, b.shape)

    def test_spectral_bound_random(self) -> None:
        for _ in range(64):
            raw_k = torch.randn(13)
            k = fla_l2norm_fp32(raw_k)
            b = torch.rand_like(k) * 2.0
            g = -torch.rand_like(k) * 3.0
            projected, stats = project_erase_gate(
                k,
                b,
                g,
                mode="decay_funded",
                step_gain_cap=1.0,
                sigma_cap_max=3.0,
            )
            p = transition_matrix(k, projected)
            sigma = torch.linalg.matrix_norm(p, ord=2)
            tau = stats["tau_effective"].double().squeeze()
            self.assertLessEqual(float(sigma), float(tau) + 2e-5)
            d = torch.diag(g.double().exp())
            step_sigma = torch.linalg.matrix_norm(p @ d, ord=2)
            self.assertLessEqual(float(step_sigma), 1.0 + 3e-5)

    def test_closed_form_sigma_matches_svd(self) -> None:
        for _ in range(64):
            k = fla_l2norm_fp32(torch.randn(17))
            b = torch.rand_like(k) * 2.0
            delta = (b * k.square()).sum()
            mean = delta / k.square().sum()
            shear2 = k.square().sum() * (
                (b - mean).square() * k.square()
            ).sum()
            actual = rank_one_transition_sigma(delta, shear2)
            expected = torch.linalg.matrix_norm(
                transition_matrix(k, b),
                ord=2,
            )
            torch.testing.assert_close(
                actual.double(),
                expected,
                rtol=2e-6,
                atol=2e-6,
            )

    def test_inactive_projection_is_exact_identity(self) -> None:
        k = fla_l2norm_fp32(torch.randn(8, 16))
        b = torch.full_like(k, 0.4)
        g = -torch.rand_like(k)
        actual, stats = project_erase_gate(
            k,
            b,
            g,
            mode="fixed_sigma",
            sigma_cap=1.5,
        )
        torch.testing.assert_close(actual, b, rtol=0, atol=0)
        self.assertFalse(bool(stats["clipped"].any()))

    def test_zero_key_is_identity(self) -> None:
        k = torch.zeros(5, 9)
        b = torch.rand_like(k)
        g = -torch.rand_like(k)
        actual, stats = project_erase_gate(k, b, g)
        torch.testing.assert_close(actual, b, rtol=0, atol=0)
        self.assertFalse(bool(stats["live"].any()))

    def test_infeasible_budget_raises(self) -> None:
        k = torch.zeros(4)
        k[0] = 1.0
        b = torch.full_like(k, 3.0)
        g = -torch.full_like(k, 0.1)
        with self.assertRaisesRegex(ValueError, "infeasible"):
            project_erase_gate(
                k,
                b,
                g,
                mode="fixed_sigma",
                sigma_cap=1.0,
                infeasible_policy="raise",
            )

    def test_relax_reports_effective_budget(self) -> None:
        k = torch.zeros(4)
        k[0] = 1.0
        b = torch.full_like(k, 3.0)
        g = -torch.full_like(k, 0.1)
        projected, stats = project_erase_gate(
            k,
            b,
            g,
            mode="fixed_sigma",
            sigma_cap=1.0,
            infeasible_policy="relax",
        )
        self.assertTrue(bool(stats["infeasible"].item()))
        self.assertAlmostEqual(float(stats["tau_effective"]), 2.0, places=6)
        sigma = torch.linalg.matrix_norm(
            transition_matrix(k, projected),
            ord=2,
        )
        self.assertLessEqual(float(sigma), 2.0 + 1e-6)

    def test_tau_one_backward_is_finite(self) -> None:
        k = fla_l2norm_fp32(torch.randn(7, requires_grad=True))
        b = torch.rand(7, requires_grad=True)
        g = torch.zeros(7, requires_grad=True)
        projected, stats = project_erase_gate(
            k,
            b,
            g,
            mode="decay_funded",
            step_gain_cap=1.0,
        )
        loss = projected.square().sum() + stats["effective_shear2"].sum()
        grads = torch.autograd.grad(loss, (k, b, g), allow_unused=True)
        self.assertTrue(all(grad is not None for grad in grads))
        self.assertTrue(all(bool(torch.isfinite(grad).all()) for grad in grads if grad is not None))

    def test_active_projection_vjp_matches_finite_difference(self) -> None:
        key = fla_l2norm_fp32(torch.randn(8)).detach().requires_grad_(True)
        gate = torch.linspace(0.05, 0.95, 8).requires_grad_(True)
        decay = (-torch.linspace(0.003, 0.006, 8)).requires_grad_(True)
        cotangent = torch.randn(8)

        def objective(
            key_value: torch.Tensor,
            gate_value: torch.Tensor,
            decay_value: torch.Tensor,
        ) -> torch.Tensor:
            projected, stats = project_erase_gate(
                key_value,
                gate_value,
                decay_value,
                mode="decay_funded",
                step_gain_cap=1.0,
            )
            self.assertTrue(bool(stats["clipped"].item()))
            return (projected * cotangent).sum()

        analytic = torch.autograd.grad(
            objective(key, gate, decay),
            (key, gate, decay),
        )
        epsilon = 2e-4
        for name, argument_index, tensor, expected in (
            ("key", 0, key, analytic[0]),
            ("gate", 1, gate, analytic[1]),
            ("decay", 2, decay, analytic[2]),
        ):
            numerical = torch.empty_like(tensor)
            base = [key.detach(), gate.detach(), decay.detach()]
            for index in range(tensor.numel()):
                plus = [value.clone() for value in base]
                minus = [value.clone() for value in base]
                plus[argument_index][index] += epsilon
                minus[argument_index][index] -= epsilon
                numerical[index] = (
                    objective(*plus) - objective(*minus)
                ) / (2.0 * epsilon)
            with self.subTest(name=name):
                torch.testing.assert_close(
                    expected,
                    numerical,
                    rtol=6e-2,
                    atol=2e-2,
                )

    def test_allow_negative_happens_before_projection(self) -> None:
        t, kdim, vdim = 3, 8, 5
        q = torch.randn(t, kdim)
        k = torch.randn(t, kdim)
        v = torch.randn(t, vdim)
        g = -torch.rand(t, kdim)
        b = torch.rand(t, kdim)
        w = torch.rand(t, vdim)
        out_neg, state_neg = gain_budgeted_gdn2_reference(
            q,
            k,
            v,
            g,
            b,
            w,
            allow_neg_eigval=True,
            budget_mode="fixed_sigma",
            sigma_cap=1.05,
        )
        out_manual, state_manual = gain_budgeted_gdn2_reference(
            q,
            k,
            v,
            g,
            b * 2.0,
            w,
            allow_neg_eigval=False,
            budget_mode="fixed_sigma",
            sigma_cap=1.05,
        )
        torch.testing.assert_close(out_neg, out_manual, rtol=0, atol=0)
        torch.testing.assert_close(state_neg, state_manual, rtol=0, atol=0)

    def test_reference_matches_manual_recurrence(self) -> None:
        t, kdim, vdim = 5, 6, 4
        q_raw = torch.randn(t, kdim)
        k_raw = torch.randn(t, kdim)
        v = torch.randn(t, vdim)
        g = -torch.rand(t, kdim)
        b = torch.rand(t, kdim)
        w = torch.rand(t, vdim)
        h0 = torch.randn(kdim, vdim) * 0.1
        actual_o, actual_h, stats = gain_budgeted_gdn2_reference(
            q_raw,
            k_raw,
            v,
            g,
            b,
            w,
            initial_state_kv=h0,
            budget_mode="fixed_sigma",
            sigma_cap=1.03,
            return_stats=True,
        )
        q = fla_l2norm_fp32(q_raw)
        k = fla_l2norm_fp32(k_raw)
        b_eff, _ = project_erase_gate(
            k,
            b,
            g,
            mode="fixed_sigma",
            sigma_cap=1.03,
        )
        state = h0.float().clone()
        expected_o = []
        for token in range(t):
            state = g[token].exp()[:, None] * state
            state = state + torch.outer(
                k[token],
                w[token] * v[token] - (b_eff[token] * k[token]) @ state,
            )
            expected_o.append(q[token] @ state)
        torch.testing.assert_close(actual_o, torch.stack(expected_o))
        torch.testing.assert_close(actual_h, state)
        self.assertIn("scale", stats)

    def test_channel_permutation_equivariance(self) -> None:
        k = fla_l2norm_fp32(torch.randn(7, 15))
        b = torch.rand_like(k)
        g = -torch.rand_like(k)
        permutation = torch.randperm(k.shape[-1])
        actual, _ = project_erase_gate(k, b, g)
        permuted, _ = project_erase_gate(
            k[:, permutation],
            b[:, permutation],
            g[:, permutation],
        )
        torch.testing.assert_close(permuted, actual[:, permutation])

    def test_near_zero_key_remains_finite(self) -> None:
        k = torch.randn(12) * 1e-10
        b = torch.rand_like(k, requires_grad=True)
        g = -torch.rand_like(k)
        actual, stats = project_erase_gate(k, b, g, norm_floor=1e-12)
        loss = actual.square().sum()
        grad = torch.autograd.grad(loss, b)[0]
        self.assertTrue(bool(torch.isfinite(actual).all()))
        self.assertTrue(bool(torch.isfinite(grad).all()))
        self.assertFalse(bool(stats["live"].item()))

    def test_positive_log_decay_is_rejected(self) -> None:
        k = fla_l2norm_fp32(torch.randn(5))
        with self.assertRaisesRegex(ValueError, "non-positive"):
            project_erase_gate(k, torch.rand_like(k), torch.ones_like(k))

    def test_invalid_shapes_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "share shape"):
            project_erase_gate(
                torch.randn(2, 4),
                torch.randn(2, 3),
                torch.randn(2, 4),
            )

    def test_key_dimension_one_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "K >= 2"):
            project_erase_gate(
                torch.randn(2, 1),
                torch.rand(2, 1),
                -torch.rand(2, 1),
            )

    def test_nonfinite_hyperparameters_are_rejected(self) -> None:
        k = fla_l2norm_fp32(torch.randn(3, 8))
        b = torch.rand_like(k)
        g = -torch.rand_like(k)
        for argument in (
            {"norm_floor": math.nan},
            {"sigma_cap": math.nan},
            {"step_gain_cap": math.nan},
            {"sigma_cap_max": math.nan},
        ):
            with self.subTest(argument=argument):
                with self.assertRaises(ValueError):
                    project_erase_gate(k, b, g, **argument)

    def test_empty_reference_sequence(self) -> None:
        output, state = gain_budgeted_gdn2_reference(
            torch.empty(0, 8),
            torch.empty(0, 8),
            torch.empty(0, 6),
            torch.empty(0, 8),
            torch.empty(0, 8),
            torch.empty(0, 6),
        )
        self.assertEqual(output.shape, (0, 6))
        self.assertEqual(state.shape, (8, 6))

    def test_reference_explicit_fla_read_scale(self) -> None:
        t, kdim, vdim = 4, 8, 6
        arguments = (
            torch.randn(t, kdim),
            torch.randn(t, kdim),
            torch.randn(t, vdim),
            -torch.rand(t, kdim),
            torch.rand(t, kdim),
            torch.rand(t, vdim),
        )
        unscaled, state = gain_budgeted_gdn2_reference(*arguments)
        fla_scaled, scaled_state = gain_budgeted_gdn2_reference(
            *arguments,
            scale=kdim**-0.5,
        )
        torch.testing.assert_close(
            fla_scaled,
            unscaled * (kdim**-0.5),
        )
        torch.testing.assert_close(scaled_state, state, rtol=0, atol=0)

    def test_reference_output_and_state_shapes(self) -> None:
        output, state = gain_budgeted_gdn2_reference(
            torch.randn(4, 8),
            torch.randn(4, 8),
            torch.randn(4, 6),
            -torch.rand(4, 8),
            torch.rand(4, 8),
            torch.rand(4, 6),
        )
        self.assertEqual(output.shape, (4, 6))
        self.assertEqual(state.shape, (8, 6))
        self.assertEqual(state.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main(verbosity=2)
