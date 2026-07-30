from __future__ import annotations

import unittest

import torch

from fast_slow_decay_gdn2 import (
    FastSlowDecayController,
    positive_causal_hazard_smooth,
)


class PositiveCausalHazardSmoothTest(unittest.TestCase):
    def test_constant_hazard_is_preserved_at_left_boundary(self) -> None:
        hazard = torch.full((2, 9, 3, 5), 0.7, dtype=torch.float64)
        kernel = torch.tensor(
            [
                [0.4, 0.3, 0.2, 0.1],
                [0.7, 0.1, 0.1, 0.1],
                [0.1, 0.2, 0.3, 0.4],
            ],
            dtype=torch.float64,
        )
        actual = positive_causal_hazard_smooth(hazard, kernel)
        torch.testing.assert_close(actual, hazard, rtol=0.0, atol=1e-12)

    def test_impulse_is_strictly_causal(self) -> None:
        hazard = torch.zeros(1, 8, 1, 1, dtype=torch.float64)
        hazard[:, 3] = 1.0
        kernel = torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.float64)
        actual = positive_causal_hazard_smooth(hazard, kernel)
        self.assertTrue(bool((actual[:, :3] == 0).all()))
        self.assertGreater(float(actual[:, 3].item()), 0.0)
        self.assertGreater(float(actual[:, 4].item()), 0.0)
        self.assertGreater(float(actual[:, 5].item()), 0.0)
        self.assertGreater(float(actual[:, 6].item()), 0.0)
        self.assertEqual(float(actual[:, 7].item()), 0.0)

    def test_positive_kernel_stays_inside_hazard_range(self) -> None:
        torch.manual_seed(7)
        hazard = torch.rand(2, 17, 4, 6, dtype=torch.float64)
        kernel = torch.rand(4, 5, dtype=torch.float64)
        actual = positive_causal_hazard_smooth(hazard, kernel)
        self.assertTrue(bool(torch.isfinite(actual).all()))
        self.assertGreaterEqual(float(actual.min()), float(hazard.min()) - 1e-12)
        self.assertLessEqual(float(actual.max()), float(hazard.max()) + 1e-12)


class FastSlowDecayControllerTest(unittest.TestCase):
    def test_external_identity_is_bitwise(self) -> None:
        torch.manual_seed(11)
        controller = FastSlowDecayController(
            heads=3,
            kernel_size=4,
            rho_init=0.1,
            current_weight_init=0.85,
        )
        log_decay = -torch.rand(2, 13, 3, 7, dtype=torch.float32)
        actual, diagnostics = controller(log_decay, mode="external_identity")
        self.assertTrue(torch.equal(actual, log_decay))
        self.assertEqual(float(diagnostics["gdn2_fast_slow_enabled"]), 0.0)

    def test_candidate_remains_valid_log_decay_and_reduces_tv(self) -> None:
        controller = FastSlowDecayController(
            heads=2,
            kernel_size=4,
            rho_init=0.5,
            current_weight_init=0.25,
        )
        hazard = torch.tensor(
            [0.1, 1.0, 0.1, 1.0, 0.1, 1.0, 0.1, 1.0],
            dtype=torch.float32,
        ).view(1, 8, 1, 1).expand(1, 8, 2, 3)
        actual, diagnostics = controller(-hazard, mode="positive_causal")
        self.assertTrue(bool((actual <= 0).all()))
        self.assertLess(
            float(diagnostics["gdn2_fast_slow_effective_tv"]),
            float(diagnostics["gdn2_fast_slow_raw_tv"]),
        )
        self.assertLess(float(diagnostics["gdn2_fast_slow_tv_ratio"]), 1.0)

    def test_controller_parameters_receive_finite_gradients(self) -> None:
        torch.manual_seed(19)
        controller = FastSlowDecayController(
            heads=2,
            kernel_size=4,
            rho_init=0.1,
            current_weight_init=0.85,
        ).double()
        log_decay = -torch.rand(
            2,
            11,
            2,
            5,
            dtype=torch.float64,
        )
        log_decay.requires_grad_(True)
        actual, _diagnostics = controller(log_decay, mode="positive_causal")
        weights = torch.linspace(0.1, 1.0, actual.numel(), dtype=actual.dtype).reshape_as(actual)
        loss = (actual * weights).sum()
        loss.backward()
        for parameter in (
            controller.kernel_logits,
            controller.rho_logit,
            log_decay,
        ):
            self.assertIsNotNone(parameter.grad)
            assert parameter.grad is not None
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(float(parameter.grad.abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
