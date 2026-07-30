from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "probe_gdn2_cell_order.py"
SPEC = importlib.util.spec_from_file_location("probe_gdn2_cell_order", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_orders_are_bijections() -> None:
    orders = MODULE.cell_orders(9, 52081)
    expected = list(range(81))
    assert set(orders) == {
        "row_major",
        "reverse",
        "column_major",
        "box_major",
        "random_fixed",
    }
    for order in orders.values():
        assert sorted(order.tolist()) == expected


def test_canonicalize_inverts_sequence_permutation() -> None:
    source = torch.arange(2 * 81 * 3).reshape(2, 81, 3)
    for permutation in MODULE.cell_orders(9, 52081).values():
        sequence = source[:, permutation]
        restored = MODULE.canonicalize(sequence, permutation)
        torch.testing.assert_close(restored, source, rtol=0.0, atol=0.0)
