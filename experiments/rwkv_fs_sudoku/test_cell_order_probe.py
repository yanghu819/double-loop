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


def test_html_accepts_strict_fla_runtime_schema() -> None:
    loop_metrics = {
        "label_exact": 0.1,
        "blank_acc": 0.5,
        "valid_sudoku": 0.1,
    }
    payload = {
        "orders": {
            "row_major": {
                "loops": {"loop1": loop_metrics},
                "blank_prediction_change_vs_row": {"loop1": 0.0},
                "wall_sec": 1.0,
            }
        },
        "loops": [1],
        "decision": {
            "address_hypothesis_supported": True,
            "label": "supported",
            "reason": "paired traversal degraded",
        },
        "checkpoint_sha256": "a" * 64,
        "source_sha": "b" * 40,
        "runtime": {
            "strict": True,
            "layers": [
                {
                    "class": "fla.layers.gdn2.GatedDeltaNet2",
                    "execution_path": "official_layer_forward",
                }
            ],
        },
        "max_paired_encoding_roundtrip_abs": 0.0,
        "board_size": 9,
    }
    rendered = MODULE.build_html(payload, [])
    assert "fla.layers.gdn2.GatedDeltaNet2/official_layer_forward" in rendered
