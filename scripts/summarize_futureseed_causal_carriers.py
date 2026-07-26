#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BACKBONES = ("rwkv", "gdn", "gdn2", "kda")
INTERNAL = {
    "rwkv": "rwkv7",
    "gdn": "fla_gdn",
    "gdn2": "gdn2",
    "kda": "kda",
}
FLA_CLASSES = {
    "gdn": "fla.layers.gated_deltanet.GatedDeltaNet",
    "gdn2": "fla.layers.gdn2.GatedDeltaNet2",
    "kda": "fla.layers.kda.KimiDeltaAttention",
}
FLA_PREFLIGHT = {
    "gdn": {
        "adapter": "fla_gdn_adapter",
        "reference": "fla_gdn_reference",
        "stack": "fla_gdn_futureseed_stack",
        "autograd_node": "ChunkGatedDeltaRuleFunctionBackward",
    },
    "gdn2": {
        "adapter": "gdn2_adapter",
        "reference": "gdn2_reference",
        "stack": "gdn2_futureseed_stack",
        "autograd_node": "ChunkGDN2FunctionBackward",
    },
    "kda": {
        "adapter": "kda_adapter",
        "reference": "kda_reference",
        "stack": "kda_futureseed_stack",
        "autograd_node": "ChunkKDAFunctionBackward",
    },
}
LABELS = {
    "rwkv": "RWKV7 TimeMix",
    "gdn": "GDN",
    "gdn2": "GDN2",
    "kda": "KDA",
}
COLORS = {
    "rwkv": "#75418a",
    "gdn": "#16705a",
    "gdn2": "#b34a3d",
    "kda": "#286aa6",
}
RANGES = {
    "b46_50": "46-50",
    "b51_55": "51-55",
    "b56_64": "56-64",
}
ARG_PATH_DIFFERENCES = {"future_seed_scale", "out_dir", "train_checkpoint_dir"}
EXPECTED_SHA = "6e51f067a18d936bda7f3d588b7b3f32625b2b18"
FS_RUN_START_UTC = "2026-07-24 15:40:20"
EXPECTED_DATA_SHA256 = {
    "/huyang2/double-loop/data/sudoku-extreme-full/train/all__inputs.npy":
        "979f0f27411dfde97b725082cd29da03bea7c3b1c6b2bd2b15bfe95b76549cbc",
    "/huyang2/double-loop/data/sudoku-extreme-full/train/all__labels.npy":
        "84e8a40dbb12d032eb9eba601c378d8388ea949430e07c00e80bc0947a59e48a",
    "/huyang2/double-loop/data/sudoku-extreme-full/test/all__inputs.npy":
        "cbc2ab9f1743d79da9c0c26273460140bf542a5fb3229cfdd80575615163bcd5",
    "/huyang2/double-loop/data/sudoku-extreme-full/test/all__labels.npy":
        "3d0bb8e73ecc5681675803c1cb518ebb4e771dac93a6741256a7a85ab8e29c2b",
}
EXPECTED_TEST_REALPATH = (
    "/huyang2/double-loop/official_eqr_sudoku_repro_20260623/"
    "data/sudoku-extreme-1k-aug-1000/test"
)
LOG_RE = re.compile(r"step=(?P<step>\d+)\s+ce=(?P<ce>[-+0-9.eE]+)")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_key_value_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or not value:
            raise AssertionError(f"Malformed key/value line in {path}: {line!r}")
        values[key] = value
    return values


def only_result(run_dir: Path) -> Path:
    paths = sorted((run_dir / "output").glob("futureseed_loop_seed*.json"))
    if len(paths) != 1:
        raise AssertionError(f"Expected one result JSON under {run_dir}, got {paths}")
    return paths[0]


def only_case(run_dir: Path) -> Path:
    paths = sorted((run_dir / "output").glob("futureseed_loop_case_seed*.html"))
    if len(paths) != 1:
        raise AssertionError(f"Expected one primary case under {run_dir}, got {paths}")
    return paths[0]


def patch_files(run_dir: Path) -> list[str]:
    text = (run_dir / "source.patch").read_text(encoding="utf-8", errors="replace")
    return sorted(set(re.findall(r"^diff --git a/(.+?) b/", text, flags=re.MULTILINE)))


def train_curve(log_path: Path) -> list[dict[str, float]]:
    values: dict[int, float] = {}
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = LOG_RE.search(line)
            if match:
                values[int(match.group("step"))] = float(match.group("ce"))
    return [{"step": step, "ce": values[step]} for step in sorted(values)]


def primary_case(run_dir: Path) -> dict[str, Any]:
    path = only_case(run_dir)
    text = path.read_text(encoding="utf-8")
    puzzle = re.search(
        r'<div class="panel"><h3>puzzle</h3>(.*?)<div class="panel"><h3>solution</h3>',
        text,
        flags=re.DOTALL,
    )
    if puzzle is None:
        raise AssertionError(f"Could not parse puzzle from {path}")
    wrong: dict[str, int] = {}
    for chunk in text.split('<div class="panel"><h3>loop ')[1:]:
        loop = re.match(r"(\d+)</h3>", chunk)
        if loop:
            wrong[f"loop{int(loop.group(1))}"] = chunk.count('class="cell wrong"')
    return {
        "filename": path.name,
        "puzzle_sha256": hashlib.sha256(puzzle.group(1).encode("utf-8")).hexdigest(),
        "wrong_by_loop": wrong,
    }


def future_trace(eval_clean: dict[str, Any], loop: int) -> dict[str, float]:
    row = eval_clean[f"loop{loop}/future_seed"]
    return {
        "fs_gate_mean": float(row["fs_gate_mean"]),
        "fs_state_norm": float(row["fs_state_norm"]),
        "fs_raw_rms_mean": float(row.get("fs_raw_rms_mean", 0.0)),
    }


def load_arm(runs_root: Path, run_name: str, public_name: str) -> dict[str, Any]:
    run_dir = runs_root / run_name
    result = read_json(only_result(run_dir))
    args = result["args"]
    metrics = result["metrics"]
    train = metrics["train"]
    config = read_json(run_dir / "config.json")
    score = read_json(run_dir / "score.json")
    checkpoint = read_json(run_dir / "output" / "checkpoint_eval_step000500.json")
    case = primary_case(run_dir)
    source_snapshot = read_key_value_file(run_dir / "source_snapshot.ref")
    case_banks: dict[str, str] = {}
    for key in RANGES:
        index = run_dir / "output" / "case_bank" / f"official_{key}" / "index.html"
        cases = index.parent / "cases.json"
        if not index.is_file() or not cases.is_file():
            raise AssertionError(f"Missing {key} case bank under {run_dir}")
        case_banks[key] = (
            f"../{run_name}/output/case_bank/official_{key}/index.html"
        )
    loops = []
    for loop in range(1, 6):
        row = metrics["eval_clean"][f"loop{loop}"]
        loops.append(
            {
                "loop": loop,
                "exact": float(row["label_exact"]),
                "blank_acc": float(row["blank_acc"]),
                "future_seed": future_trace(metrics["eval_clean"], loop),
            }
        )
    ranges: dict[str, Any] = {}
    for key in RANGES:
        source = metrics["official_eval_by_blank_range"][key]
        ranges[key] = {
            "blank_range": source["blank_range"],
            "eval_n": int(source["eval_n"]),
            "loops": [
                {
                    "loop": loop,
                    "exact": float(source["eval_clean"][f"loop{loop}"]["label_exact"]),
                    "blank_acc": float(source["eval_clean"][f"loop{loop}"]["blank_acc"]),
                    "future_seed": future_trace(source["eval_clean"], loop),
                }
                for loop in range(1, 6)
            ],
        }
    elapsed = float(checkpoint["elapsed_sec"])
    arm = {
        "public_name": public_name,
        "label": LABELS[public_name],
        "run_name": run_name,
        "run_dir": run_dir,
        "git_sha": config["git_sha"],
        "git_dirty": bool(config.get("git_dirty")),
        "source_head": (run_dir / "source_HEAD.txt").read_text(encoding="utf-8").strip(),
        "source_patch_files": patch_files(run_dir),
        "source_snapshot": source_snapshot,
        "args": args,
        "parameter_count": int(train["parameter_count"]),
        "train_ce": float(checkpoint["train"]["ce_loss"]),
        "train_sec_per_step": elapsed / 500.0,
        "peak_allocated_mib": float(train["cuda_max_memory_allocated_mb"]),
        "curve": train_curve(run_dir / "logs" / "run.log"),
        "loops": loops,
        "ranges": ranges,
        "case": case,
        "case_banks": case_banks,
        "runtime": train["backbone_runtime"],
        "fla_runtime": train.get("fla_runtime"),
        "optimizer_runtime": train["optimizer_runtime"],
        "data_identity": {
            "data_source": train["data_source"],
            "effective_batch": int(train["effective_batch"]),
            "official_train_size": int(train["official_train_size"]),
            "official_eval_size": int(train["official_eval_size"]),
            "official_sudoku_data_dir": train["official_sudoku_data_dir"],
            "official_sudoku_train_split": train["official_sudoku_train_split"],
            "official_sudoku_eval_split": train["official_sudoku_eval_split"],
            "official_sudoku_train_indices": train["official_sudoku_train_indices"],
            "eval_official": metrics["eval_official"],
            "official_eval_seed_offset": int(args["official_eval_seed_offset"]),
        },
        "fixed_holes53": checkpoint["eval_by_holes"]["holes53"],
    }
    if score["git_sha"] != arm["git_sha"]:
        raise AssertionError(f"{run_name} score source SHA differs")
    expected_score = arm["loops"][-1]["exact"]
    if score["score_key"] != "metrics.eval_clean.loop5.label_exact":
        raise AssertionError(f"{run_name} score key differs: {score['score_key']}")
    if abs(float(score["score"]) - expected_score) > 1e-9:
        raise AssertionError(f"{run_name} score file differs from result")
    return arm


def normalized_args(args: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in args.items() if key not in ARG_PATH_DIFFERENCES}


def assert_zero_future_seed(arm: dict[str, Any]) -> None:
    rows = arm["loops"] + [
        row
        for range_row in arm["ranges"].values()
        for row in range_row["loops"]
    ]
    for row in rows:
        trace = row["future_seed"]
        if abs(trace["fs_gate_mean"]) > 1e-8 or abs(trace["fs_state_norm"]) > 1e-8:
            raise AssertionError(
                f"{arm['run_name']} has active FutureSeed at loop {row['loop']}: {trace}"
            )


def assert_active_future_seed(arm: dict[str, Any]) -> None:
    rows = arm["loops"] + [
        row
        for range_row in arm["ranges"].values()
        for row in range_row["loops"]
    ]
    for row in rows:
        trace = row["future_seed"]
        if trace["fs_gate_mean"] <= 0 or trace["fs_state_norm"] <= 0:
            raise AssertionError(
                f"{arm['run_name']} did not activate FutureSeed at loop {row['loop']}: {trace}"
            )


def validate_arm(arm: dict[str, Any], expected_scale: float) -> None:
    args = arm["args"]
    if arm["git_sha"] != EXPECTED_SHA or arm["source_head"] != EXPECTED_SHA:
        raise AssertionError(f"{arm['run_name']} source is not {EXPECTED_SHA}")
    if arm["git_dirty"] or arm["source_patch_files"]:
        raise AssertionError(f"{arm['run_name']} source tree is not clean")
    snapshot = arm["source_snapshot"]
    if snapshot.get("git_sha") != EXPECTED_SHA:
        raise AssertionError(f"{arm['run_name']} source snapshot SHA differs")
    if not re.fullmatch(r"[0-9a-f]{64}", snapshot.get("sha256", "")):
        raise AssertionError(f"{arm['run_name']} source snapshot hash is invalid")
    expected_snapshot_path = (
        f"/huyang2/double-loop/runs/{arm['run_name']}/source_snapshot.tar.gz"
    )
    if snapshot.get("remote_path") != expected_snapshot_path:
        raise AssertionError(f"{arm['run_name']} source snapshot path differs")
    if args["backbone"] != INTERNAL[arm["public_name"]]:
        raise AssertionError(f"{arm['run_name']} resolved to the wrong backbone")
    expected = {
        "steps": 500,
        "batch": 32,
        "grad_accum_steps": 4,
        "d_model": 192,
        "layers": 10,
        "heads": 6,
        "head_dim": 32,
        "channel_mult": 4,
        "max_loops": 5,
        "loop_loss": "all",
        "lr": 0.0015,
        "weight_decay": 0.001,
        "optimizer_contract": "rwkv7_decay_groups",
        "shared_shell_init_seed": 52,
        "future_seed_scale": expected_scale,
        "future_seed_decay": 0.0,
        "future_seed_update": "fixed",
        "future_seed_norm_mode": "unit",
        "noise_scale": 0.0,
        "loop_feedback_scale": 0.0,
        "loop_time_scale": 0.0,
        "scratch_mode": "none",
        "hidden_agg_noise_scale": 0.0,
        "gdn_expand_v": 1.0,
        "gdn_use_short_conv": 1,
        "gdn_conv_size": 4,
        "forward_dtype": "bfloat16",
        "seed": 52,
        "eval_n": 512,
        "hole_stages": "46-50:100,51-55:400",
        "official_eval_blank_ranges": "46-50,51-55,56-64",
    }
    for key, value in expected.items():
        if args.get(key) != value:
            raise AssertionError(
                f"{arm['run_name']} config mismatch for {key}: {args.get(key)!r} != {value!r}"
            )
    if arm["data_identity"]["data_source"] != "official_sudoku":
        raise AssertionError(f"{arm['run_name']} did not use official Sudoku")
    if arm["data_identity"]["effective_batch"] != 128:
        raise AssertionError(f"{arm['run_name']} effective batch differs")
    if arm["runtime"].get("silent_fallback_allowed"):
        raise AssertionError(f"{arm['run_name']} allowed silent fallback")
    if arm["optimizer_runtime"].get("contract") != "rwkv7_decay_groups":
        raise AssertionError(f"{arm['run_name']} optimizer differs")
    if arm["public_name"] == "rwkv":
        runtime = arm["runtime"]
        if runtime.get("implementation") != "official_rwkv7_timemix_with_explicit_state_io":
            raise AssertionError(f"{arm['run_name']} is not official RWKV7")
        if runtime.get("official_source_commit") != "952102498e9ed367ea0a59ee64106916d474d30f":
            raise AssertionError(f"{arm['run_name']} RWKV source differs")
        if runtime.get("statepassing_cuda_sha256") != "59a90a0521b1851da17c008c685f959d586af1a7d28056b29a7478ab92c1c892":
            raise AssertionError(f"{arm['run_name']} RWKV CUDA kernel differs")
    else:
        fla = arm["fla_runtime"]
        if not fla or not fla.get("strict"):
            raise AssertionError(f"{arm['run_name']} did not use strict official FLA")
        if fla.get("fla_source_sha") != "fe8fce9fc6984f22905f54cfa885dce1502baf26":
            raise AssertionError(f"{arm['run_name']} FLA source differs")
        if not fla.get("backend_dispatch_disabled") or fla.get("conv_backend") != "triton":
            raise AssertionError(f"{arm['run_name']} FLA backend contract differs")
        layers = fla.get("layers", [])
        if len(layers) != 10:
            raise AssertionError(f"{arm['run_name']} did not report all official FLA layers")
        for layer in layers:
            if layer.get("class") != FLA_CLASSES[arm["public_name"]]:
                raise AssertionError(
                    f"{arm['run_name']} used unexpected FLA class: {layer.get('class')}"
                )
            if set(layer.get("conv_backends", {}).values()) != {"triton"}:
                raise AssertionError(
                    f"{arm['run_name']} used a non-Triton short convolution"
                )
    if arm["fixed_holes53"]["blank_range"] != [53, 53]:
        raise AssertionError(f"{arm['run_name']} fixed-h53 evaluator differs")
    if int(arm["fixed_holes53"]["eval_n"]) != 512:
        raise AssertionError(f"{arm['run_name']} fixed-h53 eval size differs")


def validate_pair(fs: dict[str, Any], nofs: dict[str, Any]) -> None:
    validate_arm(fs, 1.0)
    validate_arm(nofs, 0.0)
    differing_args = {
        key
        for key in set(fs["args"]) | set(nofs["args"])
        if fs["args"].get(key) != nofs["args"].get(key)
    }
    if differing_args != ARG_PATH_DIFFERENCES:
        raise AssertionError(
            f"{fs['public_name']} actual arg differences are {sorted(differing_args)}, "
            f"expected {sorted(ARG_PATH_DIFFERENCES)}"
        )
    if normalized_args(fs["args"]) != normalized_args(nofs["args"]):
        left = normalized_args(fs["args"])
        right = normalized_args(nofs["args"])
        differing = sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))
        raise AssertionError(f"{fs['public_name']} paired args differ: {differing}")
    if fs["parameter_count"] != nofs["parameter_count"]:
        raise AssertionError(f"{fs['public_name']} parameter count changed")
    for key in ("runtime", "fla_runtime", "optimizer_runtime", "data_identity"):
        if fs[key] != nofs[key]:
            raise AssertionError(f"{fs['public_name']} paired {key} differs")
    if fs["case"]["puzzle_sha256"] != nofs["case"]["puzzle_sha256"]:
        raise AssertionError(f"{fs['public_name']} primary puzzle differs")
    assert_active_future_seed(fs)
    assert_zero_future_seed(nofs)


def validate_preflight(
    fs_comparison: dict[str, Any],
    nofs_preflight: Path,
    causal_contract: dict[str, Any],
) -> dict[str, Any]:
    fla = read_json(nofs_preflight / "fla_kernel_gate.json")
    rwkv = read_json(nofs_preflight / "rwkv7_official_frontend_gate.json")
    backbone = read_json(nofs_preflight / "backbone_contract.json")
    shared = read_json(nofs_preflight / "shared_shell_initialization_gate.json")
    expected_device = "NVIDIA A800-SXM4-80GB"
    for name, payload in (
        ("FLA", fla),
        ("RWKV7", rwkv),
        ("backbone", backbone),
        ("shared-shell", shared),
        ("causal-contract", causal_contract),
    ):
        if payload.get("device") != expected_device:
            raise AssertionError(
                f"{name} preflight used {payload.get('device')!r}, expected {expected_device!r}"
            )
    if rwkv.get("cuda_visible_devices") != "0":
        raise AssertionError("RWKV7 preflight did not use CUDA_VISIBLE_DEVICES=0")
    if causal_contract.get("cuda_visible_devices") != "0":
        raise AssertionError("Causal contract did not use CUDA_VISIBLE_DEVICES=0")
    if fla.get("requested_backbone") != "all" or fla.get("requested_check") != "all":
        raise AssertionError("FLA preflight did not run the complete all-backbone gate")
    if shared.get("status") != "PASS" or int(shared.get("differing_parameter_tensors", -1)) != 0:
        raise AssertionError("Fresh shared-shell initialization gate failed")
    if shared["hashes"] != fs_comparison["shared_shell_gate"]["hashes"]:
        raise AssertionError("Fresh shared-shell parameter hashes differ from FutureSeed-on preflight")
    if fla["provenance"]["source_file_hashes"] != fs_comparison["fla_gate"]["provenance"]["source_file_hashes"]:
        raise AssertionError("Fresh official FLA source hashes differ")
    if fla["provenance"]["wheel_sha256"] != fs_comparison["fla_gate"]["provenance"]["wheel_sha256"]:
        raise AssertionError("Fresh official FLA wheel differs")
    if not fla["provenance"].get("backend_dispatch_disabled"):
        raise AssertionError("FLA backend dispatch was not disabled")
    if fla["provenance"].get("conv_backend") != "triton":
        raise AssertionError("FLA preflight did not require Triton short convolution")
    for public_name, contract in FLA_PREFLIGHT.items():
        adapter = fla[contract["adapter"]]
        if adapter.get("official_layer_class") != FLA_CLASSES[public_name]:
            raise AssertionError(f"FLA {public_name} adapter used the wrong official class")
        if adapter.get("official_chunk_autograd_node") != contract["autograd_node"]:
            raise AssertionError(f"FLA {public_name} adapter used the wrong backward node")
        if int(adapter.get("official_layer_forward_calls", 0)) != 1:
            raise AssertionError(f"FLA {public_name} official layer was not called exactly once")
        if set(adapter.get("conv_backends", {}).values()) != {"triton"}:
            raise AssertionError(f"FLA {public_name} adapter used non-Triton convolution")
        if float(adapter.get("initial_state_grad_norm", 0.0)) <= 0:
            raise AssertionError(f"FLA {public_name} initial state did not receive gradients")
        reference = fla[contract["reference"]]
        if float(reference.get("output_max_abs", float("inf"))) > 1e-3:
            raise AssertionError(f"FLA {public_name} output failed the Torch reference")
        if float(reference.get("state_max_abs", float("inf"))) > 1e-3:
            raise AssertionError(f"FLA {public_name} state failed the Torch reference")
        if float(reference.get("gradient_max_abs", float("inf"))) > 5e-2:
            raise AssertionError(f"FLA {public_name} backward failed the Torch reference")
        stack = fla[contract["stack"]]
        if float(stack.get("fs_gate_mean", 0.0)) <= 0:
            raise AssertionError(f"FLA {public_name} FutureSeed gate is inactive")
        if float(stack.get("fs_state_norm", 0.0)) <= 0:
            raise AssertionError(f"FLA {public_name} FutureSeed state is inactive")
        if float(stack.get("input_grad_norm", 0.0)) <= 0:
            raise AssertionError(f"FLA {public_name} full-stack backward is inactive")
        if stack.get("expected_missing_parameter_grads") != [
            "blocks.0.future_seed_logit"
        ]:
            raise AssertionError(f"FLA {public_name} stack has unexpected missing gradients")
    old_rwkv = fs_comparison["rwkv7_gate"]["source_and_initialization"]
    new_rwkv = rwkv["source_and_initialization"]
    for key in ("source_commit", "source_blob", "kernel_blob", "statepassing_cuda_sha256"):
        if new_rwkv[key] != old_rwkv[key]:
            raise AssertionError(f"Fresh RWKV provenance differs for {key}")
    if not rwkv["cuda_against_torch"].get("all_gradients_finite"):
        raise AssertionError("RWKV7 CUDA backward produced nonfinite gradients")
    for key in ("output", "terminal_state", "v_first"):
        if float(rwkv["cuda_against_torch"][key]["max_abs"]) > 2e-3:
            raise AssertionError(f"RWKV7 CUDA {key} failed the Torch reference")
    for name, row in rwkv["cuda_against_torch"]["gradients"].items():
        if float(row["max_abs"]) > 2e-3:
            raise AssertionError(f"RWKV7 CUDA gradient {name} failed the Torch reference")
    for key in ("output", "terminal_state", "v_first"):
        if float(rwkv["independent_formula"][key]["max_abs"]) > 1e-7:
            raise AssertionError(f"RWKV7 {key} differs from the independent formula")
    if float(rwkv["decay_parameterization"]["max_abs"]) > 1e-12:
        raise AssertionError("RWKV7 decay parameterization differs from the reference")
    if not rwkv["layer_zero_value"].get("output_finite"):
        raise AssertionError("RWKV7 official zero initialization produced nonfinite output")
    if not rwkv["layer_zero_value"].get("terminal_state_finite"):
        raise AssertionError("RWKV7 official zero initialization produced nonfinite state")
    value_contract = rwkv["value_residual_and_optimizer"]
    if float(value_contract.get("v_first_gradient_norm", 0.0)) <= 0:
        raise AssertionError("RWKV7 value-residual path did not receive gradients")
    if float(value_contract.get("v_first_output_rms_delta", 0.0)) <= 0:
        raise AssertionError("RWKV7 value-residual path did not affect output")
    if value_contract["optimizer"].get("contract") != "rwkv7_decay_groups":
        raise AssertionError("RWKV7 optimizer contract differs")
    for public_name in BACKBONES:
        old_count = int(fs_comparison["preflight"]["backbones"][public_name]["parameter_count"])
        new_count = int(backbone["backbones"][public_name]["parameter_count"])
        if old_count != new_count:
            raise AssertionError(f"Fresh {public_name} preflight parameter count differs")
        gradient = backbone["backbones"][public_name]["gradient"]
        if gradient["unexpected_missing"] or gradient["nonfinite"]:
            raise AssertionError(f"Fresh {public_name} backward gate failed")
    if causal_contract.get("status") != "PASS":
        raise AssertionError("FutureSeed causal initialization contract failed")
    if causal_contract.get("source_sha") != EXPECTED_SHA:
        raise AssertionError("FutureSeed causal contract used the wrong source")
    for public_name in BACKBONES:
        row = causal_contract["backbones"][public_name]
        if row.get("status") != "PASS" or not row.get("identical_initial_parameter_tensors"):
            raise AssertionError(f"{public_name} FutureSeed causal contract failed")
        if int(row.get("parameter_tensor_count", 0)) <= 0:
            raise AssertionError(f"{public_name} causal contract has no parameter tensors")
        if not re.fullmatch(r"[0-9a-f]{64}", row.get("initial_parameter_fingerprint", "")):
            raise AssertionError(f"{public_name} initial parameter fingerprint is invalid")
        if not row.get("identical_functional_checkpoint_tensors"):
            raise AssertionError(f"{public_name} functional checkpoint tensors differ")
        if not re.fullmatch(r"[0-9a-f]{64}", row.get("functional_checkpoint_sha256", "")):
            raise AssertionError(f"{public_name} functional checkpoint hash is invalid")
        if not re.fullmatch(r"[0-9a-f]{64}", row.get("functional_parameter_fingerprint", "")):
            raise AssertionError(f"{public_name} functional parameter fingerprint is invalid")
    return {
        "status": "PASS",
        "official_fla_source": "PASS",
        "official_rwkv7_source_and_cuda": "PASS",
        "cuda_backward": "PASS",
        "official_chunk_backward": "PASS",
        "torch_reference_alignment": "PASS",
        "gpu1_device": "PASS",
        "shared_shell_initialization": "PASS",
        "futureseed_onoff_initialization": "PASS",
        "silent_fallback": "FORBIDDEN",
    }


def validate_data_identity(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    hashes: dict[str, str] = {}
    mtimes: dict[str, str] = {}
    test_realpath = ""
    for line in text.splitlines():
        if line.startswith("test_realpath="):
            test_realpath = line.split("=", 1)[1]
            continue
        hash_match = re.fullmatch(r"([0-9a-f]{64})  (/.+)", line)
        if hash_match:
            hashes[hash_match.group(2)] = hash_match.group(1)
            continue
        mtime_match = re.fullmatch(r"mtime=(.+) size=\d+ path=(/.+)", line)
        if mtime_match:
            mtimes[mtime_match.group(2)] = mtime_match.group(1)
    if test_realpath != EXPECTED_TEST_REALPATH:
        raise AssertionError(
            f"Official test symlink target changed: {test_realpath} != {EXPECTED_TEST_REALPATH}"
        )
    if hashes != EXPECTED_DATA_SHA256:
        differing = sorted(
            data_path
            for data_path in set(hashes) | set(EXPECTED_DATA_SHA256)
            if hashes.get(data_path) != EXPECTED_DATA_SHA256.get(data_path)
        )
        raise AssertionError(f"Official Sudoku data hashes changed: {differing}")
    if set(mtimes) != set(EXPECTED_DATA_SHA256):
        raise AssertionError("Official Sudoku data mtime manifest is incomplete")
    late_files = {
        data_path: value
        for data_path, value in mtimes.items()
        if value[:19] >= FS_RUN_START_UTC
    }
    if late_files:
        raise AssertionError(
            f"Official Sudoku data changed after FutureSeed-on launch: {late_files}"
        )
    return {
        "status": "PASS",
        "test_realpath": test_realpath,
        "sha256": hashes,
        "mtimes": mtimes,
        "all_files_predate_futureseed_runs": True,
    }


def summarize_pair(fs: dict[str, Any], nofs: dict[str, Any]) -> dict[str, Any]:
    def loop_delta(source_key: str, index: int, metric: str) -> float:
        return fs[source_key][index][metric] - nofs[source_key][index][metric]

    range_rows: dict[str, Any] = {}
    for key in RANGES:
        fs_loops = fs["ranges"][key]["loops"]
        nofs_loops = nofs["ranges"][key]["loops"]
        nofs_target = nofs_loops[-1]["exact"]
        earliest = next(
            (row["loop"] for row in fs_loops if row["exact"] >= nofs_target),
            None,
        )
        range_rows[key] = {
            "fs": fs_loops,
            "nofs": nofs_loops,
            "loop5_exact_delta": fs_loops[-1]["exact"] - nofs_loops[-1]["exact"],
            "loop5_blank_delta": fs_loops[-1]["blank_acc"] - nofs_loops[-1]["blank_acc"],
            "fs_earliest_loop_reaching_nofs_loop5": earliest,
        }
    return {
        "public_name": fs["public_name"],
        "label": fs["label"],
        "color": COLORS[fs["public_name"]],
        "parameter_count": fs["parameter_count"],
        "fs_run_name": fs["run_name"],
        "nofs_run_name": nofs["run_name"],
        "train_ce": {"fs": fs["train_ce"], "nofs": nofs["train_ce"], "delta": fs["train_ce"] - nofs["train_ce"]},
        "train_sec_per_step": {
            "fs": fs["train_sec_per_step"],
            "nofs": nofs["train_sec_per_step"],
            "overhead_frac": fs["train_sec_per_step"] / nofs["train_sec_per_step"] - 1.0,
        },
        "peak_allocated_mib": {"fs": fs["peak_allocated_mib"], "nofs": nofs["peak_allocated_mib"]},
        "mixed": {
            "fs": fs["loops"],
            "nofs": nofs["loops"],
            "loop5_exact_delta": loop_delta("loops", -1, "exact"),
            "loop5_blank_delta": loop_delta("loops", -1, "blank_acc"),
            "fs_loop_gain_exact": fs["loops"][-1]["exact"] - fs["loops"][0]["exact"],
            "nofs_loop_gain_exact": nofs["loops"][-1]["exact"] - nofs["loops"][0]["exact"],
        },
        "ranges": range_rows,
        "curve": {"fs": fs["curve"], "nofs": nofs["curve"]},
        "case": {
            "puzzle_sha256": fs["case"]["puzzle_sha256"],
            "fs_wrong_by_loop": fs["case"]["wrong_by_loop"],
            "nofs_wrong_by_loop": nofs["case"]["wrong_by_loop"],
            "fs_html": f"../{fs['run_name']}/output/{fs['case']['filename']}",
            "nofs_html": f"../{nofs['run_name']}/output/{nofs['case']['filename']}",
        },
        "case_banks": {
            key: {
                "fs": fs["case_banks"][key],
                "nofs": nofs["case_banks"][key],
            }
            for key in RANGES
        },
    }


def decision(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    strong = [
        pair["public_name"]
        for pair in pairs
        if pair["ranges"]["b46_50"]["loop5_exact_delta"] >= 0.10
    ]
    deltas = {
        pair["public_name"]: pair["ranges"]["b46_50"]["loop5_exact_delta"]
        for pair in pairs
    }
    if len(strong) >= 2:
        verdict = "SUPPORTED"
        explanation = (
            f"FutureSeed improves 46-50-blank full-board exact by at least 10 points "
            f"on {len(strong)} causal carriers ({', '.join(strong)}). "
            "The cross-carrier causal gate passes."
        )
    elif all(value < 0.03 for value in deltas.values()):
        verdict = "REJECTED_AT_THIS_BUDGET"
        explanation = (
            "No carrier gains three points of 46-50-blank full-board exact. "
            "Do not claim a generic FutureSeed benefit from this budget."
        )
    else:
        verdict = "MIXED"
        explanation = (
            "FutureSeed helps too few carriers to support a generic claim. "
            "Keep only the carrier-specific evidence and do not average it into a universal result."
        )
    return {
        "verdict": verdict,
        "explanation": explanation,
        "success_threshold": "at least two carriers with b46-50 loop5 exact delta >= +0.10",
        "strong_carriers": strong,
        "b46_50_loop5_exact_delta": deltas,
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def signed_pct(value: float) -> str:
    return f"{100.0 * value:+.2f}"


def paired_chart(
    pair: dict[str, Any],
    rows_key: str,
    metric: str,
    title: str,
) -> str:
    source = pair[rows_key]
    fs_rows = source["fs"]
    nofs_rows = source["nofs"]
    values = [float(row[metric]) for row in fs_rows + nofs_rows]
    y_min = min(0.0, min(values))
    y_max = max(values)
    if y_max == y_min:
        y_max = y_min + 1.0
    width, height = 560, 260
    left, right, top, bottom = 56, 18, 42, 42
    plot_w, plot_h = width - left - right, height - top - bottom

    def x(loop: int) -> float:
        return left + (loop - 1) / 4.0 * plot_w

    def y(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
        f'<text x="{left}" y="24" class="chart-title">{html.escape(title)}</text>',
    ]
    for index in range(5):
        value = y_min + (y_max - y_min) * index / 4
        yp = y(value)
        parts.append(f'<line x1="{left}" y1="{yp:.2f}" x2="{left + plot_w}" y2="{yp:.2f}" class="grid"/>')
        parts.append(f'<text x="{left - 8}" y="{yp + 4:.2f}" text-anchor="end" class="tick">{value:.2f}</text>')
    for loop in range(1, 6):
        xp = x(loop)
        parts.append(f'<text x="{xp:.2f}" y="{top + plot_h + 22}" text-anchor="middle" class="tick">{loop}</text>')
    for label, rows, color, dash in (
        ("FS", fs_rows, pair["color"], ""),
        ("no FS", nofs_rows, "#6f7880", ' stroke-dasharray="7 5"'),
    ):
        points = " ".join(f'{x(int(row["loop"])):.2f},{y(float(row[metric])):.2f}' for row in rows)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"{dash}/>')
        for row in rows:
            parts.append(f'<circle cx="{x(int(row["loop"])):.2f}" cy="{y(float(row[metric])):.2f}" r="4" fill="{color}"/>')
        lx = left + (0 if label == "FS" else 100)
        parts.append(f'<line x1="{lx}" y1="{height - 8}" x2="{lx + 22}" y2="{height - 8}" stroke="{color}" stroke-width="3"{dash}/>')
        parts.append(f'<text x="{lx + 28}" y="{height - 4}" class="tick">{label}</text>')
    parts.append("</svg>")
    return "".join(parts)


def training_chart(pair: dict[str, Any]) -> str:
    fs_rows = pair["curve"]["fs"]
    nofs_rows = pair["curve"]["nofs"]
    rows = fs_rows + nofs_rows
    steps = sorted({int(row["step"]) for row in rows})
    values = [float(row["ce"]) for row in rows]
    x_min, x_max = min(steps), max(steps)
    y_min, y_max = min(values), max(values)
    padding = max(0.03, (y_max - y_min) * 0.08)
    y_min = max(0.0, y_min - padding)
    y_max += padding
    width, height = 1140, 260
    left, right, top, bottom = 56, 18, 42, 42
    plot_w, plot_h = width - left - right, height - top - bottom

    def x(step: int) -> float:
        return left + (step - x_min) / max(1, x_max - x_min) * plot_w

    def y(value: float) -> float:
        return top + (y_max - value) / max(1e-8, y_max - y_min) * plot_h

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Training CE by optimizer step">',
        f'<text x="{left}" y="24" class="chart-title">Training CE by optimizer step (lower is better)</text>',
    ]
    for index in range(5):
        value = y_min + (y_max - y_min) * index / 4
        yp = y(value)
        parts.append(
            f'<line x1="{left}" y1="{yp:.2f}" x2="{left + plot_w}" '
            f'y2="{yp:.2f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{yp + 4:.2f}" text-anchor="end" '
            f'class="tick">{value:.2f}</text>'
        )
    for step in steps:
        xp = x(step)
        parts.append(
            f'<text x="{xp:.2f}" y="{top + plot_h + 22}" text-anchor="middle" '
            f'class="tick">{step}</text>'
        )
    for label, curve, color, dash in (
        ("FS", fs_rows, pair["color"], ""),
        ("no FS", nofs_rows, "#6f7880", ' stroke-dasharray="7 5"'),
    ):
        points = " ".join(
            f'{x(int(row["step"])):.2f},{y(float(row["ce"])):.2f}'
            for row in curve
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            f'stroke-width="3"{dash}/>'
        )
        for row in curve:
            parts.append(
                f'<circle cx="{x(int(row["step"])):.2f}" '
                f'cy="{y(float(row["ce"])):.2f}" r="4" fill="{color}"/>'
            )
        lx = left + (0 if label == "FS" else 100)
        parts.append(
            f'<line x1="{lx}" y1="{height - 8}" x2="{lx + 22}" '
            f'y2="{height - 8}" stroke="{color}" stroke-width="3"{dash}/>'
        )
        parts.append(
            f'<text x="{lx + 28}" y="{height - 4}" class="tick">{label}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def render_html(payload: dict[str, Any]) -> str:
    pairs = payload["pairs"]
    summary_rows = "".join(
        "<tr>"
        f"<td><span class=\"swatch\" style=\"background:{pair['color']}\"></span>{pair['label']}</td>"
        f"<td>{pair['parameter_count'] / 1e6:.3f}M</td>"
        f"<td>{pair['train_ce']['fs']:.4f}</td><td>{pair['train_ce']['nofs']:.4f}</td>"
        f"<td>{pct(pair['ranges']['b46_50']['fs'][-1]['exact'])}</td>"
        f"<td>{pct(pair['ranges']['b46_50']['nofs'][-1]['exact'])}</td>"
        f"<td class=\"delta\">{signed_pct(pair['ranges']['b46_50']['loop5_exact_delta'])}</td>"
        f"<td>{pct(pair['ranges']['b51_55']['fs'][-1]['exact'])}</td>"
        f"<td>{pct(pair['ranges']['b51_55']['nofs'][-1]['exact'])}</td>"
        f"<td>{signed_pct(pair['mixed']['loop5_exact_delta'])}</td>"
        f"<td>{100.0 * pair['train_sec_per_step']['overhead_frac']:+.1f}%</td>"
        f"<td>{pair['peak_allocated_mib']['fs'] / 1024.0:.2f}</td>"
        f"<td>{pair['peak_allocated_mib']['nofs'] / 1024.0:.2f}</td>"
        f"<td>{(pair['peak_allocated_mib']['fs'] - pair['peak_allocated_mib']['nofs']) / 1024.0:+.2f}</td>"
        "</tr>"
        for pair in pairs
    )
    range_rows = "".join(
        "<tr>"
        f"<td>{pair['label']}</td><td>{label}</td>"
        f"<td>{pct(pair['ranges'][key]['fs'][-1]['exact'])}</td>"
        f"<td>{pct(pair['ranges'][key]['nofs'][-1]['exact'])}</td>"
        f"<td>{signed_pct(pair['ranges'][key]['loop5_exact_delta'])}</td>"
        f"<td>{pct(pair['ranges'][key]['fs'][-1]['blank_acc'])}</td>"
        f"<td>{pct(pair['ranges'][key]['nofs'][-1]['blank_acc'])}</td>"
        f"<td>{html.escape(str(pair['ranges'][key]['fs_earliest_loop_reaching_nofs_loop5']))}</td>"
        "</tr>"
        for pair in pairs
        for key, label in RANGES.items()
    )
    chart_sections = []
    for pair in pairs:
        range_pair = {
            "color": pair["color"],
            "range": {
                "fs": pair["ranges"]["b46_50"]["fs"],
                "nofs": pair["ranges"]["b46_50"]["nofs"],
            },
        }
        chart_sections.append(
            '<section class="carrier">'
            f"<h3>{pair['label']}</h3><div class=\"charts\">"
            f"<div class=\"wide\">{training_chart(pair)}</div>"
            f"{paired_chart(pair, 'mixed', 'exact', 'Mixed full-board exact by loop')}"
            f"{paired_chart(range_pair, 'range', 'exact', '46-50 blanks exact by loop')}"
            "</div></section>"
        )
    charts = "".join(chart_sections)
    cases = "".join(
        '<section class="case-pair">'
        f"<h3>{pair['label']}: same puzzle</h3>"
        '<div class="case-grid">'
        '<div><h4>FutureSeed</h4>'
        f"<p class=\"muted\">wrong cells {html.escape(str(pair['case']['fs_wrong_by_loop']))}</p>"
        f"<iframe src=\"{html.escape(pair['case']['fs_html'])}\" loading=\"lazy\"></iframe></div>"
        '<div><h4>No FutureSeed</h4>'
        f"<p class=\"muted\">wrong cells {html.escape(str(pair['case']['nofs_wrong_by_loop']))}</p>"
        f"<iframe src=\"{html.escape(pair['case']['nofs_html'])}\" loading=\"lazy\"></iframe></div>"
        "</div></section>"
        for pair in pairs
    )
    case_bank_rows = "".join(
        "<tr>"
        f"<td>{pair['label']}</td><td>{label}</td>"
        f"<td><a href=\"{html.escape(pair['case_banks'][key]['fs'])}\">FS cases</a></td>"
        f"<td><a href=\"{html.escape(pair['case_banks'][key]['nofs'])}\">noFS cases</a></td>"
        "</tr>"
        for pair in pairs
        for key, label in RANGES.items()
    )
    gate = payload["fairness_gate"]
    gate_rows = "".join(
        f"<tr><td>{html.escape(key.replace('_', ' '))}</td><td>{html.escape(str(value))}</td></tr>"
        for key, value in gate.items()
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FutureSeed causal carrier comparison</title>
<style>
:root{{--bg:#f3f5f6;--paper:#fff;--ink:#182127;--muted:#5b6870;--line:#d6dde1;--accent:#0d6757}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 ui-sans-serif,system-ui,sans-serif}}
main{{max-width:1480px;margin:auto;padding:26px 20px 70px}}h1{{font-size:28px;margin:0 0 5px;letter-spacing:0}}
h2{{font-size:20px;margin:30px 0 10px;letter-spacing:0}}h3{{font-size:17px;letter-spacing:0}}h4{{font-size:14px;letter-spacing:0}}
.band{{background:var(--paper);border:1px solid var(--line);border-radius:6px;padding:16px;margin-top:12px;overflow:auto}}
.verdict{{border-left:5px solid var(--accent)}}.muted{{color:var(--muted)}}table{{width:100%;border-collapse:collapse;white-space:nowrap}}
th,td{{padding:9px;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child{{text-align:left}}
.delta{{font-weight:700}}.swatch{{display:inline-block;width:10px;height:10px;margin-right:7px}}
.mechanism{{display:grid;grid-template-columns:1fr auto 1fr;gap:14px;align-items:center}}.mechanism strong{{display:block;margin-bottom:4px}}.arrow{{font-size:24px;color:var(--accent)}}
.carrier,.case-pair{{border-top:1px solid var(--line);padding-top:10px;margin-top:18px}}.charts,.case-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.wide{{grid-column:1/-1}}
svg{{width:100%;background:#fff;border:1px solid var(--line);border-radius:6px}}.grid{{stroke:#e4eaed}}.tick{{font-size:12px;fill:#4f5b63}}.chart-title{{font-size:15px;font-weight:650}}
iframe{{width:100%;height:720px;border:1px solid var(--line);border-radius:6px;background:#fff}}
a{{color:#075f93}}code{{background:#edf1f3;padding:2px 5px;border-radius:3px}}@media(max-width:950px){{.charts,.case-grid,.mechanism{{grid-template-columns:1fr}}.arrow{{transform:rotate(90deg);justify-self:start}}}}
</style></head><body><main>
<h1>FutureSeed on/off across four causal carriers</h1>
<p class="muted">Official RWKV7/GDN/GDN2/KDA, one seed, byte-identical initialization within each pair, same data order, same optimizer, same five-loop loss, GPU1 only.</p>
<div class="band verdict"><strong>{html.escape(payload['decision']['verdict'])}:</strong> {html.escape(payload['decision']['explanation'])}</div>
<div class="band"><h2>Mechanism under test</h2><div class="mechanism"><div><strong>Layer l: causal scan</strong><code>x1 -> x2 -> ... -> xT</code><br>Keep its terminal recurrent state <code>hT[l]</code>.</div><div class="arrow">-></div><div><strong>Layer l+1: seeded causal scan</strong><code>h0[l+1] = gate * normalize(hT[l])</code><br>FS-off sets this injected state to exactly zero. No reverse scan is added.</div></div></div>
<h2>Matched causal result</h2><div class="band"><table><thead><tr><th>carrier</th><th>params</th><th>CE FS</th><th>CE noFS</th><th>b46-50 FS</th><th>b46-50 noFS</th><th>exact delta pts</th><th>b51-55 FS</th><th>b51-55 noFS</th><th>mixed delta pts</th><th>FS time overhead</th><th>VRAM FS GiB</th><th>VRAM noFS GiB</th><th>VRAM delta GiB</th></tr></thead><tbody>{summary_rows}</tbody></table></div>
<h2>Difficulty ranges</h2><div class="band"><table><thead><tr><th>carrier</th><th>blanks</th><th>exact FS</th><th>exact noFS</th><th>delta pts</th><th>blank FS</th><th>blank noFS</th><th>FS loop reaching noFS L5</th></tr></thead><tbody>{range_rows}</tbody></table></div>
<h2>Loop behavior</h2>{charts}
<h2>Same puzzle, every loop</h2>{cases}
<h2>Multi-case banks</h2><div class="band"><table><thead><tr><th>carrier</th><th>blank range</th><th>FutureSeed</th><th>no FutureSeed</th></tr></thead><tbody>{case_bank_rows}</tbody></table></div>
<h2>Fail-closed fairness checks</h2><div class="band"><table><tbody>{gate_rows}</tbody></table><p>Only <code>future_seed_scale: 1 -> 0</code> and artifact output paths differ inside each pair. Initial parameter tensors are SHA256-identical. FS-off traces have zero injected gate and state; FS-on traces are nonzero. Official source, CUDA backward, data identity, parameter count, optimizer grouping, and evaluator identity are checked before this page is written.</p></div>
<h2>Interpretation boundary</h2><div class="band"><p>This is a one-seed causal mechanism gate, not a final scaling-law result. Full-board exact is the primary outcome: a lower CE or higher per-blank accuracy without more completely solved boards does not count as mechanism success. The pre-registered success rule requires a ten-point b46-50 exact gain on at least two carriers.</p></div>
</main></body></html>"""


def write_embedded_primary_cases(
    out_dir: Path,
    runs_root: Path,
    pairs: list[dict[str, Any]],
) -> None:
    embedded = out_dir / "embedded_cases"
    embedded.mkdir(parents=True, exist_ok=True)
    for pair in pairs:
        for condition, run_key in (("fs", "fs_run_name"), ("nofs", "nofs_run_name")):
            source = only_case(runs_root / pair[run_key])
            text = source.read_text(encoding="utf-8")
            if condition == "nofs":
                old_meta = "with FutureSeed and depth-loop refinement"
                new_meta = "without FutureSeed and depth-loop refinement"
                if old_meta not in text and new_meta not in text:
                    raise AssertionError(f"Unexpected no-FutureSeed case template: {source}")
                text = text.replace("FutureSeed loop case", "No-FutureSeed loop case")
                text = text.replace(old_meta, new_meta)
            destination = embedded / f"{pair['public_name']}-{condition}.html"
            destination.write_text(text, encoding="utf-8")
            pair["case"][f"{condition}_html"] = f"embedded_cases/{destination.name}"


def write_csv(path: Path, pairs: list[dict[str, Any]]) -> None:
    fields = [
        "backbone",
        "parameters",
        "train_ce_fs",
        "train_ce_nofs",
        "b46_50_exact_fs",
        "b46_50_exact_nofs",
        "b46_50_exact_delta",
        "b51_55_exact_fs",
        "b51_55_exact_nofs",
        "mixed_exact_fs",
        "mixed_exact_nofs",
        "mixed_exact_delta",
        "fs_sec_per_step",
        "nofs_sec_per_step",
        "fs_time_overhead_frac",
        "peak_allocated_mib_fs",
        "peak_allocated_mib_nofs",
        "fs_memory_overhead_mib",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for pair in pairs:
            writer.writerow(
                {
                    "backbone": pair["public_name"],
                    "parameters": pair["parameter_count"],
                    "train_ce_fs": pair["train_ce"]["fs"],
                    "train_ce_nofs": pair["train_ce"]["nofs"],
                    "b46_50_exact_fs": pair["ranges"]["b46_50"]["fs"][-1]["exact"],
                    "b46_50_exact_nofs": pair["ranges"]["b46_50"]["nofs"][-1]["exact"],
                    "b46_50_exact_delta": pair["ranges"]["b46_50"]["loop5_exact_delta"],
                    "b51_55_exact_fs": pair["ranges"]["b51_55"]["fs"][-1]["exact"],
                    "b51_55_exact_nofs": pair["ranges"]["b51_55"]["nofs"][-1]["exact"],
                    "mixed_exact_fs": pair["mixed"]["fs"][-1]["exact"],
                    "mixed_exact_nofs": pair["mixed"]["nofs"][-1]["exact"],
                    "mixed_exact_delta": pair["mixed"]["loop5_exact_delta"],
                    "fs_sec_per_step": pair["train_sec_per_step"]["fs"],
                    "nofs_sec_per_step": pair["train_sec_per_step"]["nofs"],
                    "fs_time_overhead_frac": pair["train_sec_per_step"]["overhead_frac"],
                    "peak_allocated_mib_fs": pair["peak_allocated_mib"]["fs"],
                    "peak_allocated_mib_nofs": pair["peak_allocated_mib"]["nofs"],
                    "fs_memory_overhead_mib": (
                        pair["peak_allocated_mib"]["fs"]
                        - pair["peak_allocated_mib"]["nofs"]
                    ),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--fs-comparison", type=Path, required=True)
    parser.add_argument("--nofs-preflight", type=Path, required=True)
    parser.add_argument("--causal-contract", type=Path, required=True)
    parser.add_argument("--data-identity", type=Path, required=True)
    for public_name in BACKBONES:
        parser.add_argument(f"--{public_name}-nofs-run", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    runs_root = (args.runs_root or repo / "runs").resolve()
    fs_comparison_path = args.fs_comparison if args.fs_comparison.is_absolute() else repo / args.fs_comparison
    nofs_preflight = args.nofs_preflight if args.nofs_preflight.is_absolute() else repo / args.nofs_preflight
    causal_contract_path = args.causal_contract if args.causal_contract.is_absolute() else repo / args.causal_contract
    data_identity_path = args.data_identity if args.data_identity.is_absolute() else repo / args.data_identity
    fs_comparison = read_json(fs_comparison_path)
    causal_contract = read_json(causal_contract_path)
    data_identity = validate_data_identity(data_identity_path)
    fs_names = {arm["key"]: arm["run_name"] for arm in fs_comparison["arms"]}
    nofs_names = {
        public_name: getattr(args, f"{public_name}_nofs_run")
        for public_name in BACKBONES
    }
    pairs = []
    paired_validations: dict[str, Any] = {}
    fs_snapshot_hashes: set[str] = set()
    nofs_snapshot_hashes: set[str] = set()
    for public_name in BACKBONES:
        fs = load_arm(runs_root, fs_names[public_name], public_name)
        nofs = load_arm(runs_root, nofs_names[public_name], public_name)
        validate_pair(fs, nofs)
        fs_snapshot_hashes.add(fs["source_snapshot"]["sha256"])
        nofs_snapshot_hashes.add(nofs["source_snapshot"]["sha256"])
        differing_args = sorted(
            key
            for key in set(fs["args"]) | set(nofs["args"])
            if fs["args"].get(key) != nofs["args"].get(key)
        )
        paired_validations[public_name] = {
            "status": "PASS",
            "source_sha": fs["git_sha"],
            "differing_args": differing_args,
            "allowed_differing_args": sorted(ARG_PATH_DIFFERENCES),
            "parameter_count": fs["parameter_count"],
            "puzzle_sha256": fs["case"]["puzzle_sha256"],
            "fs_result_sha256": file_sha256(only_result(fs["run_dir"])),
            "nofs_result_sha256": file_sha256(only_result(nofs["run_dir"])),
            "fs_source_snapshot": fs["source_snapshot"],
            "nofs_source_snapshot": nofs["source_snapshot"],
            "fs_loop5_trace": fs["loops"][-1]["future_seed"],
            "nofs_loop5_trace": nofs["loops"][-1]["future_seed"],
        }
        pairs.append(summarize_pair(fs, nofs))
    if len(fs_snapshot_hashes) != 1 or len(nofs_snapshot_hashes) != 1:
        raise AssertionError(
            "Source snapshots differ within a formal four-carrier suite"
        )
    out_dir = args.out_dir if args.out_dir.is_absolute() else repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_embedded_primary_cases(out_dir, runs_root, pairs)
    fairness_gate = validate_preflight(fs_comparison, nofs_preflight, causal_contract)
    fairness_gate["source_snapshot_within_suite"] = "PASS"
    fairness_gate["official_data_sha256"] = "PASS"
    fairness_gate["data_files_predate_futureseed_runs"] = "PASS"
    for public_name, validation in paired_validations.items():
        nofs_run_dir = runs_root / nofs_names[public_name]
        (nofs_run_dir / "strict_run_validation.json").write_text(
            json.dumps(
                {
                    **validation,
                    "fairness_gate": fairness_gate,
                    "official_data": data_identity,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_id": "P-BASELINE-005",
        "source_sha": EXPECTED_SHA,
        "intervention": {"future_seed_scale": {"with": 1.0, "without": 0.0}},
        "fairness_gate": fairness_gate,
        "paired_validations": paired_validations,
        "decision": decision(pairs),
        "pairs": pairs,
        "causal_contract": causal_contract,
        "data_identity": data_identity,
    }
    serializable = json.loads(json.dumps(payload, default=str))
    (out_dir / "comparison.json").write_text(
        json.dumps(serializable, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "strict_pair_validation.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "source_sha": EXPECTED_SHA,
                "fairness_gate": fairness_gate,
                "pairs": paired_validations,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "data_identity.txt").write_text(
        data_identity_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    write_csv(out_dir / "comparison.csv", pairs)
    (out_dir / "index.html").write_text(render_html(payload), encoding="utf-8")
    rows = "\n".join(
        f"| {pair['label']} | {pct(pair['ranges']['b46_50']['fs'][-1]['exact'])} | "
        f"{pct(pair['ranges']['b46_50']['nofs'][-1]['exact'])} | "
        f"{signed_pct(pair['ranges']['b46_50']['loop5_exact_delta'])} | "
        f"{100.0 * pair['train_sec_per_step']['overhead_frac']:+.1f}% | "
        f"{(pair['peak_allocated_mib']['fs'] - pair['peak_allocated_mib']['nofs']) / 1024.0:+.2f} GiB |"
        for pair in pairs
    )
    (out_dir / "README.md").write_text(
        "# FutureSeed Causal Four-Carrier Gate\n\n"
        f"Verdict: **{payload['decision']['verdict']}**. {payload['decision']['explanation']}\n\n"
        "| Carrier | b46-50 FS | b46-50 noFS | exact delta pts | FS time overhead | FS VRAM delta |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        f"{rows}\n\n"
        "Open `index.html` for loop curves, difficulty ranges, exact fairness checks, "
        "and same-puzzle visualizations.\n",
        encoding="utf-8",
    )
    print(out_dir)


if __name__ == "__main__":
    main()
