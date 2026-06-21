#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch


TOKEN_NAMES = {
    -100: "ignore",
    0: "pad",
    1: "wall",
    2: "open",
    3: "start",
    4: "goal",
    5: "path",
}

TOKEN_CLASS = {
    -100: "ignore",
    0: "pad",
    1: "wall",
    2: "open",
    3: "start",
    4: "goal",
    5: "path",
}

TOKEN_GLYPH = {
    -100: "",
    0: "",
    1: "",
    2: "",
    3: "S",
    4: "G",
    5: ".",
}


def parse_steps(raw: str) -> List[int]:
    steps = sorted({int(x.strip()) for x in raw.split(",") if x.strip()})
    if not steps or min(steps) < 1:
        raise ValueError("--steps must contain positive integers")
    return steps


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def import_official_eqr(repo: Path) -> Dict[str, Any]:
    sys.path.insert(0, str(repo))
    os.chdir(repo)

    from evaluate import (  # type: ignore
        _apply_overrides,
        _apply_runtime_arch_overrides,
        _base_config,
        _load_checkpoint,
        _seed,
    )
    from evaluators.eval_config import EvalConfig  # type: ignore
    from pretrain import create_dataloader, init_train_state  # type: ignore
    from utils.checkpoint import load_ema, load_model_state_dict  # type: ignore
    from utils.training import tree_to_device  # type: ignore

    return {
        "EvalConfig": EvalConfig,
        "_apply_overrides": _apply_overrides,
        "_apply_runtime_arch_overrides": _apply_runtime_arch_overrides,
        "_base_config": _base_config,
        "_load_checkpoint": _load_checkpoint,
        "_seed": _seed,
        "create_dataloader": create_dataloader,
        "init_train_state": init_train_state,
        "load_ema": load_ema,
        "load_model_state_dict": load_model_state_dict,
        "tree_to_device": tree_to_device,
    }


def model_inner(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod  # type: ignore[assignment]
    return getattr(model, "model", model)


def path_metrics(pred: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    valid = labels != -100
    token_acc = float(((pred == labels) & valid).sum() / max(int(valid.sum()), 1))
    exact = float(bool(np.all(pred[valid] == labels[valid])))
    true_path = labels == 5
    pred_path = pred == 5
    tp = int((true_path & pred_path).sum())
    fp = int((~true_path & pred_path & valid).sum())
    fn = int((true_path & ~pred_path).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "token_acc": token_acc,
        "exact": exact,
        "path_precision": float(precision),
        "path_recall": float(recall),
        "path_f1": float(f1),
        "path_tp": float(tp),
        "path_fp": float(fp),
        "path_fn": float(fn),
        "pred_path_frac": float(pred_path[valid].mean()) if valid.any() else 0.0,
        "true_path_frac": float(true_path[valid].mean()) if valid.any() else 0.0,
    }


def board_size(seq_len: int) -> int:
    size = int(seq_len**0.5)
    if size * size != seq_len:
        raise ValueError(f"seq_len {seq_len} is not a square")
    return size


def render_board(values: Iterable[int], labels: Iterable[int] | None = None, title: str = "") -> str:
    vals = list(int(x) for x in values)
    labs = list(int(x) for x in labels) if labels is not None else None
    size = board_size(len(vals))
    rows = [f"<div class='board-title'>{html.escape(title)}</div>", "<table class='maze'>"]
    for r in range(size):
        rows.append("<tr>")
        for c in range(size):
            idx = r * size + c
            token = vals[idx]
            cls = TOKEN_CLASS.get(token, "unknown")
            mark = ""
            if labs is not None:
                target_path = labs[idx] == 5
                pred_path = token == 5
                if pred_path and not target_path:
                    mark = " fp"
                elif target_path and not pred_path:
                    mark = " fn"
                elif target_path and pred_path:
                    mark = " tp"
            glyph = TOKEN_GLYPH.get(token, str(token))
            rows.append(
                f"<td class='{cls}{mark}' title='{idx}: {TOKEN_NAMES.get(token, token)}'>"
                f"{html.escape(str(glyph))}</td>"
            )
        rows.append("</tr>")
    rows.append("</table>")
    return "\n".join(rows)


def render_html(payload: Dict[str, Any]) -> str:
    cases = payload["cases"]
    step_keys = [str(s) for s in payload["steps"]]
    styles = """
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:#f5f5f2;color:#202124}
main{max-width:1500px;margin:0 auto;padding:26px}
h1{font-size:26px;margin:0 0 8px} h2{font-size:18px;margin:28px 0 10px} h3{font-size:15px;margin:22px 0 8px}
p{line-height:1.45}.note{background:#fff;border-left:5px solid #475569;padding:10px 12px;margin:14px 0}
.kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:16px 0}.kpi{background:#fff;border:1px solid #ddd;padding:10px;border-radius:8px}.kpi b{display:block;font-size:20px}
table.metrics{width:100%;border-collapse:collapse;background:#fff;border:1px solid #ddd;margin:10px 0 18px}
.metrics th,.metrics td{border-bottom:1px solid #e5e5e5;padding:7px 8px;text-align:left;font-size:13px}.metrics th{background:#eee}
.case{background:#fff;border:1px solid #ddd;border-radius:8px;margin:18px 0;padding:12px}
.grids{display:grid;grid-template-columns:repeat(6,minmax(188px,1fr));gap:10px;align-items:start}
.board-title{font-size:12px;font-weight:600;margin:0 0 4px}
table.maze{border-collapse:collapse;background:#fff} .maze td{width:6px;height:6px;min-width:6px;max-width:6px;padding:0;text-align:center;font-size:5px;line-height:6px;border:0}
.wall{background:#111827}.open{background:#f8fafc}.start{background:#16a34a;color:#fff;font-weight:700}.goal{background:#dc2626;color:#fff;font-weight:700}.path{background:#2563eb;color:#fff}.pad,.ignore{background:#d1d5db}
.tp{box-shadow:inset 0 0 0 1px #22c55e}.fp{background:#f97316!important;color:#111}.fn{box-shadow:inset 0 0 0 1px #dc2626;background:#fee2e2!important;color:#111}
code{background:#eee;padding:1px 4px;border-radius:4px}.small{color:#5f6368;font-size:12px}.bad{color:#b42318}.good{color:#067647}
"""
    rows = []
    for step in step_keys:
        m = payload["aggregate_by_step"][step]
        rows.append(
            "<tr>"
            f"<td>loop{step}</td>"
            f"<td>{m['token_acc']:.4f}</td>"
            f"<td>{m['exact']:.4f}</td>"
            f"<td>{m['path_f1']:.4f}</td>"
            f"<td>{m['path_precision']:.4f}</td>"
            f"<td>{m['path_recall']:.4f}</td>"
            f"<td>{m['path_fp']:.1f}</td>"
            f"<td>{m['path_fn']:.1f}</td>"
            f"<td>{m['pred_path_frac']:.4f}</td>"
            "</tr>"
        )

    case_html = []
    for case in cases:
        final_m = case["metrics"][step_keys[-1]]
        case_html.append("<section class='case'>")
        case_html.append(
            f"<h3>Case {case['case_index']} "
            f"<span class='small'>loop{step_keys[-1]} path_f1={final_m['path_f1']:.3f}, "
            f"FP={final_m['path_fp']:.0f}, FN={final_m['path_fn']:.0f}</span></h3>"
        )
        case_html.append("<div class='grids'>")
        case_html.append(render_board(case["input"], None, "Input"))
        case_html.append(render_board(case["label"], None, "Target"))
        for step in step_keys:
            m = case["metrics"][step]
            title = f"loop{step} F1={m['path_f1']:.2f} FP={m['path_fp']:.0f} FN={m['path_fn']:.0f}"
            case_html.append(render_board(case["predictions"][step], case["label"], title))
        case_html.append("</div></section>")

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(payload['run_name'])}</title><style>{styles}</style></head>
<body><main>
<h1>{html.escape(payload['run_name'])}</h1>
<p>Official EqR case visualization from checkpoint <code>{html.escape(payload['checkpoint'])}</code>.</p>
<div class="note">Orange cells are predicted PATH false positives; pale red outlined cells are missed true-path cells. This is a read-only visualization runner using the official EqR checkpoint, official dataset, official model code, and the same EMA loading path as <code>evaluate.py</code>.</div>
<div class="kpis">
<div class="kpi"><b>{payload['condition']}</b><span>condition</span></div>
<div class="kpi"><b>{payload['processed_cases']}</b><span>processed cases</span></div>
<div class="kpi"><b>{payload['steps'][-1]}</b><span>max loop</span></div>
<div class="kpi"><b>{payload['aggregate_by_step'][step_keys[-1]]['path_f1']:.4f}</b><span>final path F1</span></div>
<div class="kpi"><b>{payload['elapsed_sec']:.1f}s</b><span>visualization runtime</span></div>
</div>
<h2>Aggregate Loop Metrics</h2>
<table class="metrics"><thead><tr><th>loop</th><th>token acc</th><th>exact</th><th>path F1</th><th>path precision</th><th>path recall</th><th>path FP</th><th>path FN</th><th>pred path frac</th></tr></thead><tbody>
{''.join(rows)}
</tbody></table>
<h2>Hard Cases</h2>
{''.join(case_html)}
</main></body></html>"""


def mean_metrics(items: List[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted(items[0]) if items else []
    return {k: float(np.mean([item[k] for item in items])) for k in keys}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    t0 = time.time()
    repo = Path(args.repo).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("DISABLE_COMPILE", "1")
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    api = import_official_eqr(repo)
    steps = parse_steps(args.steps_text)
    max_step = max(steps)
    ckpt, ckpt_path = api["_load_checkpoint"](str(checkpoint))
    eval_cfg = api["EvalConfig"](
        checkpoint=str(checkpoint),
        global_batch_size=args.batch_size,
        seed=args.seed,
        dataset_data_path=args.dataset_data_path,
        load_strict=False,
        loss_head_name="losses@InferenceLossHead",
    )
    config = api["_base_config"](eval_cfg, ckpt, ckpt_path, 0)
    overrides: Dict[str, Any] = {
        "arch.halt_max_steps": max_step,
        "arch.noise_scale": args.noise_scale,
        "arch.H_init_std": args.h_init_std,
        "arch.L_init_std": args.l_init_std,
    }
    api["_apply_overrides"](config, overrides, rank=0)
    api["_seed"](config.seed, 0)

    train_loader, train_metadata = api["create_dataloader"](
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=0,
        world_size=1,
    )
    del train_loader
    eval_loader, eval_metadata = api["create_dataloader"](
        config,
        "test",
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=0,
        world_size=1,
    )
    del eval_metadata

    train_state = api["init_train_state"](config, train_metadata, rank=0, world_size=1)
    api["load_model_state_dict"](
        train_state.model,
        ckpt.get("model", ckpt),
        strict=bool(getattr(config, "load_strict", False)),
        assign=True,
    )
    train_state.step = int(ckpt.get("step", 0))
    if config.ema and "ema" in ckpt:
        train_state = api["load_ema"](ckpt, train_state, config, 0)
    train_state.model.eval()
    api["_apply_runtime_arch_overrides"](train_state.model, overrides, 0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_state.model.to(device)
    selected_step_set = set(steps)

    all_records: List[Dict[str, Any]] = []
    processed = 0
    with torch.no_grad():
        for _set_name, batch, _batch_size in eval_loader:
            batch = api["tree_to_device"](batch, device)
            carry = model.initial_carry(batch)
            preds_by_step: Dict[int, np.ndarray] = {}
            for step in range(1, max_step + 1):
                carry, _loss, _metrics, detached, _all_finish = model(
                    carry=carry,
                    batch=batch,
                    return_keys=["preds"],
                )
                if step in selected_step_set:
                    preds_by_step[step] = detached["preds"].detach().cpu().numpy()

            inputs = batch["inputs"].detach().cpu().numpy()
            labels = batch["labels"].detach().cpu().numpy()
            batch_n = int(inputs.shape[0])
            for i in range(batch_n):
                rec_metrics = {
                    str(step): path_metrics(preds_by_step[step][i], labels[i])
                    for step in steps
                }
                final = rec_metrics[str(max_step)]
                all_records.append(
                    {
                        "case_index": processed + i,
                        "hardness": final["path_fp"] + final["path_fn"],
                        "input": inputs[i].astype(int).tolist(),
                        "label": labels[i].astype(int).tolist(),
                        "predictions": {
                            str(step): preds_by_step[step][i].astype(int).tolist()
                            for step in steps
                        },
                        "metrics": rec_metrics,
                    }
                )
            processed += batch_n
            if processed >= args.max_cases:
                break

    all_records.sort(key=lambda r: (r["hardness"], 1.0 - r["metrics"][str(max_step)]["path_f1"]), reverse=True)
    kept_cases = all_records[: args.num_vis_cases]
    aggregate = {
        str(step): mean_metrics([record["metrics"][str(step)] for record in all_records])
        for step in steps
    }
    payload = {
        "run_name": args.run_name,
        "condition": args.condition,
        "repo": str(repo),
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(ckpt.get("step", 0)),
        "dataset": str(config.dataset.data_path),
        "seed": int(config.seed),
        "batch_size": int(config.global_batch_size),
        "steps": steps,
        "processed_cases": len(all_records),
        "num_visualized_cases": len(kept_cases),
        "aggregate_by_step": aggregate,
        "cases": kept_cases,
        "elapsed_sec": time.time() - t0,
        "notes": args.notes,
    }
    (out_dir / "cases.json").write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")
    (out_dir / "index.html").write_text(render_html(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Render official EqR Maze case trajectories from a checkpoint.")
    parser.add_argument("--repo", required=True, help="Path to an official EqR clone.")
    parser.add_argument("--checkpoint", required=True, help="Path to an EqR checkpoint .pth.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--condition", default="base")
    parser.add_argument("--dataset-data-path", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-cases", type=int, default=128)
    parser.add_argument("--num-vis-cases", type=int, default=6)
    parser.add_argument("--steps-text", "--steps", default="1,4,8,16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-scale", type=float, default=0.5)
    parser.add_argument("--h-init-std", type=float, default=1.0)
    parser.add_argument("--l-init-std", type=float, default=1.0)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps({k: payload[k] for k in ("run_name", "condition", "processed_cases", "elapsed_sec")}, indent=2))


if __name__ == "__main__":
    main()
