from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any


CAUSAL_ARM = "causal_gdn2"
FUTURE_SEED_ARM = "future_seed_gdn2"
ARMS = (CAUSAL_ARM, FUTURE_SEED_ARM)
ARM_LABELS = {
    CAUSAL_ARM: "Causal GDN2",
    FUTURE_SEED_ARM: "GDN2 + FutureSeed",
}
METRICS = ("masked_accuracy", "masked_ce", "masked_exact")
METRIC_LABELS = {
    "masked_accuracy": "Masked accuracy",
    "masked_ce": "Masked CE",
    "masked_exact": "Exact windows",
}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"JSON input does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _require_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{context} must be finite")
    return number


def _case_rows(payload: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        payload = payload["cases"]
    if not isinstance(payload, list):
        raise ValueError(f"Expected a case list in {path}")
    if not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"Every case in {path} must be an object")
    return payload


def _validate_case(case: dict[str, Any], arm: str, row_index: int) -> None:
    context = f"{arm} case row {row_index}"
    required = {
        "case_index",
        "case_id",
        "errors",
        "tokens",
        "masked_tokens",
    }
    missing = sorted(required - case.keys())
    if missing:
        raise ValueError(f"{context} is missing keys: {', '.join(missing)}")
    if isinstance(case["case_index"], bool) or not isinstance(
        case["case_index"], int
    ):
        raise ValueError(f"{context}.case_index must be an integer")
    if not isinstance(case["case_id"], str) or not case["case_id"]:
        raise ValueError(f"{context}.case_id must be a non-empty string")
    if (
        isinstance(case["errors"], bool)
        or not isinstance(case["errors"], int)
        or case["errors"] < 0
    ):
        raise ValueError(f"{context}.errors must be a non-negative integer")
    if not isinstance(case["tokens"], list) or not all(
        isinstance(token, str) for token in case["tokens"]
    ):
        raise ValueError(f"{context}.tokens must be a list of strings")
    if not isinstance(case["masked_tokens"], list):
        raise ValueError(f"{context}.masked_tokens must be a list")

    seen_positions: set[int] = set()
    computed_errors = 0
    masked_required = {
        "position",
        "target",
        "prediction",
        "correct",
        "true_probability",
        "top5",
    }
    for masked_index, masked in enumerate(case["masked_tokens"]):
        masked_context = f"{context}.masked_tokens[{masked_index}]"
        if not isinstance(masked, dict):
            raise ValueError(f"{masked_context} must be an object")
        missing = sorted(masked_required - masked.keys())
        if missing:
            raise ValueError(
                f"{masked_context} is missing keys: {', '.join(missing)}"
            )
        position = masked["position"]
        if isinstance(position, bool) or not isinstance(position, int):
            raise ValueError(f"{masked_context}.position must be an integer")
        if position < 0 or position >= len(case["tokens"]):
            raise ValueError(f"{masked_context}.position is outside the token window")
        if position in seen_positions:
            raise ValueError(f"{context} repeats masked position {position}")
        seen_positions.add(position)
        if not isinstance(masked["target"], str) or not isinstance(
            masked["prediction"], str
        ):
            raise ValueError(
                f"{masked_context}.target and prediction must be strings"
            )
        if not isinstance(masked["correct"], bool):
            raise ValueError(f"{masked_context}.correct must be a boolean")
        probability = _require_number(
            masked["true_probability"], f"{masked_context}.true_probability"
        )
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"{masked_context}.true_probability must be between zero and one"
            )
        if not isinstance(masked["top5"], list):
            raise ValueError(f"{masked_context}.top5 must be a list")
        computed_errors += int(not masked["correct"])
    if computed_errors != case["errors"]:
        raise ValueError(
            f"{context}.errors is {case['errors']}, but masked_tokens contain "
            f"{computed_errors} errors"
        )


def _index_cases(rows: list[dict[str, Any]], arm: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row_index, case in enumerate(rows):
        _validate_case(case, arm, row_index)
        case_id = case["case_id"]
        if case_id in indexed:
            raise ValueError(f"{arm} contains duplicate case_id {case_id!r}")
        indexed[case_id] = case
    return indexed


def _validate_comparison(comparison: Any) -> dict[str, Any]:
    if not isinstance(comparison, dict):
        raise ValueError("Comparison JSON must contain an object")
    arms = comparison.get("arms")
    if not isinstance(arms, dict):
        raise ValueError("Comparison JSON is missing arms")
    for arm in ARMS:
        score = arms.get(arm)
        if not isinstance(score, dict) or not isinstance(score.get("metrics"), dict):
            raise ValueError(f"Comparison JSON is missing arms.{arm}.metrics")
        for metric in METRICS:
            if metric not in score["metrics"]:
                raise ValueError(
                    f"Comparison JSON is missing arms.{arm}.metrics.{metric}"
                )
            _require_number(
                score["metrics"][metric], f"arms.{arm}.metrics.{metric}"
            )
    if not isinstance(comparison.get("future_seed_vs_causal"), dict):
        raise ValueError("Comparison JSON is missing future_seed_vs_causal deltas")
    return comparison


def _masked_by_position(case: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["position"]): row for row in case["masked_tokens"]}


def _pair_case(
    causal: dict[str, Any], future_seed: dict[str, Any]
) -> dict[str, Any]:
    case_id = causal["case_id"]
    if causal["case_index"] != future_seed["case_index"]:
        raise ValueError(f"Matched case {case_id} has different case_index values")
    if causal["tokens"] != future_seed["tokens"]:
        raise ValueError(f"Matched case {case_id} has different original tokens")

    causal_masked = _masked_by_position(causal)
    future_masked = _masked_by_position(future_seed)
    if set(causal_masked) != set(future_masked):
        raise ValueError(f"Matched case {case_id} has different masked positions")
    for position in causal_masked:
        if causal_masked[position]["target"] != future_masked[position]["target"]:
            raise ValueError(
                f"Matched case {case_id} has different target at position {position}"
            )

    repair_positions = sorted(
        position
        for position in causal_masked
        if not causal_masked[position]["correct"]
        and future_masked[position]["correct"]
    )
    regression_positions = sorted(
        position
        for position in causal_masked
        if causal_masked[position]["correct"]
        and not future_masked[position]["correct"]
    )
    changed_wrong_positions = sorted(
        position
        for position in causal_masked
        if not causal_masked[position]["correct"]
        and not future_masked[position]["correct"]
        and causal_masked[position]["prediction"]
        != future_masked[position]["prediction"]
    )
    prediction_change_positions = sorted(
        position
        for position in causal_masked
        if causal_masked[position]["prediction"]
        != future_masked[position]["prediction"]
    )
    probability_deltas = [
        float(future_masked[position]["true_probability"])
        - float(causal_masked[position]["true_probability"])
        for position in causal_masked
    ]
    net_error_reduction = int(causal["errors"]) - int(future_seed["errors"])
    transition_balance = len(repair_positions) - len(regression_positions)
    if net_error_reduction != transition_balance:
        raise ValueError(
            f"Matched case {case_id} has inconsistent error and transition counts"
        )

    if repair_positions and regression_positions:
        classification = "mixed"
    elif repair_positions:
        classification = "repair"
    elif regression_positions:
        classification = "regression"
    elif prediction_change_positions:
        classification = "changed"
    else:
        classification = "stable"
    return {
        "case_index": causal["case_index"],
        "case_id": case_id,
        "classification": classification,
        "repair_positions": repair_positions,
        "regression_positions": regression_positions,
        "changed_wrong_positions": changed_wrong_positions,
        "prediction_change_positions": prediction_change_positions,
        "repair_count": len(repair_positions),
        "regression_count": len(regression_positions),
        "net_error_reduction": net_error_reduction,
        "mean_target_probability_delta": (
            sum(probability_deltas) / len(probability_deltas)
            if probability_deltas
            else 0.0
        ),
        "arms": {
            CAUSAL_ARM: causal,
            FUTURE_SEED_ARM: future_seed,
        },
    }


def pair_cases(
    cases_by_arm: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    causal_ids = set(cases_by_arm[CAUSAL_ARM])
    future_seed_ids = set(cases_by_arm[FUTURE_SEED_ARM])
    if causal_ids != future_seed_ids:
        raise ValueError(
            "The two case files contain different case IDs: "
            f"causal_only={sorted(causal_ids - future_seed_ids)} "
            f"future_seed_only={sorted(future_seed_ids - causal_ids)}"
        )
    if not causal_ids:
        raise ValueError("The two case files contain no common case_id values")
    return [
        _pair_case(
            cases_by_arm[CAUSAL_ARM][case_id],
            cases_by_arm[FUTURE_SEED_ARM][case_id],
        )
        for case_id in sorted(causal_ids)
    ]


def _case_tie_breaker(case: dict[str, Any]) -> tuple[int, str]:
    return int(case["case_index"]), str(case["case_id"])


def select_cases(paired: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("Case limit must be positive")
    limit = min(limit, len(paired))
    repairs = sorted(
        (case for case in paired if case["repair_count"]),
        key=lambda case: (
            -int(case["net_error_reduction"]),
            -int(case["repair_count"]),
            int(case["regression_count"]),
            -int(case["arms"][CAUSAL_ARM]["errors"]),
            *_case_tie_breaker(case),
        ),
    )
    regressions = sorted(
        (case for case in paired if case["regression_count"]),
        key=lambda case: (
            int(case["repair_count"] > 0),
            -int(case["regression_count"]),
            int(case["repair_count"]),
            -int(case["arms"][FUTURE_SEED_ARM]["errors"]),
            *_case_tie_breaker(case),
        ),
    )
    remaining = sorted(
        paired,
        key=lambda case: (
            -(int(case["repair_count"]) + int(case["regression_count"])),
            -abs(int(case["net_error_reduction"])),
            -len(case["prediction_change_positions"]),
            -max(
                int(case["arms"][CAUSAL_ARM]["errors"]),
                int(case["arms"][FUTURE_SEED_ARM]["errors"]),
            ),
            -abs(float(case["mean_target_probability_delta"])),
            *_case_tie_breaker(case),
        ),
    )

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add(candidates: list[dict[str, Any]], count: int, bucket: str) -> None:
        added = 0
        for candidate in candidates:
            if added == count or len(selected) == limit:
                return
            if candidate["case_id"] in selected_ids:
                continue
            row = dict(candidate)
            row["selection_bucket"] = bucket
            selected.append(row)
            selected_ids.add(candidate["case_id"])
            added += 1

    regression_reserve = 0
    if regressions and limit > 1:
        regression_reserve = min(len(regressions), max(1, limit // 3))
    add(repairs, limit - regression_reserve, "repair_priority")
    add(regressions, regression_reserve, "regression_watch")
    add(repairs, limit - len(selected), "repair_priority")
    add(regressions, limit - len(selected), "regression_watch")
    add(remaining, limit - len(selected), "most_changed_or_hard")

    for rank, case in enumerate(selected, start=1):
        case["selection_rank"] = rank
    return selected


def _metric_delta(comparison: dict[str, Any], metric: str) -> float:
    deltas = comparison["future_seed_vs_causal"]
    aliases = (
        f"{metric}_delta",
        f"{metric.removeprefix('masked_')}_delta",
        metric,
    )
    for key in aliases:
        if key in deltas:
            return _require_number(deltas[key], f"future_seed_vs_causal.{key}")
    causal = comparison["arms"][CAUSAL_ARM]["metrics"][metric]
    future_seed = comparison["arms"][FUTURE_SEED_ARM]["metrics"][metric]
    return float(future_seed) - float(causal)


def _format_metric(value: Any) -> str:
    return f"{float(value):.4f}"


def _format_delta(value: float) -> str:
    if abs(value) < 0.00005:
        return "0.0000"
    return f"{value:+.4f}"


def _delta_class(metric: str, value: float) -> str:
    if abs(value) < 1e-15:
        return "neutral"
    improvement = value < 0.0 if metric == "masked_ce" else value > 0.0
    return "good" if improvement else "bad"


def _top5_item(item: Any) -> tuple[str, Any | None]:
    if isinstance(item, dict):
        token = item.get(
            "token",
            item.get("prediction", item.get("text", item.get("token_id", "?"))),
        )
        probability = item.get("probability", item.get("prob"))
        return str(token), probability
    if isinstance(item, (list, tuple)) and len(item) == 2:
        return str(item[0]), item[1]
    return str(item), None


def _top5_html(items: list[Any]) -> str:
    rendered = []
    for item in items[:5]:
        token, probability = _top5_item(item)
        if probability is None:
            probability_html = ""
        else:
            try:
                probability_html = f"<span>{float(probability):.3f}</span>"
            except (TypeError, ValueError):
                probability_html = f"<span>{html.escape(str(probability))}</span>"
        rendered.append(
            f"<li><code>{html.escape(token)}</code>{probability_html}</li>"
        )
    return f'<ol class="top5">{"".join(rendered)}</ol>'


def _token_outcome(causal: dict[str, Any], future_seed: dict[str, Any]) -> str:
    if not causal["correct"] and future_seed["correct"]:
        return "repair"
    if causal["correct"] and not future_seed["correct"]:
        return "regression"
    if causal["correct"] and future_seed["correct"]:
        return "both-correct"
    return "both-wrong"


def _token_stream(case: dict[str, Any]) -> str:
    causal = case["arms"][CAUSAL_ARM]
    future_seed = case["arms"][FUTURE_SEED_ARM]
    causal_masked = _masked_by_position(causal)
    future_masked = _masked_by_position(future_seed)
    rendered = []
    for position, token in enumerate(causal["tokens"]):
        if position not in causal_masked:
            rendered.append(f'<span class="token">{html.escape(token)}</span>')
            continue
        causal_token = causal_masked[position]
        future_token = future_masked[position]
        outcome = _token_outcome(causal_token, future_token)
        title = (
            f"masked position {position}; target={causal_token['target']}; "
            f"causal={causal_token['prediction']}; "
            f"FutureSeed={future_token['prediction']}"
        )
        rendered.append(
            f'<span class="token masked-token {outcome}" '
            f'title="{html.escape(title, quote=True)}">'
            f'<b class="mask-index">m{position}</b>{html.escape(token)}</span>'
        )
    return "".join(rendered)


def _masked_rows(case: dict[str, Any]) -> str:
    causal = _masked_by_position(case["arms"][CAUSAL_ARM])
    future_seed = _masked_by_position(case["arms"][FUTURE_SEED_ARM])
    rows = []
    for position in sorted(causal):
        causal_token = causal[position]
        future_token = future_seed[position]
        outcome = _token_outcome(causal_token, future_token)
        if outcome == "repair":
            outcome_label = "FS repair"
        elif outcome == "regression":
            outcome_label = "FS regression"
        elif outcome == "both-correct":
            outcome_label = "Both correct"
        elif causal_token["prediction"] != future_token["prediction"]:
            outcome_label = "Both wrong, changed"
        else:
            outcome_label = "Both wrong"
        probability_delta = float(future_token["true_probability"]) - float(
            causal_token["true_probability"]
        )
        rows.append(
            f'<tr class="transition-{outcome}">'
            f"<td>{position}</td>"
            f"<td><code>{html.escape(causal_token['target'])}</code></td>"
            f'<td class="prediction {"correct" if causal_token["correct"] else "wrong"}">'
            f"<code>{html.escape(causal_token['prediction'])}</code></td>"
            f"<td>{float(causal_token['true_probability']):.4f}</td>"
            f"<td>{_top5_html(causal_token['top5'])}</td>"
            f'<td class="prediction {"correct" if future_token["correct"] else "wrong"}">'
            f"<code>{html.escape(future_token['prediction'])}</code></td>"
            f"<td>{float(future_token['true_probability']):.4f}</td>"
            f"<td>{_top5_html(future_token['top5'])}</td>"
            f'<td><strong class="outcome {outcome}">{outcome_label}</strong>'
            f'<small class="prob-delta">P(target) {probability_delta:+.4f}</small></td>'
            "</tr>"
        )
    return "".join(rows)


def _case_html(case: dict[str, Any]) -> str:
    causal = case["arms"][CAUSAL_ARM]
    future_seed = case["arms"][FUTURE_SEED_ARM]
    masked_count = len(causal["masked_tokens"])
    net = int(case["net_error_reduction"])
    if net > 0:
        net_text = f"FS repairs {net} net error{'s' if net != 1 else ''}"
        net_class = "good"
    elif net < 0:
        net_text = f"FS adds {-net} net error{'s' if net != -1 else ''}"
        net_class = "bad"
    else:
        net_text = "No net error change"
        net_class = "neutral"
    return f"""
<article class="case case-{case['classification']}">
  <div class="case-header">
    <div>
      <p class="case-kicker">Selected {case['selection_rank']:02d} / {html.escape(case['selection_bucket'].replace('_', ' '))}</p>
      <h2>Window {case['case_index']} <code>{html.escape(case['case_id'])}</code></h2>
    </div>
    <div class="case-counts">
      <span>Causal <b>{causal['errors']}/{masked_count}</b></span>
      <span>FutureSeed <b>{future_seed['errors']}/{masked_count}</b></span>
      <span class="{net_class}">{html.escape(net_text)}</span>
    </div>
  </div>
  <div class="token-stream">{_token_stream(case)}</div>
  <div class="table-wrap"><table class="tokens-table">
    <thead><tr>
      <th>Pos</th><th>Target</th>
      <th>Causal prediction</th><th>Causal P(target)</th><th>Causal top 5</th>
      <th>FutureSeed prediction</th><th>FS P(target)</th><th>FS top 5</th><th>Transition</th>
    </tr></thead>
    <tbody>{_masked_rows(case)}</tbody>
  </table></div>
</article>"""


def render_html(
    comparison: dict[str, Any],
    selected: list[dict[str, Any]],
    common_case_count: int,
) -> str:
    metric_rows = []
    for metric in METRICS:
        causal_value = comparison["arms"][CAUSAL_ARM]["metrics"][metric]
        future_value = comparison["arms"][FUTURE_SEED_ARM]["metrics"][metric]
        delta = _metric_delta(comparison, metric)
        metric_rows.append(
            "<tr>"
            f"<th>{METRIC_LABELS[metric]}</th>"
            f"<td>{_format_metric(causal_value)}</td>"
            f"<td>{_format_metric(future_value)}</td>"
            f'<td class="delta {_delta_class(metric, delta)}">{_format_delta(delta)}</td>'
            "</tr>"
        )
    repair_events = sum(int(case["repair_count"]) for case in selected)
    regression_events = sum(int(case["regression_count"]) for case in selected)
    case_sections = "".join(_case_html(case) for case in selected)
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light">
<title>P-CAUSAL-019: WordPiece FutureSeed MLM</title>
<style>
:root {{ --ink:#1d2421; --muted:#626c67; --line:#d7ddda; --soft:#f4f6f5; --paper:#fff; --good:#126448; --good-bg:#e3f2eb; --bad:#a23828; --bad-bg:#fae8e4; --warn:#8a5b12; --warn-bg:#fbf0d8; --focus:#285f9c; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:var(--soft); font:14px/1.45 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; letter-spacing:0; }}
main {{ width:min(1480px,calc(100% - 32px)); margin:0 auto; padding:28px 0 64px; }}
h1 {{ margin:0; font-size:28px; font-weight:720; }} h2 {{ margin:0; font-size:16px; }}
p {{ margin:0; }} code {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:.92em; }}
.lede {{ color:var(--muted); margin-top:5px; max-width:920px; }}
.summary {{ margin:22px 0 18px; display:grid; grid-template-columns:minmax(520px,1.35fr) minmax(300px,.65fr); gap:14px; align-items:stretch; }}
.summary-panel,.selection-panel {{ background:var(--paper); border:1px solid var(--line); border-radius:5px; overflow:hidden; }}
.summary h2,.selection-panel h2 {{ padding:11px 14px; border-bottom:1px solid var(--line); background:#eef1ef; }}
.summary table {{ width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }}
.summary th,.summary td {{ padding:9px 14px; border-bottom:1px solid var(--line); text-align:right; }}
.summary tbody tr:last-child th,.summary tbody tr:last-child td {{ border-bottom:0; }} .summary th {{ text-align:left; font-weight:600; }}
.summary thead th {{ color:var(--muted); font-size:12px; background:#fafbfa; text-align:right; }} .summary thead th:first-child {{ text-align:left; }}
.selection-facts {{ display:grid; grid-template-columns:1fr 1fr; }} .selection-facts div {{ padding:12px 14px; border-bottom:1px solid var(--line); }}
.selection-facts div:nth-child(odd) {{ border-right:1px solid var(--line); }} .selection-facts div:nth-last-child(-n+2) {{ border-bottom:0; }}
.selection-facts dt {{ color:var(--muted); font-size:12px; }} .selection-facts dd {{ margin:2px 0 0; font-size:20px; font-weight:700; font-variant-numeric:tabular-nums; }}
.good {{ color:var(--good); }} .bad {{ color:var(--bad); }} .neutral {{ color:var(--muted); }} .delta {{ font-weight:750; font-variant-numeric:tabular-nums; }}
.legend {{ display:flex; flex-wrap:wrap; gap:14px; color:var(--muted); margin:0 0 14px; font-size:12px; }} .legend span {{ display:flex; align-items:center; gap:6px; }}
.legend i {{ width:11px; height:11px; display:inline-block; border:2px solid; background:#fff; }} .legend .repair {{ border-color:var(--good); }} .legend .regression {{ border-color:var(--bad); }} .legend .both-correct {{ border-color:var(--focus); }} .legend .both-wrong {{ border-color:var(--warn); }}
.case {{ background:var(--paper); border:1px solid var(--line); border-radius:5px; margin:0 0 16px; overflow:hidden; }}
.case-header {{ min-height:68px; display:flex; align-items:center; justify-content:space-between; gap:18px; padding:12px 14px; border-bottom:1px solid var(--line); }}
.case-kicker {{ color:var(--muted); font-size:11px; text-transform:uppercase; margin-bottom:3px; }} .case h2 code {{ color:var(--muted); font-weight:500; margin-left:6px; overflow-wrap:anywhere; }}
.case-counts {{ display:flex; flex-wrap:wrap; justify-content:flex-end; gap:8px 15px; font-variant-numeric:tabular-nums; }} .case-counts span {{ white-space:nowrap; }}
.token-stream {{ display:flex; flex-wrap:wrap; align-items:flex-end; gap:5px 3px; padding:15px 14px; background:#fafbfa; border-bottom:1px solid var(--line); font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; overflow-wrap:anywhere; }}
.token {{ min-height:25px; display:inline-flex; align-items:flex-end; padding:3px 2px; border:1px solid transparent; }}
.masked-token {{ position:relative; min-height:42px; padding:14px 5px 3px; border-width:2px; background:#fff; }} .mask-index {{ position:absolute; top:1px; left:4px; color:var(--muted); font:9px/1 ui-sans-serif,system-ui,sans-serif; }}
.masked-token.repair {{ border-color:var(--good); background:var(--good-bg); }} .masked-token.regression {{ border-color:var(--bad); background:var(--bad-bg); }} .masked-token.both-correct {{ border-color:var(--focus); }} .masked-token.both-wrong {{ border-color:var(--warn); background:var(--warn-bg); }}
.table-wrap {{ width:100%; overflow:auto; }} .tokens-table {{ width:100%; min-width:1320px; border-collapse:collapse; table-layout:fixed; font-size:12px; }}
.tokens-table th,.tokens-table td {{ padding:8px 9px; border-bottom:1px solid var(--line); vertical-align:top; text-align:left; font-variant-numeric:tabular-nums; overflow-wrap:anywhere; }}
.tokens-table thead th {{ position:sticky; top:0; z-index:1; color:var(--muted); background:#eef1ef; font-size:11px; }} .tokens-table tbody tr:last-child td {{ border-bottom:0; }}
.tokens-table th:nth-child(1) {{ width:48px; }} .tokens-table th:nth-child(2) {{ width:105px; }} .tokens-table th:nth-child(3),.tokens-table th:nth-child(6) {{ width:125px; }} .tokens-table th:nth-child(4),.tokens-table th:nth-child(7) {{ width:100px; }} .tokens-table th:nth-child(5),.tokens-table th:nth-child(8) {{ width:260px; }} .tokens-table th:nth-child(9) {{ width:145px; }}
.prediction.correct {{ color:var(--good); background:var(--good-bg); font-weight:750; }} .prediction.wrong {{ color:var(--bad); background:var(--bad-bg); font-weight:750; }}
.top5 {{ list-style:none; margin:0; padding:0; display:grid; grid-template-columns:1fr 1fr; gap:2px 9px; }} .top5 li {{ display:flex; justify-content:space-between; gap:6px; min-width:0; }} .top5 code {{ overflow-wrap:anywhere; }} .top5 span {{ color:var(--muted); white-space:nowrap; }}
.outcome {{ display:block; font-size:11px; }} .outcome.repair {{ color:var(--good); }} .outcome.regression {{ color:var(--bad); }} .outcome.both-correct {{ color:var(--focus); }} .outcome.both-wrong {{ color:var(--warn); }} .prob-delta {{ color:var(--muted); display:block; margin-top:3px; white-space:nowrap; }}
@media (max-width:850px) {{ main {{ width:min(100% - 18px,1480px); padding-top:18px; }} h1 {{ font-size:23px; }} .summary {{ grid-template-columns:1fr; }} .case-header {{ align-items:flex-start; flex-direction:column; }} .case-counts {{ justify-content:flex-start; }} }}
@media (max-width:560px) {{ .summary {{ grid-template-columns:minmax(0,1fr); }} .summary-panel {{ overflow-x:auto; }} .summary table {{ min-width:510px; }} .selection-facts {{ grid-template-columns:1fr; }} .selection-facts div,.selection-facts div:nth-child(odd),.selection-facts div:nth-last-child(-n+2) {{ border-right:0; border-bottom:1px solid var(--line); }} .selection-facts div:last-child {{ border-bottom:0; }} }}
</style></head><body><main>
<header><h1>WordPiece masked-token recovery</h1><p class="lede">P-CAUSAL-019 compares matched official-FLA causal GDN2 and native FutureSeed on identical validation windows, masks, and targets.</p></header>
<section class="summary">
  <div class="summary-panel"><h2>Registered endpoint metrics</h2><table>
    <thead><tr><th>Metric</th><th>{ARM_LABELS[CAUSAL_ARM]}</th><th>{ARM_LABELS[FUTURE_SEED_ARM]}</th><th>FS - causal</th></tr></thead>
    <tbody>{''.join(metric_rows)}</tbody>
  </table></div>
  <div class="selection-panel"><h2>Same-window audit</h2><dl class="selection-facts">
    <div><dt>Common windows</dt><dd>{common_case_count}</dd></div>
    <div><dt>Selected windows</dt><dd>{len(selected)}</dd></div>
    <div><dt>FS repair events</dt><dd class="good">{repair_events}</dd></div>
    <div><dt>FS regression events</dt><dd class="bad">{regression_events}</dd></div>
  </dl></div>
</section>
<div class="legend"><span><i class="repair"></i>FS repair</span><span><i class="regression"></i>FS regression</span><span><i class="both-correct"></i>both correct</span><span><i class="both-wrong"></i>both wrong</span></div>
{case_sections}
</main></body></html>
"""


def _discover_cases(comparison_path: Path, arm: str) -> Path:
    parent = comparison_path.parent
    candidates = (
        parent / arm / "cases.json",
        parent / "arms" / arm / "cases.json",
        parent / f"{arm}_cases.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    choices = ", ".join(str(candidate) for candidate in candidates)
    raise ValueError(f"Could not find {arm} cases; checked {choices}")


def _selected_payload(
    comparison: dict[str, Any],
    selected: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    requested_cases: int,
) -> dict[str, Any]:
    return {
        "arms": list(ARMS),
        "metrics": {
            arm: {
                metric: comparison["arms"][arm]["metrics"][metric]
                for metric in METRICS
            }
            for arm in ARMS
        },
        "future_seed_vs_causal": {
            f"{metric}_delta": _metric_delta(comparison, metric)
            for metric in METRICS
        },
        "selection": {
            "policy": (
                "same-window FutureSeed repairs first, with regression coverage, "
                "then the most changed or hardest remaining windows"
            ),
            "requested_cases": requested_cases,
            "common_cases": len(paired),
            "repair_cases_available": sum(
                int(case["repair_count"] > 0) for case in paired
            ),
            "regression_cases_available": sum(
                int(case["regression_count"] > 0) for case in paired
            ),
            "selected_cases": len(selected),
        },
        "cases": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render matched causal GDN2/FutureSeed WordPiece MLM cases as a "
            "standalone HTML report."
        )
    )
    parser.add_argument(
        "--comparison",
        "--comparison-json",
        dest="comparison",
        type=Path,
        required=True,
    )
    parser.add_argument("--causal-cases", type=Path)
    parser.add_argument(
        "--future-seed-cases", "--future-cases", dest="future_seed_cases", type=Path
    )
    parser.add_argument("--html", "--output-html", dest="html", type=Path)
    parser.add_argument("--output-dir", "--visual-dir", dest="output_dir", type=Path)
    parser.add_argument("--selected-cases", type=Path)
    parser.add_argument("--cases", type=int, default=12)
    args = parser.parse_args()

    if args.cases <= 0:
        parser.error("--cases must be positive")
    if args.html is not None and args.output_dir is not None:
        parser.error("use either --html or --output-dir, not both")

    comparison_path = args.comparison
    causal_path = args.causal_cases or _discover_cases(
        comparison_path, CAUSAL_ARM
    )
    future_seed_path = args.future_seed_cases or _discover_cases(
        comparison_path, FUTURE_SEED_ARM
    )
    if args.html is not None:
        html_path = args.html
    elif args.output_dir is not None:
        html_path = args.output_dir / "index.html"
    else:
        html_path = comparison_path.parent / "index.html"
    selected_path = args.selected_cases or html_path.parent / "selected_cases.json"

    comparison = _validate_comparison(_read_json(comparison_path))
    cases_by_arm = {
        CAUSAL_ARM: _index_cases(
            _case_rows(_read_json(causal_path), causal_path), CAUSAL_ARM
        ),
        FUTURE_SEED_ARM: _index_cases(
            _case_rows(_read_json(future_seed_path), future_seed_path),
            FUTURE_SEED_ARM,
        ),
    }
    paired = pair_cases(cases_by_arm)
    selected = select_cases(paired, args.cases)

    html_path.parent.mkdir(parents=True, exist_ok=True)
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(
        render_html(comparison, selected, len(paired)), encoding="utf-8"
    )
    selected_path.write_text(
        json.dumps(
            _selected_payload(comparison, selected, paired, args.cases),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "html": str(html_path),
                "selected_cases": str(selected_path),
                "common_cases": len(paired),
                "selected": len(selected),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
