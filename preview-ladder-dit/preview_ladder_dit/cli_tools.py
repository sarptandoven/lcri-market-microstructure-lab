from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping

from .fixtures import FIXTURE_CASES, write_fixtures
from .commitments import commitment_loss, extract_commitments
from .metrics import preview_final_consistency_report
from .product import build_preview_scorecard
from .schema import REPORT_SCHEMA_VERSION, load_and_validate_run_report, validate_run_report

PRIMARY_METRICS = (
    "preview_final_l1",
    "boundary_consistency_error",
    "boundary_signed_bias",
    "temporal_flicker_delta",
    "background_preservation_error",
    "background_temporal_leak",
    "trajectory_center_error",
    "trajectory_acceleration_drift",
    "mask_occupancy_delta",
    "temporal_edge_jitter",
    "confidence_weighted_l1",
    "preview_confidence_release_rate",
    "occupancy_iou_error",
    "low_frequency_drift",
    "local_temporal_residual_flicker",
    "commitment_weighted_error",
)

DEFAULT_WEIGHTS = {
    "preview_final_l1": 1.00,
    "boundary_consistency_error": 1.25,
    "boundary_signed_bias": 0.75,
    "temporal_flicker_delta": 1.25,
    "background_preservation_error": 1.50,
    "background_temporal_leak": 1.00,
    "trajectory_center_error": 1.00,
    "trajectory_acceleration_drift": 0.75,
    "mask_occupancy_delta": 0.75,
    "temporal_edge_jitter": 0.75,
    "confidence_weighted_l1": 0.75,
    "preview_confidence_release_rate": 0.25,
    "occupancy_iou_error": 0.50,
    "low_frequency_drift": 0.75,
    "local_temporal_residual_flicker": 0.75,
    "commitment_weighted_error": 1.00,
}


def generate_fixtures_command(out: str | Path, *, cases: list[str] | None = None, frames: int = 8, height: int = 32, width: int = 32) -> dict[str, Any]:
    selected = cases or list(FIXTURE_CASES)
    paths = write_fixtures(out, cases=selected, frames=frames, height=height, width=width)
    return {
        "ok": True,
        "command": "generate-fixtures",
        "manifest": str(Path(out) / "manifest.json"),
        "files": [str(p) for p in paths],
        "case_count": len(paths),
    }


def evaluate_fixtures_command(fixtures: str | Path, out: str | Path) -> dict[str, Any]:
    fixture_paths = _collect_fixture_paths(fixtures)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_paths: list[Path] = []
    rows: list[dict[str, Any]] = []
    for fixture_path in fixture_paths:
        data = json.loads(fixture_path.read_text(encoding="utf-8"))
        metrics = preview_final_consistency_report(
            source=data["source"], preview=data["preview"], final=data["final"], mask=data["mask"]
        ).to_dict()
        commitment_packet = extract_commitments(data["preview"], data["mask"])
        metrics["commitment_packet_loss"] = commitment_loss(data["preview"], data["final"], data["mask"], commitment_packet)
        score = aggregate_consistency_score(metrics)
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "run_id": f"synthetic-evaluate-{data['case']}",
            "task_id": f"synthetic-{data['case']}",
            "task": {"task_type": "synthetic_masked_replacement", "case": data["case"], "seed": 0},
            "submission": {
                "method_name": "synthetic_fixture_evaluator",
                "method_version": "0.1.0",
                "backend_family": "synthetic",
                "preview_mode": "fixture_preview_array",
                "final_mode": "fixture_final_array",
                "report_schema_version": REPORT_SCHEMA_VERSION,
                "artifact_hashes": {
                    "fixture_json": _sha256_bytes(fixture_path.read_bytes()),
                    "preview_array": _sha256_json(data["preview"]),
                    "final_array": _sha256_json(data["final"]),
                    "mask_array": _sha256_json(data["mask"]),
                },
            },
            "artifacts": {
                "preview": None,
                "final": None,
                "commitments": commitment_packet.to_dict(),
                "uncertainty": {"source": "deterministic_boundary_threshold_proxy", "mean_boundary_uncertainty": commitment_packet.mean_boundary_uncertainty},
            },
            "metrics": {**metrics, "aggregate_consistency_score": score},
            "latency_log": {"events": []},
            "model": {"backend": "fixture_evaluator", "commitment_policy": "accepted_preview_is_contract", "scheduler": "deterministic_fixture"},
            "environment": {"fixture": str(fixture_path), "case": data["case"]},
        }
        validate_run_report(report)
        path = out_dir / f"report-{data['case']}.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        report_paths.append(path)
        rows.append({"case": data["case"], **metrics, "aggregate_consistency_score": score})
    summary = summarize_rows(rows)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_metrics_csv(out_dir / "metrics.csv", rows)
    return {"ok": True, "reports": [str(p) for p in report_paths], "summary": str(summary_path), "metrics_csv": str(out_dir / "metrics.csv")}


def validate_submission_command(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    report_paths = _collect_report_paths(root)
    errors: list[str] = []
    warnings: list[str] = []
    metric_presence = {metric: 0 for metric in PRIMARY_METRICS}
    aggregate_scores: list[float] = []
    for report_path in report_paths:
        try:
            report = load_and_validate_run_report(report_path)
            metrics = report["metrics"]
            for metric in PRIMARY_METRICS:
                if metric in metrics:
                    metric_presence[metric] += 1
                else:
                    errors.append(f"{report_path}: missing metric {metric}")
            if "aggregate_consistency_score" in metrics:
                aggregate_scores.append(float(metrics["aggregate_consistency_score"]))
            if not report.get("latency_log", {}).get("events"):
                warnings.append(f"{report_path}: no latency events, acceptable for synthetic fixtures but not for product benchmark submissions")
        except Exception as exc:  # validation command should return all failures, not fail fast
            errors.append(f"{report_path}: {exc}")
    if not report_paths:
        errors.append(f"no report JSON files found under {root}")
    result = {
        "ok": not errors,
        "report_count": len(report_paths),
        "metric_presence": metric_presence,
        "aggregate_score_mean": statistics.fmean(aggregate_scores) if aggregate_scores else None,
        "warnings": warnings,
        "errors": errors,
    }
    (root / "validation.json" if root.is_dir() else root.with_suffix(".validation.json")).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def compare_command(left: str | Path, right: str | Path, out: str | Path | None = None) -> dict[str, Any]:
    left_rows = load_metric_rows(left)
    right_rows = load_metric_rows(right)
    left_by_task = {r["task_id"]: r for r in left_rows}
    right_by_task = {r["task_id"]: r for r in right_rows}
    common = sorted(set(left_by_task) & set(right_by_task))
    if not common:
        raise ValueError("no common task_id values to compare")
    per_task: list[dict[str, Any]] = []
    for task_id in common:
        row = {"task_id": task_id}
        for metric in PRIMARY_METRICS + ("aggregate_consistency_score",):
            if metric in left_by_task[task_id] and metric in right_by_task[task_id]:
                lval = float(left_by_task[task_id][metric])
                rval = float(right_by_task[task_id][metric])
                row[f"{metric}.left"] = lval
                row[f"{metric}.right"] = rval
                row[f"{metric}.delta_right_minus_left"] = rval - lval
        per_task.append(row)
    metric_deltas: dict[str, float] = {}
    for metric in PRIMARY_METRICS + ("aggregate_consistency_score",):
        deltas = [r[f"{metric}.delta_right_minus_left"] for r in per_task if f"{metric}.delta_right_minus_left" in r]
        if deltas:
            metric_deltas[metric] = statistics.fmean(deltas)
    result = {
        "ok": True,
        "task_count": len(common),
        "lower_is_better": list(PRIMARY_METRICS) + ["aggregate_consistency_score"],
        "mean_delta_right_minus_left": metric_deltas,
        "winner_by_aggregate": "right" if metric_deltas.get("aggregate_consistency_score", math.inf) < 0 else "left_or_tie",
        "per_task": per_task,
    }
    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def paper_figures_command(reports: str | Path, out: str | Path) -> dict[str, Any]:
    rows = load_metric_rows(reports)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = out_dir / "figure_metrics_by_task.csv"
    _write_metrics_csv(metrics_csv, rows)
    rank_rows = sorted(rows, key=lambda r: float(r.get("aggregate_consistency_score", aggregate_consistency_score_from_row(r))))
    rank_csv = out_dir / "figure_ranked_methods.csv"
    _write_metrics_csv(rank_csv, rank_rows)
    spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "description": "Preview-Final Consistency aggregate score by task. Lower is better.",
        "data": {"url": metrics_csv.name},
        "mark": "bar",
        "encoding": {
            "x": {"field": "task_id", "type": "nominal", "sort": None},
            "y": {"field": "aggregate_consistency_score", "type": "quantitative"},
            "tooltip": [{"field": m, "type": "quantitative"} for m in ("aggregate_consistency_score",) + PRIMARY_METRICS],
        },
    }
    spec_path = out_dir / "vega_preview_final_score.json"
    spec_path.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")
    caption_path = out_dir / "captions.md"
    caption_path.write_text(
        "Figure 1. Preview-Final Consistency score across masked replacement stress cases. "
        "The score is a weighted sum of lower-is-better structural, boundary, temporal, background, and mask occupancy errors.\n",
        encoding="utf-8",
    )
    return {"ok": True, "files": [str(metrics_csv), str(rank_csv), str(spec_path), str(caption_path)]}


def export_paper_tables_command(reports: str | Path, out: str | Path) -> dict[str, Any]:
    """Export stable reviewer-facing CSV tables from validated reports.

    ``paper-figures`` emits plot-oriented artifacts. This command emits tables
    intended for paper drafts and release notes: ranked per-task metrics, metric
    summary statistics, and one aggregate row keyed by the primary synthetic
    preview-final consistency score.
    """

    report_items = _load_report_items(reports)
    if not report_items:
        raise ValueError(f"no report JSON files found under {reports}")

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_task_rows: list[dict[str, Any]] = []
    for report_path, report in report_items:
        metrics = dict(report["metrics"])
        if "aggregate_consistency_score" not in metrics:
            metrics["aggregate_consistency_score"] = aggregate_consistency_score(metrics)
        per_task_rows.append(
            {
                "rank": 0,
                "task_id": report["task_id"],
                "run_id": report["run_id"],
                "case": report.get("task", {}).get("case", ""),
                "method_name": report.get("submission", {}).get("method_name", ""),
                "backend_family": report.get("submission", {}).get("backend_family", report.get("model", {}).get("backend", "")),
                "report_path": str(report_path),
                **metrics,
            }
        )
    per_task_rows.sort(key=lambda row: float(row["aggregate_consistency_score"]))
    for index, row in enumerate(per_task_rows, start=1):
        row["rank"] = index

    metric_rows: list[dict[str, Any]] = []
    for metric in PRIMARY_METRICS + ("aggregate_consistency_score",):
        values = [float(row[metric]) for row in per_task_rows if metric in row and row[metric] != ""]
        if values:
            metric_rows.append(
                {
                    "metric": metric,
                    "direction": "lower_is_better",
                    "count": len(values),
                    "mean": statistics.fmean(values),
                    "min": min(values),
                    "p95": percentile(values, 95),
                    "max": max(values),
                }
            )

    aggregate_score_row = next(row for row in metric_rows if row["metric"] == "aggregate_consistency_score")
    aggregate_row: dict[str, Any] = {
        "report_count": len(per_task_rows),
        "task_count": len({row["task_id"] for row in per_task_rows}),
        "method_name": _single_or_mixed(row.get("method_name", "") for row in per_task_rows),
        "backend_family": _single_or_mixed(row.get("backend_family", "") for row in per_task_rows),
        "best_task_id": per_task_rows[0]["task_id"],
        "worst_task_id": per_task_rows[-1]["task_id"],
        "primary_score": "aggregate_consistency_score",
        "primary_score_mean": aggregate_score_row["mean"],
        "primary_score_p95": aggregate_score_row["p95"],
    }
    for metric_row in metric_rows:
        metric = str(metric_row["metric"])
        aggregate_row[f"{metric}_mean"] = metric_row["mean"]
        aggregate_row[f"{metric}_p95"] = metric_row["p95"]

    aggregate_path = out_dir / "table_aggregate_results.csv"
    metric_summary_path = out_dir / "table_metric_summary.csv"
    per_task_path = out_dir / "table_per_task_metrics.csv"
    _write_metrics_csv(aggregate_path, [aggregate_row])
    _write_metrics_csv(metric_summary_path, metric_rows)
    _write_metrics_csv(per_task_path, per_task_rows)

    return {
        "ok": True,
        "report_count": len(report_items),
        "files": [str(aggregate_path), str(metric_summary_path), str(per_task_path)],
    }


def run_paper_demo_command(
    out: str | Path,
    *,
    cases: list[str] | None = None,
    frames: int = 8,
    height: int = 32,
    width: int = 32,
) -> dict[str, Any]:
    """Generate a complete synthetic evidence bundle for the paper path."""

    from .benchmark import run_benchmark, write_synthetic_benchmark

    out_dir = Path(out)
    fixtures_dir = out_dir / "fixtures"
    reports_dir = out_dir / "reports"
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    selected = cases or list(FIXTURE_CASES)
    manifest = write_synthetic_benchmark(fixtures_dir, cases=selected, frames=frames, height=height, width=width)
    benchmark_summary = run_benchmark(manifest, reports_dir)
    validation = validate_submission_command(reports_dir)
    figures = paper_figures_command(reports_dir, figures_dir)
    tables = export_paper_tables_command(reports_dir, tables_dir)
    report_paths = _collect_report_paths(reports_dir)
    scorecards = [build_preview_scorecard(load_and_validate_run_report(path)).to_dict() for path in report_paths]
    trust_summary = {
        "report_count": len(scorecards),
        "mean_trust_score": statistics.fmean(float(card["trust_score"]) for card in scorecards) if scorecards else 0.0,
        "pass_rate": statistics.fmean(1.0 if card["passed"] else 0.0 for card in scorecards) if scorecards else 0.0,
        "scorecards": scorecards,
    }
    trust_path = reports_dir / "trust_summary.json"
    trust_path.write_text(json.dumps(trust_summary, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "ok": validation["ok"],
        "command": "run-paper-demo",
        "manifest": str(manifest),
        "reports_dir": str(reports_dir),
        "figures_dir": str(figures_dir),
        "tables_dir": str(tables_dir),
        "benchmark_summary": benchmark_summary,
        "validation": validation,
        "figures": figures,
        "tables": tables,
        "trust_summary": trust_summary,
        "trust_summary_path": str(trust_path),
    }


def inspect_report_command(report: str | Path) -> dict[str, Any]:
    data = load_and_validate_run_report(report)
    metrics = data["metrics"]
    sorted_metrics = sorted(metrics.items(), key=lambda kv: abs(float(kv[1])), reverse=True)
    events = data.get("latency_log", {}).get("events", [])
    latency_total = sum(float(e.get("duration_s", float(e["ended_at_s"]) - float(e["started_at_s"]))) for e in events)
    warnings = []
    for metric in PRIMARY_METRICS:
        if metric not in metrics:
            warnings.append(f"missing primary metric {metric}")
    if metrics.get("background_preservation_error", 0) > metrics.get("preview_final_l1", 0) and metrics.get("background_preservation_error", 0) > 0.01:
        warnings.append("background error dominates, likely leaked edits outside mask")
    return {
        "ok": True,
        "run_id": data["run_id"],
        "task_id": data["task_id"],
        "sorted_metrics": sorted_metrics,
        "latency_total_s": latency_total,
        "latency_event_count": len(events),
        "warnings": warnings,
    }


def aggregate_consistency_score(metrics: Mapping[str, float]) -> float:
    return sum(float(metrics.get(metric, 0.0)) * weight for metric, weight in DEFAULT_WEIGHTS.items())


def aggregate_consistency_score_from_row(row: Mapping[str, Any]) -> float:
    return aggregate_consistency_score({m: float(row.get(m, 0.0)) for m in PRIMARY_METRICS})


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"case_count": len(rows), "lower_is_better": list(PRIMARY_METRICS) + ["aggregate_consistency_score"], "metrics": {}}
    for metric in PRIMARY_METRICS + ("aggregate_consistency_score",):
        values = [float(r[metric]) for r in rows if metric in r]
        if values:
            summary["metrics"][metric] = {"mean": statistics.fmean(values), "min": min(values), "max": max(values)}
    return summary


def percentile(values: Iterable[float], pct: float) -> float:
    sorted_values = sorted(float(value) for value in values)
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[int(rank)]
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def load_metric_rows(path: str | Path) -> list[dict[str, Any]]:
    root = Path(path)
    if root.is_file() and root.suffix == ".csv":
        with root.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    rows: list[dict[str, Any]] = []
    for report_path in _collect_report_paths(root):
        report = load_and_validate_run_report(report_path)
        metrics = dict(report["metrics"])
        if "aggregate_consistency_score" not in metrics:
            metrics["aggregate_consistency_score"] = aggregate_consistency_score(metrics)
        rows.append({"task_id": report["task_id"], "run_id": report["run_id"], **metrics})
    return rows


def _write_metrics_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for preferred in ("rank", "case", "task_id", "run_id", "method_name", "backend_family", "report_path") + PRIMARY_METRICS + ("aggregate_consistency_score",):
        if any(preferred in row for row in rows):
            keys.append(preferred)
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _load_report_items(path: str | Path) -> list[tuple[Path, dict[str, Any]]]:
    return [(report_path, load_and_validate_run_report(report_path)) for report_path in _collect_report_paths(path)]


def _single_or_mixed(values: Iterable[Any]) -> str:
    normalized = sorted({str(value) for value in values if str(value)})
    if not normalized:
        return ""
    if len(normalized) == 1:
        return normalized[0]
    return "mixed"


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _collect_fixture_paths(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    paths = sorted(p for p in root.glob("*.json") if p.name != "manifest.json" and not p.name.startswith("report-"))
    if not paths:
        raise ValueError(f"no fixture JSON files found under {root}")
    return paths


def _collect_report_paths(path: str | Path) -> list[Path]:
    root = Path(path)
    if root.is_file():
        return [root]
    candidates: Iterable[Path] = root.rglob("*.json")
    return sorted(p for p in candidates if p.name.startswith("report-") or p.name.endswith(".report.json"))
