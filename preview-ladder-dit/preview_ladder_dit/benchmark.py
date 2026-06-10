from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .fixtures import EXPECTED_METRIC_BANDS, FIXTURE_CASES, FIXTURE_METADATA, write_fixtures
from .harness import run_synthetic
from .product import build_preview_scorecard
from .schema import load_and_validate_run_report

BENCHMARK_MANIFEST_VERSION = "preview-ladder-benchmark/v0.1"


@dataclass(frozen=True)
class BenchmarkTaskSpec:
    """Public benchmark task descriptor.

    The initial implementation supports deterministic synthetic tasks. The same
    envelope is intentionally compatible with later real-video tasks by keeping
    artifact paths, edit metadata, method metadata, and expected metric bands in
    explicit fields rather than hard-coding fixture internals into the harness.
    """

    task_id: str
    task_type: str
    fixture_case: str
    fixture_uri: str
    prompt: str
    mask_id: str
    expected_metric_bands: Mapping[str, tuple[float, float]]
    failure_axis: str
    expected_detector_metrics: tuple[str, ...]
    generation_seed: int
    difficulty: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["expected_metric_bands"] = {k: list(v) for k, v in self.expected_metric_bands.items()}
        data["expected_detector_metrics"] = list(self.expected_detector_metrics)
        return data


@dataclass(frozen=True)
class BenchmarkManifest:
    schema_version: str
    benchmark_id: str
    tasks: tuple[BenchmarkTaskSpec, ...]
    metrics: tuple[str, ...]
    protocol: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "benchmark_id": self.benchmark_id,
            "tasks": [task.to_dict() for task in self.tasks],
            "metrics": list(self.metrics),
            "protocol": dict(self.protocol),
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def write_synthetic_benchmark(
    out_dir: str | Path,
    *,
    cases: Sequence[str] = FIXTURE_CASES,
    frames: int = 8,
    height: int = 32,
    width: int = 32,
    benchmark_id: str = "synthetic-preview-trust-v0.1",
) -> Path:
    out = Path(out_dir)
    fixture_dir = out / "fixtures"
    paths = write_fixtures(fixture_dir, cases=cases, frames=frames, height=height, width=width)
    tasks = []
    for path in paths:
        case = path.stem
        metadata = FIXTURE_METADATA[case]
        tasks.append(
            BenchmarkTaskSpec(
                task_id=f"synthetic-{case}",
                task_type="synthetic_masked_replacement",
                fixture_case=case,
                fixture_uri=str(path.relative_to(out)),
                prompt=f"Deterministic masked replacement fixture: {case}",
                mask_id="primary_edit_mask",
                expected_metric_bands=EXPECTED_METRIC_BANDS.get(case, {}),
                failure_axis=str(metadata["failure_axis"]),
                expected_detector_metrics=tuple(str(metric) for metric in metadata["expected_detector_metrics"]),
                generation_seed=int(metadata["generation_seed"]),
                difficulty=str(metadata["difficulty"]),
            )
        )
    manifest = BenchmarkManifest(
        schema_version=BENCHMARK_MANIFEST_VERSION,
        benchmark_id=benchmark_id,
        tasks=tuple(tasks),
        metrics=(
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
        ),
        protocol={
            "acceptance_unit": "preview-final task pair",
            "primary_score": "preview_trust_gap",
            "metric_direction": "lower_is_better",
            "ltx_optional": True,
            "frames": frames,
            "height": height,
            "width": width,
        },
    )
    manifest_path = out / "benchmark_manifest.json"
    manifest.write_json(manifest_path)
    validate_benchmark_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
    return manifest_path


def run_benchmark(manifest_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_benchmark_manifest(manifest)
    cases = [task["fixture_case"] for task in manifest["tasks"]]
    protocol = manifest.get("protocol", {})
    report_paths = run_synthetic(
        out_dir,
        cases=cases,
        frames=int(protocol.get("frames", 8)),
        height=int(protocol.get("height", 32)),
        width=int(protocol.get("width", 32)),
    )
    reports: list[dict[str, Any]] = []
    rejected_reports: list[dict[str, str]] = []
    for path in report_paths:
        try:
            reports.append(load_and_validate_run_report(path))
        except Exception as exc:
            rejected_reports.append({"path": str(path), "reason": str(exc)})
    if not reports:
        raise ValueError("benchmark produced no valid reports")
    task_by_case = {task["fixture_case"]: task for task in manifest["tasks"]}
    band_violations = []
    rows = []
    scorecards = []
    for report in reports:
        case = str(report.get("task", {}).get("case", report["task_id"].replace("synthetic-", "")))
        task = task_by_case.get(case, {})
        violations = metric_band_violations(report["metrics"], task.get("expected_metric_bands", {}))
        band_violations.extend({"task_id": report["task_id"], **v} for v in violations)
        scorecard = build_preview_scorecard(report)
        scorecards.append(scorecard)
        rows.append(
            {
                "task_id": report["task_id"],
                "case": case,
                "violations": len(violations),
                "trust_score": scorecard.trust_score,
                "passed": scorecard.passed,
                **report["metrics"],
            }
        )
    metric_items = [r["metrics"] for r in reports]
    metric_means = _mean_metrics(metric_items)
    summary = {
        "schema_version": "preview-ladder-benchmark-result/v0.1",
        "benchmark_id": manifest["benchmark_id"],
        "reports": [str(path) for path in report_paths],
        "task_count": len(reports),
        "valid_report_count": len(reports),
        "rejected_report_count": len(rejected_reports),
        "rejected_reports": rejected_reports,
        "band_violation_count": len(band_violations),
        "band_violations": band_violations,
        "mean_trust_score": sum(card.trust_score for card in scorecards) / len(scorecards),
        "pass_rate": sum(1 for card in scorecards if card.passed) / len(scorecards),
        "metric_means": metric_means,
        "metric_p95s": _p95_metrics(metric_items),
        "mean_metrics": metric_means,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(out / "benchmark_metrics.csv", rows)
    return summary


def validate_benchmark_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != BENCHMARK_MANIFEST_VERSION:
        raise ValueError(f"schema_version must be {BENCHMARK_MANIFEST_VERSION}")
    if not isinstance(manifest.get("benchmark_id"), str) or not manifest["benchmark_id"]:
        raise ValueError("benchmark_id must be a non-empty string")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks must be a non-empty list")
    seen: set[str] = set()
    for task in tasks:
        for key in ("task_id", "task_type", "fixture_case", "fixture_uri", "prompt", "mask_id", "failure_axis", "difficulty"):
            if not isinstance(task.get(key), str) or not task[key]:
                raise ValueError(f"task missing non-empty {key}")
        detector_metrics = task.get("expected_detector_metrics")
        if not isinstance(detector_metrics, list) or not detector_metrics or not all(isinstance(metric, str) and metric for metric in detector_metrics):
            raise ValueError("expected_detector_metrics must be a non-empty list of metric names")
        if not isinstance(task.get("generation_seed"), int):
            raise ValueError("generation_seed must be an integer")
        if task["task_id"] in seen:
            raise ValueError(f"duplicate task_id {task['task_id']}")
        seen.add(task["task_id"])
        if task["fixture_case"] not in FIXTURE_CASES:
            raise ValueError(f"unknown fixture_case {task['fixture_case']}")
        bands = task.get("expected_metric_bands", {})
        if not isinstance(bands, Mapping):
            raise ValueError("expected_metric_bands must be an object")
        for metric, band in bands.items():
            if not isinstance(metric, str) or not isinstance(band, (list, tuple)) or len(band) != 2:
                raise ValueError("metric bands must map names to [min, max]")
            if float(band[0]) > float(band[1]):
                raise ValueError(f"metric band min exceeds max for {metric}")


def metric_band_violations(metrics: Mapping[str, float], bands: Mapping[str, Sequence[float]]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for metric, band in bands.items():
        if metric not in metrics:
            violations.append({"metric": metric, "reason": "missing"})
            continue
        value = float(metrics[metric])
        lower, upper = float(band[0]), float(band[1])
        if value < lower or value > upper:
            violations.append({"metric": metric, "value": value, "min": lower, "max": upper, "reason": "outside_expected_band"})
    return violations


def _mean_metrics(items: Sequence[Mapping[str, float]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for metrics in items:
        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {key: totals[key] / counts[key] for key in sorted(totals)}


def _p95_metrics(items: Sequence[Mapping[str, float]]) -> dict[str, float]:
    by_metric: dict[str, list[float]] = {}
    for metrics in items:
        for key, value in metrics.items():
            by_metric.setdefault(key, []).append(float(value))
    return {key: _nearest_rank_percentile(values, 0.95) for key, values in sorted(by_metric.items())}


def _nearest_rank_percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    if percentile < 0.0 or percentile > 1.0:
        raise ValueError("percentile must be between 0 and 1")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in rows for key in row})
    lines = [",".join(keys)]
    for row in rows:
        lines.append(",".join(str(row.get(key, "")) for key in keys))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
