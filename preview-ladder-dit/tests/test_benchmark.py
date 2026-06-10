import json

from preview_ladder_dit.benchmark import (
    metric_band_violations,
    run_benchmark,
    validate_benchmark_manifest,
    write_synthetic_benchmark,
)
from preview_ladder_dit.fixtures import EXPECTED_METRIC_BANDS, make_fixture
from preview_ladder_dit.metrics import preview_final_consistency_report


def test_expected_metric_bands_cover_registered_synthetic_failures():
    for case, bands in EXPECTED_METRIC_BANDS.items():
        fixture = make_fixture(case, frames=8, height=32, width=32)
        metrics = preview_final_consistency_report(
            source=fixture.source, preview=fixture.preview, final=fixture.final, mask=fixture.mask
        ).to_dict()
        assert not metric_band_violations(metrics, bands), case


def test_write_and_run_synthetic_benchmark(tmp_path):
    manifest_path = write_synthetic_benchmark(
        tmp_path / "bench", cases=["clean", "thin_structure_boundary", "shadow_leak"], frames=4, height=16, width=16
    )
    manifest = json.loads(manifest_path.read_text())
    validate_benchmark_manifest(manifest)
    assert manifest["schema_version"] == "preview-ladder-benchmark/v0.1"
    assert [task["fixture_case"] for task in manifest["tasks"]] == ["clean", "thin_structure_boundary", "shadow_leak"]

    summary = run_benchmark(manifest_path, tmp_path / "run")
    assert summary["task_count"] == 3
    assert summary["valid_report_count"] == 3
    assert summary["rejected_report_count"] == 0
    assert 0.0 <= summary["mean_trust_score"] <= 1.0
    assert 0.0 <= summary["pass_rate"] <= 1.0
    assert "preview_final_l1" in summary["metric_means"]
    assert "preview_final_l1" in summary["metric_p95s"]
    assert summary["metric_p95s"]["preview_final_l1"] >= summary["metric_means"]["preview_final_l1"]
    assert summary["mean_metrics"] == summary["metric_means"]
    assert summary["band_violation_count"] == 0
    first_task = manifest["tasks"][0]
    assert first_task["failure_axis"] == "control"
    assert "preview_final_l1" in first_task["expected_detector_metrics"]
    assert isinstance(first_task["generation_seed"], int)
    assert first_task["difficulty"] in {"control", "easy", "medium", "hard"}
    assert (tmp_path / "run" / "benchmark_summary.json").exists()
    assert (tmp_path / "run" / "benchmark_metrics.csv").exists()
    csv_text = (tmp_path / "run" / "benchmark_metrics.csv").read_text()
    assert "trust_score" in csv_text
    assert "passed" in csv_text
