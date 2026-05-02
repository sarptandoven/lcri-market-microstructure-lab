import json

import pandas as pd

from lcri_lab.reporting import (
    build_artifact_manifest,
    collect_artifact_metadata,
    missing_artifacts,
    verify_artifact_manifest,
    write_json,
    write_research_summary,
)


def test_build_artifact_manifest_records_run_config_and_outputs() -> None:
    manifest = build_artifact_manifest(
        rows=100,
        train_rows=70,
        heldout_rows=30,
        seed=7,
        train_frac=0.7,
        model_artifact_version=2,
        artifacts=["metrics.csv"],
        artifact_metadata={"metrics.csv": {"size_bytes": 10, "sha256": "abc"}},
    )

    assert manifest["run"] == {
        "rows": 100,
        "train_rows": 70,
        "heldout_rows": 30,
        "seed": 7,
        "train_frac": 0.7,
    }
    assert manifest["model"] == {"artifact_version": 2}
    assert manifest["artifacts"] == ["metrics.csv"]
    assert manifest["artifact_metadata"] == {"metrics.csv": {"size_bytes": 10, "sha256": "abc"}}


def test_collect_artifact_metadata_records_size_and_digest(tmp_path) -> None:
    (tmp_path / "metrics.csv").write_text("signal,value\n")

    metadata = collect_artifact_metadata(tmp_path, ["metrics.csv", "missing.csv"])

    assert set(metadata) == {"metrics.csv"}
    assert metadata["metrics.csv"]["size_bytes"] == len("signal,value\n")
    assert len(metadata["metrics.csv"]["sha256"]) == 64


def test_verify_artifact_manifest_reports_checksum_mismatch(tmp_path) -> None:
    (tmp_path / "metrics.csv").write_text("signal,value\n")
    manifest = {
        "artifacts": ["metrics.csv"],
        "artifact_metadata": {"metrics.csv": {"size_bytes": 1, "sha256": "0" * 64}},
    }

    errors = verify_artifact_manifest(tmp_path, manifest)

    assert "size mismatch: metrics.csv" in errors
    assert "sha256 mismatch: metrics.csv" in errors


def test_missing_artifacts_reports_absent_paths(tmp_path) -> None:
    (tmp_path / "metrics.csv").write_text("signal\n")
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "plot.png").write_text("png")

    missing = missing_artifacts(
        tmp_path,
        ["metrics.csv", "transition_lift.csv", "figures/plot.png"],
    )

    assert missing == ["transition_lift.csv"]


def test_write_json_writes_sorted_pretty_payload(tmp_path) -> None:
    path = tmp_path / "payload.json"

    write_json(path, {"b": 2, "a": True})

    assert json.loads(path.read_text()) == {"a": True, "b": 2}
    assert path.read_text().startswith('{\n  "a"')


def test_write_research_summary_includes_metrics_and_robustness(tmp_path) -> None:
    path = tmp_path / "summary.md"
    metrics = pd.DataFrame(
        [
            {
                "signal": "lcri",
                "directional_accuracy": 0.61,
                "brier_score": 0.22,
                "rank_correlation": 0.18,
            }
        ]
    )
    transition_lift = pd.DataFrame(
        [
            {
                "segment": "transition",
                "rows": 4,
                "directional_accuracy_lift": 0.25,
            }
        ]
    )

    write_research_summary(
        path,
        rows=100,
        train_rows=70,
        heldout_rows=30,
        seed=7,
        train_frac=0.7,
        metrics=metrics,
        transition_lift=transition_lift,
        transition_robustness={"passes_transition_robustness": True},
    )

    text = path.read_text()
    assert "# LCRI Research Summary" in text
    assert "- seed: 7" in text
    assert "| signal | directional_accuracy | brier_score | rank_correlation |" in text
    assert "| lcri | 0.610000 | 0.220000 | 0.180000 |" in text
    assert "- passes_transition_robustness: true" in text
