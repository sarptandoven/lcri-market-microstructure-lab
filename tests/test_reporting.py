import json

import pandas as pd

from lcri_lab.reporting import build_artifact_manifest, write_json, write_research_summary


def test_build_artifact_manifest_records_run_config_and_outputs() -> None:
    manifest = build_artifact_manifest(
        rows=100,
        train_rows=70,
        heldout_rows=30,
        seed=7,
        train_frac=0.7,
        model_artifact_version=2,
        artifacts=["metrics.csv"],
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
