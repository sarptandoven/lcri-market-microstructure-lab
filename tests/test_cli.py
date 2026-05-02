import json
from pathlib import Path

import pandas as pd
import pytest

from lcri_lab.cli import describe_model, fit_model, normalize_snapshots, score_model, verify_report
from lcri_lab.simulator import SimulationConfig, simulate_order_books


def _write_snapshots(path: Path, rows: int = 150) -> None:
    simulate_order_books(SimulationConfig(rows=rows, seed=13)).to_csv(path, index=False)


def test_normalize_snapshots_writes_derived_state(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "normalized.csv"
    pd.DataFrame(
        {
            "bid_px_1": [99.99, 100.00],
            "ask_px_1": [100.01, 100.03],
            "bid_sz_1": [10.0, 12.0],
            "ask_sz_1": [9.0, 11.0],
        }
    ).to_csv(input_path, index=False)

    normalize_snapshots(input_path, output_path, tick_size=0.01, levels=1, derive_state=True)

    columns = pd.read_csv(output_path).columns
    assert {"mid", "spread_ticks", "volatility", "replenishment_rate"}.issubset(columns)


def test_fit_model_persists_requested_ridge(tmp_path: Path) -> None:
    snapshots = tmp_path / "snapshots.csv"
    model_path = tmp_path / "model.json"
    _write_snapshots(snapshots)

    fit_model(snapshots, model_path, levels=5, ridge=0.25, probability_scale=2.5)

    payload = json.loads(model_path.read_text())
    assert payload["config"]["ridge"] == 0.25
    assert payload["config"]["probability_scale"] == 2.5


def test_describe_model_prints_artifact_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshots = tmp_path / "snapshots.csv"
    model_path = tmp_path / "model.json"
    _write_snapshots(snapshots)
    fit_model(snapshots, model_path, levels=5, ridge=0.25)

    describe_model(model_path)
    output = capsys.readouterr().out

    assert "schema_version: 2" in output
    assert "levels: 5" in output
    assert "features: 14" in output


def test_score_model_writes_selected_columns(tmp_path: Path) -> None:
    snapshots = tmp_path / "snapshots.csv"
    model_path = tmp_path / "model.json"
    output_path = tmp_path / "scores.csv"
    _write_snapshots(snapshots)
    fit_model(snapshots, model_path, levels=5)

    score_model(snapshots, model_path, output_path, columns=["timestamp", "lcri", "lcri_probability"])

    assert list(pd.read_csv(output_path).columns) == ["timestamp", "lcri", "lcri_probability"]


def test_verify_report_accepts_intact_manifest(tmp_path: Path) -> None:
    artifact = tmp_path / "metrics.csv"
    artifact.write_text("signal,value\n")
    manifest = {
        "artifacts": ["metrics.csv"],
        "artifact_metadata": {
            "metrics.csv": {
                "size_bytes": artifact.stat().st_size,
                "sha256": "48d81bac0e2bd3054a99e5fa3a1ebaac0d7e1d23e7903b11950ad14e8d5878c4",
            }
        },
    }
    (tmp_path / "artifact_manifest.json").write_text(json.dumps(manifest))

    verify_report(tmp_path)


def test_verify_report_rejects_changed_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "metrics.csv"
    artifact.write_text("signal,value\n")
    manifest = {
        "artifacts": ["metrics.csv"],
        "artifact_metadata": {
            "metrics.csv": {
                "size_bytes": artifact.stat().st_size,
                "sha256": "0" * 64,
            }
        },
    }
    (tmp_path / "artifact_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="sha256 mismatch"):
        verify_report(tmp_path)


def test_score_model_rejects_unknown_columns(tmp_path: Path) -> None:
    snapshots = tmp_path / "snapshots.csv"
    model_path = tmp_path / "model.json"
    _write_snapshots(snapshots)
    fit_model(snapshots, model_path, levels=5)

    with pytest.raises(ValueError, match="unavailable"):
        score_model(snapshots, model_path, tmp_path / "scores.csv", columns=["missing"])
