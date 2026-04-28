import json
from pathlib import Path

import pandas as pd
import pytest

from lcri_lab.cli import fit_model, score_model
from lcri_lab.simulator import SimulationConfig, simulate_order_books


def _write_snapshots(path: Path, rows: int = 150) -> None:
    simulate_order_books(SimulationConfig(rows=rows, seed=13)).to_csv(path, index=False)


def test_fit_model_persists_requested_ridge(tmp_path: Path) -> None:
    snapshots = tmp_path / "snapshots.csv"
    model_path = tmp_path / "model.json"
    _write_snapshots(snapshots)

    fit_model(snapshots, model_path, levels=5, ridge=0.25, probability_scale=2.5)

    payload = json.loads(model_path.read_text())
    assert payload["config"]["ridge"] == 0.25
    assert payload["config"]["probability_scale"] == 2.5


def test_score_model_writes_selected_columns(tmp_path: Path) -> None:
    snapshots = tmp_path / "snapshots.csv"
    model_path = tmp_path / "model.json"
    output_path = tmp_path / "scores.csv"
    _write_snapshots(snapshots)
    fit_model(snapshots, model_path, levels=5)

    score_model(snapshots, model_path, output_path, columns=["timestamp", "lcri", "lcri_probability"])

    assert list(pd.read_csv(output_path).columns) == ["timestamp", "lcri", "lcri_probability"]


def test_score_model_rejects_unknown_columns(tmp_path: Path) -> None:
    snapshots = tmp_path / "snapshots.csv"
    model_path = tmp_path / "model.json"
    _write_snapshots(snapshots)
    fit_model(snapshots, model_path, levels=5)

    with pytest.raises(ValueError, match="unavailable"):
        score_model(snapshots, model_path, tmp_path / "scores.csv", columns=["missing"])
