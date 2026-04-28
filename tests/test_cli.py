import json
from pathlib import Path

from lcri_lab.cli import fit_model
from lcri_lab.simulator import SimulationConfig, simulate_order_books


def test_fit_model_persists_requested_ridge(tmp_path: Path) -> None:
    snapshots = tmp_path / "snapshots.csv"
    model_path = tmp_path / "model.json"
    simulate_order_books(SimulationConfig(rows=150, seed=13)).to_csv(snapshots, index=False)

    fit_model(snapshots, model_path, levels=5, ridge=0.25)

    payload = json.loads(model_path.read_text())
    assert payload["config"]["ridge"] == 0.25
