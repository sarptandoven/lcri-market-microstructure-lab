import json
from pathlib import Path

import pytest

from lcri_lab.cli import run_demo


def test_run_demo_writes_reports(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_demo(rows=750, seed=3, train_frac=0.65, output=tmp_path)
    output = capsys.readouterr().out

    assert "train fraction: 0.65" in output

    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "regime_metrics.csv").exists()
    assert (tmp_path / "transition_metrics.csv").exists()
    assert (tmp_path / "transition_lift.csv").exists()
    assert (tmp_path / "transition_robustness.json").exists()
    assert (tmp_path / "research_summary.md").exists()
    assert (tmp_path / "artifact_manifest.json").exists()
    assert (tmp_path / "sample_snapshots.csv").exists()
    assert (tmp_path / "figures" / "raw_vs_lcri_scatter.png").exists()
    assert (tmp_path / "figures" / "regime_signal_quality.png").exists()
    assert (tmp_path / "figures" / "transition_signal_quality.png").exists()
    assert (tmp_path / "figures" / "calibration_curve.png").exists()

    robustness = json.loads((tmp_path / "transition_robustness.json").read_text())
    assert "passes_transition_robustness" in robustness
    manifest = json.loads((tmp_path / "artifact_manifest.json").read_text())
    assert manifest["run"]["seed"] == 3
    assert "research_summary.md" in manifest["artifacts"]
    summary = (tmp_path / "research_summary.md").read_text()
    assert "# LCRI Research Summary" in summary
    assert "## Transition robustness" in summary


def test_run_demo_rejects_invalid_train_fraction(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="train_frac"):
        run_demo(rows=100, seed=3, train_frac=1.0, output=tmp_path)
