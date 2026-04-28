from pathlib import Path

from lcri_lab.cli import run_demo


def test_run_demo_writes_reports(tmp_path: Path) -> None:
    run_demo(rows=750, seed=3, output=tmp_path)

    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "regime_metrics.csv").exists()
    assert (tmp_path / "sample_snapshots.csv").exists()
    assert (tmp_path / "figures" / "raw_vs_lcri_scatter.png").exists()
    assert (tmp_path / "figures" / "regime_signal_quality.png").exists()
    assert (tmp_path / "figures" / "calibration_curve.png").exists()
