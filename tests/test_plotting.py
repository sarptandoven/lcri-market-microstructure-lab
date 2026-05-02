from pathlib import Path

import pandas as pd

from lcri_lab.plotting import write_figures


def test_write_figures_keeps_heldout_outputs_optional(tmp_path: Path) -> None:
    frame = _scored_frame()
    regime_table = _regime_table()

    write_figures(frame, regime_table, tmp_path)

    assert (tmp_path / "raw_vs_lcri_scatter.png").exists()
    assert (tmp_path / "regime_signal_quality.png").exists()
    assert (tmp_path / "calibration_curve.png").exists()
    assert not (tmp_path / "heldout_calibration_curve.png").exists()
    assert not (tmp_path / "heldout_transition_signal_quality.png").exists()


def test_write_figures_writes_heldout_transition_plot(tmp_path: Path) -> None:
    frame = _scored_frame()
    regime_table = _regime_table()
    transition_table = _transition_table()

    write_figures(
        frame,
        regime_table,
        tmp_path,
        transition_table=transition_table,
        heldout_transition_table=transition_table,
        heldout_frame=frame,
    )

    assert (tmp_path / "transition_signal_quality.png").exists()
    assert (tmp_path / "heldout_transition_signal_quality.png").exists()
    assert (tmp_path / "heldout_calibration_curve.png").exists()


def _scored_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "raw_imbalance": [0.1, -0.2, 0.3, -0.4],
            "lcri": [0.2, -0.1, 0.5, -0.6],
            "future_direction": [1, 0, 1, 0],
        }
    )


def _regime_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "regime": ["thick", "thick"],
            "signal": ["raw_imbalance", "lcri"],
            "directional_accuracy": [0.5, 0.75],
        }
    )


def _transition_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment": ["stable", "stable"],
            "signal": ["raw_imbalance", "lcri"],
            "directional_accuracy": [0.5, 0.75],
        }
    )
