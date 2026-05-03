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
    assert not (tmp_path / "generalization_gap.png").exists()
    assert not (tmp_path / "regime_generalization_gap.png").exists()
    assert not (tmp_path / "transition_generalization_gap.png").exists()


def test_write_figures_writes_lcri_severity_scope_plot(tmp_path: Path) -> None:
    frame = _scored_frame()
    regime_table = _regime_table()

    write_figures(
        frame,
        regime_table,
        tmp_path,
        lcri_generalization_severity_by_scope=_severity_scope_table(),
    )

    assert (tmp_path / "lcri_generalization_severity_by_scope.png").exists()


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
        generalization_gap=_generalization_gap_table(),
        regime_generalization_gap=_regime_generalization_gap_table(),
        transition_generalization_gap=_transition_generalization_gap_table(),
    )

    assert (tmp_path / "transition_signal_quality.png").exists()
    assert (tmp_path / "heldout_transition_signal_quality.png").exists()
    assert (tmp_path / "heldout_calibration_curve.png").exists()
    assert (tmp_path / "generalization_gap.png").exists()
    assert (tmp_path / "regime_generalization_gap.png").exists()
    assert (tmp_path / "transition_generalization_gap.png").exists()


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


def _severity_scope_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition"],
            "stable_rows": [1, 0, 0],
            "warning_rows": [0, 1, 1],
            "critical_rows": [0, 1, 0],
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


def _generalization_gap_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal": ["raw_imbalance", "lcri"],
            "directional_accuracy_gap": [0.05, 0.03],
            "brier_score_gap": [0.02, 0.01],
            "rank_correlation_gap": [0.04, 0.02],
        }
    )


def _regime_generalization_gap_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "regime": ["thin", "thin", "thick", "thick"],
            "signal": ["raw_imbalance", "lcri", "raw_imbalance", "lcri"],
            "directional_accuracy_gap": [0.02, 0.08, -0.01, 0.03],
        }
    )


def _transition_generalization_gap_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment": ["stable", "stable", "transition", "transition"],
            "signal": ["raw_imbalance", "lcri", "raw_imbalance", "lcri"],
            "directional_accuracy_gap": [0.01, 0.03, 0.02, 0.07],
        }
    )
