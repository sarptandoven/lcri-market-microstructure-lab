import pandas as pd
import pytest

from lcri_lab.evaluation import (
    calibration_curve,
    evaluate_signals,
    regime_metrics,
    summarize_signal_lift,
)
from lcri_lab.model import LCRIModel
from lcri_lab.simulator import SimulationConfig, simulate_order_books


def test_summarize_signal_lift_reports_metric_deltas() -> None:
    books = simulate_order_books(SimulationConfig(rows=300, seed=22))
    scored = LCRIModel().fit(books.iloc[:200]).score_frame(books.iloc[200:])

    summary = summarize_signal_lift(scored)

    assert set(summary) == {
        "directional_accuracy_lift",
        "brier_score_reduction",
        "rank_correlation_lift",
    }
    assert all(isinstance(value, float) for value in summary.values())


def test_calibration_curve_rejects_non_positive_bins() -> None:
    books = simulate_order_books(SimulationConfig(rows=120, seed=21))
    scored = LCRIModel().fit(books.iloc[:80]).score_frame(books.iloc[80:])

    with pytest.raises(ValueError, match="bins"):
        calibration_curve(scored, signal="lcri", bins=0)


def test_evaluation_rejects_empty_frames() -> None:
    with pytest.raises(ValueError, match="empty"):
        evaluate_signals(pd.DataFrame())
    with pytest.raises(ValueError, match="empty"):
        regime_metrics(pd.DataFrame())


def test_evaluation_rejects_missing_columns() -> None:
    frame = pd.DataFrame(
        {
            "raw_imbalance": [0.1, -0.2],
            "future_direction": [1, 0],
        }
    )

    with pytest.raises(ValueError, match="lcri"):
        evaluate_signals(frame)
    with pytest.raises(ValueError, match="regime"):
        regime_metrics(frame.assign(lcri=[0.3, -0.4]))


def test_calibration_curve_rejects_missing_signal() -> None:
    frame = pd.DataFrame({"future_direction": [1, 0]})

    with pytest.raises(ValueError, match="missing_signal"):
        calibration_curve(frame, signal="missing_signal")
