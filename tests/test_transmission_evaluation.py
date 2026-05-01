import pandas as pd
import pytest

from lcri_lab.evaluation import compare_transmission_signal, evaluate_signals


def test_evaluate_signals_accepts_transmission_pressure() -> None:
    frame = pd.DataFrame(
        {
            "raw_imbalance": [0.2, -0.1, 0.4],
            "lcri": [0.4, -0.3, 0.6],
            "transmission_pressure": [0.5, -0.2, 0.7],
            "future_direction": [1, 0, 1],
        }
    )

    result = evaluate_signals(frame, signals=["lcri", "transmission_pressure"])

    assert result["signal"].tolist() == ["lcri", "transmission_pressure"]
    assert result["directional_accuracy"].tolist() == [1.0, 1.0]


def test_compare_transmission_signal_returns_metric_deltas() -> None:
    frame = pd.DataFrame(
        {
            "lcri": [1.0, 0.5, -0.2, -1.0],
            "transmission_pressure": [0.8, -0.4, -0.6, -0.7],
            "future_direction": [1, 0, 0, 0],
        }
    )

    result = compare_transmission_signal(frame)

    assert set(result) == {
        "directional_accuracy_delta",
        "brier_score_delta",
        "rank_correlation_delta",
    }
    assert result["directional_accuracy_delta"] == pytest.approx(0.25)
