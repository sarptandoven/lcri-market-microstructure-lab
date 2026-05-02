import pandas as pd
import pytest

from lcri_lab.evaluation import (
    calibration_curve,
    evaluate_signals,
    generalization_gap_leaderboard,
    generalization_overview,
    regime_generalization_gap,
    regime_metrics,
    signal_generalization_gap,
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


def test_generalization_gap_leaderboard_ranks_all_scopes() -> None:
    signal_gap = pd.DataFrame(
        {"signal": ["lcri"], "directional_accuracy_gap": [0.05]}
    )
    regime_gap = pd.DataFrame(
        {"regime": ["thin"], "signal": ["lcri"], "directional_accuracy_gap": [0.08]}
    )
    transition_gap = pd.DataFrame(
        {
            "segment": ["transition"],
            "signal": ["lcri"],
            "directional_accuracy_gap": [0.04],
        }
    )

    output = generalization_gap_leaderboard(signal_gap, regime_gap, transition_gap)

    assert output.loc[0, "scope"] == "regime"
    assert output.loc[0, "context"] == "thin"
    assert output.loc[0, "directional_accuracy_gap"] == pytest.approx(0.08)


def test_generalization_overview_summarizes_gap_tables() -> None:
    signal_gap = pd.DataFrame({"directional_accuracy_gap": [0.02, 0.05]})
    regime_gap = pd.DataFrame({"directional_accuracy_gap": [0.03, 0.08]})
    transition_gap = pd.DataFrame({"directional_accuracy_gap": [0.01, 0.04]})

    output = generalization_overview(signal_gap, regime_gap, transition_gap)

    assert output["signal_rows"] == 2
    assert output["regime_rows"] == 2
    assert output["transition_rows"] == 2
    assert output["max_signal_directional_accuracy_gap"] == pytest.approx(0.05)
    assert output["max_regime_directional_accuracy_gap"] == pytest.approx(0.08)
    assert output["max_transition_directional_accuracy_gap"] == pytest.approx(0.04)


def test_regime_generalization_gap_compares_matching_regime_signals() -> None:
    metrics = pd.DataFrame(
        {
            "regime": ["thin", "thin"],
            "signal": ["raw_imbalance", "lcri"],
            "directional_accuracy": [0.50, 0.70],
            "brier_score": [0.35, 0.22],
            "rank_correlation": [0.05, 0.25],
        }
    )
    heldout = pd.DataFrame(
        {
            "regime": ["thin", "thin"],
            "signal": ["raw_imbalance", "lcri"],
            "directional_accuracy": [0.48, 0.62],
            "brier_score": [0.36, 0.27],
            "rank_correlation": [0.04, 0.18],
        }
    )

    output = regime_generalization_gap(metrics, heldout).set_index(["regime", "signal"])

    assert output.loc[("thin", "lcri"), "directional_accuracy_gap"] == pytest.approx(0.08)
    assert output.loc[("thin", "lcri"), "brier_score_gap"] == pytest.approx(0.05)
    assert output.loc[("thin", "lcri"), "rank_correlation_gap"] == pytest.approx(0.07)


def test_signal_generalization_gap_compares_full_and_heldout_metrics() -> None:
    metrics = pd.DataFrame(
        {
            "signal": ["raw_imbalance", "lcri"],
            "directional_accuracy": [0.60, 0.70],
            "brier_score": [0.30, 0.20],
            "rank_correlation": [0.10, 0.25],
        }
    )
    heldout = pd.DataFrame(
        {
            "signal": ["raw_imbalance", "lcri"],
            "directional_accuracy": [0.55, 0.65],
            "brier_score": [0.32, 0.24],
            "rank_correlation": [0.08, 0.20],
        }
    )

    output = signal_generalization_gap(metrics, heldout).set_index("signal")

    assert output.loc["lcri", "directional_accuracy_gap"] == pytest.approx(0.05)
    assert output.loc["lcri", "brier_score_gap"] == pytest.approx(0.04)
    assert output.loc["lcri", "rank_correlation_gap"] == pytest.approx(0.05)


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
