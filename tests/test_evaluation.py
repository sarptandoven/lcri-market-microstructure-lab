import pandas as pd
import pytest

from lcri_lab.evaluation import (
    calibration_curve,
    evaluate_signals,
    generalization_gap_leaderboard,
    generalization_overview,
    lcri_gap_delta_flags,
    lcri_gap_delta_scorecard,
    lcri_gap_delta_summary,
    lcri_generalization_gate_decision,
    lcri_generalization_gap_delta,
    lcri_generalization_gap_leaderboard,
    lcri_generalization_scope_summary,
    lcri_generalization_severity,
    lcri_generalization_severity_by_scope,
    lcri_generalization_severity_summary,
    lcri_worst_generalization_context,
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


def test_lcri_generalization_gap_leaderboard_filters_other_signals() -> None:
    signal_gap = pd.DataFrame(
        {
            "signal": ["raw_imbalance", "lcri"],
            "directional_accuracy_gap": [0.09, 0.04],
        }
    )
    regime_gap = pd.DataFrame(
        {
            "regime": ["thin", "thin"],
            "signal": ["raw_imbalance", "lcri"],
            "directional_accuracy_gap": [0.07, 0.08],
        }
    )
    transition_gap = pd.DataFrame(
        {
            "segment": ["stable"],
            "signal": ["raw_imbalance"],
            "directional_accuracy_gap": [0.10],
        }
    )

    output = lcri_generalization_gap_leaderboard(signal_gap, regime_gap, transition_gap)

    assert list(output["signal"]) == ["lcri", "lcri"]
    assert output.loc[0, "scope"] == "regime"
    assert output.loc[0, "context"] == "thin"


def test_lcri_generalization_scope_summary_groups_gap_rows() -> None:
    leaderboard = pd.DataFrame(
        {
            "scope": ["signal", "regime", "regime"],
            "context": ["all", "thin", "stressed"],
            "signal": ["lcri", "lcri", "lcri"],
            "directional_accuracy_gap": [0.03, 0.05, 0.09],
        }
    )

    output = lcri_generalization_scope_summary(leaderboard).set_index("scope")

    assert output.loc["regime", "rows"] == 2
    assert output.loc["regime", "mean_directional_accuracy_gap"] == pytest.approx(0.07)
    assert output.loc["signal", "max_directional_accuracy_gap"] == pytest.approx(0.03)


def test_lcri_worst_generalization_context_reports_max_gap() -> None:
    leaderboard = pd.DataFrame(
        {
            "scope": ["signal", "transition"],
            "context": ["all", "stable"],
            "signal": ["lcri", "lcri"],
            "directional_accuracy_gap": [0.03, 0.08],
        }
    )

    output = lcri_worst_generalization_context(leaderboard)

    assert output == {
        "scope": "transition",
        "context": "stable",
        "directional_accuracy_gap": pytest.approx(0.08),
    }


def test_lcri_generalization_severity_labels_gap_rows() -> None:
    leaderboard = pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition"],
            "context": ["all", "thin", "stable"],
            "directional_accuracy_gap": [0.01, 0.03, 0.07],
        }
    )

    output = lcri_generalization_severity(leaderboard)

    assert list(output["severity"]) == ["stable", "warning", "critical"]


def test_lcri_generalization_severity_rejects_bad_thresholds() -> None:
    with pytest.raises(ValueError, match="thresholds"):
        lcri_generalization_severity(pd.DataFrame(), warning_gap=0.05, critical_gap=0.02)


def test_lcri_generalization_severity_by_scope_counts_labels() -> None:
    severity = pd.DataFrame(
        {
            "scope": ["signal", "regime", "regime", "transition"],
            "severity": ["stable", "warning", "critical", "warning"],
        }
    )

    output = lcri_generalization_severity_by_scope(severity).set_index("scope")

    assert output.loc["regime", "rows"] == 2
    assert output.loc["regime", "warning_rows"] == 1
    assert output.loc["regime", "critical_rows"] == 1
    assert output.loc["signal", "stable_rows"] == 1


def test_lcri_generalization_severity_summary_counts_labels() -> None:
    severity = pd.DataFrame(
        {"severity": ["stable", "warning", "warning", "critical"]}
    )

    output = lcri_generalization_severity_summary(severity)

    assert output == {
        "rows": 4,
        "stable_rows": 1,
        "warning_rows": 2,
        "critical_rows": 1,
        "passes_lcri_generalization_gate": False,
    }


def test_lcri_generalization_severity_summary_passes_without_critical_rows() -> None:
    severity = pd.DataFrame({"severity": ["stable", "warning"]})

    output = lcri_generalization_severity_summary(severity)

    assert output["critical_rows"] == 0
    assert output["passes_lcri_generalization_gate"] is True


def test_lcri_generalization_gate_decision_blocks_critical_rows() -> None:
    severity_summary = {
        "rows": 3,
        "stable_rows": 1,
        "warning_rows": 1,
        "critical_rows": 1,
        "passes_lcri_generalization_gate": False,
    }
    worst_context = {
        "scope": "regime",
        "context": "thin",
        "directional_accuracy_gap": 0.07,
    }

    output = lcri_generalization_gate_decision(severity_summary, worst_context)

    assert output["passes"] is False
    assert output["decision"] == "block"
    assert output["worst_context"] == "thin"
    assert "critical" in str(output["reason"])


def test_lcri_generalization_gate_decision_rejects_incomplete_inputs() -> None:
    with pytest.raises(ValueError, match="severity summary"):
        lcri_generalization_gate_decision({}, {"scope": "signal"})


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


def test_lcri_generalization_gap_delta_compares_raw_gap_stability() -> None:
    signal_gap = pd.DataFrame(
        {
            "signal": ["raw_imbalance", "lcri"],
            "directional_accuracy_gap": [0.08, 0.05],
        }
    )
    regime_gap = pd.DataFrame(
        {
            "regime": ["thin", "thin"],
            "signal": ["raw_imbalance", "lcri"],
            "directional_accuracy_gap": [0.03, 0.07],
        }
    )
    transition_gap = pd.DataFrame(
        {
            "segment": ["transition", "transition"],
            "signal": ["raw_imbalance", "lcri"],
            "directional_accuracy_gap": [0.06, 0.02],
        }
    )

    output = lcri_generalization_gap_delta(signal_gap, regime_gap, transition_gap)

    assert output.loc[0, "scope"] == "transition"
    assert output.loc[0, "context"] == "transition"
    assert output.loc[0, "raw_minus_lcri_directional_accuracy_gap"] == pytest.approx(0.04)
    assert output.loc[2, "scope"] == "regime"
    assert output.loc[2, "raw_minus_lcri_directional_accuracy_gap"] == pytest.approx(-0.04)


def test_lcri_gap_delta_summary_identifies_stability_edges() -> None:
    gap_delta = pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition", "regime"],
            "context": ["all", "thin", "transition", "stressed"],
            "raw_minus_lcri_directional_accuracy_gap": [0.03, -0.04, 0.01, 0.0],
        }
    )

    output = lcri_gap_delta_summary(gap_delta)

    assert output["rows"] == 4
    assert output["lcri_more_stable_rows"] == 2
    assert output["lcri_less_stable_rows"] == 1
    assert output["lcri_equal_stability_rows"] == 1
    assert output["max_lcri_stability_edge"] == pytest.approx(0.03)
    assert output["max_lcri_stability_edge_context"] == "signal:all"
    assert output["max_lcri_instability_edge"] == pytest.approx(-0.04)
    assert output["max_lcri_instability_edge_context"] == "regime:thin"


def test_lcri_gap_delta_scorecard_reports_relative_stability_shares() -> None:
    gap_delta = pd.DataFrame(
        {"raw_minus_lcri_directional_accuracy_gap": [0.03, -0.01, 0.0, 0.06]}
    )

    output = lcri_gap_delta_scorecard(gap_delta)

    assert output["rows"] == 4
    assert output["mean_raw_minus_lcri_directional_accuracy_gap"] == pytest.approx(0.02)
    assert output["median_raw_minus_lcri_directional_accuracy_gap"] == pytest.approx(0.015)
    assert output["lcri_more_stable_share"] == pytest.approx(0.5)
    assert output["lcri_less_stable_share"] == pytest.approx(0.25)


def test_lcri_gap_delta_flags_label_stability_direction() -> None:
    gap_delta = pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition"],
            "context": ["all", "thin", "stable"],
            "raw_minus_lcri_directional_accuracy_gap": [0.03, -0.04, 0.0],
        }
    )

    output = lcri_gap_delta_flags(gap_delta)

    assert list(output["stability_flag"]) == [
        "lcri_more_stable",
        "lcri_less_stable",
        "lcri_equal_stability",
    ]


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
