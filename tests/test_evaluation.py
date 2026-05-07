import pandas as pd
import pytest

from lcri_lab.evaluation import (
    calibration_curve,
    calibration_error_summary,
    calibration_gate_decision,
    calibration_monotonicity_pressure,
    evaluate_signals,
    generalization_fragility_diagnostics,
    generalization_fragility_summary,
    generalization_gap_leaderboard,
    generalization_overview,
    generalization_stability_confidence_intervals,
    generalization_stability_confidence_summary,
    lcri_ci_confidence_coverage_scorecard,
    lcri_ci_confidence_coverage_summary,
    lcri_ci_gate_contradiction_diagnostics,
    lcri_ci_gate_contradiction_summary,
    lcri_contradiction_review_packet,
    lcri_contradiction_review_packet_summary,
    lcri_cross_artifact_evidence_index,
    lcri_cross_artifact_evidence_index_summary,
    lcri_evidence_release_checklist,
    lcri_evidence_release_checklist_summary,
    lcri_evidence_lineage_map,
    lcri_evidence_lineage_map_summary,
    lcri_owner_handoff_packet,
    lcri_owner_handoff_packet_summary,
    lcri_uncertainty_weighted_review_priority,
    lcri_uncertainty_weighted_review_priority_summary,
    lcri_fragility_gate_alignment,
    lcri_fragility_gate_scorecard,
    lcri_gap_delta_dominant_scopes,
    lcri_gap_delta_flags,
    lcri_gap_delta_improvements,
    lcri_gap_delta_regressions,
    lcri_gap_delta_scorecard,
    lcri_gap_delta_scope_extremes,
    lcri_gap_delta_scope_summary,
    lcri_gap_delta_summary,
    lcri_generalization_blocker_summary,
    lcri_generalization_critical_contexts,
    lcri_generalization_gate_decision,
    lcri_generalization_gap_delta,
    lcri_generalization_gap_leaderboard,
    lcri_generalization_scope_gate_decisions,
    lcri_scope_gate_decision_summary,
    lcri_scope_stability_contradiction_summary,
    lcri_scope_stability_contradictions,
    lcri_generalization_scope_risk,
    lcri_generalization_scope_summary,
    lcri_generalization_severity,
    lcri_generalization_severity_by_scope,
    lcri_generalization_severity_summary,
    lcri_worst_generalization_context,
    regime_generalization_gap,
    regime_metrics,
    signal_generalization_gap,
    signal_quantile_monotonicity,
    signal_quantile_monotonicity_summary,
    summarize_signal_lift,
)
from lcri_lab.model import LCRIModel
from lcri_lab.simulator import SimulationConfig, simulate_order_books


def test_signal_quantile_monotonicity_flags_reversals() -> None:
    frame = pd.DataFrame(
        {
            "lcri": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "future_direction": [0, 1, 1, 1, 0, 0],
        }
    )

    output = signal_quantile_monotonicity(frame, "lcri", quantiles=3)

    assert output["rows"].tolist() == [2, 2, 2]
    assert output["observed_frequency"].tolist() == pytest.approx([0.5, 1.0, 0.0])
    assert output["adjacent_frequency_slope"].tolist() == pytest.approx([0.0, 0.5, -1.0])
    assert output["monotonicity_violation"].tolist() == [False, False, True]


def test_signal_quantile_monotonicity_summary_gates_violations() -> None:
    monotonicity = pd.DataFrame(
        {
            "quantile": [0, 1, 2],
            "observed_frequency": [0.2, 0.8, 0.5],
            "adjacent_frequency_slope": [0.0, 0.6, -0.3],
            "monotonicity_violation": [False, False, True],
        }
    )

    summary = signal_quantile_monotonicity_summary(monotonicity)

    assert summary == {
        "quantiles": 3,
        "monotonicity_violation_rows": 1,
        "worst_negative_slope": pytest.approx(-0.3),
        "worst_negative_slope_quantile": "2",
        "mean_observed_frequency": pytest.approx(0.5),
        "passes_monotonicity_gate": False,
    }


def test_signal_quantile_monotonicity_rejects_constant_signal() -> None:
    frame = pd.DataFrame({"lcri": [1.0, 1.0, 1.0], "future_direction": [0, 1, 1]})

    with pytest.raises(ValueError, match="signal must vary"):
        signal_quantile_monotonicity(frame, "lcri", quantiles=3)


def test_calibration_monotonicity_pressure_flags_fractured_miscalibration() -> None:
    calibration = pd.DataFrame(
        {
            "bin": [0, 1, 2],
            "predicted_probability": [0.2, 0.5, 0.8],
            "observed_frequency": [0.25, 0.7, 0.4],
            "rows": [10, 10, 10],
        }
    )
    monotonicity = pd.DataFrame(
        {
            "quantile": [0, 1, 2],
            "observed_frequency": [0.25, 0.7, 0.4],
            "adjacent_frequency_slope": [0.0, 0.45, -0.3],
            "monotonicity_violation": [False, False, True],
            "rows": [10, 10, 10],
        }
    )

    output = calibration_monotonicity_pressure(calibration, monotonicity)

    assert output.loc[2, "pressure_label"] == "fractured_miscalibrated"
    assert output.loc[2, "calibration_residual"] == pytest.approx(-0.4)
    assert output.loc[2, "fracture_pressure"] == pytest.approx(0.3 * 0.4 * (1 / 3) ** 0.5)


def test_calibration_monotonicity_pressure_rejects_negative_threshold() -> None:
    with pytest.raises(ValueError, match="residual_threshold"):
        calibration_monotonicity_pressure(pd.DataFrame(), pd.DataFrame(), residual_threshold=-0.1)


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


def test_evaluate_signals_rejects_non_finite_inputs() -> None:
    frame = pd.DataFrame(
        {
            "raw_imbalance": [0.1, float("nan")],
            "future_direction": [1.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="finite"):
        evaluate_signals(frame, signals=["raw_imbalance"])


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


def test_generalization_gap_leaderboard_rejects_invalid_limit() -> None:
    for limit in [0, 1.5]:
        with pytest.raises(ValueError, match="limit"):
            generalization_gap_leaderboard(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), limit=limit)


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


def test_lcri_generalization_gap_leaderboard_rejects_invalid_limit() -> None:
    for limit in [0, 1.5]:
        with pytest.raises(ValueError, match="limit"):
            lcri_generalization_gap_leaderboard(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), limit=limit)


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


def test_generalization_fragility_diagnostics_scales_gaps_by_heldout_uncertainty() -> None:
    metrics = pd.DataFrame(
        {
            "signal": ["raw_imbalance", "lcri"],
            "rows": [1000, 1000],
            "directional_accuracy": [0.60, 0.70],
        }
    )
    heldout_metrics = pd.DataFrame(
        {
            "signal": ["raw_imbalance", "lcri"],
            "rows": [100, 100],
            "directional_accuracy": [0.56, 0.44],
        }
    )
    regime = pd.DataFrame(
        {"regime": ["thin"], "signal": ["lcri"], "rows": [500], "directional_accuracy": [0.62]}
    )
    heldout_regime = pd.DataFrame(
        {"regime": ["thin"], "signal": ["lcri"], "rows": [20], "directional_accuracy": [0.60]}
    )
    transition = pd.DataFrame(
        {"segment": ["stable"], "signal": ["lcri"], "rows": [500], "directional_accuracy": [0.61]}
    )
    heldout_transition = pd.DataFrame(
        {"segment": ["stable"], "signal": ["lcri"], "rows": [20], "directional_accuracy": [0.60]}
    )

    output = generalization_fragility_diagnostics(
        metrics,
        heldout_metrics,
        regime,
        heldout_regime,
        transition,
        heldout_transition,
    )

    worst = output.iloc[0]
    assert worst["scope"] == "signal"
    assert worst["signal"] == "lcri"
    assert worst["directional_accuracy_gap"] == pytest.approx(0.26)
    assert worst["fragility_label"] == "fragile"


def test_generalization_fragility_summary_counts_labels() -> None:
    diagnostics = pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition"],
            "context": ["all", "thin", "stable"],
            "signal": ["lcri", "lcri", "raw_imbalance"],
            "abs_gap_to_se_ratio": [3.5, 2.1, 0.5],
            "fragility_label": ["fragile", "watch", "stable"],
        }
    )

    summary = generalization_fragility_summary(diagnostics)

    assert summary == {
        "rows": 3,
        "stable_rows": 1,
        "watch_rows": 1,
        "fragile_rows": 1,
        "max_abs_gap_to_se_ratio": 3.5,
        "most_fragile_context": "signal:all:lcri",
    }


def test_generalization_stability_confidence_intervals_bound_heldout_accuracy() -> None:
    diagnostics = pd.DataFrame(
        {
            "scope": ["signal", "regime"],
            "context": ["all", "thin"],
            "signal": ["lcri", "lcri"],
            "heldout_rows": [100, 25],
            "heldout_directional_accuracy": [0.60, 0.96],
            "heldout_directional_accuracy_se": [0.05, 0.04],
            "directional_accuracy_gap": [0.12, 0.02],
        }
    )

    output = generalization_stability_confidence_intervals(diagnostics).set_index("context")

    assert output.loc["all", "confidence_level"] == pytest.approx(0.950004209703559)
    assert output.loc["all", "heldout_directional_accuracy_ci_lower"] == pytest.approx(0.502)
    assert output.loc["all", "heldout_directional_accuracy_ci_upper"] == pytest.approx(0.698)
    assert output.loc["all", "heldout_directional_accuracy_ci_width"] == pytest.approx(0.196)
    assert bool(output.loc["all", "gap_exceeds_ci_half_width"])
    assert output.loc["thin", "heldout_directional_accuracy_ci_upper"] == pytest.approx(1.0)


def test_generalization_stability_confidence_intervals_rejects_invalid_z_score() -> None:
    for z_score in [0.0, float("nan")]:
        with pytest.raises(ValueError, match="z_score"):
            generalization_stability_confidence_intervals(pd.DataFrame(), z_score=z_score)


def test_generalization_stability_confidence_summary_counts_gap_flags() -> None:
    intervals = pd.DataFrame(
        {
            "scope": ["signal", "regime"],
            "context": ["all", "thin"],
            "signal": ["lcri", "raw_imbalance"],
            "heldout_directional_accuracy_ci_width": [0.20, 0.40],
            "gap_exceeds_ci_half_width": [True, False],
        }
    )

    output = generalization_stability_confidence_summary(intervals)

    assert output["rows"] == 2
    assert output["gap_exceeds_ci_half_width_rows"] == 1
    assert output["mean_ci_width"] == pytest.approx(0.30)
    assert output["max_ci_width"] == pytest.approx(0.40)
    assert output["widest_interval_context"] == "regime:thin:raw_imbalance"


def test_lcri_ci_gate_contradiction_diagnostics_flags_uncertainty_disagreements() -> None:
    severity = pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition", "regime"],
            "context": ["all", "thin", "open", "deep"],
            "signal": ["lcri", "lcri", "lcri", "raw_imbalance"],
            "directional_accuracy_gap": [0.07, 0.03, 0.01, 0.20],
            "severity": ["critical", "warning", "stable", "critical"],
        }
    )
    intervals = pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition", "regime"],
            "context": ["all", "thin", "open", "deep"],
            "signal": ["lcri", "lcri", "lcri", "raw_imbalance"],
            "heldout_rows": [100, 40, 80, 50],
            "confidence_level": [0.95, 0.95, 0.95, 0.95],
            "heldout_directional_accuracy_ci_width": [0.20, 0.22, 0.01, 0.10],
            "gap_exceeds_ci_half_width": [False, False, True, True],
        }
    )

    output = lcri_ci_gate_contradiction_diagnostics(severity, intervals).set_index(
        ["scope", "context"]
    )

    assert output.loc[("signal", "all"), "ci_gate_label"] == "gate_blocks_inside_ci"
    assert output.loc[("signal", "all"), "review_priority"] == 3
    assert output.loc[("regime", "thin"), "ci_gate_label"] == "gate_warns_inside_ci"
    assert output.loc[("regime", "thin"), "ci_half_width"] == pytest.approx(0.11)
    assert output.loc[("transition", "open"), "ci_gate_label"] == "stable_gap_outside_ci"
    assert ("regime", "deep") not in output.index


def test_lcri_ci_gate_contradiction_summary_prioritizes_blocking_ci_disagreement() -> None:
    diagnostics = pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition", "regime"],
            "context": ["all", "thin", "open", "deep"],
            "directional_accuracy_gap": [0.07, 0.03, 0.01, 0.06],
            "ci_gate_label": [
                "gate_blocks_inside_ci",
                "gate_warns_inside_ci",
                "stable_gap_outside_ci",
                "aligned",
            ],
            "review_priority": [3, 2, 2, 2],
        }
    )

    summary = lcri_ci_gate_contradiction_summary(diagnostics)

    assert summary == {
        "rows": 4,
        "aligned_rows": 1,
        "contradiction_rows": 3,
        "gate_blocks_inside_ci_rows": 1,
        "gate_warns_inside_ci_rows": 1,
        "stable_gap_outside_ci_rows": 1,
        "max_review_priority": 3,
        "worst_ci_gate_context": "signal:all:gate_blocks_inside_ci",
    }


def test_lcri_ci_confidence_coverage_scorecard_prioritizes_scope_review() -> None:
    intervals = pd.DataFrame(
        {
            "scope": ["signal", "regime", "regime", "transition", "ignored"],
            "context": ["all", "thin", "deep", "open", "raw"],
            "signal": ["lcri", "lcri", "lcri", "lcri", "raw_imbalance"],
            "heldout_directional_accuracy_ci_width": [0.10, 0.24, 0.30, 0.22, 0.50],
            "gap_exceeds_ci_half_width": [False, False, True, True, True],
        }
    )
    diagnostics = pd.DataFrame(
        {
            "scope": ["signal", "regime", "regime", "transition"],
            "context": ["all", "thin", "deep", "open"],
            "ci_gate_label": [
                "gate_blocks_inside_ci",
                "aligned",
                "gate_warns_inside_ci",
                "aligned",
            ],
            "review_priority": [3, 1, 2, 1],
        }
    )

    output = lcri_ci_confidence_coverage_scorecard(intervals, diagnostics).set_index("scope")

    assert list(output.index) == ["signal", "regime", "transition"]
    assert output.loc["signal", "coverage_label"] == "blocking_ci_gate_review"
    assert output.loc["regime", "coverage_label"] == "ci_gate_contradiction_review"
    assert output.loc["regime", "wide_ci_rows"] == 2
    assert output.loc["regime", "wide_ci_share"] == pytest.approx(1.0)
    assert output.loc["transition", "coverage_label"] == "wide_ci_review"


def test_lcri_ci_confidence_coverage_summary_counts_review_scopes() -> None:
    scorecard = pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition", "session"],
            "coverage_label": [
                "blocking_ci_gate_review",
                "ci_gate_contradiction_review",
                "wide_ci_review",
                "adequate_ci_coverage",
            ],
            "ci_gate_contradiction_rows": [1, 2, 0, 0],
            "wide_ci_rows": [0, 2, 1, 0],
            "high_priority_ci_gate_rows": [1, 0, 0, 0],
            "wide_ci_share": [0.0, 1.0, 0.5, 0.0],
            "max_ci_width": [0.10, 0.30, 0.22, 0.08],
        }
    )

    output = lcri_ci_confidence_coverage_summary(scorecard)

    assert output == {
        "scopes": 4,
        "review_scopes": 3,
        "blocking_review_scopes": 1,
        "contradiction_review_scopes": 1,
        "wide_ci_review_scopes": 1,
        "total_ci_gate_contradiction_rows": 3,
        "total_wide_ci_rows": 3,
        "worst_ci_confidence_scope": "signal:blocking_ci_gate_review",
    }


def test_lcri_fragility_gate_alignment_flags_cross_diagnostic_contradictions() -> None:
    fragility = pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition"],
            "context": ["all", "thin", "open"],
            "signal": ["lcri", "lcri", "lcri"],
            "heldout_rows": [80, 40, 30],
            "heldout_directional_accuracy_se": [0.05, 0.04, 0.03],
            "abs_gap_to_se_ratio": [0.8, 3.4, 2.1],
            "fragility_label": ["stable", "fragile", "watch"],
        }
    )
    severity = pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition"],
            "context": ["all", "thin", "open"],
            "signal": ["lcri", "lcri", "lcri"],
            "directional_accuracy_gap": [0.07, 0.03, 0.01],
            "severity": ["critical", "warning", "stable"],
        }
    )

    output = lcri_fragility_gate_alignment(fragility, severity).set_index(["scope", "context"])

    assert output.loc[("signal", "all"), "alignment_label"] == "gate_blocks_stable_slice"
    assert output.loc[("regime", "thin"), "alignment_label"] == "uncertainty_fragile_noncritical"
    assert output.loc[("transition", "open"), "alignment_label"] == "uncertainty_watch_stable_gap"
    assert "review" in output.loc[("regime", "thin"), "review_note"]


def test_lcri_fragility_gate_scorecard_counts_review_rows() -> None:
    alignment = pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition", "regime"],
            "context": ["all", "thin", "open", "deep"],
            "severity": ["critical", "warning", "stable", "critical"],
            "abs_gap_to_se_ratio": [0.8, 3.4, 2.1, 1.1],
            "alignment_label": [
                "gate_blocks_stable_slice",
                "uncertainty_fragile_noncritical",
                "uncertainty_watch_stable_gap",
                "aligned",
            ],
        }
    )

    output = lcri_fragility_gate_scorecard(alignment)

    assert output["rows"] == 4
    assert output["aligned_rows"] == 1
    assert output["review_required_rows"] == 3
    assert output["gate_blocks_stable_slice_rows"] == 1
    assert output["uncertainty_fragile_noncritical_rows"] == 1
    assert output["uncertainty_watch_stable_gap_rows"] == 1
    assert output["critical_rows"] == 2
    assert output["critical_stable_slice_share"] == pytest.approx(0.5)
    assert output["max_abs_gap_to_se_ratio"] == pytest.approx(3.4)
    assert output["worst_review_context"] == "regime:thin:uncertainty_fragile_noncritical"


def test_lcri_scope_stability_contradictions_flags_cross_scope_conflicts() -> None:
    scope_decisions = pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition"],
            "rows": [1, 3, 2],
            "decision": ["block", "pass", "warn"],
        }
    )
    gap_delta_scope_summary = pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition"],
            "lcri_more_stable_share": [1.0, 0.5, 0.0],
            "lcri_less_stable_share": [0.0, 0.5, 1.0],
        }
    )
    fragility_alignment = pd.DataFrame(
        {
            "scope": ["signal", "regime", "regime", "transition"],
            "alignment_label": [
                "aligned",
                "uncertainty_fragile_noncritical",
                "aligned",
                "gate_blocks_stable_slice",
            ],
        }
    )

    output = lcri_scope_stability_contradictions(
        scope_decisions,
        gap_delta_scope_summary,
        fragility_alignment,
    ).set_index("scope")

    assert output.loc["signal", "contradiction_label"] == "gate_blocks_despite_relative_stability"
    assert output.loc["regime", "contradiction_label"] == "pass_scope_with_relative_regressions"
    assert output.loc["transition", "contradiction_label"] == "warning_scope_with_broad_relative_regression"
    assert output.loc["regime", "fragility_review_required_rows"] == 1


def test_lcri_scope_stability_contradiction_summary_counts_labels() -> None:
    contradictions = pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition", "micro"],
            "contradiction_label": [
                "gate_blocks_despite_relative_stability",
                "pass_scope_with_relative_regressions",
                "warning_scope_with_broad_relative_regression",
                "aligned",
            ],
            "fragility_review_required_rows": [0, 2, 1, 3],
            "lcri_less_stable_share": [0.0, 0.5, 1.0, 0.0],
        }
    )

    summary = lcri_scope_stability_contradiction_summary(contradictions)

    assert summary["scopes"] == 4
    assert summary["aligned_scopes"] == 1
    assert summary["contradiction_scopes"] == 3
    assert summary["gate_blocks_despite_relative_stability_scopes"] == 1
    assert summary["pass_scope_with_relative_regressions_scopes"] == 1
    assert summary["warning_scope_with_broad_relative_regression_scopes"] == 1
    assert summary["fragility_review_required_rows"] == 6
    assert summary["worst_contradiction_scope"] == "regime:pass_scope_with_relative_regressions"


def test_lcri_contradiction_review_packet_links_scope_to_row_evidence() -> None:
    contradictions = pd.DataFrame(
        {
            "scope": ["signal", "regime"],
            "contradiction_label": ["gate_blocks_despite_relative_stability", "aligned"],
            "decision": ["block", "warn"],
            "rows": [1, 2],
            "lcri_less_stable_share": [0.0, 0.5],
            "fragility_review_required_rows": [0, 1],
        }
    )
    severity = pd.DataFrame(
        {
            "scope": ["signal", "regime", "regime"],
            "context": ["all", "thin", "deep"],
            "directional_accuracy_gap": [0.07, 0.03, 0.05],
            "severity": ["critical", "warning", "critical"],
        }
    )
    gap_delta = pd.DataFrame(
        {
            "scope": ["signal", "regime", "regime"],
            "context": ["all", "thin", "deep"],
            "raw_minus_lcri_directional_accuracy_gap": [0.02, -0.04, 0.01],
        }
    )
    fragility_alignment = pd.DataFrame(
        {
            "scope": ["signal", "regime", "regime"],
            "context": ["all", "thin", "deep"],
            "alignment_label": ["aligned", "uncertainty_fragile_noncritical", "aligned"],
            "abs_gap_to_se_ratio": [0.8, 3.2, 1.1],
        }
    )

    output = lcri_contradiction_review_packet(
        contradictions,
        severity,
        gap_delta,
        fragility_alignment,
    ).set_index("scope")

    assert output.loc["signal", "review_priority"] == 2
    assert output.loc["signal", "worst_gate_context"] == "all"
    assert output.loc["regime", "review_priority"] == 1
    assert output.loc["regime", "worst_gate_context"] == "deep"
    assert output.loc["regime", "worst_delta_context"] == "thin"
    assert output.loc["regime", "worst_fragility_alignment_label"] == "uncertainty_fragile_noncritical"
    assert output.loc["regime", "worst_fragility_abs_gap_to_se_ratio"] == pytest.approx(3.2)


def test_lcri_contradiction_review_packet_summary_prioritizes_review_scopes() -> None:
    packet = pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition"],
            "contradiction_label": [
                "gate_blocks_despite_relative_stability",
                "aligned",
                "pass_scope_with_relative_regressions",
            ],
            "fragility_review_required_rows": [0, 2, 1],
            "worst_fragility_abs_gap_to_se_ratio": [0.8, 4.1, 3.2],
            "review_priority": [2, 1, 3],
        }
    )

    summary = lcri_contradiction_review_packet_summary(packet)

    assert summary["scopes"] == 3
    assert summary["high_priority_scopes"] == 1
    assert summary["medium_priority_scopes"] == 1
    assert summary["low_priority_scopes"] == 1
    assert summary["fragility_review_required_rows"] == 3
    assert summary["max_review_priority"] == 3
    assert summary["worst_review_scope"] == "transition:pass_scope_with_relative_regressions"
    assert summary["worst_fragility_scope"] == "regime:4.100000"


def test_lcri_uncertainty_weighted_review_priority_combines_ci_and_packet_evidence() -> None:
    packet = pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition"],
            "contradiction_label": [
                "gate_blocks_despite_relative_stability",
                "aligned",
                "pass_scope_with_relative_regressions",
            ],
            "fragility_review_required_rows": [0, 2, 1],
            "worst_fragility_abs_gap_to_se_ratio": [0.8, 4.1, 3.2],
            "review_priority": [2, 1, 3],
        }
    )
    scorecard = pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition"],
            "coverage_label": [
                "blocking_ci_gate_review",
                "wide_ci_review",
                "ci_gate_contradiction_review",
            ],
            "mean_ci_width": [0.10, 0.24, 0.18],
            "max_ci_width": [0.12, 0.30, 0.25],
            "wide_ci_share": [0.0, 1.0, 0.5],
            "ci_gate_contradiction_rows": [1, 0, 2],
            "high_priority_ci_gate_rows": [1, 0, 0],
        }
    )

    output = lcri_uncertainty_weighted_review_priority(packet, scorecard)

    assert list(output["scope"]) == ["transition", "regime", "signal"]
    by_scope = output.set_index("scope")
    assert by_scope.loc["transition", "priority_label"] == "critical"
    assert by_scope.loc["transition", "uncertainty_weighted_priority"] == pytest.approx(6.6)
    assert by_scope.loc["regime", "priority_label"] == "high"
    assert by_scope.loc["signal", "base_review_priority"] == 2
    assert "blocking_ci_gate_review" in by_scope.loc["signal", "review_note"]


def test_lcri_uncertainty_weighted_review_priority_summary_counts_labels() -> None:
    priorities = pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition", "session"],
            "priority_label": ["critical", "high", "medium", "low"],
            "uncertainty_weighted_priority": [6.4, 5.1, 3.0, 1.0],
        }
    )

    summary = lcri_uncertainty_weighted_review_priority_summary(priorities)

    assert summary == {
        "scopes": 4,
        "critical_priority_scopes": 1,
        "high_priority_scopes": 1,
        "medium_priority_scopes": 1,
        "low_priority_scopes": 1,
        "max_uncertainty_weighted_priority": 6.4,
        "worst_uncertainty_weighted_scope": "signal:critical",
    }


def test_lcri_cross_artifact_evidence_index_joins_scope_review_signals() -> None:
    severity_by_scope = pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition"],
            "rows": [1, 2, 2],
            "stable_rows": [1, 0, 0],
            "warning_rows": [0, 1, 0],
            "critical_rows": [0, 1, 2],
        }
    )
    gate_decisions = pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition"],
            "rows": [1, 2, 2],
            "decision": ["pass", "block", "block"],
            "reason": ["ok", "critical", "critical"],
        }
    )
    delta_summary = pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition"],
            "rows": [1, 2, 2],
            "mean_raw_minus_lcri_gap": [0.02, -0.03, -0.04],
            "min_raw_minus_lcri_gap": [0.02, -0.05, -0.06],
            "max_raw_minus_lcri_gap": [0.02, 0.01, -0.02],
            "lcri_more_stable_share": [1.0, 0.5, 0.0],
            "lcri_less_stable_share": [0.0, 0.5, 1.0],
        }
    )
    contradictions = pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition"],
            "decision": ["pass", "block", "block"],
            "rows": [1, 2, 2],
            "lcri_more_stable_share": [1.0, 0.5, 0.0],
            "lcri_less_stable_share": [0.0, 0.5, 1.0],
            "fragility_review_required_rows": [0, 1, 3],
            "contradiction_label": [
                "aligned",
                "mixed_stability_block_scope",
                "block_scope_with_relative_regressions",
            ],
            "review_note": ["ok", "review", "review"],
        }
    )
    ci_scorecard = pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition"],
            "coverage_label": ["adequate_ci", "wide_ci_review", "blocking_ci_gate_review"],
            "mean_ci_width": [0.05, 0.20, 0.25],
            "max_ci_width": [0.07, 0.25, 0.35],
            "wide_ci_share": [0.0, 0.5, 1.0],
            "ci_gate_contradiction_rows": [0, 1, 2],
            "high_priority_ci_gate_rows": [0, 0, 1],
            "max_ci_gate_review_priority": [0, 2, 3],
            "review_note": ["ok", "review", "review"],
        }
    )
    priorities = pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition"],
            "contradiction_label": ["aligned", "mixed", "regression"],
            "base_review_priority": [0, 2, 3],
            "fragility_review_required_rows": [0, 1, 3],
            "worst_fragility_abs_gap_to_se_ratio": [0.0, 2.2, 4.5],
            "coverage_label": ["adequate_ci", "wide_ci_review", "blocking_ci_gate_review"],
            "mean_ci_width": [0.05, 0.20, 0.25],
            "max_ci_width": [0.07, 0.25, 0.35],
            "wide_ci_share": [0.0, 0.5, 1.0],
            "ci_gate_contradiction_rows": [0, 1, 2],
            "high_priority_ci_gate_rows": [0, 0, 1],
            "uncertainty_weighted_priority": [0.5, 4.0, 6.5],
            "priority_label": ["low", "medium", "critical"],
            "review_note": ["low", "medium", "critical"],
        }
    )

    output = lcri_cross_artifact_evidence_index(
        severity_by_scope,
        gate_decisions,
        delta_summary,
        contradictions,
        ci_scorecard,
        priorities,
    )

    assert list(output["scope"]) == ["transition", "regime", "signal"]
    by_scope = output.set_index("scope")
    assert by_scope.loc["transition", "evidence_label"] == "urgent"
    assert by_scope.loc["transition", "evidence_score"] > by_scope.loc["regime", "evidence_score"]
    assert by_scope.loc["signal", "evidence_label"] == "aligned"
    assert "owner review first" in by_scope.loc["transition", "review_note"]


def test_lcri_cross_artifact_evidence_index_summary_counts_labels() -> None:
    evidence = pd.DataFrame(
        {
            "scope": ["transition", "regime", "session", "signal"],
            "evidence_label": ["urgent", "review", "monitor", "aligned"],
            "evidence_score": [11.5, 6.0, 3.0, 0.3],
        }
    )

    summary = lcri_cross_artifact_evidence_index_summary(evidence)

    assert summary == {
        "scopes": 4,
        "urgent_scopes": 1,
        "review_scopes": 1,
        "monitor_scopes": 1,
        "aligned_scopes": 1,
        "max_evidence_score": 11.5,
        "worst_evidence_scope": "transition:urgent",
    }


def test_lcri_evidence_release_checklist_converts_index_to_owner_actions() -> None:
    evidence = pd.DataFrame(
        {
            "scope": ["transition", "regime", "session", "signal"],
            "gate_decision": ["block", "warn", "pass", "pass"],
            "evidence_label": ["urgent", "review", "monitor", "aligned"],
            "evidence_score": [11.5, 6.0, 3.0, 0.3],
            "priority_label": ["critical", "medium", "low", "low"],
        }
    )

    checklist = lcri_evidence_release_checklist(evidence)

    assert list(checklist["scope"]) == ["transition", "regime", "session", "signal"]
    by_scope = checklist.set_index("scope")
    assert by_scope.loc["transition", "check_status"] == "blocked"
    assert by_scope.loc["regime", "check_status"] == "needs_review"
    assert by_scope.loc["session", "check_status"] == "monitor"
    assert by_scope.loc["signal", "check_status"] == "ready"
    assert "waive transition evidence" in by_scope.loc["transition", "required_action"]
    assert by_scope.loc["signal", "source_artifact"] == "lcri_cross_artifact_evidence_index.csv"


def test_lcri_evidence_release_checklist_summary_counts_statuses() -> None:
    checklist = pd.DataFrame(
        {
            "scope": ["transition", "regime", "session", "signal"],
            "check_status": ["blocked", "needs_review", "monitor", "ready"],
            "evidence_score": [11.5, 6.0, 3.0, 0.3],
        }
    )

    summary = lcri_evidence_release_checklist_summary(checklist)

    assert summary == {
        "items": 4,
        "blocked_items": 1,
        "review_items": 1,
        "monitor_items": 1,
        "ready_items": 1,
        "max_evidence_score": 11.5,
        "worst_check_scope": "transition:blocked",
        "release_ready": False,
    }


def test_lcri_owner_handoff_packet_prioritizes_checklist_with_evidence_context() -> None:
    evidence = pd.DataFrame(
        {
            "scope": ["transition", "regime", "session", "signal"],
            "critical_rows": [2, 0, 0, 0],
            "warning_rows": [1, 2, 0, 0],
            "fragility_review_required_rows": [1, 1, 0, 0],
            "ci_gate_contradiction_rows": [2, 0, 0, 0],
            "high_priority_ci_gate_rows": [1, 0, 0, 0],
            "lcri_less_stable_share": [0.75, 0.25, 0.10, 0.0],
        }
    )
    checklist = pd.DataFrame(
        {
            "scope": ["transition", "regime", "session", "signal"],
            "check_status": ["blocked", "needs_review", "monitor", "ready"],
            "gate_decision": ["block", "warn", "pass", "pass"],
            "evidence_label": ["urgent", "review", "monitor", "aligned"],
            "evidence_score": [11.5, 6.0, 3.0, 0.3],
            "priority_label": ["critical", "medium", "low", "low"],
            "required_action": ["fix transition", "review regime", "note session", "sign signal"],
            "source_artifact": ["lcri_cross_artifact_evidence_index.csv"] * 4,
        }
    )

    packet = lcri_owner_handoff_packet(evidence, checklist)

    assert list(packet["scope"]) == ["transition", "regime", "session", "signal"]
    assert list(packet["handoff_rank"]) == [1, 2, 3, 4]
    by_scope = packet.set_index("scope")
    assert by_scope.loc["transition", "handoff_status"] == "immediate_owner_decision"
    assert by_scope.loc["regime", "handoff_status"] == "owner_review"
    assert by_scope.loc["session", "handoff_status"] == "release_note_monitor"
    assert by_scope.loc["signal", "handoff_status"] == "signoff_ready"
    assert by_scope.loc["transition", "critical_rows"] == 2
    assert "waive/fix posture" in by_scope.loc["transition", "owner_queue"]
    assert by_scope.loc["signal", "checklist_source_artifact"] == "lcri_evidence_release_checklist.csv"


def test_lcri_owner_handoff_packet_summary_counts_queue_statuses() -> None:
    packet = pd.DataFrame(
        {
            "scope": ["transition", "regime", "session", "signal"],
            "handoff_status": [
                "immediate_owner_decision",
                "owner_review",
                "release_note_monitor",
                "signoff_ready",
            ],
            "evidence_score": [11.5, 6.0, 3.0, 0.3],
        }
    )

    summary = lcri_owner_handoff_packet_summary(packet)

    assert summary == {
        "items": 4,
        "immediate_items": 1,
        "review_items": 1,
        "monitor_items": 1,
        "signoff_items": 1,
        "max_evidence_score": 11.5,
        "top_handoff_scope": "transition:immediate_owner_decision",
        "handoff_clear": False,
    }


def test_lcri_evidence_lineage_map_traces_owner_artifact_chain() -> None:
    evidence = pd.DataFrame(
        {
            "scope": ["transition", "signal"],
            "evidence_label": ["urgent", "aligned"],
            "evidence_score": [11.5, 0.3],
        }
    )
    checklist = pd.DataFrame(
        {
            "scope": ["transition", "signal"],
            "check_status": ["blocked", "ready"],
            "source_artifact": ["lcri_cross_artifact_evidence_index.csv"] * 2,
        }
    )
    handoff = pd.DataFrame(
        {
            "scope": ["transition", "signal"],
            "handoff_status": ["immediate_owner_decision", "signoff_ready"],
            "evidence_source_artifact": ["lcri_cross_artifact_evidence_index.csv"] * 2,
            "checklist_source_artifact": ["lcri_evidence_release_checklist.csv"] * 2,
        }
    )

    lineage = lcri_evidence_lineage_map(evidence, checklist, handoff)

    assert list(lineage["scope"]) == ["transition", "signal"]
    by_scope = lineage.set_index("scope")
    assert by_scope.loc["transition", "lineage_status"] == "complete"
    assert by_scope.loc["transition", "handoff_status"] == "immediate_owner_decision"
    assert "index to release checklist to handoff" in by_scope.loc["signal", "lineage_note"]


def test_lcri_evidence_lineage_map_flags_stale_source_references() -> None:
    evidence = pd.DataFrame(
        {"scope": ["signal"], "evidence_label": ["aligned"], "evidence_score": [0.3]}
    )
    checklist = pd.DataFrame(
        {
            "scope": ["signal"],
            "check_status": ["ready"],
            "source_artifact": ["stale_evidence.csv"],
        }
    )
    handoff = pd.DataFrame(
        {
            "scope": ["signal"],
            "handoff_status": ["signoff_ready"],
            "evidence_source_artifact": ["lcri_cross_artifact_evidence_index.csv"],
            "checklist_source_artifact": ["lcri_evidence_release_checklist.csv"],
        }
    )

    lineage = lcri_evidence_lineage_map(evidence, checklist, handoff)

    assert lineage.loc[0, "lineage_status"] == "source_mismatch"
    assert "stale or unexpected" in lineage.loc[0, "lineage_note"]


def test_lcri_evidence_lineage_map_summary_counts_chain_health() -> None:
    lineage = pd.DataFrame(
        {
            "scope": ["transition", "regime", "signal"],
            "lineage_status": ["complete", "source_mismatch", "incomplete_lineage"],
            "evidence_score": [11.5, 6.0, 0.3],
        }
    )

    summary = lcri_evidence_lineage_map_summary(lineage)

    assert summary == {
        "scopes": 3,
        "complete_scopes": 1,
        "source_mismatch_scopes": 1,
        "incomplete_scopes": 1,
        "max_evidence_score": 11.5,
        "worst_lineage_scope": "transition:complete",
        "lineage_clear": False,
    }


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


def test_lcri_generalization_blocker_summary_reports_worst_blocker() -> None:
    critical_contexts = pd.DataFrame(
        {
            "scope": ["regime", "transition"],
            "context": ["thin", "transition"],
            "directional_accuracy_gap": [0.08, 0.06],
        }
    )

    output = lcri_generalization_blocker_summary(critical_contexts)

    assert output["critical_rows"] == 2
    assert output["critical_scopes"] == "regime,transition"
    assert output["max_critical_gap"] == pytest.approx(0.08)
    assert output["max_critical_context"] == "regime:thin"


def test_lcri_generalization_critical_contexts_sorts_blocking_rows() -> None:
    severity = pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition"],
            "context": ["all", "thin", "transition"],
            "directional_accuracy_gap": [0.03, 0.08, 0.06],
            "severity": ["warning", "critical", "critical"],
        }
    )

    output = lcri_generalization_critical_contexts(severity)

    assert list(output["context"]) == ["thin", "transition"]
    assert list(output["severity"]) == ["critical", "critical"]


def test_lcri_generalization_critical_contexts_handles_no_critical_rows() -> None:
    severity = pd.DataFrame(
        {"directional_accuracy_gap": [0.01], "severity": ["stable"]}
    )

    output = lcri_generalization_critical_contexts(severity)

    assert output.empty
    assert "severity" in output.columns


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


def test_lcri_generalization_scope_risk_reports_warning_and_critical_rates() -> None:
    severity_by_scope = pd.DataFrame(
        {
            "scope": ["regime", "signal"],
            "rows": [4, 2],
            "stable_rows": [1, 2],
            "warning_rows": [2, 0],
            "critical_rows": [1, 0],
        }
    )

    output = lcri_generalization_scope_risk(severity_by_scope).set_index("scope")

    assert output.loc["regime", "warning_or_critical_share"] == pytest.approx(0.75)
    assert output.loc["regime", "critical_share"] == pytest.approx(0.25)
    assert output.loc["signal", "critical_share"] == pytest.approx(0.0)


def test_lcri_generalization_scope_gate_decisions_assigns_scope_actions() -> None:
    scope_risk = pd.DataFrame(
        {
            "scope": ["regime", "signal", "transition"],
            "rows": [2, 1, 1],
            "warning_or_critical_share": [1.0, 0.0, 1.0],
            "critical_share": [0.5, 0.0, 0.0],
        }
    )

    output = lcri_generalization_scope_gate_decisions(scope_risk).set_index("scope")

    assert output.loc["regime", "decision"] == "block"
    assert output.loc["signal", "decision"] == "pass"
    assert output.loc["transition", "decision"] == "warn"


def test_lcri_scope_gate_decision_summary_lists_action_scopes() -> None:
    decisions = pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition"],
            "decision": ["pass", "block", "warn"],
        }
    )

    output = lcri_scope_gate_decision_summary(decisions)

    assert output["scopes"] == 3
    assert output["pass_scopes"] == 1
    assert output["warn_scopes"] == 1
    assert output["block_scopes"] == 1
    assert output["blocked_scope_names"] == "regime"
    assert output["warn_scope_names"] == "transition"


def test_lcri_scope_gate_decision_summary_handles_empty_decisions() -> None:
    output = lcri_scope_gate_decision_summary(pd.DataFrame())

    assert output["scopes"] == 0
    assert output["blocked_scope_names"] == "none"
    assert output["warn_scope_names"] == "none"


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


def test_lcri_gap_delta_improvements_sorts_best_positive_edges() -> None:
    gap_delta = pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition"],
            "context": ["all", "thin", "transition"],
            "raw_minus_lcri_directional_accuracy_gap": [0.02, -0.04, 0.06],
        }
    )

    output = lcri_gap_delta_improvements(gap_delta)

    assert list(output["context"]) == ["transition", "all"]
    assert list(output["raw_minus_lcri_directional_accuracy_gap"]) == [0.06, 0.02]


def test_lcri_gap_delta_improvements_handles_no_positive_edges() -> None:
    gap_delta = pd.DataFrame(
        {"raw_minus_lcri_directional_accuracy_gap": [-0.02, 0.0]}
    )

    assert lcri_gap_delta_improvements(gap_delta).empty


def test_lcri_gap_delta_regressions_sorts_worst_negative_edges() -> None:
    gap_delta = pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition"],
            "context": ["all", "thin", "transition"],
            "raw_minus_lcri_directional_accuracy_gap": [0.02, -0.04, -0.01],
        }
    )

    output = lcri_gap_delta_regressions(gap_delta)

    assert list(output["context"]) == ["thin", "transition"]
    assert list(output["raw_minus_lcri_directional_accuracy_gap"]) == [-0.04, -0.01]


def test_lcri_gap_delta_regressions_handles_all_stable_edges() -> None:
    gap_delta = pd.DataFrame(
        {"raw_minus_lcri_directional_accuracy_gap": [0.02, 0.0]}
    )

    assert lcri_gap_delta_regressions(gap_delta).empty


def test_lcri_gap_delta_scope_summary_groups_relative_stability() -> None:
    gap_delta = pd.DataFrame(
        {
            "scope": ["signal", "regime", "regime"],
            "raw_minus_lcri_directional_accuracy_gap": [0.02, -0.04, 0.06],
        }
    )

    output = lcri_gap_delta_scope_summary(gap_delta).set_index("scope")

    assert output.loc["regime", "rows"] == 2
    assert output.loc["regime", "mean_raw_minus_lcri_gap"] == pytest.approx(0.01)
    assert output.loc["regime", "min_raw_minus_lcri_gap"] == pytest.approx(-0.04)
    assert output.loc["regime", "max_raw_minus_lcri_gap"] == pytest.approx(0.06)
    assert output.loc["regime", "lcri_more_stable_share"] == pytest.approx(0.5)
    assert output.loc["regime", "lcri_less_stable_share"] == pytest.approx(0.5)


def test_lcri_gap_delta_dominant_scopes_identifies_edge_and_drag() -> None:
    scope_summary = pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition"],
            "mean_raw_minus_lcri_gap": [0.01, -0.03, 0.05],
        }
    )

    output = lcri_gap_delta_dominant_scopes(scope_summary)

    assert output["best_scope"] == "transition"
    assert output["best_mean_raw_minus_lcri_gap"] == pytest.approx(0.05)
    assert output["worst_scope"] == "regime"
    assert output["worst_mean_raw_minus_lcri_gap"] == pytest.approx(-0.03)


def test_lcri_gap_delta_dominant_scopes_handles_empty_summary() -> None:
    output = lcri_gap_delta_dominant_scopes(pd.DataFrame())

    assert output == {
        "best_scope": "none",
        "best_mean_raw_minus_lcri_gap": 0.0,
        "worst_scope": "none",
        "worst_mean_raw_minus_lcri_gap": 0.0,
    }


def test_lcri_gap_delta_scope_extremes_selects_best_and_worst_contexts() -> None:
    gap_delta = pd.DataFrame(
        {
            "scope": ["regime", "regime", "signal", "signal"],
            "context": ["thin", "deep", "all", "stress"],
            "raw_minus_lcri_directional_accuracy_gap": [-0.04, 0.06, 0.01, -0.02],
        }
    )

    output = lcri_gap_delta_scope_extremes(gap_delta).set_index("scope")

    assert output.loc["regime", "best_context"] == "deep"
    assert output.loc["regime", "best_raw_minus_lcri_gap"] == pytest.approx(0.06)
    assert output.loc["regime", "worst_context"] == "thin"
    assert output.loc["regime", "worst_raw_minus_lcri_gap"] == pytest.approx(-0.04)
    assert output.loc["signal", "best_context"] == "all"
    assert output.loc["signal", "worst_context"] == "stress"


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


def test_calibration_error_summary_reports_weighted_error() -> None:
    curve = pd.DataFrame(
        {
            "bin": [0, 1, 2],
            "predicted_probability": [0.20, 0.50, 0.80],
            "observed_frequency": [0.10, 0.75, 0.60],
            "rows": [10, 30, 60],
        }
    )

    summary = calibration_error_summary(curve)

    assert summary == {
        "bins": 3,
        "rows": 100,
        "expected_calibration_error": pytest.approx(0.205),
        "max_calibration_error": pytest.approx(0.25),
        "worst_calibration_bin": "1.0",
    }


def test_calibration_gate_decision_blocks_error_breaches() -> None:
    summary = {
        "bins": 10,
        "rows": 1000,
        "expected_calibration_error": 0.08,
        "max_calibration_error": 0.20,
        "worst_calibration_bin": "7",
    }

    decision = calibration_gate_decision(summary, max_ece=0.05, max_bin_error=0.15)

    assert decision["passes"] is False
    assert decision["decision"] == "block"
    assert decision["worst_calibration_bin"] == "7"
    assert "ECE 0.0800 exceeds 0.0500" in decision["reason"]
    assert "max bin error 0.2000 exceeds 0.1500" in decision["reason"]


def test_calibration_gate_decision_passes_clean_summary() -> None:
    summary = {
        "bins": 8,
        "rows": 500,
        "expected_calibration_error": 0.02,
        "max_calibration_error": 0.06,
        "worst_calibration_bin": "3",
    }

    decision = calibration_gate_decision(summary)

    assert decision["passes"] is True
    assert decision["decision"] == "pass"
    assert decision["reason"] == "calibration passed release thresholds"


def test_calibration_error_summary_rejects_negative_rows() -> None:
    curve = pd.DataFrame(
        {
            "bin": [0],
            "predicted_probability": [0.2],
            "observed_frequency": [0.3],
            "rows": [-1],
        }
    )

    with pytest.raises(ValueError, match="row counts"):
        calibration_error_summary(curve)


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
