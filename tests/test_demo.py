import json
from pathlib import Path

import pandas as pd
import pytest

from lcri_lab.cli import run_demo, verify_report


def test_run_demo_writes_reports(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_demo(rows=750, seed=3, train_frac=0.65, output=tmp_path)
    output = capsys.readouterr().out

    assert "train fraction: 0.65" in output

    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "heldout_metrics.csv").exists()
    assert (tmp_path / "generalization_gap.csv").exists()
    assert (tmp_path / "regime_metrics.csv").exists()
    assert (tmp_path / "heldout_regime_metrics.csv").exists()
    assert (tmp_path / "regime_generalization_gap.csv").exists()
    assert (tmp_path / "transition_metrics.csv").exists()
    assert (tmp_path / "heldout_transition_metrics.csv").exists()
    assert (tmp_path / "transition_generalization_gap.csv").exists()
    assert (tmp_path / "generalization_overview.json").exists()
    assert (tmp_path / "generalization_gap_leaderboard.csv").exists()
    assert (tmp_path / "lcri_generalization_gap_leaderboard.csv").exists()
    assert (tmp_path / "lcri_generalization_scope_summary.csv").exists()
    assert (tmp_path / "lcri_generalization_severity.csv").exists()
    assert (tmp_path / "lcri_generalization_severity_by_scope.csv").exists()
    assert (tmp_path / "lcri_generalization_scope_risk.csv").exists()
    assert (tmp_path / "lcri_generalization_scope_gate_decisions.csv").exists()
    assert (tmp_path / "lcri_generalization_scope_gate_decision_summary.json").exists()
    assert (tmp_path / "lcri_generalization_critical_contexts.csv").exists()
    assert (tmp_path / "lcri_generalization_blocker_summary.json").exists()
    assert (tmp_path / "lcri_generalization_severity_summary.json").exists()
    assert (tmp_path / "lcri_worst_generalization_context.json").exists()
    assert (tmp_path / "lcri_generalization_gate_decision.json").exists()
    assert (tmp_path / "lcri_generalization_gap_delta.csv").exists()
    assert (tmp_path / "lcri_gap_delta_dominant_scopes.json").exists()
    assert (tmp_path / "lcri_gap_delta_flags.csv").exists()
    assert (tmp_path / "lcri_gap_delta_improvements.csv").exists()
    assert (tmp_path / "lcri_gap_delta_regressions.csv").exists()
    assert (tmp_path / "lcri_gap_delta_scorecard.json").exists()
    assert (tmp_path / "lcri_gap_delta_scope_extremes.csv").exists()
    assert (tmp_path / "lcri_gap_delta_scope_summary.csv").exists()
    assert (tmp_path / "lcri_gap_delta_summary.json").exists()
    assert (tmp_path / "transition_lift.csv").exists()
    assert (tmp_path / "heldout_transition_lift.csv").exists()
    assert (tmp_path / "pressure_memory_decay_summary.csv").exists()
    assert (tmp_path / "heldout_pressure_memory_decay_summary.csv").exists()
    assert (tmp_path / "hidden_resiliency_asymmetry_summary.json").exists()
    assert (tmp_path / "heldout_hidden_resiliency_asymmetry_summary.json").exists()
    assert (tmp_path / "adverse_selection_phase_shift_summary.csv").exists()
    assert (tmp_path / "heldout_adverse_selection_phase_shift_summary.csv").exists()
    assert (tmp_path / "phase_shift_artifact_review.csv").exists()
    assert (tmp_path / "heldout_phase_shift_artifact_review.csv").exists()
    assert (tmp_path / "lcri_signal_monotonicity.csv").exists()
    assert (tmp_path / "heldout_lcri_signal_monotonicity.csv").exists()
    assert (tmp_path / "lcri_signal_monotonicity_summary.json").exists()
    assert (tmp_path / "heldout_lcri_signal_monotonicity_summary.json").exists()
    assert (tmp_path / "lcri_calibration_curve.csv").exists()
    assert (tmp_path / "heldout_lcri_calibration_curve.csv").exists()
    assert (tmp_path / "lcri_calibration_summary.json").exists()
    assert (tmp_path / "heldout_lcri_calibration_summary.json").exists()
    assert (tmp_path / "lcri_calibration_gate.json").exists()
    assert (tmp_path / "heldout_lcri_calibration_gate.json").exists()
    assert (tmp_path / "transition_robustness.json").exists()
    assert (tmp_path / "heldout_transition_robustness.json").exists()
    assert (tmp_path / "research_summary.md").exists()
    assert (tmp_path / "artifact_manifest.json").exists()
    assert (tmp_path / "artifact_metadata_summary.json").exists()
    assert (tmp_path / "sample_snapshots.csv").exists()
    assert (tmp_path / "figures" / "raw_vs_lcri_scatter.png").exists()
    assert (tmp_path / "figures" / "regime_signal_quality.png").exists()
    assert (tmp_path / "figures" / "transition_signal_quality.png").exists()
    assert (tmp_path / "figures" / "heldout_transition_signal_quality.png").exists()
    assert (tmp_path / "figures" / "calibration_curve.png").exists()
    assert (tmp_path / "figures" / "heldout_calibration_curve.png").exists()
    assert (tmp_path / "figures" / "generalization_gap.png").exists()
    assert (tmp_path / "figures" / "regime_generalization_gap.png").exists()
    assert (tmp_path / "figures" / "transition_generalization_gap.png").exists()
    assert (tmp_path / "figures" / "generalization_stability_confidence_intervals.png").exists()
    assert (tmp_path / "figures" / "lcri_generalization_gap_delta.png").exists()
    assert (tmp_path / "figures" / "lcri_generalization_severity_by_scope.png").exists()
    assert (tmp_path / "figures" / "lcri_ci_gate_contradiction_diagnostics.png").exists()
    assert (tmp_path / "lcri_ci_confidence_coverage_scorecard.csv").exists()
    assert (tmp_path / "lcri_ci_confidence_coverage_summary.json").exists()
    assert (tmp_path / "figures" / "lcri_ci_confidence_coverage_scorecard.png").exists()
    assert (tmp_path / "figures" / "lcri_gap_delta_scope_summary.png").exists()
    assert (tmp_path / "figures" / "lcri_contradiction_review_packet.png").exists()
    assert (tmp_path / "figures" / "lcri_uncertainty_weighted_review_priority.png").exists()
    assert (tmp_path / "figures" / "lcri_cross_artifact_evidence_index.png").exists()
    assert (tmp_path / "figures" / "lcri_evidence_release_checklist.png").exists()
    assert (tmp_path / "figures" / "lcri_owner_handoff_packet.png").exists()
    assert (tmp_path / "figures" / "lcri_evidence_lineage_map.png").exists()
    assert (tmp_path / "lcri_owner_handoff_packet.md").exists()
    assert (tmp_path / "lcri_evidence_release_checklist.csv").exists()
    assert (tmp_path / "lcri_evidence_release_checklist_summary.json").exists()
    assert (tmp_path / "alpha_event_windows.csv").exists()
    assert (tmp_path / "alpha_event_regime_summary.csv").exists()
    assert (tmp_path / "alpha_event_window_regime_summary.csv").exists()
    assert (tmp_path / "alpha_event_window_summary.json").exists()
    assert (tmp_path / "alpha_event_score_weighted_drift.json").exists()
    assert (tmp_path / "alpha_event_drift_gate.json").exists()
    assert (tmp_path / "alpha_event_release_review_packet.csv").exists()
    assert (tmp_path / "alpha_event_review_verification_summary.json").exists()
    assert (tmp_path / "execution_adjusted_edge_summary.json").exists()
    assert (tmp_path / "heldout_execution_adjusted_edge_summary.json").exists()
    assert (tmp_path / "execution_publishability_review_packet.csv").exists()
    assert (tmp_path / "heldout_execution_publishability_review_packet.csv").exists()
    assert (tmp_path / "execution_adjusted_lcri_side_attribution.csv").exists()
    assert (tmp_path / "heldout_execution_adjusted_lcri_side_attribution.csv").exists()
    assert (tmp_path / "execution_publishability_release_gate.json").exists()
    assert (tmp_path / "heldout_execution_publishability_release_gate.json").exists()
    assert (tmp_path / "passive_fill_event_windows.csv").exists()
    assert (tmp_path / "passive_fill_event_lead_lag_profile.csv").exists()
    assert (tmp_path / "passive_fill_event_lead_lag_scorecard.csv").exists()
    assert (tmp_path / "passive_fill_event_regime_summary.csv").exists()
    assert (tmp_path / "passive_fill_event_transition_summary.csv").exists()
    assert (tmp_path / "passive_fill_event_lifecycle_summary.csv").exists()
    assert (tmp_path / "passive_fill_event_lifecycle_policy_curve.csv").exists()
    assert (tmp_path / "passive_fill_event_transition_policy_curve.csv").exists()
    assert (tmp_path / "passive_fill_event_toxicity_scorecard.json").exists()
    assert (tmp_path / "passive_fill_event_lifecycle_toxicity_scorecard.json").exists()
    assert (tmp_path / "passive_fill_event_transition_toxicity_scorecard.json").exists()
    assert (tmp_path / "heldout_passive_fill_event_windows.csv").exists()
    assert (tmp_path / "heldout_passive_fill_event_lead_lag_profile.csv").exists()
    assert (tmp_path / "heldout_passive_fill_event_lead_lag_scorecard.csv").exists()
    assert (tmp_path / "heldout_passive_fill_event_regime_summary.csv").exists()
    assert (tmp_path / "heldout_passive_fill_event_transition_summary.csv").exists()
    assert (tmp_path / "heldout_passive_fill_event_lifecycle_summary.csv").exists()
    assert (tmp_path / "heldout_passive_fill_event_lifecycle_policy_curve.csv").exists()
    assert (tmp_path / "heldout_passive_fill_event_transition_policy_curve.csv").exists()
    assert (tmp_path / "heldout_passive_fill_event_toxicity_scorecard.json").exists()
    assert (tmp_path / "heldout_passive_fill_event_lifecycle_toxicity_scorecard.json").exists()
    assert (tmp_path / "heldout_passive_fill_event_transition_toxicity_scorecard.json").exists()
    assert (tmp_path / "passive_fill_calibration_curve.csv").exists()
    assert (tmp_path / "heldout_passive_fill_calibration_curve.csv").exists()
    assert (tmp_path / "passive_fill_calibration_summary.json").exists()
    assert (tmp_path / "heldout_passive_fill_calibration_summary.json").exists()
    assert (tmp_path / "passive_fill_realization_horizon_sweep.csv").exists()
    assert (tmp_path / "heldout_passive_fill_realization_horizon_sweep.csv").exists()
    assert (tmp_path / "passive_fill_threshold_policy_curve.csv").exists()
    assert (tmp_path / "heldout_passive_fill_threshold_policy_curve.csv").exists()
    assert (tmp_path / "queue_position_fill_surface.csv").exists()
    assert (tmp_path / "heldout_queue_position_fill_surface.csv").exists()
    assert (tmp_path / "queue_position_fraction_sweep.csv").exists()
    assert (tmp_path / "heldout_queue_position_fraction_sweep.csv").exists()
    assert (tmp_path / "queue_position_regime_fraction_sweep.csv").exists()
    assert (tmp_path / "heldout_queue_position_regime_fraction_sweep.csv").exists()
    assert (tmp_path / "queue_position_capacity_frontier.json").exists()
    assert (tmp_path / "heldout_queue_position_capacity_frontier.json").exists()
    assert (tmp_path / "queue_position_regime_capacity_frontier.csv").exists()
    assert (tmp_path / "heldout_queue_position_regime_capacity_frontier.csv").exists()
    assert (tmp_path / "queue_position_regime_capacity_concentration.json").exists()
    assert (tmp_path / "heldout_queue_position_regime_capacity_concentration.json").exists()
    assert (tmp_path / "queue_position_capacity_stability.json").exists()
    assert (tmp_path / "queue_position_edge_decay.csv").exists()
    assert (tmp_path / "heldout_queue_position_edge_decay.csv").exists()
    assert (tmp_path / "execution_adjusted_sample.csv").exists()

    monotonicity = json.loads((tmp_path / "lcri_signal_monotonicity_summary.json").read_text())
    assert "passes_monotonicity_gate" in monotonicity
    calibration_gate = json.loads((tmp_path / "lcri_calibration_gate.json").read_text())
    assert "expected_calibration_error" in calibration_gate
    robustness = json.loads((tmp_path / "transition_robustness.json").read_text())
    assert "passes_transition_robustness" in robustness
    metadata_summary = json.loads((tmp_path / "artifact_metadata_summary.json").read_text())
    sample_columns = set(pd.read_csv(tmp_path / "sample_snapshots.csv", nrows=1).columns)
    assert {
        "pressure_memory_half_life",
        "pressure_memory_decay_ratio",
        "pressure_memory_decay_event",
        "bid_queue_ahead",
        "ask_queue_ahead",
        "bid_fill_probability",
        "ask_fill_probability",
        "execution_adjusted_edge_ticks",
        "best_execution_side",
        "alpha_event_window_regime",
        "alpha_event_distance",
    }.issubset(sample_columns)
    execution_sample_columns = set(pd.read_csv(tmp_path / "execution_adjusted_sample.csv", nrows=1).columns)
    assert {
        "lcri",
        "publishable_side",
        "best_execution_side",
        "execution_adjusted_edge_ticks",
        "bid_fill_probability",
        "ask_fill_probability",
    }.issubset(execution_sample_columns)
    execution_summary = json.loads((tmp_path / "execution_adjusted_edge_summary.json").read_text())
    assert execution_summary["rows"] == 750
    assert "tradable_share" in execution_summary
    assert "publishable_side_conflict_share" in execution_summary
    release_gate = json.loads((tmp_path / "execution_publishability_release_gate.json").read_text())
    assert release_gate["decision"] in {"pass", "review", "block"}
    assert "weighted_conflict_share" in release_gate
    assert "quality_gate_label" in release_gate
    assert "capacity_stability_label" in release_gate
    heldout_release_gate = json.loads(
        (tmp_path / "heldout_execution_publishability_release_gate.json").read_text()
    )
    assert heldout_release_gate["decision"] in {"pass", "review", "block"}
    assert "release_gate_label" in heldout_release_gate
    event_toxicity = json.loads((tmp_path / "passive_fill_event_toxicity_scorecard.json").read_text())
    assert "event_toxicity_label" in event_toxicity
    assert "weighted_mean_post_minus_pre_realized_edge" in event_toxicity
    transition_toxicity = json.loads(
        (tmp_path / "passive_fill_event_transition_toxicity_scorecard.json").read_text()
    )
    assert "transition_toxicity_label" in transition_toxicity
    assert "worst_transition" in transition_toxicity
    assert "weighted_mean_post_minus_pre_realized_edge" in transition_toxicity
    lifecycle_toxicity = json.loads(
        (tmp_path / "passive_fill_event_lifecycle_toxicity_scorecard.json").read_text()
    )
    assert "lifecycle_toxicity_gate_label" in lifecycle_toxicity
    assert "worst_lifecycle_path" in lifecycle_toxicity
    assert "weighted_mean_event_adverse_fill_probability" in lifecycle_toxicity
    lead_lag_scorecard_columns = set(
        pd.read_csv(tmp_path / "passive_fill_event_lead_lag_scorecard.csv", nrows=1).columns
    )
    assert {
        "event_regime",
        "pre_cumulative_mean_edge_ticks",
        "post_cumulative_mean_edge_ticks",
        "lead_lag_decay_ticks",
        "toxicity_inversion",
        "warning_label",
    }.issubset(lead_lag_scorecard_columns)
    event_transition_columns = set(
        pd.read_csv(tmp_path / "passive_fill_event_transition_summary.csv", nrows=1).columns
    )
    assert {
        "regime_transition",
        "event_regimes",
        "events",
        "adverse_post_edge_share",
        "mean_event_adverse_fill_probability",
        "worst_post_minus_pre_realized_edge",
    }.issubset(event_transition_columns)
    event_lifecycle_columns = set(
        pd.read_csv(tmp_path / "passive_fill_event_lifecycle_summary.csv", nrows=1).columns
    )
    assert {
        "lifecycle_path",
        "pre_window_regime",
        "event_regime",
        "post_window_regime",
        "events",
        "adverse_post_edge_share",
        "mean_post_minus_pre_realized_edge",
    }.issubset(event_lifecycle_columns)
    event_transition_policy_columns = set(
        pd.read_csv(tmp_path / "passive_fill_event_transition_policy_curve.csv", nrows=1).columns
    )
    assert {
        "regime_transition",
        "threshold",
        "candidate_events",
        "event_share",
        "mean_post_minus_pre_realized_edge",
        "adverse_post_edge_share",
        "policy_label",
    }.issubset(event_transition_policy_columns)
    execution_packet_columns = set(
        pd.read_csv(tmp_path / "execution_publishability_review_packet.csv", nrows=1).columns
    )
    assert {
        "publishable_side",
        "best_execution_side",
        "mean_best_fill_probability",
        "mean_edge_drag_ticks",
        "review_priority",
        "review_note",
    }.issubset(execution_packet_columns)
    execution_lcri_side_columns = set(
        pd.read_csv(tmp_path / "execution_adjusted_lcri_side_attribution.csv", nrows=1).columns
    )
    assert {
        "lcri_side",
        "tradable_rows",
        "execution_conflict_share",
        "mean_signal_confidence",
        "mean_fill_probability_advantage",
        "dominant_execution_side",
        "review_label",
    }.issubset(execution_lcri_side_columns)
    passive_fill_calibration_columns = set(
        pd.read_csv(tmp_path / "passive_fill_calibration_curve.csv", nrows=1).columns
    )
    assert {
        "regime",
        "bin",
        "mean_predicted_fill_probability",
        "realized_fill_rate",
        "absolute_calibration_error",
        "brier_score",
    }.issubset(passive_fill_calibration_columns)
    passive_fill_calibration_summary = json.loads(
        (tmp_path / "passive_fill_calibration_summary.json").read_text()
    )
    assert "expected_calibration_error" in passive_fill_calibration_summary
    assert "weighted_brier_score" in passive_fill_calibration_summary
    assert passive_fill_calibration_summary["realization_horizon_snapshots"] == 2
    horizon_sweep_columns = set(
        pd.read_csv(tmp_path / "passive_fill_realization_horizon_sweep.csv", nrows=1).columns
    )
    assert {
        "horizon",
        "weighted_realized_fill_rate",
        "weighted_brier_score",
        "realized_fill_rate_gap_vs_shortest",
        "horizon_stability_label",
    }.issubset(horizon_sweep_columns)
    threshold_policy_columns = set(
        pd.read_csv(tmp_path / "passive_fill_threshold_policy_curve.csv", nrows=1).columns
    )
    assert {
        "threshold",
        "candidate_rows",
        "trade_share",
        "mean_predicted_fill_probability",
        "realized_fill_rate",
        "weighted_brier_score",
        "mean_realized_edge_ticks",
        "policy_label",
    }.issubset(threshold_policy_columns)
    queue_surface_columns = set(
        pd.read_csv(tmp_path / "queue_position_fill_surface.csv", nrows=1).columns
    )
    assert {
        "regime",
        "queue_bin",
        "fill_probability_bin",
        "mean_queue_share",
        "realized_fill_rate",
        "absolute_calibration_error",
        "mean_execution_adjusted_edge_ticks",
    }.issubset(queue_surface_columns)
    sweep_columns = set(pd.read_csv(tmp_path / "queue_position_fraction_sweep.csv", nrows=1).columns)
    assert {
        "queue_position_fraction",
        "mean_bid_fill_probability",
        "mean_ask_fill_probability",
        "mean_execution_adjusted_edge_ticks",
        "tradable_share",
        "dominant_execution_side",
    }.issubset(sweep_columns)
    capacity_frontier = json.loads((tmp_path / "queue_position_capacity_frontier.json").read_text())
    assert "max_viable_queue_position_fraction" in capacity_frontier
    assert "capacity_label" in capacity_frontier
    regime_sweep_columns = set(
        pd.read_csv(tmp_path / "queue_position_regime_fraction_sweep.csv", nrows=1).columns
    )
    assert {
        "pressure_memory_decay_state",
        "queue_position_fraction",
        "mean_execution_adjusted_edge_ticks",
        "tradable_share",
    }.issubset(regime_sweep_columns)
    regime_frontier_columns = set(
        pd.read_csv(tmp_path / "queue_position_regime_capacity_frontier.csv", nrows=1).columns
    )
    assert {
        "pressure_memory_decay_state",
        "max_viable_queue_position_fraction",
        "capacity_shortfall_fraction",
        "capacity_brittleness_label",
    }.issubset(regime_frontier_columns)
    concentration = json.loads((tmp_path / "queue_position_regime_capacity_concentration.json").read_text())
    assert "capacity_concentration_label" in concentration
    assert "front_only_or_no_capacity_share" in concentration
    capacity_stability = json.loads((tmp_path / "queue_position_capacity_stability.json").read_text())
    assert "capacity_fraction_gap" in capacity_stability
    assert "capacity_stability_label" in capacity_stability
    edge_decay_columns = set(pd.read_csv(tmp_path / "queue_position_edge_decay.csv", nrows=1).columns)
    assert {
        "regime",
        "front_mean_queue_share",
        "back_mean_queue_share",
        "fill_rate_decay",
        "edge_decay_ticks",
        "queue_decay_label",
    }.issubset(edge_decay_columns)
    assert metadata_summary["artifacts_with_metadata"] > 0
    assert metadata_summary["total_size_bytes"] > 0
    assert metadata_summary["largest_artifact"] != "none"
    manifest = json.loads((tmp_path / "artifact_manifest.json").read_text())
    assert manifest["run"]["seed"] == 3
    assert manifest["model"]["artifact_version"] == 3
    assert "research_summary.md" in manifest["artifacts"]
    assert manifest["artifact_metadata"]["metrics.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["heldout_metrics.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["generalization_gap.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["heldout_regime_metrics.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["regime_generalization_gap.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["heldout_transition_metrics.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["transition_generalization_gap.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["generalization_overview.json"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["generalization_gap_leaderboard.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["lcri_generalization_gap_leaderboard.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["lcri_generalization_scope_summary.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["lcri_generalization_severity.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["lcri_generalization_severity_by_scope.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["lcri_generalization_scope_risk.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["lcri_generalization_scope_gate_decisions.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["lcri_generalization_critical_contexts.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["lcri_generalization_blocker_summary.json"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["lcri_generalization_severity_summary.json"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["lcri_worst_generalization_context.json"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["lcri_generalization_gate_decision.json"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["lcri_generalization_gap_delta.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["figures/generalization_stability_confidence_intervals.png"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["figures/lcri_generalization_severity_by_scope.png"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["figures/lcri_ci_gate_contradiction_diagnostics.png"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["lcri_ci_confidence_coverage_scorecard.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["lcri_ci_confidence_coverage_summary.json"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["figures/lcri_ci_confidence_coverage_scorecard.png"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["figures/lcri_gap_delta_scope_summary.png"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["figures/lcri_contradiction_review_packet.png"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["figures/lcri_uncertainty_weighted_review_priority.png"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["figures/lcri_cross_artifact_evidence_index.png"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["figures/lcri_evidence_release_checklist.png"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["figures/lcri_owner_handoff_packet.png"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["figures/lcri_evidence_lineage_map.png"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["lcri_owner_handoff_packet.md"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["lcri_evidence_release_checklist.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["lcri_evidence_release_checklist_summary.json"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["lcri_gap_delta_dominant_scopes.json"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["pressure_memory_decay_summary.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["heldout_pressure_memory_decay_summary.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["hidden_resiliency_asymmetry_summary.json"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["adverse_selection_phase_shift_summary.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["phase_shift_artifact_review.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["lcri_gap_delta_flags.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["lcri_gap_delta_improvements.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["lcri_gap_delta_regressions.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["lcri_gap_delta_scorecard.json"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["lcri_gap_delta_scope_extremes.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["lcri_gap_delta_scope_summary.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["artifact_metadata_summary.json"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["lcri_gap_delta_summary.json"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["heldout_transition_lift.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["alpha_event_release_review_packet.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["alpha_event_window_regime_summary.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["alpha_event_drift_gate.json"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["execution_adjusted_edge_summary.json"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["heldout_execution_adjusted_edge_summary.json"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["execution_publishability_review_packet.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["heldout_execution_publishability_review_packet.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["execution_adjusted_lcri_side_attribution.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["heldout_execution_adjusted_lcri_side_attribution.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["execution_publishability_release_gate.json"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["heldout_execution_publishability_release_gate.json"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["passive_fill_event_windows.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["passive_fill_event_regime_summary.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["passive_fill_event_lifecycle_summary.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["passive_fill_event_toxicity_scorecard.json"]["size_bytes"] > 0
    assert (
        manifest["artifact_metadata"]["passive_fill_event_lifecycle_toxicity_scorecard.json"][
            "size_bytes"
        ]
        > 0
    )
    assert manifest["artifact_metadata"]["passive_fill_event_transition_policy_curve.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["passive_fill_realization_horizon_sweep.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["passive_fill_threshold_policy_curve.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["heldout_passive_fill_event_windows.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["heldout_passive_fill_event_regime_summary.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["heldout_passive_fill_event_toxicity_scorecard.json"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["heldout_passive_fill_realization_horizon_sweep.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["heldout_passive_fill_threshold_policy_curve.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["queue_position_fill_surface.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["heldout_queue_position_fill_surface.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["queue_position_fraction_sweep.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["heldout_queue_position_fraction_sweep.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["queue_position_regime_fraction_sweep.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["heldout_queue_position_regime_fraction_sweep.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["queue_position_capacity_frontier.json"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["heldout_queue_position_capacity_frontier.json"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["queue_position_regime_capacity_frontier.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["heldout_queue_position_regime_capacity_frontier.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["queue_position_regime_capacity_concentration.json"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["heldout_queue_position_regime_capacity_concentration.json"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["queue_position_capacity_stability.json"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["queue_position_edge_decay.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["heldout_queue_position_edge_decay.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["execution_adjusted_sample.csv"]["size_bytes"] > 0
    assert len(manifest["artifact_metadata"]["metrics.csv"]["sha256"]) == 64
    summary = (tmp_path / "research_summary.md").read_text()
    assert "# LCRI Research Summary" in summary
    assert "## Heldout signal quality" in summary
    assert "## Signal generalization gap" in summary
    assert "## Regime generalization gap" in summary
    assert "## Transition generalization gap" in summary
    assert "## Generalization overview" in summary
    assert "## Generalization gap leaderboard" in summary
    assert "## LCRI generalization gap leaderboard" in summary
    assert "## LCRI generalization scope summary" in summary
    assert "## LCRI generalization severity" in summary
    assert "## LCRI CI gate contradiction diagnostics" in summary
    assert "## LCRI CI gate contradiction summary" in summary
    assert "## LCRI CI confidence coverage scorecard" in summary
    assert "## LCRI CI confidence coverage summary" in summary
    assert "## LCRI generalization severity by scope" in summary
    assert "## LCRI generalization severity summary" in summary
    assert "## LCRI worst generalization context" in summary
    assert "## LCRI generalization gate decision" in summary
    assert "## LCRI generalization gap delta" in summary
    assert "## LCRI gap delta flags" in summary
    assert "## LCRI gap delta scorecard" in summary
    assert "## LCRI gap delta summary" in summary
    assert "## LCRI evidence release checklist" in summary
    assert "## LCRI evidence release checklist summary" in summary
    assert "## Alpha event release review packet" in summary
    assert "## Alpha event-window row regimes" in summary
    assert "## Alpha event drift gate" in summary
    assert "## Transition robustness" in summary
    assert "## Heldout transition lift" in summary
    assert "## Hidden resiliency asymmetry summary" in summary
    assert "## Adverse selection phase-shift summary" in summary
    assert "## Execution-adjusted edge summary" in summary
    assert "## Execution publishability release gate" in summary
    assert "## Passive-fill event-window regime diagnostics" in summary
    assert "## Passive-fill event-window toxicity scorecard" in summary
    assert "## Passive-fill event lifecycle toxicity scorecard" in summary
    assert "## Passive-fill realization horizon sweep" in summary
    assert "## Queue-position fill calibration surface" in summary
    assert "## Queue-position fraction sweep" in summary
    assert "## Queue-position capacity frontier" in summary
    assert "## Queue-position capacity stability" in summary
    assert "## Queue-position edge decay" in summary

    verify_report(tmp_path)


def test_run_demo_rejects_invalid_train_fraction(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="train_frac"):
        run_demo(rows=100, seed=3, train_frac=1.0, output=tmp_path)
