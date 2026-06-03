from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lcri_lab.evaluation import (
    calibration_curve,
    calibration_error_summary,
    calibration_fracture_gate_decision,
    calibration_gate_decision,
    calibration_monotonicity_pressure,
    calibration_monotonicity_pressure_summary,
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
    transition_conditioned_metrics,
    transition_generalization_gap,
    transition_robustness_summary,
    transition_signal_lift,
)
from lcri_lab.absorption import add_shadow_absorption
from lcri_lab.baseline import (
    baseline_nonlinear_extrapolation_risk,
    baseline_nonlinear_extrapolation_risk_summary,
    baseline_nonlinear_feature_ablation,
    baseline_nonlinear_feature_ablation_summary,
    baseline_nonlinear_stress_surface,
    baseline_nonlinear_stress_surface_summary,
    baseline_regime_basis_comparison,
    baseline_regime_publishability_summary,
    baseline_stress_residual_drift,
    baseline_tail_lift_diagnostics,
)
from lcri_lab.alpha import (
    add_alpha_event_window_regimes,
    add_microstructure_alpha_stack,
    alpha_event_drift_gate,
    alpha_event_regime_summary,
    alpha_event_release_review_packet,
    alpha_event_score_weighted_drift,
    alpha_event_window_diagnostics,
    alpha_event_window_regime_summary,
    alpha_event_window_summary,
)
from lcri_lab.execution import (
    add_execution_adjusted_edge,
    add_passive_fill_event_window_regimes,
    add_passive_fill_probabilities,
    add_queue_position_features,
    add_queue_position_realized_fill_proxy,
    execution_adjusted_edge_component_attribution,
    execution_adjusted_edge_summary,
    execution_adjusted_lcri_event_window_attribution,
    execution_adjusted_lcri_event_window_release_scorecard,
    execution_adjusted_lcri_quantile_diagnostics,
    execution_adjusted_lcri_regime_attribution,
    execution_adjusted_lcri_side_attribution,
    execution_adjusted_lcri_side_release_scorecard,
    execution_publishability_release_gate,
    execution_publishability_review_packet,
    passive_fill_calibration_curve,
    passive_fill_calibration_summary,
    passive_fill_event_lead_lag_profile,
    passive_fill_event_lead_lag_scorecard,
    passive_fill_event_lifecycle_policy_curve,
    passive_fill_event_lifecycle_scorecard,
    passive_fill_event_lifecycle_summary,
    passive_fill_event_policy_stability,
    passive_fill_event_policy_stability_scorecard,
    passive_fill_event_regime_summary,
    passive_fill_event_toxicity_scorecard,
    passive_fill_event_transition_policy_curve,
    passive_fill_event_transition_scorecard,
    passive_fill_event_transition_summary,
    passive_fill_event_window_diagnostics,
    passive_fill_event_window_regime_scorecard,
    passive_fill_event_window_regime_summary,
    passive_fill_event_window_sensitivity,
    passive_fill_event_window_transition_matrix,
    passive_fill_event_window_transition_scorecard,
    passive_fill_proxy_disagreement,
    passive_fill_realization_horizon_sweep,
    passive_fill_threshold_policy_curve,
    queue_position_adverse_selection_policy_frontier,
    queue_position_adverse_selection_policy_summary,
    queue_position_calibration_drift,
    queue_position_calibration_stability,
    queue_position_calibration_stability_summary,
    queue_position_capacity_frontier,
    queue_position_capacity_stability,
    queue_position_edge_decay,
    queue_position_execution_quality_gate,
    queue_position_expected_value_frontier,
    queue_position_expected_value_policy_drift,
    queue_position_expected_value_policy_scorecard,
    queue_position_expected_value_policy_selection,
    queue_position_expected_value_stress_summary,
    queue_position_expected_value_stress_table,
    queue_position_fill_calibration_surface,
    queue_position_fill_monotonicity_scorecard,
    queue_position_fill_surface,
    queue_position_fraction_sweep,
    queue_position_latency_edge_survival,
    queue_position_latency_edge_survival_scorecard,
    queue_position_latency_regime_surface,
    queue_position_latency_release_scorecard,
    queue_position_lcri_tail_adverse_selection_release_scorecard,
    queue_position_lcri_tail_adverse_selection_surface,
    queue_position_lcri_tail_fill_residuals,
    queue_position_unfilled_opportunity_curve,
    queue_position_unfilled_opportunity_scorecard,
    queue_position_path_drawdown_episodes,
    queue_position_path_drawdown_summary,
    queue_position_regime_capacity_concentration,
    queue_position_regime_capacity_frontier,
    queue_position_regime_capacity_stability,
    queue_position_regime_capacity_stability_summary,
    queue_position_regime_fraction_sweep,
)
from lcri_lab.features import add_regime_transition_features
from lcri_lab.ingest import normalize_l2_snapshots
from lcri_lab.labels import add_transaction_cost_labels
from lcri_lab.memory import (
    add_liquidity_memory_half_life,
    add_pressure_memory,
    adverse_selection_phase_shift_summary,
    classify_phase_shift_artifacts,
    hidden_resiliency_asymmetry_summary,
    pressure_memory_decay_summary,
)
from lcri_lab.model import ARTIFACT_VERSION, LCRIModel, ModelConfig
from lcri_lab.plotting import write_figures
from lcri_lab.publishability import add_publishability_gate
from lcri_lab.reporting import (
    artifact_coverage_matrix,
    artifact_coverage_summary,
    build_artifact_manifest,
    collect_artifact_metadata,
    missing_artifacts,
    summarize_artifact_metadata,
    summarize_verification_errors,
    alpha_event_review_verification_summary,
    verify_alpha_event_review_artifacts,
    verify_alpha_event_review_verification_summary,
    verify_artifact_coverage_matrix,
    verify_artifact_manifest,
    verify_artifact_metadata_summary,
    verify_baseline_regime_publishability_summary,
    verify_baseline_stress_residual_drift,
    verify_baseline_tail_lift_diagnostics,
    verify_figure_artifacts,
    verify_generalization_fragility_consistency,
    verify_generalization_fragility_diagnostics,
    verify_generalization_fragility_summary,
    verify_generalization_stability_confidence_consistency,
    verify_generalization_stability_confidence_intervals,
    verify_generalization_stability_confidence_summary,
    verify_hidden_resiliency_asymmetry_summary,
    verify_adverse_selection_phase_shift_summary,
    verify_execution_adjusted_edge_component_attribution,
    verify_execution_adjusted_lcri_event_window_release_scorecard,
    verify_execution_adjusted_lcri_quantile_diagnostics,
    verify_execution_adjusted_lcri_side_attribution,
    verify_execution_publishability_review_artifacts,
    verify_execution_publishability_release_gate,
    verify_queue_position_fill_monotonicity_scorecard,
    verify_queue_position_trade_confirmation_regime_scorecard,
    verify_queue_position_trade_confirmation_release_scorecard,
    verify_queue_position_unfilled_opportunity_scorecard,
    verify_trade_confirmed_passive_fill_latency_summary,
    verify_passive_fill_realization_horizon_sweep,
    verify_queue_position_lcri_tail_fill_residuals,
    verify_queue_position_path_drawdown_artifacts,
    verify_passive_fill_event_lifecycle_policy_curve,
    verify_passive_fill_event_policy_stability,
    verify_passive_fill_event_policy_stability_scorecard,
    verify_passive_fill_event_window_regime_scorecard,
    verify_passive_fill_event_transition_policy_curve,
    verify_passive_fill_threshold_policy_curve,
    verify_phase_shift_artifact_review,
    verify_lcri_fragility_gate_alignment,
    verify_lcri_fragility_gate_scorecard,
    verify_lcri_fracture_reversal_gate,
    verify_lcri_reversal_transition_gate_consistency,
    verify_lcri_ci_confidence_coverage_consistency,
    verify_lcri_ci_confidence_coverage_scorecard,
    verify_lcri_ci_confidence_coverage_summary,
    verify_lcri_ci_gate_contradiction_diagnostics,
    verify_lcri_ci_gate_contradiction_summary,
    verify_lcri_ci_gate_contradiction_consistency,
    verify_generalization_overview,
    verify_lcri_gap_delta_dominant_scopes,
    verify_lcri_gap_delta_consistency,
    verify_lcri_gap_delta_flags,
    verify_lcri_gap_delta_improvements,
    verify_lcri_gap_delta_regressions,
    verify_lcri_gap_delta_scorecard,
    verify_lcri_gap_delta_scope_extremes,
    verify_lcri_gap_delta_scope_summary,
    verify_lcri_gap_delta_summary,
    verify_lcri_contradiction_review_packet,
    verify_lcri_contradiction_review_packet_summary,
    verify_lcri_cross_artifact_evidence_index,
    verify_lcri_cross_artifact_evidence_index_summary,
    verify_lcri_cross_artifact_evidence_index_consistency,
    verify_lcri_evidence_release_checklist,
    verify_lcri_evidence_release_checklist_summary,
    verify_lcri_evidence_release_checklist_consistency,
    verify_lcri_evidence_lineage_map,
    verify_lcri_evidence_lineage_map_summary,
    verify_lcri_evidence_lineage_map_consistency,
    verify_lcri_owner_handoff_packet,
    verify_lcri_owner_handoff_packet_summary,
    verify_lcri_owner_handoff_packet_consistency,
    verify_lcri_owner_handoff_markdown_packet,
    verify_lcri_uncertainty_weighted_review_priority,
    verify_lcri_uncertainty_weighted_review_priority_summary,
    verify_lcri_uncertainty_weighted_review_priority_consistency,
    verify_lcri_generalization_blocker_summary,
    verify_lcri_generalization_critical_contexts,
    verify_lcri_generalization_gate_decision,
    verify_lcri_generalization_gate_decision_consistency,
    verify_lcri_generalization_gap_delta,
    verify_lcri_generalization_gap_leaderboard,
    verify_lcri_generalization_scope_gate_decision_summary,
    verify_lcri_generalization_scope_gate_consistency,
    verify_lcri_generalization_scope_gate_decisions,
    verify_lcri_generalization_scope_risk,
    verify_lcri_generalization_scope_summary,
    verify_lcri_generalization_severity,
    verify_lcri_generalization_severity_by_scope,
    verify_lcri_generalization_severity_consistency,
    verify_lcri_generalization_severity_summary,
    verify_lcri_scope_stability_contradiction_summary,
    verify_lcri_scope_stability_contradictions,
    verify_lcri_scope_stability_contradictions_consistency,
    verify_lcri_worst_generalization_context,
    verify_pressure_memory_decay_summary,
    verify_research_summary_sections,
    write_json,
    write_lcri_owner_handoff_markdown_packet,
    write_research_summary,
)
from lcri_lab.reversal import (
    add_queue_reversal_risk,
    add_reversal_lead_lag_coupling,
    fracture_reversal_release_gate,
    reversal_coupling_regime_stress,
    reversal_stress_concentration_summary,
    reversal_transition_gate_diagnostics,
)
from lcri_lab.simulator import SimulationConfig, simulate_order_books


def main() -> None:
    parser = argparse.ArgumentParser(prog="lcri-lab")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("run-demo", help="run the synthetic LCRI research workflow")
    demo.add_argument("--rows", type=int, default=20_000)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--train-frac", type=float, default=0.70)
    demo.add_argument("--passive-fill-horizon", type=int, default=2)
    demo.add_argument("--output", type=Path, default=Path("reports"))

    normalize = subparsers.add_parser("normalize", help="normalize flat L2 snapshots")
    normalize.add_argument("--input", type=Path, required=True)
    normalize.add_argument("--output", type=Path, required=True)
    normalize.add_argument("--tick-size", type=float, required=True)
    normalize.add_argument("--levels", type=int, default=5)
    normalize.add_argument("--derive-state", action="store_true")

    fit = subparsers.add_parser("fit", help="fit an LCRI model from order book snapshots")
    fit.add_argument("--input", type=Path, required=True)
    fit.add_argument("--model", type=Path, required=True)
    fit.add_argument("--levels", type=int, default=5)
    fit.add_argument("--ridge", type=float, default=1e-3)
    fit.add_argument("--probability-scale", type=float, default=1.0)

    score = subparsers.add_parser("score", help="score order book snapshots with a fitted model")
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--model", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--columns", help="comma-separated output columns; defaults to all columns")

    describe = subparsers.add_parser("describe-model", help="print fitted model artifact metadata")
    describe.add_argument("--model", type=Path, required=True)

    verify = subparsers.add_parser("verify-report", help="verify generated report artifacts")
    verify.add_argument("--report-dir", type=Path, default=Path("reports"))

    args = parser.parse_args()
    if args.command == "run-demo":
        run_demo(
            rows=args.rows,
            seed=args.seed,
            train_frac=args.train_frac,
            output=args.output,
            passive_fill_horizon=args.passive_fill_horizon,
        )
    elif args.command == "normalize":
        normalize_snapshots(
            input_path=args.input,
            output_path=args.output,
            tick_size=args.tick_size,
            levels=args.levels,
            derive_state=args.derive_state,
        )
    elif args.command == "fit":
        fit_model(
            input_path=args.input,
            model_path=args.model,
            levels=args.levels,
            ridge=args.ridge,
            probability_scale=args.probability_scale,
        )
    elif args.command == "score":
        columns = args.columns.split(",") if args.columns else None
        score_model(input_path=args.input, model_path=args.model, output_path=args.output, columns=columns)
    elif args.command == "describe-model":
        describe_model(model_path=args.model)
    elif args.command == "verify-report":
        verify_report(report_dir=args.report_dir)


def run_demo(
    rows: int,
    seed: int,
    output: Path,
    train_frac: float = 0.70,
    passive_fill_horizon: int = 2,
) -> None:
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be between 0 and 1")
    if not isinstance(passive_fill_horizon, int) or isinstance(passive_fill_horizon, bool):
        raise ValueError("passive_fill_horizon must be an integer")
    if passive_fill_horizon < 1:
        raise ValueError("passive_fill_horizon must be at least 1")

    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(parents=True, exist_ok=True)

    simulation_config = SimulationConfig(rows=rows, seed=seed)
    books = simulate_order_books(simulation_config)
    train = books.sample(frac=train_frac, random_state=seed)
    heldout = books.drop(index=train.index)
    model = LCRIModel().fit(train)
    scored = _add_execution_adjusted_stack(
        _add_reversal_pressure_stack(add_regime_transition_features(model.score_frame(books))),
        tick_size=simulation_config.tick_size,
    )
    heldout_scored = _add_execution_adjusted_stack(
        _add_reversal_pressure_stack(add_regime_transition_features(model.score_frame(heldout))),
        tick_size=simulation_config.tick_size,
    )

    metrics = evaluate_signals(scored)
    heldout_metrics = evaluate_signals(heldout_scored)
    generalization_gap = signal_generalization_gap(metrics, heldout_metrics)
    by_regime = regime_metrics(scored)
    heldout_by_regime = regime_metrics(heldout_scored)
    regime_gap = regime_generalization_gap(by_regime, heldout_by_regime)
    by_transition = transition_conditioned_metrics(scored)
    heldout_by_transition = transition_conditioned_metrics(heldout_scored)
    transition_gap = transition_generalization_gap(by_transition, heldout_by_transition)
    fragility_diagnostics = generalization_fragility_diagnostics(
        metrics.assign(rows=len(scored)),
        heldout_metrics.assign(rows=len(heldout_scored)),
        by_regime,
        heldout_by_regime,
        by_transition,
        heldout_by_transition,
    )
    fragility_summary = generalization_fragility_summary(fragility_diagnostics)
    stability_confidence = generalization_stability_confidence_intervals(fragility_diagnostics)
    stability_confidence_summary = generalization_stability_confidence_summary(stability_confidence)
    overview = generalization_overview(generalization_gap, regime_gap, transition_gap)
    gap_leaderboard = generalization_gap_leaderboard(generalization_gap, regime_gap, transition_gap)
    lcri_gap_leaderboard = lcri_generalization_gap_leaderboard(
        generalization_gap,
        regime_gap,
        transition_gap,
    )
    lcri_generalization_scope_summary_table = lcri_generalization_scope_summary(lcri_gap_leaderboard)
    lcri_gap_severity = lcri_generalization_severity(lcri_gap_leaderboard)
    lcri_fragility_alignment = lcri_fragility_gate_alignment(fragility_diagnostics, lcri_gap_severity)
    lcri_fragility_scorecard = lcri_fragility_gate_scorecard(lcri_fragility_alignment)
    lcri_ci_gate_diagnostics = lcri_ci_gate_contradiction_diagnostics(
        lcri_gap_severity,
        stability_confidence,
    )
    lcri_ci_gate_summary = lcri_ci_gate_contradiction_summary(lcri_ci_gate_diagnostics)
    lcri_ci_confidence_scorecard = lcri_ci_confidence_coverage_scorecard(
        stability_confidence,
        lcri_ci_gate_diagnostics,
    )
    lcri_ci_confidence_summary = lcri_ci_confidence_coverage_summary(
        lcri_ci_confidence_scorecard
    )
    lcri_critical_contexts = lcri_generalization_critical_contexts(lcri_gap_severity)
    lcri_gap_severity_by_scope = lcri_generalization_severity_by_scope(lcri_gap_severity)
    lcri_scope_risk = lcri_generalization_scope_risk(lcri_gap_severity_by_scope)
    lcri_scope_gate_decisions = lcri_generalization_scope_gate_decisions(lcri_scope_risk)
    lcri_scope_gate_summary = lcri_scope_gate_decision_summary(lcri_scope_gate_decisions)
    lcri_blocker_summary = lcri_generalization_blocker_summary(lcri_critical_contexts)
    lcri_gap_severity_summary = lcri_generalization_severity_summary(lcri_gap_severity)
    lcri_worst_gap_context = lcri_worst_generalization_context(lcri_gap_leaderboard)
    lcri_gate_decision = lcri_generalization_gate_decision(
        lcri_gap_severity_summary,
        lcri_worst_gap_context,
    )
    lcri_gap_delta = lcri_generalization_gap_delta(generalization_gap, regime_gap, transition_gap)
    lcri_gap_flags = lcri_gap_delta_flags(lcri_gap_delta)
    lcri_gap_improvements = lcri_gap_delta_improvements(lcri_gap_delta)
    lcri_gap_regressions = lcri_gap_delta_regressions(lcri_gap_delta)
    lcri_gap_scorecard = lcri_gap_delta_scorecard(lcri_gap_delta)
    lcri_gap_scope_extremes = lcri_gap_delta_scope_extremes(lcri_gap_delta)
    lcri_gap_delta_scope_summary_table = lcri_gap_delta_scope_summary(lcri_gap_delta)
    lcri_gap_dominant_scopes = lcri_gap_delta_dominant_scopes(lcri_gap_delta_scope_summary_table)
    lcri_gap_summary = lcri_gap_delta_summary(lcri_gap_delta)
    lcri_scope_contradictions = lcri_scope_stability_contradictions(
        lcri_scope_gate_decisions,
        lcri_gap_delta_scope_summary_table,
        lcri_fragility_alignment,
    )
    lcri_scope_contradiction_summary = lcri_scope_stability_contradiction_summary(
        lcri_scope_contradictions
    )
    lcri_review_packet = lcri_contradiction_review_packet(
        lcri_scope_contradictions,
        lcri_gap_severity,
        lcri_gap_delta,
        lcri_fragility_alignment,
    )
    lcri_review_packet_summary = lcri_contradiction_review_packet_summary(lcri_review_packet)
    lcri_uncertainty_priority = lcri_uncertainty_weighted_review_priority(
        lcri_review_packet,
        lcri_ci_confidence_scorecard,
    )
    lcri_uncertainty_priority_summary = lcri_uncertainty_weighted_review_priority_summary(
        lcri_uncertainty_priority
    )
    lcri_evidence_index = lcri_cross_artifact_evidence_index(
        lcri_gap_severity_by_scope,
        lcri_scope_gate_decisions,
        lcri_gap_delta_scope_summary_table,
        lcri_scope_contradictions,
        lcri_ci_confidence_scorecard,
        lcri_uncertainty_priority,
    )
    lcri_evidence_index_summary = lcri_cross_artifact_evidence_index_summary(
        lcri_evidence_index
    )
    lcri_release_checklist = lcri_evidence_release_checklist(lcri_evidence_index)
    lcri_release_checklist_summary = lcri_evidence_release_checklist_summary(
        lcri_release_checklist
    )
    lcri_owner_handoff = lcri_owner_handoff_packet(lcri_evidence_index, lcri_release_checklist)
    lcri_owner_handoff_summary = lcri_owner_handoff_packet_summary(lcri_owner_handoff)
    lcri_lineage_map = lcri_evidence_lineage_map(
        lcri_evidence_index,
        lcri_release_checklist,
        lcri_owner_handoff,
    )
    lcri_lineage_map_summary = lcri_evidence_lineage_map_summary(lcri_lineage_map)
    transition_lift = transition_signal_lift(scored)
    heldout_transition_lift = transition_signal_lift(heldout_scored)
    memory_decay_summary = pressure_memory_decay_summary(scored)
    heldout_memory_decay_summary = pressure_memory_decay_summary(heldout_scored)
    resiliency_asymmetry = hidden_resiliency_asymmetry_summary(scored)
    heldout_resiliency_asymmetry = hidden_resiliency_asymmetry_summary(heldout_scored)
    adverse_phase_shift = adverse_selection_phase_shift_summary(scored, return_col="future_return_ticks")
    heldout_adverse_phase_shift = adverse_selection_phase_shift_summary(
        heldout_scored, return_col="future_return_ticks"
    )
    phase_shift_artifacts = classify_phase_shift_artifacts(adverse_phase_shift)
    heldout_phase_shift_artifacts = classify_phase_shift_artifacts(heldout_adverse_phase_shift)
    lcri_monotonicity = signal_quantile_monotonicity(scored, "lcri")
    heldout_lcri_monotonicity = signal_quantile_monotonicity(heldout_scored, "lcri")
    lcri_monotonicity_summary = signal_quantile_monotonicity_summary(lcri_monotonicity)
    heldout_lcri_monotonicity_summary = signal_quantile_monotonicity_summary(
        heldout_lcri_monotonicity
    )
    lcri_calibration = calibration_curve(scored, "lcri")
    heldout_lcri_calibration = calibration_curve(heldout_scored, "lcri")
    lcri_calibration_summary = calibration_error_summary(lcri_calibration)
    heldout_lcri_calibration_summary = calibration_error_summary(heldout_lcri_calibration)
    lcri_calibration_gate = calibration_gate_decision(lcri_calibration_summary)
    heldout_lcri_calibration_gate = calibration_gate_decision(heldout_lcri_calibration_summary)
    lcri_calibration_fracture_pressure = calibration_monotonicity_pressure(
        lcri_calibration,
        lcri_monotonicity,
    )
    heldout_lcri_calibration_fracture_pressure = calibration_monotonicity_pressure(
        heldout_lcri_calibration,
        heldout_lcri_monotonicity,
    )
    lcri_calibration_fracture_pressure_summary = calibration_monotonicity_pressure_summary(
        lcri_calibration_fracture_pressure
    )
    heldout_lcri_calibration_fracture_pressure_summary = calibration_monotonicity_pressure_summary(
        heldout_lcri_calibration_fracture_pressure
    )
    lcri_calibration_fracture_gate = calibration_fracture_gate_decision(
        lcri_calibration_fracture_pressure_summary,
        heldout_lcri_calibration_fracture_pressure_summary,
    )
    reversal_stress = reversal_coupling_regime_stress(scored)
    heldout_reversal_stress = reversal_coupling_regime_stress(heldout_scored)
    reversal_stress_summary = reversal_stress_concentration_summary(reversal_stress)
    heldout_reversal_stress_summary = reversal_stress_concentration_summary(heldout_reversal_stress)
    fracture_reversal_gate = fracture_reversal_release_gate(
        reversal_stress_summary,
        lcri_calibration_fracture_gate,
        heldout_reversal_summary=heldout_reversal_stress_summary,
    )
    reversal_transition_gate = reversal_transition_gate_diagnostics(scored, fracture_reversal_gate)
    heldout_reversal_transition_gate = reversal_transition_gate_diagnostics(
        heldout_scored,
        fracture_reversal_gate,
    )
    transition_robustness = transition_robustness_summary(scored)
    heldout_transition_robustness = transition_robustness_summary(heldout_scored)
    scored = add_alpha_event_window_regimes(scored, return_col="future_return_ticks")
    heldout_scored = add_alpha_event_window_regimes(heldout_scored, return_col="future_return_ticks")
    alpha_window_regime_summary = alpha_event_window_regime_summary(
        scored,
        return_col="future_return_ticks",
    )
    alpha_events = alpha_event_window_diagnostics(
        scored,
        return_col="future_return_ticks",
        regime_col="pressure_memory_decay_state",
    )
    alpha_event_regimes = alpha_event_regime_summary(alpha_events)
    alpha_event_summary = alpha_event_window_summary(alpha_events)
    alpha_weighted_drift = alpha_event_score_weighted_drift(alpha_events)
    alpha_drift_gate = alpha_event_drift_gate(alpha_event_summary)
    alpha_release_packet = alpha_event_release_review_packet(
        alpha_drift_gate,
        alpha_weighted_drift,
        alpha_event_regimes,
    )
    alpha_release_packet_for_summary = alpha_release_packet.copy()
    try:
        alpha_release_packet_for_summary["top_weighted_event_index"] = pd.to_numeric(
            alpha_release_packet_for_summary["top_weighted_event_index"],
        )
    except (TypeError, ValueError):
        pass
    execution_summary = execution_adjusted_edge_summary(scored)
    heldout_execution_summary = execution_adjusted_edge_summary(heldout_scored)
    execution_edge_component_attribution = execution_adjusted_edge_component_attribution(scored)
    heldout_execution_edge_component_attribution = execution_adjusted_edge_component_attribution(
        heldout_scored
    )
    execution_publishability_packet = execution_publishability_review_packet(scored)
    heldout_execution_publishability_packet = execution_publishability_review_packet(heldout_scored)
    scored = add_passive_fill_event_window_regimes(
        scored,
        threshold=0.75,
        window=3,
        group_cols="pressure_memory_decay_state",
    )
    heldout_scored = add_passive_fill_event_window_regimes(
        heldout_scored,
        threshold=0.75,
        window=3,
        group_cols="pressure_memory_decay_state",
    )
    passive_fill_events = passive_fill_event_window_diagnostics(
        scored,
        threshold=0.75,
        window=3,
        regime_col="pressure_memory_decay_state",
    )
    passive_fill_event_lead_lag = passive_fill_event_lead_lag_profile(
        scored,
        threshold=0.75,
        window=3,
        regime_col="pressure_memory_decay_state",
    )
    passive_fill_event_lead_lag_warnings = passive_fill_event_lead_lag_scorecard(
        passive_fill_event_lead_lag
    )
    passive_fill_event_regimes = passive_fill_event_regime_summary(passive_fill_events)
    passive_fill_event_window_regimes = passive_fill_event_window_regime_summary(scored)
    passive_fill_event_window_regime_gate = passive_fill_event_window_regime_scorecard(
        passive_fill_event_window_regimes
    )
    passive_fill_event_window_transitions = passive_fill_event_window_transition_matrix(scored)
    passive_fill_event_window_transition_gate = passive_fill_event_window_transition_scorecard(
        passive_fill_event_window_transitions
    )
    passive_fill_event_window_surface = passive_fill_event_window_sensitivity(
        scored,
        thresholds=(0.60, 0.70, 0.75, 0.80, 0.90),
        windows=(1, 3, 5),
        regime_col="pressure_memory_decay_state",
    )
    passive_fill_event_transitions = passive_fill_event_transition_summary(passive_fill_events)
    passive_fill_event_lifecycle = passive_fill_event_lifecycle_summary(passive_fill_events)
    passive_fill_event_lifecycle_policy = passive_fill_event_lifecycle_policy_curve(passive_fill_events)
    passive_fill_event_transition_policy = passive_fill_event_transition_policy_curve(passive_fill_events)
    passive_fill_event_toxicity = passive_fill_event_toxicity_scorecard(passive_fill_event_regimes)
    passive_fill_event_lifecycle_toxicity = passive_fill_event_lifecycle_scorecard(
        passive_fill_event_lifecycle
    )
    passive_fill_event_transition_toxicity = passive_fill_event_transition_scorecard(
        passive_fill_event_transitions
    )
    heldout_passive_fill_events = passive_fill_event_window_diagnostics(
        heldout_scored,
        threshold=0.75,
        window=3,
        regime_col="pressure_memory_decay_state",
    )
    heldout_passive_fill_event_lead_lag = passive_fill_event_lead_lag_profile(
        heldout_scored,
        threshold=0.75,
        window=3,
        regime_col="pressure_memory_decay_state",
    )
    heldout_passive_fill_event_lead_lag_warnings = passive_fill_event_lead_lag_scorecard(
        heldout_passive_fill_event_lead_lag
    )
    heldout_passive_fill_event_regimes = passive_fill_event_regime_summary(
        heldout_passive_fill_events
    )
    heldout_passive_fill_event_window_regimes = passive_fill_event_window_regime_summary(
        heldout_scored
    )
    heldout_passive_fill_event_window_regime_gate = passive_fill_event_window_regime_scorecard(
        heldout_passive_fill_event_window_regimes
    )
    heldout_passive_fill_event_window_transitions = passive_fill_event_window_transition_matrix(
        heldout_scored
    )
    heldout_passive_fill_event_window_transition_gate = passive_fill_event_window_transition_scorecard(
        heldout_passive_fill_event_window_transitions
    )
    heldout_passive_fill_event_window_surface = passive_fill_event_window_sensitivity(
        heldout_scored,
        thresholds=(0.60, 0.70, 0.75, 0.80, 0.90),
        windows=(1, 3, 5),
        regime_col="pressure_memory_decay_state",
    )
    heldout_passive_fill_event_transitions = passive_fill_event_transition_summary(
        heldout_passive_fill_events
    )
    heldout_passive_fill_event_lifecycle = passive_fill_event_lifecycle_summary(
        heldout_passive_fill_events
    )
    heldout_passive_fill_event_lifecycle_policy = passive_fill_event_lifecycle_policy_curve(
        heldout_passive_fill_events
    )
    heldout_passive_fill_event_transition_policy = passive_fill_event_transition_policy_curve(
        heldout_passive_fill_events
    )
    heldout_passive_fill_event_toxicity = passive_fill_event_toxicity_scorecard(
        heldout_passive_fill_event_regimes
    )
    heldout_passive_fill_event_lifecycle_toxicity = passive_fill_event_lifecycle_scorecard(
        heldout_passive_fill_event_lifecycle
    )
    heldout_passive_fill_event_transition_toxicity = passive_fill_event_transition_scorecard(
        heldout_passive_fill_event_transitions
    )
    passive_fill_event_lifecycle_policy_stability = passive_fill_event_policy_stability(
        passive_fill_event_lifecycle_policy,
        heldout_passive_fill_event_lifecycle_policy,
    )
    passive_fill_event_transition_policy_stability = passive_fill_event_policy_stability(
        passive_fill_event_transition_policy,
        heldout_passive_fill_event_transition_policy,
        path_col="regime_transition",
    )
    passive_fill_event_lifecycle_policy_stability_scorecard = (
        passive_fill_event_policy_stability_scorecard(passive_fill_event_lifecycle_policy_stability)
    )
    passive_fill_event_transition_policy_stability_scorecard = (
        passive_fill_event_policy_stability_scorecard(
            passive_fill_event_transition_policy_stability,
            path_col="regime_transition",
        )
    )
    passive_fill_labeled = _add_passive_fill_realization_proxy(
        scored,
        horizon=passive_fill_horizon,
    )
    heldout_passive_fill_labeled = _add_passive_fill_realization_proxy(
        heldout_scored,
        horizon=passive_fill_horizon,
    )
    passive_fill_event_labeled = add_queue_position_realized_fill_proxy(
        passive_fill_labeled,
        horizon=1,
        bid_realized_col="bid_event_fill",
        ask_realized_col="ask_event_fill",
    )
    heldout_passive_fill_event_labeled = add_queue_position_realized_fill_proxy(
        heldout_passive_fill_labeled,
        horizon=1,
        bid_realized_col="bid_event_fill",
        ask_realized_col="ask_event_fill",
    )
    passive_fill_proxy_audit = passive_fill_proxy_disagreement(
        passive_fill_event_labeled,
        snapshot_cols=("bid_realized_fill", "ask_realized_fill"),
        event_cols=("bid_event_fill", "ask_event_fill"),
    )
    heldout_passive_fill_proxy_audit = passive_fill_proxy_disagreement(
        heldout_passive_fill_event_labeled,
        snapshot_cols=("bid_realized_fill", "ask_realized_fill"),
        event_cols=("bid_event_fill", "ask_event_fill"),
    )
    passive_fill_calibration = passive_fill_calibration_curve(
        passive_fill_labeled,
        regime_col="pressure_memory_decay_state",
    )
    heldout_passive_fill_calibration = passive_fill_calibration_curve(
        heldout_passive_fill_labeled,
        regime_col="pressure_memory_decay_state",
    )
    passive_fill_calibration_stats = passive_fill_calibration_summary(passive_fill_calibration)
    passive_fill_calibration_stats["realization_horizon_snapshots"] = passive_fill_horizon
    heldout_passive_fill_calibration_stats = passive_fill_calibration_summary(
        heldout_passive_fill_calibration
    )
    heldout_passive_fill_calibration_stats["realization_horizon_snapshots"] = passive_fill_horizon
    passive_fill_horizons = tuple(sorted({1, passive_fill_horizon, 5}))
    passive_fill_horizon_sweep = passive_fill_realization_horizon_sweep(
        scored,
        horizons=passive_fill_horizons,
        regime_col="pressure_memory_decay_state",
    )
    heldout_passive_fill_horizon_sweep = passive_fill_realization_horizon_sweep(
        heldout_scored,
        horizons=passive_fill_horizons,
        regime_col="pressure_memory_decay_state",
    )
    passive_fill_threshold_policy = passive_fill_threshold_policy_curve(passive_fill_labeled)
    heldout_passive_fill_threshold_policy = passive_fill_threshold_policy_curve(
        heldout_passive_fill_labeled
    )
    queue_fill_surface = queue_position_fill_surface(
        passive_fill_labeled,
        regime_col="pressure_memory_decay_state",
    )
    heldout_queue_fill_surface = queue_position_fill_surface(
        heldout_passive_fill_labeled,
        regime_col="pressure_memory_decay_state",
    )
    queue_fill_calibration_surface = queue_position_fill_calibration_surface(
        passive_fill_labeled,
        regime_col="pressure_memory_decay_state",
    )
    heldout_queue_fill_calibration_surface = queue_position_fill_calibration_surface(
        heldout_passive_fill_labeled,
        regime_col="pressure_memory_decay_state",
    )
    queue_fill_monotonicity_scorecard = queue_position_fill_monotonicity_scorecard(
        queue_fill_calibration_surface,
        regime_col="pressure_memory_decay_state",
    )
    heldout_queue_fill_monotonicity_scorecard = queue_position_fill_monotonicity_scorecard(
        heldout_queue_fill_calibration_surface,
        regime_col="pressure_memory_decay_state",
    )
    queue_latency_regime_surface = queue_position_latency_regime_surface(passive_fill_labeled)
    heldout_queue_latency_regime_surface = queue_position_latency_regime_surface(
        heldout_passive_fill_labeled
    )
    queue_latency_edge_survival = queue_position_latency_edge_survival(passive_fill_labeled)
    heldout_queue_latency_edge_survival = queue_position_latency_edge_survival(
        heldout_passive_fill_labeled
    )
    queue_latency_edge_survival_scorecard = queue_position_latency_edge_survival_scorecard(
        queue_latency_edge_survival
    )
    heldout_queue_latency_edge_survival_scorecard = queue_position_latency_edge_survival_scorecard(
        heldout_queue_latency_edge_survival
    )
    queue_latency_release_scorecard = queue_position_latency_release_scorecard(
        queue_latency_regime_surface
    )
    heldout_queue_latency_release_scorecard = queue_position_latency_release_scorecard(
        heldout_queue_latency_regime_surface
    )
    queue_fraction_sweep = queue_position_fraction_sweep(scored)
    heldout_queue_fraction_sweep = queue_position_fraction_sweep(heldout_scored)
    queue_regime_fraction_sweep = queue_position_regime_fraction_sweep(
        scored,
        regime_col="pressure_memory_decay_state",
    )
    heldout_queue_regime_fraction_sweep = queue_position_regime_fraction_sweep(
        heldout_scored,
        regime_col="pressure_memory_decay_state",
    )
    queue_capacity_frontier = queue_position_capacity_frontier(
        queue_fraction_sweep,
        min_edge_ticks=0.0,
        min_tradable_share=0.50,
    )
    heldout_queue_capacity_frontier = queue_position_capacity_frontier(
        heldout_queue_fraction_sweep,
        min_edge_ticks=0.0,
        min_tradable_share=0.50,
    )
    queue_regime_capacity_frontier = queue_position_regime_capacity_frontier(
        queue_regime_fraction_sweep,
        regime_col="pressure_memory_decay_state",
        min_edge_ticks=0.0,
        min_tradable_share=0.50,
    )
    heldout_queue_regime_capacity_frontier = queue_position_regime_capacity_frontier(
        heldout_queue_regime_fraction_sweep,
        regime_col="pressure_memory_decay_state",
        min_edge_ticks=0.0,
        min_tradable_share=0.50,
    )
    queue_regime_capacity_concentration = queue_position_regime_capacity_concentration(
        queue_regime_capacity_frontier,
        regime_col="pressure_memory_decay_state",
    )
    heldout_queue_regime_capacity_concentration = queue_position_regime_capacity_concentration(
        heldout_queue_regime_capacity_frontier,
        regime_col="pressure_memory_decay_state",
    )
    queue_capacity_stability = queue_position_capacity_stability(
        queue_capacity_frontier,
        heldout_queue_capacity_frontier,
    )
    queue_regime_capacity_stability = queue_position_regime_capacity_stability(
        queue_regime_capacity_frontier,
        heldout_queue_regime_capacity_frontier,
        regime_col="pressure_memory_decay_state",
    )
    queue_regime_capacity_stability_summary = queue_position_regime_capacity_stability_summary(
        queue_regime_capacity_stability,
        regime_col="pressure_memory_decay_state",
    )
    queue_edge_decay = queue_position_edge_decay(queue_fill_surface)
    heldout_queue_edge_decay = queue_position_edge_decay(heldout_queue_fill_surface)
    queue_calibration_drift = queue_position_calibration_drift(
        queue_fill_calibration_surface,
        regime_col="pressure_memory_decay_state",
    )
    heldout_queue_calibration_drift = queue_position_calibration_drift(
        heldout_queue_fill_calibration_surface,
        regime_col="pressure_memory_decay_state",
    )
    queue_calibration_stability = queue_position_calibration_stability(
        queue_fill_calibration_surface,
        heldout_queue_fill_calibration_surface,
        regime_col="pressure_memory_decay_state",
    )
    queue_calibration_stability_summary = queue_position_calibration_stability_summary(
        queue_calibration_stability,
        regime_col="pressure_memory_decay_state",
    )
    queue_adverse_policy_frontier = queue_position_adverse_selection_policy_frontier(
        passive_fill_labeled
    )
    heldout_queue_adverse_policy_frontier = queue_position_adverse_selection_policy_frontier(
        heldout_passive_fill_labeled
    )
    queue_adverse_policy_summary = queue_position_adverse_selection_policy_summary(
        queue_adverse_policy_frontier,
        min_trade_share=0.05,
        min_realized_fill_rate=0.60,
        min_mean_realized_edge_ticks=0.0,
        max_mean_adverse_fill_probability=0.40,
        max_toxicity_filtered_share=0.75,
    )
    heldout_queue_adverse_policy_summary = queue_position_adverse_selection_policy_summary(
        heldout_queue_adverse_policy_frontier,
        min_trade_share=0.05,
        min_realized_fill_rate=0.60,
        min_mean_realized_edge_ticks=0.0,
        max_mean_adverse_fill_probability=0.40,
        max_toxicity_filtered_share=0.75,
    )
    queue_expected_value_frontier = queue_position_expected_value_frontier(
        scored,
        regime_col="pressure_memory_decay_state",
    )
    heldout_queue_expected_value_frontier = queue_position_expected_value_frontier(
        heldout_scored,
        regime_col="pressure_memory_decay_state",
    )
    queue_expected_value_policy_selection = queue_position_expected_value_policy_selection(
        queue_expected_value_frontier,
        min_candidate_share=0.05,
    )
    heldout_queue_expected_value_policy_selection = queue_position_expected_value_policy_selection(
        heldout_queue_expected_value_frontier,
        min_candidate_share=0.05,
    )
    queue_expected_value_policy_drift = queue_position_expected_value_policy_drift(
        queue_expected_value_policy_selection,
        heldout_queue_expected_value_policy_selection,
        max_threshold_drift=0.10,
        max_ev_decay_ratio=0.50,
        min_holdout_candidate_share=0.05,
    )
    queue_expected_value_policy_scorecard = queue_position_expected_value_policy_scorecard(
        queue_expected_value_policy_selection
    )
    heldout_queue_expected_value_policy_scorecard = queue_position_expected_value_policy_scorecard(
        heldout_queue_expected_value_policy_selection
    )
    queue_expected_value_stress_table = queue_position_expected_value_stress_table(
        queue_expected_value_policy_selection,
        min_candidate_share=0.05,
        min_stressed_expected_value_ticks=0.0,
    )
    heldout_queue_expected_value_stress_table = queue_position_expected_value_stress_table(
        heldout_queue_expected_value_policy_selection,
        min_candidate_share=0.05,
        min_stressed_expected_value_ticks=0.0,
    )
    queue_expected_value_stress_summary = queue_position_expected_value_stress_summary(
        queue_expected_value_stress_table,
        min_candidate_weighted_ev_ticks=0.0,
    )
    heldout_queue_expected_value_stress_summary = queue_position_expected_value_stress_summary(
        heldout_queue_expected_value_stress_table,
        min_candidate_weighted_ev_ticks=0.0,
    )
    queue_execution_quality_gate = queue_position_execution_quality_gate(
        queue_fill_surface,
        queue_edge_decay,
        drift=queue_calibration_drift,
    )
    heldout_queue_execution_quality_gate = queue_position_execution_quality_gate(
        heldout_queue_fill_surface,
        heldout_queue_edge_decay,
        drift=heldout_queue_calibration_drift,
    )
    execution_lcri_side_attribution = execution_adjusted_lcri_side_attribution(scored)
    heldout_execution_lcri_side_attribution = execution_adjusted_lcri_side_attribution(heldout_scored)
    execution_lcri_side_release_scorecard = execution_adjusted_lcri_side_release_scorecard(
        execution_lcri_side_attribution
    )
    heldout_execution_lcri_side_release_scorecard = execution_adjusted_lcri_side_release_scorecard(
        heldout_execution_lcri_side_attribution
    )
    execution_lcri_regime_attribution = execution_adjusted_lcri_regime_attribution(scored)
    heldout_execution_lcri_regime_attribution = execution_adjusted_lcri_regime_attribution(heldout_scored)
    execution_lcri_event_window_attribution = execution_adjusted_lcri_event_window_attribution(scored)
    heldout_execution_lcri_event_window_attribution = execution_adjusted_lcri_event_window_attribution(
        heldout_scored
    )
    execution_lcri_event_window_release_scorecard = (
        execution_adjusted_lcri_event_window_release_scorecard(
            execution_lcri_event_window_attribution
        )
    )
    heldout_execution_lcri_event_window_release_scorecard = (
        execution_adjusted_lcri_event_window_release_scorecard(
            heldout_execution_lcri_event_window_attribution
        )
    )
    execution_publishability_gate = execution_publishability_release_gate(
        execution_publishability_packet,
        quality_gate=queue_execution_quality_gate,
        capacity_stability=queue_capacity_stability,
        regime_capacity_stability=queue_regime_capacity_stability_summary,
        lcri_regime_attribution=execution_lcri_regime_attribution,
        lcri_event_window_scorecard=execution_lcri_event_window_release_scorecard,
    )
    heldout_execution_publishability_gate = execution_publishability_release_gate(
        heldout_execution_publishability_packet,
        quality_gate=heldout_queue_execution_quality_gate,
        capacity_stability=queue_capacity_stability,
        regime_capacity_stability=queue_regime_capacity_stability_summary,
        lcri_regime_attribution=heldout_execution_lcri_regime_attribution,
        lcri_event_window_scorecard=heldout_execution_lcri_event_window_release_scorecard,
    )
    execution_lcri_quantile_diagnostics = execution_adjusted_lcri_quantile_diagnostics(scored)
    heldout_execution_lcri_quantile_diagnostics = execution_adjusted_lcri_quantile_diagnostics(
        heldout_scored
    )
    queue_lcri_tail_fill_residuals = queue_position_lcri_tail_fill_residuals(passive_fill_labeled)
    heldout_queue_lcri_tail_fill_residuals = queue_position_lcri_tail_fill_residuals(
        heldout_passive_fill_labeled
    )
    queue_unfilled_opportunity_curve = queue_position_unfilled_opportunity_curve(
        passive_fill_labeled,
        group_cols="passive_fill_event_window_regime",
    )
    heldout_queue_unfilled_opportunity_curve = queue_position_unfilled_opportunity_curve(
        heldout_passive_fill_labeled,
        group_cols="passive_fill_event_window_regime",
    )
    queue_unfilled_opportunity_scorecard = queue_position_unfilled_opportunity_scorecard(
        queue_unfilled_opportunity_curve,
        min_tail_bin=4,
        min_tail_rows=20,
    )
    heldout_queue_unfilled_opportunity_scorecard = queue_position_unfilled_opportunity_scorecard(
        heldout_queue_unfilled_opportunity_curve,
        min_tail_bin=4,
        min_tail_rows=20,
    )
    queue_lcri_tail_adverse_selection_surface = queue_position_lcri_tail_adverse_selection_surface(
        passive_fill_labeled
    )
    heldout_queue_lcri_tail_adverse_selection_surface = (
        queue_position_lcri_tail_adverse_selection_surface(heldout_passive_fill_labeled)
    )
    queue_lcri_tail_adverse_selection_release_scorecard = (
        queue_position_lcri_tail_adverse_selection_release_scorecard(
            queue_lcri_tail_adverse_selection_surface
        )
    )
    heldout_queue_lcri_tail_adverse_selection_release_scorecard = (
        queue_position_lcri_tail_adverse_selection_release_scorecard(
            heldout_queue_lcri_tail_adverse_selection_surface
        )
    )
    queue_path_drawdown_episodes = queue_position_path_drawdown_episodes(
        scored,
        group_cols="pressure_memory_decay_state",
        event_window_col="passive_fill_event_window_regime",
    )
    heldout_queue_path_drawdown_episodes = queue_position_path_drawdown_episodes(
        heldout_scored,
        group_cols="pressure_memory_decay_state",
        event_window_col="passive_fill_event_window_regime",
    )
    queue_path_drawdown_summary = queue_position_path_drawdown_summary(queue_path_drawdown_episodes)
    heldout_queue_path_drawdown_summary = queue_position_path_drawdown_summary(
        heldout_queue_path_drawdown_episodes
    )
    baseline_regime_basis = baseline_regime_basis_comparison(
        scored,
        train_window=max(200, rows // 2),
        test_window=max(100, rows // 4),
        step=max(100, rows // 4),
    )
    baseline_regime_publishability = baseline_regime_publishability_summary(
        baseline_regime_basis,
        min_regime_lift=0.0,
    )
    baseline_tail_lift = baseline_tail_lift_diagnostics(
        scored,
        feature="liquidity_void_x_volatility",
        train_fraction=train_frac,
        min_tail_lift=0.0,
    )
    baseline_residual_drift = baseline_stress_residual_drift(
        scored,
        feature="liquidity_void_x_volatility",
        train_fraction=train_frac,
    )
    baseline_nonlinear_surface = baseline_nonlinear_stress_surface(
        scored,
        train_fraction=train_frac,
        stress_cols=("spread_ticks", "volatility"),
        bins=3,
    )
    baseline_nonlinear_surface_summary = baseline_nonlinear_stress_surface_summary(
        baseline_nonlinear_surface
    )
    baseline_nonlinear_ablation = baseline_nonlinear_feature_ablation(
        scored,
        train_fraction=train_frac,
    )
    baseline_nonlinear_ablation_summary = baseline_nonlinear_feature_ablation_summary(
        baseline_nonlinear_ablation,
        min_material_terms=1,
        min_total_positive_drag_share=0.10,
    )
    baseline_nonlinear_extrapolation = baseline_nonlinear_extrapolation_risk(
        scored.loc[train.index],
        heldout_scored,
        train_quantile=0.90,
        max_safe_out_of_support_share=0.15,
    )
    baseline_nonlinear_extrapolation_summary = baseline_nonlinear_extrapolation_risk_summary(
        baseline_nonlinear_extrapolation,
        max_safe_risky_terms=1,
        max_safe_out_of_support_share=0.15,
    )

    artifact_paths = [
        "lcri-model.json",
        "sample_snapshots.csv",
        "metrics.csv",
        "heldout_metrics.csv",
        "generalization_gap.csv",
        "baseline_regime_basis_comparison.csv",
        "baseline_regime_publishability_summary.json",
        "baseline_tail_lift_diagnostics.csv",
        "baseline_stress_residual_drift.csv",
        "baseline_nonlinear_stress_surface.csv",
        "baseline_nonlinear_stress_surface_summary.json",
        "baseline_nonlinear_feature_ablation.csv",
        "baseline_nonlinear_feature_ablation_summary.json",
        "baseline_nonlinear_extrapolation_risk.csv",
        "baseline_nonlinear_extrapolation_risk_summary.json",
        "regime_metrics.csv",
        "heldout_regime_metrics.csv",
        "regime_generalization_gap.csv",
        "transition_metrics.csv",
        "heldout_transition_metrics.csv",
        "transition_generalization_gap.csv",
        "generalization_fragility_diagnostics.csv",
        "generalization_fragility_summary.json",
        "generalization_stability_confidence_intervals.csv",
        "generalization_stability_confidence_summary.json",
        "generalization_overview.json",
        "generalization_gap_leaderboard.csv",
        "lcri_generalization_gap_leaderboard.csv",
        "lcri_generalization_scope_summary.csv",
        "lcri_generalization_severity.csv",
        "lcri_fragility_gate_alignment.csv",
        "lcri_fragility_gate_scorecard.json",
        "lcri_ci_gate_contradiction_diagnostics.csv",
        "lcri_ci_gate_contradiction_summary.json",
        "lcri_ci_confidence_coverage_scorecard.csv",
        "lcri_ci_confidence_coverage_summary.json",
        "lcri_generalization_severity_by_scope.csv",
        "lcri_generalization_scope_risk.csv",
        "lcri_generalization_scope_gate_decisions.csv",
        "lcri_generalization_scope_gate_decision_summary.json",
        "lcri_generalization_critical_contexts.csv",
        "lcri_generalization_blocker_summary.json",
        "lcri_generalization_severity_summary.json",
        "lcri_worst_generalization_context.json",
        "lcri_generalization_gate_decision.json",
        "lcri_generalization_gap_delta.csv",
        "lcri_gap_delta_dominant_scopes.json",
        "lcri_gap_delta_flags.csv",
        "lcri_gap_delta_improvements.csv",
        "lcri_gap_delta_regressions.csv",
        "lcri_gap_delta_scorecard.json",
        "lcri_gap_delta_scope_extremes.csv",
        "lcri_gap_delta_scope_summary.csv",
        "lcri_gap_delta_summary.json",
        "lcri_scope_stability_contradictions.csv",
        "lcri_scope_stability_contradiction_summary.json",
        "lcri_contradiction_review_packet.csv",
        "lcri_contradiction_review_packet_summary.json",
        "lcri_uncertainty_weighted_review_priority.csv",
        "lcri_uncertainty_weighted_review_priority_summary.json",
        "lcri_cross_artifact_evidence_index.csv",
        "lcri_cross_artifact_evidence_index_summary.json",
        "lcri_evidence_release_checklist.csv",
        "lcri_evidence_release_checklist_summary.json",
        "lcri_owner_handoff_packet.csv",
        "lcri_owner_handoff_packet_summary.json",
        "lcri_owner_handoff_packet.md",
        "lcri_evidence_lineage_map.csv",
        "lcri_evidence_lineage_map_summary.json",
        "transition_lift.csv",
        "heldout_transition_lift.csv",
        "pressure_memory_decay_summary.csv",
        "heldout_pressure_memory_decay_summary.csv",
        "hidden_resiliency_asymmetry_summary.json",
        "heldout_hidden_resiliency_asymmetry_summary.json",
        "adverse_selection_phase_shift_summary.csv",
        "heldout_adverse_selection_phase_shift_summary.csv",
        "phase_shift_artifact_review.csv",
        "heldout_phase_shift_artifact_review.csv",
        "lcri_signal_monotonicity.csv",
        "heldout_lcri_signal_monotonicity.csv",
        "lcri_signal_monotonicity_summary.json",
        "heldout_lcri_signal_monotonicity_summary.json",
        "lcri_calibration_curve.csv",
        "heldout_lcri_calibration_curve.csv",
        "lcri_calibration_summary.json",
        "heldout_lcri_calibration_summary.json",
        "lcri_calibration_gate.json",
        "heldout_lcri_calibration_gate.json",
        "lcri_calibration_fracture_pressure.csv",
        "heldout_lcri_calibration_fracture_pressure.csv",
        "lcri_calibration_fracture_pressure_summary.json",
        "heldout_lcri_calibration_fracture_pressure_summary.json",
        "lcri_calibration_fracture_gate.json",
        "lcri_reversal_stress_concentration.csv",
        "heldout_lcri_reversal_stress_concentration.csv",
        "lcri_reversal_stress_concentration_summary.json",
        "heldout_lcri_reversal_stress_concentration_summary.json",
        "lcri_fracture_reversal_gate.json",
        "lcri_reversal_transition_gate.csv",
        "heldout_lcri_reversal_transition_gate.csv",
        "transition_robustness.json",
        "heldout_transition_robustness.json",
        "alpha_event_windows.csv",
        "alpha_event_regime_summary.csv",
        "alpha_event_window_regime_summary.csv",
        "alpha_event_window_summary.json",
        "alpha_event_score_weighted_drift.json",
        "alpha_event_drift_gate.json",
        "alpha_event_release_review_packet.csv",
        "alpha_event_review_verification_summary.json",
        "execution_adjusted_edge_summary.json",
        "heldout_execution_adjusted_edge_summary.json",
        "execution_adjusted_edge_component_attribution.csv",
        "heldout_execution_adjusted_edge_component_attribution.csv",
        "execution_adjusted_lcri_side_attribution.csv",
        "heldout_execution_adjusted_lcri_side_attribution.csv",
        "execution_adjusted_lcri_side_release_scorecard.json",
        "heldout_execution_adjusted_lcri_side_release_scorecard.json",
        "execution_adjusted_lcri_regime_attribution.csv",
        "heldout_execution_adjusted_lcri_regime_attribution.csv",
        "execution_adjusted_lcri_quantile_diagnostics.csv",
        "heldout_execution_adjusted_lcri_quantile_diagnostics.csv",
        "queue_position_lcri_tail_fill_residuals.csv",
        "heldout_queue_position_lcri_tail_fill_residuals.csv",
        "queue_position_unfilled_opportunity_curve.csv",
        "heldout_queue_position_unfilled_opportunity_curve.csv",
        "queue_position_unfilled_opportunity_scorecard.json",
        "heldout_queue_position_unfilled_opportunity_scorecard.json",
        "queue_position_lcri_tail_adverse_selection_surface.csv",
        "heldout_queue_position_lcri_tail_adverse_selection_surface.csv",
        "queue_position_lcri_tail_adverse_selection_release_scorecard.json",
        "heldout_queue_position_lcri_tail_adverse_selection_release_scorecard.json",
        "execution_adjusted_lcri_event_window_attribution.csv",
        "heldout_execution_adjusted_lcri_event_window_attribution.csv",
        "execution_adjusted_lcri_event_window_release_scorecard.json",
        "heldout_execution_adjusted_lcri_event_window_release_scorecard.json",
        "execution_publishability_review_packet.csv",
        "heldout_execution_publishability_review_packet.csv",
        "execution_publishability_release_gate.json",
        "heldout_execution_publishability_release_gate.json",
        "passive_fill_event_windows.csv",
        "passive_fill_event_lead_lag_profile.csv",
        "passive_fill_event_lead_lag_scorecard.csv",
        "passive_fill_event_regime_summary.csv",
        "passive_fill_event_window_regime_summary.csv",
        "passive_fill_event_window_regime_scorecard.json",
        "passive_fill_event_window_transition_matrix.csv",
        "passive_fill_event_window_transition_scorecard.json",
        "passive_fill_event_window_sensitivity.csv",
        "passive_fill_event_transition_summary.csv",
        "passive_fill_event_lifecycle_summary.csv",
        "passive_fill_event_lifecycle_policy_curve.csv",
        "passive_fill_event_transition_policy_curve.csv",
        "passive_fill_event_lifecycle_policy_stability.csv",
        "passive_fill_event_transition_policy_stability.csv",
        "passive_fill_event_lifecycle_policy_stability_scorecard.json",
        "passive_fill_event_transition_policy_stability_scorecard.json",
        "passive_fill_event_toxicity_scorecard.json",
        "passive_fill_event_lifecycle_toxicity_scorecard.json",
        "passive_fill_event_transition_toxicity_scorecard.json",
        "heldout_passive_fill_event_windows.csv",
        "heldout_passive_fill_event_lead_lag_profile.csv",
        "heldout_passive_fill_event_lead_lag_scorecard.csv",
        "heldout_passive_fill_event_regime_summary.csv",
        "heldout_passive_fill_event_window_regime_summary.csv",
        "heldout_passive_fill_event_window_regime_scorecard.json",
        "heldout_passive_fill_event_window_transition_matrix.csv",
        "heldout_passive_fill_event_window_transition_scorecard.json",
        "heldout_passive_fill_event_window_sensitivity.csv",
        "heldout_passive_fill_event_transition_summary.csv",
        "heldout_passive_fill_event_lifecycle_summary.csv",
        "heldout_passive_fill_event_lifecycle_policy_curve.csv",
        "heldout_passive_fill_event_transition_policy_curve.csv",
        "heldout_passive_fill_event_toxicity_scorecard.json",
        "heldout_passive_fill_event_lifecycle_toxicity_scorecard.json",
        "heldout_passive_fill_event_transition_toxicity_scorecard.json",
        "passive_fill_calibration_curve.csv",
        "heldout_passive_fill_calibration_curve.csv",
        "passive_fill_calibration_summary.json",
        "heldout_passive_fill_calibration_summary.json",
        "passive_fill_realization_horizon_sweep.csv",
        "heldout_passive_fill_realization_horizon_sweep.csv",
        "passive_fill_proxy_disagreement.csv",
        "heldout_passive_fill_proxy_disagreement.csv",
        "passive_fill_threshold_policy_curve.csv",
        "heldout_passive_fill_threshold_policy_curve.csv",
        "queue_position_fill_surface.csv",
        "heldout_queue_position_fill_surface.csv",
        "queue_position_fill_calibration_surface.csv",
        "heldout_queue_position_fill_calibration_surface.csv",
        "queue_position_fill_monotonicity_scorecard.csv",
        "heldout_queue_position_fill_monotonicity_scorecard.csv",
        "queue_position_latency_regime_surface.csv",
        "heldout_queue_position_latency_regime_surface.csv",
        "queue_position_latency_edge_survival.csv",
        "heldout_queue_position_latency_edge_survival.csv",
        "queue_position_latency_edge_survival_scorecard.json",
        "heldout_queue_position_latency_edge_survival_scorecard.json",
        "queue_position_latency_release_scorecard.json",
        "heldout_queue_position_latency_release_scorecard.json",
        "queue_position_fraction_sweep.csv",
        "heldout_queue_position_fraction_sweep.csv",
        "queue_position_regime_fraction_sweep.csv",
        "heldout_queue_position_regime_fraction_sweep.csv",
        "queue_position_capacity_frontier.json",
        "heldout_queue_position_capacity_frontier.json",
        "queue_position_regime_capacity_frontier.csv",
        "heldout_queue_position_regime_capacity_frontier.csv",
        "queue_position_regime_capacity_concentration.json",
        "heldout_queue_position_regime_capacity_concentration.json",
        "queue_position_capacity_stability.json",
        "queue_position_regime_capacity_stability.csv",
        "queue_position_regime_capacity_stability_summary.json",
        "queue_position_edge_decay.csv",
        "heldout_queue_position_edge_decay.csv",
        "queue_position_calibration_drift.csv",
        "heldout_queue_position_calibration_drift.csv",
        "queue_position_calibration_stability.csv",
        "queue_position_calibration_stability_summary.json",
        "queue_position_adverse_selection_policy_frontier.csv",
        "heldout_queue_position_adverse_selection_policy_frontier.csv",
        "queue_position_adverse_selection_policy_summary.json",
        "heldout_queue_position_adverse_selection_policy_summary.json",
        "queue_position_expected_value_frontier.csv",
        "heldout_queue_position_expected_value_frontier.csv",
        "queue_position_expected_value_policy_selection.csv",
        "heldout_queue_position_expected_value_policy_selection.csv",
        "queue_position_expected_value_policy_drift.csv",
        "queue_position_expected_value_policy_scorecard.csv",
        "heldout_queue_position_expected_value_policy_scorecard.csv",
        "queue_position_expected_value_stress_table.csv",
        "heldout_queue_position_expected_value_stress_table.csv",
        "queue_position_expected_value_stress_summary.json",
        "heldout_queue_position_expected_value_stress_summary.json",
        "queue_position_path_drawdown_episodes.csv",
        "heldout_queue_position_path_drawdown_episodes.csv",
        "queue_position_path_drawdown_summary.json",
        "heldout_queue_position_path_drawdown_summary.json",
        "execution_adjusted_sample.csv",
        "research_summary.md",
        "artifact_coverage_matrix.csv",
        "artifact_coverage_summary.json",
        "artifact_metadata_summary.json",
        "figures/raw_vs_lcri_scatter.png",
        "figures/regime_signal_quality.png",
        "figures/transition_signal_quality.png",
        "figures/heldout_transition_signal_quality.png",
        "figures/calibration_curve.png",
        "figures/heldout_calibration_curve.png",
        "figures/generalization_gap.png",
        "figures/regime_generalization_gap.png",
        "figures/transition_generalization_gap.png",
        "figures/generalization_fragility_diagnostics.png",
        "figures/generalization_stability_confidence_intervals.png",
        "figures/lcri_generalization_gap_delta.png",
        "figures/lcri_generalization_severity_by_scope.png",
        "figures/lcri_ci_gate_contradiction_diagnostics.png",
        "figures/lcri_ci_confidence_coverage_scorecard.png",
        "figures/lcri_gap_delta_scope_summary.png",
        "figures/lcri_contradiction_review_packet.png",
        "figures/lcri_uncertainty_weighted_review_priority.png",
        "figures/lcri_cross_artifact_evidence_index.png",
        "figures/lcri_evidence_release_checklist.png",
        "figures/lcri_owner_handoff_packet.png",
        "figures/lcri_evidence_lineage_map.png",
        "figures/lcri_calibration_fracture_pressure.png",
        "figures/lcri_reversal_transition_gate.png",
    ]

    model.save(output / "lcri-model.json")
    scored.head(500).to_csv(output / "sample_snapshots.csv", index=False)
    metrics.to_csv(output / "metrics.csv", index=False)
    heldout_metrics.to_csv(output / "heldout_metrics.csv", index=False)
    generalization_gap.to_csv(output / "generalization_gap.csv", index=False)
    baseline_regime_basis.to_csv(output / "baseline_regime_basis_comparison.csv", index=False)
    write_json(
        output / "baseline_regime_publishability_summary.json",
        baseline_regime_publishability,
    )
    baseline_tail_lift.to_csv(output / "baseline_tail_lift_diagnostics.csv", index=False)
    baseline_residual_drift.to_csv(output / "baseline_stress_residual_drift.csv", index=False)
    baseline_nonlinear_surface.to_csv(output / "baseline_nonlinear_stress_surface.csv", index=False)
    write_json(
        output / "baseline_nonlinear_stress_surface_summary.json",
        baseline_nonlinear_surface_summary,
    )
    baseline_nonlinear_ablation.to_csv(
        output / "baseline_nonlinear_feature_ablation.csv", index=False
    )
    write_json(
        output / "baseline_nonlinear_feature_ablation_summary.json",
        baseline_nonlinear_ablation_summary,
    )
    baseline_nonlinear_extrapolation.to_csv(
        output / "baseline_nonlinear_extrapolation_risk.csv", index=False
    )
    write_json(
        output / "baseline_nonlinear_extrapolation_risk_summary.json",
        baseline_nonlinear_extrapolation_summary,
    )
    by_regime.to_csv(output / "regime_metrics.csv", index=False)
    heldout_by_regime.to_csv(output / "heldout_regime_metrics.csv", index=False)
    regime_gap.to_csv(output / "regime_generalization_gap.csv", index=False)
    by_transition.to_csv(output / "transition_metrics.csv", index=False)
    heldout_by_transition.to_csv(output / "heldout_transition_metrics.csv", index=False)
    transition_gap.to_csv(output / "transition_generalization_gap.csv", index=False)
    fragility_diagnostics.to_csv(output / "generalization_fragility_diagnostics.csv", index=False)
    write_json(output / "generalization_fragility_summary.json", fragility_summary)
    stability_confidence.to_csv(output / "generalization_stability_confidence_intervals.csv", index=False)
    write_json(
        output / "generalization_stability_confidence_summary.json",
        stability_confidence_summary,
    )
    write_json(output / "generalization_overview.json", overview)
    gap_leaderboard.to_csv(output / "generalization_gap_leaderboard.csv", index=False)
    lcri_gap_leaderboard.to_csv(output / "lcri_generalization_gap_leaderboard.csv", index=False)
    lcri_generalization_scope_summary_table.to_csv(output / "lcri_generalization_scope_summary.csv", index=False)
    lcri_gap_severity.to_csv(output / "lcri_generalization_severity.csv", index=False)
    lcri_fragility_alignment.to_csv(output / "lcri_fragility_gate_alignment.csv", index=False)
    write_json(output / "lcri_fragility_gate_scorecard.json", lcri_fragility_scorecard)
    lcri_ci_gate_diagnostics.to_csv(
        output / "lcri_ci_gate_contradiction_diagnostics.csv", index=False
    )
    write_json(output / "lcri_ci_gate_contradiction_summary.json", lcri_ci_gate_summary)
    lcri_ci_confidence_scorecard.to_csv(
        output / "lcri_ci_confidence_coverage_scorecard.csv", index=False
    )
    write_json(
        output / "lcri_ci_confidence_coverage_summary.json",
        lcri_ci_confidence_summary,
    )
    lcri_gap_severity_by_scope.to_csv(output / "lcri_generalization_severity_by_scope.csv", index=False)
    lcri_scope_risk.to_csv(output / "lcri_generalization_scope_risk.csv", index=False)
    lcri_scope_gate_decisions.to_csv(output / "lcri_generalization_scope_gate_decisions.csv", index=False)
    write_json(output / "lcri_generalization_scope_gate_decision_summary.json", lcri_scope_gate_summary)
    lcri_critical_contexts.to_csv(output / "lcri_generalization_critical_contexts.csv", index=False)
    write_json(output / "lcri_generalization_blocker_summary.json", lcri_blocker_summary)
    write_json(output / "lcri_generalization_severity_summary.json", lcri_gap_severity_summary)
    write_json(output / "lcri_worst_generalization_context.json", lcri_worst_gap_context)
    write_json(output / "lcri_generalization_gate_decision.json", lcri_gate_decision)
    lcri_gap_delta.to_csv(output / "lcri_generalization_gap_delta.csv", index=False)
    write_json(output / "lcri_gap_delta_dominant_scopes.json", lcri_gap_dominant_scopes)
    lcri_gap_flags.to_csv(output / "lcri_gap_delta_flags.csv", index=False)
    lcri_gap_improvements.to_csv(output / "lcri_gap_delta_improvements.csv", index=False)
    lcri_gap_regressions.to_csv(output / "lcri_gap_delta_regressions.csv", index=False)
    write_json(output / "lcri_gap_delta_scorecard.json", lcri_gap_scorecard)
    lcri_gap_scope_extremes.to_csv(output / "lcri_gap_delta_scope_extremes.csv", index=False)
    lcri_gap_delta_scope_summary_table.to_csv(output / "lcri_gap_delta_scope_summary.csv", index=False)
    write_json(output / "lcri_gap_delta_summary.json", lcri_gap_summary)
    lcri_scope_contradictions.to_csv(output / "lcri_scope_stability_contradictions.csv", index=False)
    write_json(
        output / "lcri_scope_stability_contradiction_summary.json",
        lcri_scope_contradiction_summary,
    )
    lcri_review_packet.to_csv(output / "lcri_contradiction_review_packet.csv", index=False)
    write_json(output / "lcri_contradiction_review_packet_summary.json", lcri_review_packet_summary)
    lcri_uncertainty_priority.to_csv(
        output / "lcri_uncertainty_weighted_review_priority.csv", index=False
    )
    write_json(
        output / "lcri_uncertainty_weighted_review_priority_summary.json",
        lcri_uncertainty_priority_summary,
    )
    lcri_evidence_index.to_csv(output / "lcri_cross_artifact_evidence_index.csv", index=False)
    write_json(
        output / "lcri_cross_artifact_evidence_index_summary.json",
        lcri_evidence_index_summary,
    )
    lcri_release_checklist.to_csv(output / "lcri_evidence_release_checklist.csv", index=False)
    write_json(
        output / "lcri_evidence_release_checklist_summary.json",
        lcri_release_checklist_summary,
    )
    lcri_owner_handoff.to_csv(output / "lcri_owner_handoff_packet.csv", index=False)
    write_json(output / "lcri_owner_handoff_packet_summary.json", lcri_owner_handoff_summary)
    write_lcri_owner_handoff_markdown_packet(
        output / "lcri_owner_handoff_packet.md",
        packet=lcri_owner_handoff,
        summary=lcri_owner_handoff_summary,
    )
    lcri_lineage_map.to_csv(output / "lcri_evidence_lineage_map.csv", index=False)
    write_json(output / "lcri_evidence_lineage_map_summary.json", lcri_lineage_map_summary)
    transition_lift.to_csv(output / "transition_lift.csv", index=False)
    heldout_transition_lift.to_csv(output / "heldout_transition_lift.csv", index=False)
    memory_decay_summary.to_csv(output / "pressure_memory_decay_summary.csv", index=False)
    heldout_memory_decay_summary.to_csv(
        output / "heldout_pressure_memory_decay_summary.csv", index=False
    )
    write_json(output / "hidden_resiliency_asymmetry_summary.json", resiliency_asymmetry)
    write_json(
        output / "heldout_hidden_resiliency_asymmetry_summary.json",
        heldout_resiliency_asymmetry,
    )
    adverse_phase_shift.to_csv(output / "adverse_selection_phase_shift_summary.csv", index=False)
    heldout_adverse_phase_shift.to_csv(
        output / "heldout_adverse_selection_phase_shift_summary.csv", index=False
    )
    phase_shift_artifacts.to_csv(output / "phase_shift_artifact_review.csv", index=False)
    heldout_phase_shift_artifacts.to_csv(
        output / "heldout_phase_shift_artifact_review.csv", index=False
    )
    lcri_monotonicity.to_csv(output / "lcri_signal_monotonicity.csv", index=False)
    heldout_lcri_monotonicity.to_csv(
        output / "heldout_lcri_signal_monotonicity.csv", index=False
    )
    write_json(output / "lcri_signal_monotonicity_summary.json", lcri_monotonicity_summary)
    write_json(
        output / "heldout_lcri_signal_monotonicity_summary.json",
        heldout_lcri_monotonicity_summary,
    )
    lcri_calibration.to_csv(output / "lcri_calibration_curve.csv", index=False)
    heldout_lcri_calibration.to_csv(output / "heldout_lcri_calibration_curve.csv", index=False)
    write_json(output / "lcri_calibration_summary.json", lcri_calibration_summary)
    write_json(output / "heldout_lcri_calibration_summary.json", heldout_lcri_calibration_summary)
    write_json(output / "lcri_calibration_gate.json", lcri_calibration_gate)
    write_json(output / "heldout_lcri_calibration_gate.json", heldout_lcri_calibration_gate)
    lcri_calibration_fracture_pressure.to_csv(
        output / "lcri_calibration_fracture_pressure.csv", index=False
    )
    heldout_lcri_calibration_fracture_pressure.to_csv(
        output / "heldout_lcri_calibration_fracture_pressure.csv", index=False
    )
    write_json(
        output / "lcri_calibration_fracture_pressure_summary.json",
        lcri_calibration_fracture_pressure_summary,
    )
    write_json(
        output / "heldout_lcri_calibration_fracture_pressure_summary.json",
        heldout_lcri_calibration_fracture_pressure_summary,
    )
    write_json(output / "lcri_calibration_fracture_gate.json", lcri_calibration_fracture_gate)
    reversal_stress.to_csv(output / "lcri_reversal_stress_concentration.csv", index=False)
    heldout_reversal_stress.to_csv(
        output / "heldout_lcri_reversal_stress_concentration.csv", index=False
    )
    write_json(output / "lcri_reversal_stress_concentration_summary.json", reversal_stress_summary)
    write_json(
        output / "heldout_lcri_reversal_stress_concentration_summary.json",
        heldout_reversal_stress_summary,
    )
    write_json(output / "lcri_fracture_reversal_gate.json", fracture_reversal_gate)
    reversal_transition_gate.to_csv(output / "lcri_reversal_transition_gate.csv", index=False)
    heldout_reversal_transition_gate.to_csv(
        output / "heldout_lcri_reversal_transition_gate.csv",
        index=False,
    )
    write_json(output / "transition_robustness.json", transition_robustness)
    write_json(output / "heldout_transition_robustness.json", heldout_transition_robustness)
    alpha_events.to_csv(output / "alpha_event_windows.csv", index=False)
    alpha_event_regimes.to_csv(output / "alpha_event_regime_summary.csv", index=False)
    alpha_window_regime_summary.to_csv(
        output / "alpha_event_window_regime_summary.csv", index=False
    )
    write_json(output / "alpha_event_window_summary.json", alpha_event_summary)
    write_json(output / "alpha_event_score_weighted_drift.json", alpha_weighted_drift)
    write_json(output / "alpha_event_drift_gate.json", alpha_drift_gate)
    alpha_release_packet.to_csv(output / "alpha_event_release_review_packet.csv", index=False)
    alpha_event_verification_summary = alpha_event_review_verification_summary(output)
    write_json(output / "alpha_event_review_verification_summary.json", alpha_event_verification_summary)
    write_json(output / "execution_adjusted_edge_summary.json", execution_summary)
    write_json(output / "heldout_execution_adjusted_edge_summary.json", heldout_execution_summary)
    execution_edge_component_attribution.to_csv(
        output / "execution_adjusted_edge_component_attribution.csv", index=False
    )
    heldout_execution_edge_component_attribution.to_csv(
        output / "heldout_execution_adjusted_edge_component_attribution.csv", index=False
    )
    execution_publishability_packet.to_csv(
        output / "execution_publishability_review_packet.csv", index=False
    )
    heldout_execution_publishability_packet.to_csv(
        output / "heldout_execution_publishability_review_packet.csv", index=False
    )
    write_json(output / "execution_publishability_release_gate.json", execution_publishability_gate)
    write_json(
        output / "heldout_execution_publishability_release_gate.json",
        heldout_execution_publishability_gate,
    )
    passive_fill_events.to_csv(output / "passive_fill_event_windows.csv", index=False)
    passive_fill_event_lead_lag.to_csv(
        output / "passive_fill_event_lead_lag_profile.csv", index=False
    )
    passive_fill_event_lead_lag_warnings.to_csv(
        output / "passive_fill_event_lead_lag_scorecard.csv", index=False
    )
    passive_fill_event_regimes.to_csv(
        output / "passive_fill_event_regime_summary.csv", index=False
    )
    passive_fill_event_window_regimes.to_csv(
        output / "passive_fill_event_window_regime_summary.csv", index=False
    )
    write_json(
        output / "passive_fill_event_window_regime_scorecard.json",
        passive_fill_event_window_regime_gate,
    )
    passive_fill_event_window_transitions.to_csv(
        output / "passive_fill_event_window_transition_matrix.csv", index=False
    )
    write_json(
        output / "passive_fill_event_window_transition_scorecard.json",
        passive_fill_event_window_transition_gate,
    )
    passive_fill_event_window_surface.to_csv(
        output / "passive_fill_event_window_sensitivity.csv", index=False
    )
    passive_fill_event_transitions.to_csv(
        output / "passive_fill_event_transition_summary.csv", index=False
    )
    passive_fill_event_lifecycle.to_csv(
        output / "passive_fill_event_lifecycle_summary.csv", index=False
    )
    passive_fill_event_lifecycle_policy.to_csv(
        output / "passive_fill_event_lifecycle_policy_curve.csv", index=False
    )
    passive_fill_event_transition_policy.to_csv(
        output / "passive_fill_event_transition_policy_curve.csv", index=False
    )
    passive_fill_event_lifecycle_policy_stability.to_csv(
        output / "passive_fill_event_lifecycle_policy_stability.csv", index=False
    )
    passive_fill_event_transition_policy_stability.to_csv(
        output / "passive_fill_event_transition_policy_stability.csv", index=False
    )
    write_json(
        output / "passive_fill_event_lifecycle_policy_stability_scorecard.json",
        passive_fill_event_lifecycle_policy_stability_scorecard,
    )
    write_json(
        output / "passive_fill_event_transition_policy_stability_scorecard.json",
        passive_fill_event_transition_policy_stability_scorecard,
    )
    write_json(output / "passive_fill_event_toxicity_scorecard.json", passive_fill_event_toxicity)
    write_json(
        output / "passive_fill_event_lifecycle_toxicity_scorecard.json",
        passive_fill_event_lifecycle_toxicity,
    )
    write_json(
        output / "passive_fill_event_transition_toxicity_scorecard.json",
        passive_fill_event_transition_toxicity,
    )
    heldout_passive_fill_events.to_csv(
        output / "heldout_passive_fill_event_windows.csv", index=False
    )
    heldout_passive_fill_event_lead_lag.to_csv(
        output / "heldout_passive_fill_event_lead_lag_profile.csv", index=False
    )
    heldout_passive_fill_event_lead_lag_warnings.to_csv(
        output / "heldout_passive_fill_event_lead_lag_scorecard.csv", index=False
    )
    heldout_passive_fill_event_regimes.to_csv(
        output / "heldout_passive_fill_event_regime_summary.csv", index=False
    )
    heldout_passive_fill_event_window_regimes.to_csv(
        output / "heldout_passive_fill_event_window_regime_summary.csv", index=False
    )
    write_json(
        output / "heldout_passive_fill_event_window_regime_scorecard.json",
        heldout_passive_fill_event_window_regime_gate,
    )
    heldout_passive_fill_event_window_transitions.to_csv(
        output / "heldout_passive_fill_event_window_transition_matrix.csv", index=False
    )
    write_json(
        output / "heldout_passive_fill_event_window_transition_scorecard.json",
        heldout_passive_fill_event_window_transition_gate,
    )
    heldout_passive_fill_event_window_surface.to_csv(
        output / "heldout_passive_fill_event_window_sensitivity.csv", index=False
    )
    heldout_passive_fill_event_transitions.to_csv(
        output / "heldout_passive_fill_event_transition_summary.csv", index=False
    )
    heldout_passive_fill_event_lifecycle.to_csv(
        output / "heldout_passive_fill_event_lifecycle_summary.csv", index=False
    )
    heldout_passive_fill_event_lifecycle_policy.to_csv(
        output / "heldout_passive_fill_event_lifecycle_policy_curve.csv", index=False
    )
    heldout_passive_fill_event_transition_policy.to_csv(
        output / "heldout_passive_fill_event_transition_policy_curve.csv", index=False
    )
    write_json(
        output / "heldout_passive_fill_event_toxicity_scorecard.json",
        heldout_passive_fill_event_toxicity,
    )
    write_json(
        output / "heldout_passive_fill_event_lifecycle_toxicity_scorecard.json",
        heldout_passive_fill_event_lifecycle_toxicity,
    )
    write_json(
        output / "heldout_passive_fill_event_transition_toxicity_scorecard.json",
        heldout_passive_fill_event_transition_toxicity,
    )
    passive_fill_calibration.to_csv(output / "passive_fill_calibration_curve.csv", index=False)
    heldout_passive_fill_calibration.to_csv(
        output / "heldout_passive_fill_calibration_curve.csv", index=False
    )
    write_json(output / "passive_fill_calibration_summary.json", passive_fill_calibration_stats)
    write_json(
        output / "heldout_passive_fill_calibration_summary.json",
        heldout_passive_fill_calibration_stats,
    )
    passive_fill_horizon_sweep.to_csv(
        output / "passive_fill_realization_horizon_sweep.csv", index=False
    )
    heldout_passive_fill_horizon_sweep.to_csv(
        output / "heldout_passive_fill_realization_horizon_sweep.csv", index=False
    )
    passive_fill_proxy_audit.to_csv(output / "passive_fill_proxy_disagreement.csv", index=False)
    heldout_passive_fill_proxy_audit.to_csv(
        output / "heldout_passive_fill_proxy_disagreement.csv", index=False
    )
    passive_fill_threshold_policy.to_csv(
        output / "passive_fill_threshold_policy_curve.csv", index=False
    )
    heldout_passive_fill_threshold_policy.to_csv(
        output / "heldout_passive_fill_threshold_policy_curve.csv", index=False
    )
    queue_fill_surface.to_csv(output / "queue_position_fill_surface.csv", index=False)
    heldout_queue_fill_surface.to_csv(
        output / "heldout_queue_position_fill_surface.csv", index=False
    )
    queue_fill_calibration_surface.to_csv(
        output / "queue_position_fill_calibration_surface.csv", index=False
    )
    heldout_queue_fill_calibration_surface.to_csv(
        output / "heldout_queue_position_fill_calibration_surface.csv", index=False
    )
    queue_fill_monotonicity_scorecard.to_csv(
        output / "queue_position_fill_monotonicity_scorecard.csv", index=False
    )
    heldout_queue_fill_monotonicity_scorecard.to_csv(
        output / "heldout_queue_position_fill_monotonicity_scorecard.csv", index=False
    )
    queue_latency_regime_surface.to_csv(
        output / "queue_position_latency_regime_surface.csv", index=False
    )
    heldout_queue_latency_regime_surface.to_csv(
        output / "heldout_queue_position_latency_regime_surface.csv", index=False
    )
    queue_latency_edge_survival.to_csv(
        output / "queue_position_latency_edge_survival.csv", index=False
    )
    heldout_queue_latency_edge_survival.to_csv(
        output / "heldout_queue_position_latency_edge_survival.csv", index=False
    )
    write_json(
        output / "queue_position_latency_edge_survival_scorecard.json",
        queue_latency_edge_survival_scorecard,
    )
    write_json(
        output / "heldout_queue_position_latency_edge_survival_scorecard.json",
        heldout_queue_latency_edge_survival_scorecard,
    )
    write_json(
        output / "queue_position_latency_release_scorecard.json",
        queue_latency_release_scorecard,
    )
    write_json(
        output / "heldout_queue_position_latency_release_scorecard.json",
        heldout_queue_latency_release_scorecard,
    )
    queue_fraction_sweep.to_csv(output / "queue_position_fraction_sweep.csv", index=False)
    heldout_queue_fraction_sweep.to_csv(
        output / "heldout_queue_position_fraction_sweep.csv", index=False
    )
    queue_regime_fraction_sweep.to_csv(
        output / "queue_position_regime_fraction_sweep.csv", index=False
    )
    heldout_queue_regime_fraction_sweep.to_csv(
        output / "heldout_queue_position_regime_fraction_sweep.csv", index=False
    )
    write_json(output / "queue_position_capacity_frontier.json", queue_capacity_frontier)
    write_json(
        output / "heldout_queue_position_capacity_frontier.json", heldout_queue_capacity_frontier
    )
    queue_regime_capacity_frontier.to_csv(
        output / "queue_position_regime_capacity_frontier.csv", index=False
    )
    heldout_queue_regime_capacity_frontier.to_csv(
        output / "heldout_queue_position_regime_capacity_frontier.csv", index=False
    )
    write_json(
        output / "queue_position_regime_capacity_concentration.json",
        queue_regime_capacity_concentration,
    )
    write_json(
        output / "heldout_queue_position_regime_capacity_concentration.json",
        heldout_queue_regime_capacity_concentration,
    )
    write_json(output / "queue_position_capacity_stability.json", queue_capacity_stability)
    queue_regime_capacity_stability.to_csv(
        output / "queue_position_regime_capacity_stability.csv", index=False
    )
    write_json(
        output / "queue_position_regime_capacity_stability_summary.json",
        queue_regime_capacity_stability_summary,
    )
    queue_edge_decay.to_csv(output / "queue_position_edge_decay.csv", index=False)
    heldout_queue_edge_decay.to_csv(output / "heldout_queue_position_edge_decay.csv", index=False)
    queue_calibration_drift.to_csv(output / "queue_position_calibration_drift.csv", index=False)
    heldout_queue_calibration_drift.to_csv(
        output / "heldout_queue_position_calibration_drift.csv", index=False
    )
    queue_calibration_stability.to_csv(
        output / "queue_position_calibration_stability.csv", index=False
    )
    write_json(
        output / "queue_position_calibration_stability_summary.json",
        queue_calibration_stability_summary,
    )
    queue_adverse_policy_frontier.to_csv(
        output / "queue_position_adverse_selection_policy_frontier.csv", index=False
    )
    heldout_queue_adverse_policy_frontier.to_csv(
        output / "heldout_queue_position_adverse_selection_policy_frontier.csv", index=False
    )
    write_json(
        output / "queue_position_adverse_selection_policy_summary.json",
        queue_adverse_policy_summary,
    )
    write_json(
        output / "heldout_queue_position_adverse_selection_policy_summary.json",
        heldout_queue_adverse_policy_summary,
    )
    queue_expected_value_frontier.to_csv(
        output / "queue_position_expected_value_frontier.csv", index=False
    )
    heldout_queue_expected_value_frontier.to_csv(
        output / "heldout_queue_position_expected_value_frontier.csv", index=False
    )
    queue_expected_value_policy_selection.to_csv(
        output / "queue_position_expected_value_policy_selection.csv", index=False
    )
    heldout_queue_expected_value_policy_selection.to_csv(
        output / "heldout_queue_position_expected_value_policy_selection.csv", index=False
    )
    queue_expected_value_policy_drift.to_csv(
        output / "queue_position_expected_value_policy_drift.csv", index=False
    )
    queue_expected_value_policy_scorecard.to_csv(
        output / "queue_position_expected_value_policy_scorecard.csv", index=False
    )
    heldout_queue_expected_value_policy_scorecard.to_csv(
        output / "heldout_queue_position_expected_value_policy_scorecard.csv", index=False
    )
    queue_expected_value_stress_table.to_csv(
        output / "queue_position_expected_value_stress_table.csv", index=False
    )
    heldout_queue_expected_value_stress_table.to_csv(
        output / "heldout_queue_position_expected_value_stress_table.csv", index=False
    )
    write_json(
        output / "queue_position_expected_value_stress_summary.json",
        queue_expected_value_stress_summary,
    )
    write_json(
        output / "heldout_queue_position_expected_value_stress_summary.json",
        heldout_queue_expected_value_stress_summary,
    )
    execution_lcri_side_attribution.to_csv(
        output / "execution_adjusted_lcri_side_attribution.csv", index=False
    )
    heldout_execution_lcri_side_attribution.to_csv(
        output / "heldout_execution_adjusted_lcri_side_attribution.csv", index=False
    )
    write_json(
        output / "execution_adjusted_lcri_side_release_scorecard.json",
        execution_lcri_side_release_scorecard,
    )
    write_json(
        output / "heldout_execution_adjusted_lcri_side_release_scorecard.json",
        heldout_execution_lcri_side_release_scorecard,
    )
    execution_lcri_regime_attribution.to_csv(
        output / "execution_adjusted_lcri_regime_attribution.csv", index=False
    )
    heldout_execution_lcri_regime_attribution.to_csv(
        output / "heldout_execution_adjusted_lcri_regime_attribution.csv", index=False
    )
    execution_lcri_quantile_diagnostics.to_csv(
        output / "execution_adjusted_lcri_quantile_diagnostics.csv", index=False
    )
    heldout_execution_lcri_quantile_diagnostics.to_csv(
        output / "heldout_execution_adjusted_lcri_quantile_diagnostics.csv", index=False
    )
    queue_lcri_tail_fill_residuals.to_csv(
        output / "queue_position_lcri_tail_fill_residuals.csv", index=False
    )
    heldout_queue_lcri_tail_fill_residuals.to_csv(
        output / "heldout_queue_position_lcri_tail_fill_residuals.csv", index=False
    )
    queue_unfilled_opportunity_curve.to_csv(
        output / "queue_position_unfilled_opportunity_curve.csv", index=False
    )
    heldout_queue_unfilled_opportunity_curve.to_csv(
        output / "heldout_queue_position_unfilled_opportunity_curve.csv", index=False
    )
    write_json(
        output / "queue_position_unfilled_opportunity_scorecard.json",
        queue_unfilled_opportunity_scorecard,
    )
    write_json(
        output / "heldout_queue_position_unfilled_opportunity_scorecard.json",
        heldout_queue_unfilled_opportunity_scorecard,
    )
    queue_lcri_tail_adverse_selection_surface.to_csv(
        output / "queue_position_lcri_tail_adverse_selection_surface.csv", index=False
    )
    heldout_queue_lcri_tail_adverse_selection_surface.to_csv(
        output / "heldout_queue_position_lcri_tail_adverse_selection_surface.csv", index=False
    )
    write_json(
        output / "queue_position_lcri_tail_adverse_selection_release_scorecard.json",
        queue_lcri_tail_adverse_selection_release_scorecard,
    )
    write_json(
        output / "heldout_queue_position_lcri_tail_adverse_selection_release_scorecard.json",
        heldout_queue_lcri_tail_adverse_selection_release_scorecard,
    )
    queue_path_drawdown_episodes.to_csv(
        output / "queue_position_path_drawdown_episodes.csv", index=False
    )
    heldout_queue_path_drawdown_episodes.to_csv(
        output / "heldout_queue_position_path_drawdown_episodes.csv", index=False
    )
    write_json(output / "queue_position_path_drawdown_summary.json", queue_path_drawdown_summary)
    write_json(
        output / "heldout_queue_position_path_drawdown_summary.json",
        heldout_queue_path_drawdown_summary,
    )
    execution_lcri_event_window_attribution.to_csv(
        output / "execution_adjusted_lcri_event_window_attribution.csv", index=False
    )
    heldout_execution_lcri_event_window_attribution.to_csv(
        output / "heldout_execution_adjusted_lcri_event_window_attribution.csv", index=False
    )
    write_json(
        output / "execution_adjusted_lcri_event_window_release_scorecard.json",
        execution_lcri_event_window_release_scorecard,
    )
    write_json(
        output / "heldout_execution_adjusted_lcri_event_window_release_scorecard.json",
        heldout_execution_lcri_event_window_release_scorecard,
    )
    scored[
        [
            "lcri",
            "lcri_probability",
            "publishable_side",
            "best_execution_side",
            "execution_adjusted_edge_ticks",
            "long_fill_adjusted_edge_ticks",
            "short_fill_adjusted_edge_ticks",
            "bid_fill_probability",
            "ask_fill_probability",
            "bid_adverse_fill_probability",
            "ask_adverse_fill_probability",
        ]
    ].head(500).to_csv(output / "execution_adjusted_sample.csv", index=False)
    write_figures(
        scored,
        by_regime,
        output / "figures",
        transition_table=by_transition,
        heldout_transition_table=heldout_by_transition,
        heldout_frame=heldout_scored,
        generalization_gap=generalization_gap,
        regime_generalization_gap=regime_gap,
        transition_generalization_gap=transition_gap,
        generalization_fragility_diagnostics=fragility_diagnostics,
        generalization_stability_confidence_intervals=stability_confidence,
        lcri_generalization_gap_delta=lcri_gap_delta,
        lcri_generalization_severity_by_scope=lcri_gap_severity_by_scope,
        lcri_ci_gate_contradiction_diagnostics=lcri_ci_gate_diagnostics,
        lcri_ci_confidence_coverage_scorecard=lcri_ci_confidence_scorecard,
        lcri_gap_delta_scope_summary=lcri_gap_delta_scope_summary_table,
        lcri_contradiction_review_packet=lcri_review_packet,
        lcri_uncertainty_weighted_review_priority=lcri_uncertainty_priority,
        lcri_cross_artifact_evidence_index=lcri_evidence_index,
        lcri_evidence_release_checklist=lcri_release_checklist,
        lcri_owner_handoff_packet=lcri_owner_handoff,
        lcri_evidence_lineage_map=lcri_lineage_map,
        lcri_calibration_fracture_pressure=lcri_calibration_fracture_pressure,
        lcri_reversal_transition_gate=reversal_transition_gate,
    )

    heldout_rows = len(books) - len(train)
    write_research_summary(
        output / "research_summary.md",
        rows=len(books),
        train_rows=len(train),
        heldout_rows=heldout_rows,
        seed=seed,
        train_frac=train_frac,
        metrics=metrics,
        heldout_metrics=heldout_metrics,
        generalization_gap=generalization_gap,
        baseline_tail_lift_diagnostics=baseline_tail_lift,
        baseline_stress_residual_drift=baseline_residual_drift,
        baseline_nonlinear_extrapolation_risk=baseline_nonlinear_extrapolation,
        baseline_nonlinear_extrapolation_risk_summary=baseline_nonlinear_extrapolation_summary,
        baseline_regime_publishability_summary=baseline_regime_publishability,
        regime_generalization_gap=regime_gap,
        transition_generalization_gap=transition_gap,
        generalization_fragility_diagnostics=fragility_diagnostics,
        generalization_fragility_summary=fragility_summary,
        generalization_overview=overview,
        generalization_gap_leaderboard=gap_leaderboard,
        lcri_generalization_gap_leaderboard=lcri_gap_leaderboard,
        lcri_generalization_scope_summary=lcri_generalization_scope_summary_table,
        lcri_generalization_severity=lcri_gap_severity,
        lcri_fragility_gate_alignment=lcri_fragility_alignment,
        lcri_fragility_gate_scorecard=lcri_fragility_scorecard,
        lcri_ci_gate_contradiction_diagnostics=lcri_ci_gate_diagnostics,
        lcri_ci_gate_contradiction_summary=lcri_ci_gate_summary,
        lcri_ci_confidence_coverage_scorecard=lcri_ci_confidence_scorecard,
        lcri_ci_confidence_coverage_summary=lcri_ci_confidence_summary,
        lcri_generalization_severity_by_scope=lcri_gap_severity_by_scope,
        lcri_generalization_severity_summary=lcri_gap_severity_summary,
        lcri_worst_generalization_context=lcri_worst_gap_context,
        lcri_generalization_gate_decision=lcri_gate_decision,
        lcri_generalization_gap_delta=lcri_gap_delta,
        lcri_gap_delta_flags=lcri_gap_flags,
        lcri_gap_delta_scorecard=lcri_gap_scorecard,
        lcri_gap_delta_summary=lcri_gap_summary,
        lcri_scope_stability_contradictions=lcri_scope_contradictions,
        lcri_scope_stability_contradiction_summary=lcri_scope_contradiction_summary,
        lcri_contradiction_review_packet=lcri_review_packet,
        lcri_contradiction_review_packet_summary=lcri_review_packet_summary,
        lcri_uncertainty_weighted_review_priority=lcri_uncertainty_priority,
        lcri_uncertainty_weighted_review_priority_summary=lcri_uncertainty_priority_summary,
        lcri_cross_artifact_evidence_index=lcri_evidence_index,
        lcri_cross_artifact_evidence_index_summary=lcri_evidence_index_summary,
        lcri_evidence_release_checklist=lcri_release_checklist,
        lcri_evidence_release_checklist_summary=lcri_release_checklist_summary,
        lcri_owner_handoff_packet=lcri_owner_handoff,
        lcri_owner_handoff_packet_summary=lcri_owner_handoff_summary,
        transition_lift=transition_lift,
        transition_robustness=transition_robustness,
        heldout_transition_lift=heldout_transition_lift,
        lcri_signal_monotonicity=lcri_monotonicity,
        heldout_lcri_signal_monotonicity=heldout_lcri_monotonicity,
        lcri_signal_monotonicity_summary=lcri_monotonicity_summary,
        heldout_lcri_signal_monotonicity_summary=heldout_lcri_monotonicity_summary,
        lcri_calibration_curve=lcri_calibration,
        heldout_lcri_calibration_curve=heldout_lcri_calibration,
        lcri_calibration_gate=lcri_calibration_gate,
        heldout_lcri_calibration_gate=heldout_lcri_calibration_gate,
        lcri_calibration_fracture_pressure=lcri_calibration_fracture_pressure,
        heldout_lcri_calibration_fracture_pressure=heldout_lcri_calibration_fracture_pressure,
        lcri_calibration_fracture_pressure_summary=lcri_calibration_fracture_pressure_summary,
        heldout_lcri_calibration_fracture_pressure_summary=heldout_lcri_calibration_fracture_pressure_summary,
        lcri_calibration_fracture_gate=lcri_calibration_fracture_gate,
        lcri_reversal_stress_summary=reversal_stress_summary,
        heldout_lcri_reversal_stress_summary=heldout_reversal_stress_summary,
        lcri_fracture_reversal_gate=fracture_reversal_gate,
        lcri_reversal_transition_gate=reversal_transition_gate,
        heldout_lcri_reversal_transition_gate=heldout_reversal_transition_gate,
        heldout_transition_robustness=heldout_transition_robustness,
        alpha_event_release_review_packet=alpha_release_packet_for_summary,
        alpha_event_window_regime_summary=alpha_window_regime_summary,
        alpha_event_window_summary=alpha_event_summary,
        alpha_event_drift_gate=alpha_drift_gate,
        alpha_event_review_verification_summary=alpha_event_verification_summary,
        hidden_resiliency_asymmetry_summary=resiliency_asymmetry,
        heldout_hidden_resiliency_asymmetry_summary=heldout_resiliency_asymmetry,
        adverse_selection_phase_shift_summary=adverse_phase_shift,
        heldout_adverse_selection_phase_shift_summary=heldout_adverse_phase_shift,
    )
    _append_execution_adjusted_summary(
        output / "research_summary.md",
        execution_summary=execution_summary,
        heldout_execution_summary=heldout_execution_summary,
        passive_fill_event_regimes=passive_fill_event_regimes,
        heldout_passive_fill_event_regimes=heldout_passive_fill_event_regimes,
        passive_fill_event_toxicity=passive_fill_event_toxicity,
        heldout_passive_fill_event_toxicity=heldout_passive_fill_event_toxicity,
        passive_fill_event_lifecycle_toxicity=passive_fill_event_lifecycle_toxicity,
        heldout_passive_fill_event_lifecycle_toxicity=heldout_passive_fill_event_lifecycle_toxicity,
        passive_fill_event_transition_toxicity=passive_fill_event_transition_toxicity,
        heldout_passive_fill_event_transition_toxicity=heldout_passive_fill_event_transition_toxicity,
        passive_fill_calibration_summary=passive_fill_calibration_stats,
        heldout_passive_fill_calibration_summary=heldout_passive_fill_calibration_stats,
        execution_publishability_gate=execution_publishability_gate,
        heldout_execution_publishability_gate=heldout_execution_publishability_gate,
        passive_fill_horizon_sweep=passive_fill_horizon_sweep,
        heldout_passive_fill_horizon_sweep=heldout_passive_fill_horizon_sweep,
        passive_fill_proxy_audit=passive_fill_proxy_audit,
        heldout_passive_fill_proxy_audit=heldout_passive_fill_proxy_audit,
        queue_fill_surface=queue_fill_surface,
        heldout_queue_fill_surface=heldout_queue_fill_surface,
        queue_fraction_sweep=queue_fraction_sweep,
        heldout_queue_fraction_sweep=heldout_queue_fraction_sweep,
        queue_capacity_frontier=queue_capacity_frontier,
        heldout_queue_capacity_frontier=heldout_queue_capacity_frontier,
        queue_capacity_stability=queue_capacity_stability,
        queue_edge_decay=queue_edge_decay,
        heldout_queue_edge_decay=heldout_queue_edge_decay,
        queue_calibration_drift=queue_calibration_drift,
        heldout_queue_calibration_drift=heldout_queue_calibration_drift,
        queue_calibration_stability_summary=queue_calibration_stability_summary,
        queue_adverse_policy_summary=queue_adverse_policy_summary,
        heldout_queue_adverse_policy_summary=heldout_queue_adverse_policy_summary,
    )
    coverage_matrix = artifact_coverage_matrix(artifact_paths)
    coverage_matrix.to_csv(output / "artifact_coverage_matrix.csv", index=False)
    write_json(output / "artifact_coverage_summary.json", artifact_coverage_summary(coverage_matrix))
    preliminary_metadata = collect_artifact_metadata(
        output,
        [path for path in artifact_paths if path != "artifact_metadata_summary.json"],
    )
    write_json(
        output / "artifact_metadata_summary.json",
        summarize_artifact_metadata(preliminary_metadata),
    )
    artifact_metadata = collect_artifact_metadata(output, artifact_paths)
    manifest = build_artifact_manifest(
        rows=len(books),
        train_rows=len(train),
        heldout_rows=heldout_rows,
        seed=seed,
        train_frac=train_frac,
        model_artifact_version=ARTIFACT_VERSION,
        artifacts=artifact_paths,
        artifact_metadata=artifact_metadata,
    )
    write_json(output / "artifact_manifest.json", manifest)
    missing = missing_artifacts(output, [*artifact_paths, "artifact_manifest.json"])
    if missing:
        raise RuntimeError(f"missing generated artifacts: {missing}")

    print("Wrote research artifacts")
    print(f"rows: {len(books)} total, {len(train)} train, {heldout_rows} held out")
    print(f"train fraction: {train_frac:.2f}")
    print(f"model: {output / 'lcri-model.json'}")
    print(f"metrics: {output / 'metrics.csv'}")
    print(f"heldout metrics: {output / 'heldout_metrics.csv'}")
    print(f"generalization gap: {output / 'generalization_gap.csv'}")
    print(f"regime metrics: {output / 'regime_metrics.csv'}")
    print(f"heldout regime metrics: {output / 'heldout_regime_metrics.csv'}")
    print(f"regime generalization gap: {output / 'regime_generalization_gap.csv'}")
    print(f"transition metrics: {output / 'transition_metrics.csv'}")
    print(f"heldout transition metrics: {output / 'heldout_transition_metrics.csv'}")
    print(f"transition generalization gap: {output / 'transition_generalization_gap.csv'}")
    print(
        "generalization stability confidence intervals: "
        f"{output / 'generalization_stability_confidence_intervals.csv'}"
    )
    print(
        "generalization stability confidence interval figure: "
        f"{output / 'figures' / 'generalization_stability_confidence_intervals.png'}"
    )
    print(f"generalization overview: {output / 'generalization_overview.json'}")
    print(f"generalization gap leaderboard: {output / 'generalization_gap_leaderboard.csv'}")
    print(f"lcri generalization gap leaderboard: {output / 'lcri_generalization_gap_leaderboard.csv'}")
    print(f"lcri generalization scope summary: {output / 'lcri_generalization_scope_summary.csv'}")
    print(f"lcri generalization severity: {output / 'lcri_generalization_severity.csv'}")
    print(f"lcri fragility gate alignment: {output / 'lcri_fragility_gate_alignment.csv'}")
    print(f"lcri fragility gate scorecard: {output / 'lcri_fragility_gate_scorecard.json'}")
    print(
        "lcri CI gate contradiction diagnostics: "
        f"{output / 'lcri_ci_gate_contradiction_diagnostics.csv'}"
    )
    print(
        "lcri CI gate contradiction summary: "
        f"{output / 'lcri_ci_gate_contradiction_summary.json'}"
    )
    print(
        "lcri CI confidence coverage scorecard: "
        f"{output / 'lcri_ci_confidence_coverage_scorecard.csv'}"
    )
    print(
        "lcri CI confidence coverage summary: "
        f"{output / 'lcri_ci_confidence_coverage_summary.json'}"
    )
    print(
        "lcri CI gate contradiction figure: "
        f"{output / 'figures' / 'lcri_ci_gate_contradiction_diagnostics.png'}"
    )
    print(
        "lcri CI confidence coverage figure: "
        f"{output / 'figures' / 'lcri_ci_confidence_coverage_scorecard.png'}"
    )
    print(f"lcri generalization severity by scope: {output / 'lcri_generalization_severity_by_scope.csv'}")
    print(f"lcri generalization severity summary: {output / 'lcri_generalization_severity_summary.json'}")
    print(f"lcri worst generalization context: {output / 'lcri_worst_generalization_context.json'}")
    print(f"lcri generalization gate decision: {output / 'lcri_generalization_gate_decision.json'}")
    print(f"lcri generalization gap delta: {output / 'lcri_generalization_gap_delta.csv'}")
    print(f"lcri gap delta dominant scopes: {output / 'lcri_gap_delta_dominant_scopes.json'}")
    print(f"lcri gap delta flags: {output / 'lcri_gap_delta_flags.csv'}")
    print(f"lcri gap delta scorecard: {output / 'lcri_gap_delta_scorecard.json'}")
    print(f"lcri gap delta summary: {output / 'lcri_gap_delta_summary.json'}")
    print(f"lcri scope stability contradictions: {output / 'lcri_scope_stability_contradictions.csv'}")
    print(
        "lcri scope stability contradiction summary: "
        f"{output / 'lcri_scope_stability_contradiction_summary.json'}"
    )
    print(f"lcri contradiction review packet: {output / 'lcri_contradiction_review_packet.csv'}")
    print(
        "lcri contradiction review packet summary: "
        f"{output / 'lcri_contradiction_review_packet_summary.json'}"
    )
    print(
        "lcri uncertainty-weighted review priority: "
        f"{output / 'lcri_uncertainty_weighted_review_priority.csv'}"
    )
    print(
        "lcri uncertainty-weighted review priority summary: "
        f"{output / 'lcri_uncertainty_weighted_review_priority_summary.json'}"
    )
    print(f"lcri cross-artifact evidence index: {output / 'lcri_cross_artifact_evidence_index.csv'}")
    print(
        "lcri cross-artifact evidence index summary: "
        f"{output / 'lcri_cross_artifact_evidence_index_summary.json'}"
    )
    print(f"lcri evidence release checklist: {output / 'lcri_evidence_release_checklist.csv'}")
    print(
        "lcri evidence release checklist summary: "
        f"{output / 'lcri_evidence_release_checklist_summary.json'}"
    )
    print(f"lcri owner handoff packet: {output / 'lcri_owner_handoff_packet.csv'}")
    print(f"lcri owner handoff markdown packet: {output / 'lcri_owner_handoff_packet.md'}")
    print(
        "lcri owner handoff packet summary: "
        f"{output / 'lcri_owner_handoff_packet_summary.json'}"
    )
    print(f"lcri evidence lineage map: {output / 'lcri_evidence_lineage_map.csv'}")
    print(f"lcri evidence lineage map figure: {output / 'figures' / 'lcri_evidence_lineage_map.png'}")
    print(f"transition lift: {output / 'transition_lift.csv'}")
    print(f"alpha event release review packet: {output / 'alpha_event_release_review_packet.csv'}")
    print(f"heldout transition lift: {output / 'heldout_transition_lift.csv'}")
    print(f"pressure memory decay summary: {output / 'pressure_memory_decay_summary.csv'}")
    print(
        "heldout pressure memory decay summary: "
        f"{output / 'heldout_pressure_memory_decay_summary.csv'}"
    )
    print(
        "hidden resiliency asymmetry summary: "
        f"{output / 'hidden_resiliency_asymmetry_summary.json'}"
    )
    print(f"adverse selection phase shift summary: {output / 'adverse_selection_phase_shift_summary.csv'}")
    print(f"lcri reversal stress concentration: {output / 'lcri_reversal_stress_concentration.csv'}")
    print(
        "lcri reversal stress concentration summary: "
        f"{output / 'lcri_reversal_stress_concentration_summary.json'}"
    )
    print(f"lcri fracture reversal gate: {output / 'lcri_fracture_reversal_gate.json'}")
    print(f"transition robustness: {output / 'transition_robustness.json'}")
    print(f"heldout transition robustness: {output / 'heldout_transition_robustness.json'}")
    print(
        "execution-adjusted LCRI quantile diagnostics: "
        f"{output / 'execution_adjusted_lcri_quantile_diagnostics.csv'}"
    )
    print(
        "heldout execution-adjusted LCRI quantile diagnostics: "
        f"{output / 'heldout_execution_adjusted_lcri_quantile_diagnostics.csv'}"
    )
    print(f"execution publishability release gate: {output / 'execution_publishability_release_gate.json'}")
    print(
        "heldout execution publishability release gate: "
        f"{output / 'heldout_execution_publishability_release_gate.json'}"
    )
    print(f"summary: {output / 'research_summary.md'}")
    print(f"artifact coverage matrix: {output / 'artifact_coverage_matrix.csv'}")
    print(f"manifest: {output / 'artifact_manifest.json'}")
    print(f"figures: {output / 'figures'}")
    print()
    print(metrics.to_string(index=False))


def _add_reversal_pressure_stack(frame: pd.DataFrame) -> pd.DataFrame:
    pressure_stack = add_liquidity_memory_half_life(
        add_pressure_memory(frame),
        group_col="regime",
    )
    reversal_stack = add_reversal_lead_lag_coupling(
        add_queue_reversal_risk(add_shadow_absorption(pressure_stack)),
        group_col="regime",
    )
    return add_microstructure_alpha_stack(
        reversal_stack,
        return_col="future_return_ticks",
        depth_col="total_depth",
    )


def _add_execution_adjusted_stack(frame: pd.DataFrame, *, tick_size: float) -> pd.DataFrame:
    """Attach queue-fill, publishability, and execution-adjusted LCRI diagnostics."""
    labeled = add_transaction_cost_labels(frame, tick_size=tick_size)
    gated = add_publishability_gate(labeled)
    queued = add_queue_position_features(gated)
    fills = add_passive_fill_probabilities(queued)
    execution_adjusted = add_execution_adjusted_edge(fills)
    return add_passive_fill_event_window_regimes(execution_adjusted)


def _add_passive_fill_realization_proxy(frame: pd.DataFrame, *, horizon: int = 1) -> pd.DataFrame:
    """Attach demo passive-fill labels from queue depletion over a snapshot horizon.

    The synthetic demo has snapshots rather than order-level executions. For the
    queue-position calibration surface we infer conservative passive fills from
    visible best-level depletion versus the estimated queue ahead, including full
    depletion when a future snapshot loses the current best bid/ask level. This
    is still a snapshot proxy, but it is queue-position-aware rather than only a
    next-mid touch label.
    """
    return add_queue_position_realized_fill_proxy(frame, horizon=horizon)


def _append_execution_adjusted_summary(
    path: Path,
    *,
    execution_summary: dict[str, float | int | str],
    heldout_execution_summary: dict[str, float | int | str],
    passive_fill_event_regimes: pd.DataFrame,
    heldout_passive_fill_event_regimes: pd.DataFrame,
    passive_fill_event_toxicity: dict[str, float | int | str],
    heldout_passive_fill_event_toxicity: dict[str, float | int | str],
    passive_fill_event_lifecycle_toxicity: dict[str, float | int | str],
    heldout_passive_fill_event_lifecycle_toxicity: dict[str, float | int | str],
    passive_fill_event_transition_toxicity: dict[str, float | int | str],
    heldout_passive_fill_event_transition_toxicity: dict[str, float | int | str],
    passive_fill_calibration_summary: dict[str, float | int | str],
    heldout_passive_fill_calibration_summary: dict[str, float | int | str],
    execution_publishability_gate: dict[str, float | int | str | bool],
    heldout_execution_publishability_gate: dict[str, float | int | str | bool],
    passive_fill_horizon_sweep: pd.DataFrame,
    heldout_passive_fill_horizon_sweep: pd.DataFrame,
    passive_fill_proxy_audit: pd.DataFrame,
    heldout_passive_fill_proxy_audit: pd.DataFrame,
    queue_fill_surface: pd.DataFrame,
    heldout_queue_fill_surface: pd.DataFrame,
    queue_fraction_sweep: pd.DataFrame,
    heldout_queue_fraction_sweep: pd.DataFrame,
    queue_capacity_frontier: dict[str, float | int | str],
    heldout_queue_capacity_frontier: dict[str, float | int | str],
    queue_capacity_stability: dict[str, float | int | str | bool],
    queue_edge_decay: pd.DataFrame,
    heldout_queue_edge_decay: pd.DataFrame,
    queue_calibration_drift: pd.DataFrame,
    heldout_queue_calibration_drift: pd.DataFrame,
    queue_calibration_stability_summary: dict[str, float | int | str],
    queue_adverse_policy_summary: dict[str, float | int | str],
    heldout_queue_adverse_policy_summary: dict[str, float | int | str],
) -> None:
    regime_lines = _passive_fill_regime_summary_lines(passive_fill_event_regimes)
    heldout_regime_lines = _passive_fill_regime_summary_lines(heldout_passive_fill_event_regimes)
    toxicity_lines = _passive_fill_event_toxicity_lines(passive_fill_event_toxicity)
    heldout_toxicity_lines = _passive_fill_event_toxicity_lines(
        heldout_passive_fill_event_toxicity
    )
    transition_toxicity_lines = _passive_fill_event_transition_toxicity_lines(
        passive_fill_event_transition_toxicity
    )
    lifecycle_toxicity_lines = _passive_fill_event_lifecycle_toxicity_lines(
        passive_fill_event_lifecycle_toxicity
    )
    heldout_lifecycle_toxicity_lines = _passive_fill_event_lifecycle_toxicity_lines(
        heldout_passive_fill_event_lifecycle_toxicity
    )
    heldout_transition_toxicity_lines = _passive_fill_event_transition_toxicity_lines(
        heldout_passive_fill_event_transition_toxicity
    )
    passive_calibration_lines = _passive_fill_calibration_summary_lines(
        passive_fill_calibration_summary
    )
    heldout_passive_calibration_lines = _passive_fill_calibration_summary_lines(
        heldout_passive_fill_calibration_summary
    )
    execution_gate_lines = _execution_publishability_release_gate_lines(
        execution_publishability_gate
    )
    heldout_execution_gate_lines = _execution_publishability_release_gate_lines(
        heldout_execution_publishability_gate
    )
    passive_horizon_lines = _passive_fill_horizon_sweep_lines(passive_fill_horizon_sweep)
    heldout_passive_horizon_lines = _passive_fill_horizon_sweep_lines(
        heldout_passive_fill_horizon_sweep
    )
    passive_proxy_audit_lines = _passive_fill_proxy_disagreement_lines(passive_fill_proxy_audit)
    heldout_passive_proxy_audit_lines = _passive_fill_proxy_disagreement_lines(
        heldout_passive_fill_proxy_audit
    )
    queue_surface_lines = _queue_position_fill_surface_lines(queue_fill_surface)
    heldout_queue_surface_lines = _queue_position_fill_surface_lines(heldout_queue_fill_surface)
    queue_sweep_lines = _queue_position_fraction_sweep_lines(queue_fraction_sweep)
    heldout_queue_sweep_lines = _queue_position_fraction_sweep_lines(heldout_queue_fraction_sweep)
    queue_capacity_lines = _queue_position_capacity_frontier_lines(queue_capacity_frontier)
    heldout_queue_capacity_lines = _queue_position_capacity_frontier_lines(
        heldout_queue_capacity_frontier
    )
    queue_capacity_stability_lines = _queue_position_capacity_stability_lines(
        queue_capacity_stability
    )
    queue_decay_lines = _queue_position_edge_decay_lines(queue_edge_decay)
    heldout_queue_decay_lines = _queue_position_edge_decay_lines(heldout_queue_edge_decay)
    queue_drift_lines = _queue_position_calibration_drift_lines(queue_calibration_drift)
    heldout_queue_drift_lines = _queue_position_calibration_drift_lines(
        heldout_queue_calibration_drift
    )
    queue_calibration_stability_lines = _queue_position_calibration_stability_summary_lines(
        queue_calibration_stability_summary
    )
    queue_adverse_policy_lines = _queue_position_adverse_policy_summary_lines(
        queue_adverse_policy_summary
    )
    heldout_queue_adverse_policy_lines = _queue_position_adverse_policy_summary_lines(
        heldout_queue_adverse_policy_summary
    )
    lines = [
        "",
        "## Execution-adjusted edge summary",
        "",
        *[f"- {key}: {value}" for key, value in execution_summary.items()],
        "",
        "## Heldout execution-adjusted edge summary",
        "",
        *[f"- {key}: {value}" for key, value in heldout_execution_summary.items()],
        "",
        "## Passive-fill event-window regime diagnostics",
        "",
        *regime_lines,
        "",
        "## Heldout passive-fill event-window regime diagnostics",
        "",
        *heldout_regime_lines,
        "",
        "## Passive-fill event-window toxicity scorecard",
        "",
        *toxicity_lines,
        "",
        "## Heldout passive-fill event-window toxicity scorecard",
        "",
        *heldout_toxicity_lines,
        "",
        "## Passive-fill event lifecycle toxicity scorecard",
        "",
        *lifecycle_toxicity_lines,
        "",
        "## Heldout passive-fill event lifecycle toxicity scorecard",
        "",
        *heldout_lifecycle_toxicity_lines,
        "",
        "## Passive-fill transition toxicity scorecard",
        "",
        *transition_toxicity_lines,
        "",
        "## Heldout passive-fill transition toxicity scorecard",
        "",
        *heldout_transition_toxicity_lines,
        "",
        "## Passive-fill calibration summary",
        "",
        *passive_calibration_lines,
        "",
        "## Heldout passive-fill calibration summary",
        "",
        *heldout_passive_calibration_lines,
        "",
        "## Execution publishability release gate",
        "",
        *execution_gate_lines,
        "",
        "## Heldout execution publishability release gate",
        "",
        *heldout_execution_gate_lines,
        "",
        "## Passive-fill realization horizon sweep",
        "",
        *passive_horizon_lines,
        "",
        "## Heldout passive-fill realization horizon sweep",
        "",
        *heldout_passive_horizon_lines,
        "",
        "## Passive-fill proxy disagreement audit",
        "",
        *passive_proxy_audit_lines,
        "",
        "## Heldout passive-fill proxy disagreement audit",
        "",
        *heldout_passive_proxy_audit_lines,
        "",
        "## Queue-position fill calibration surface",
        "",
        *queue_surface_lines,
        "",
        "## Heldout queue-position fill calibration surface",
        "",
        *heldout_queue_surface_lines,
        "",
        "## Queue-position fraction sweep",
        "",
        *queue_sweep_lines,
        "",
        "## Heldout queue-position fraction sweep",
        "",
        *heldout_queue_sweep_lines,
        "",
        "## Queue-position capacity frontier",
        "",
        *queue_capacity_lines,
        "",
        "## Heldout queue-position capacity frontier",
        "",
        *heldout_queue_capacity_lines,
        "",
        "## Queue-position capacity stability",
        "",
        *queue_capacity_stability_lines,
        "",
        "## Queue-position edge decay",
        "",
        *queue_decay_lines,
        "",
        "## Heldout queue-position edge decay",
        "",
        *heldout_queue_decay_lines,
        "",
        "## Queue-position calibration drift",
        "",
        *queue_drift_lines,
        "",
        "## Heldout queue-position calibration drift",
        "",
        *heldout_queue_drift_lines,
        "",
        "## Queue-position calibration stability summary",
        "",
        *queue_calibration_stability_lines,
        "",
        "## Queue-position adverse-selection policy summary",
        "",
        *queue_adverse_policy_lines,
        "",
        "## Heldout queue-position adverse-selection policy summary",
        "",
        *heldout_queue_adverse_policy_lines,
        "",
    ]
    path.write_text(path.read_text() + "\n".join(lines))


def _passive_fill_regime_summary_lines(summary: pd.DataFrame) -> list[str]:
    if summary.empty:
        return ["- no high-probability passive-fill events at threshold 0.75"]
    rows = []
    for row in summary.head(5).to_dict("records"):
        rows.append(
            "- "
            f"{row['event_regime']}: events={row['events']}, "
            f"adverse_post_edge_share={row['adverse_post_edge_share']:.3f}, "
            f"mean_post_minus_pre_realized_edge="
            f"{row['mean_post_minus_pre_realized_edge']:.3f}, "
            f"worst_post_minus_pre_realized_edge="
            f"{row['worst_post_minus_pre_realized_edge']:.3f}"
        )
    return rows


def _passive_fill_event_toxicity_lines(summary: dict[str, float | int | str]) -> list[str]:
    if not summary or int(summary.get("total_events", 0)) == 0:
        return ["- no passive-fill event toxicity windows"]
    keys = [
        "event_toxicity_label",
        "total_events",
        "eligible_regimes",
        "blocked_regimes",
        "worst_regime",
        "worst_adverse_post_edge_share",
        "worst_mean_post_minus_pre_realized_edge",
        "weighted_mean_event_fill_probability",
        "weighted_mean_event_adverse_fill_probability",
        "weighted_mean_post_minus_pre_realized_edge",
    ]
    return [f"- {key}: {summary.get(key, 'n/a')}" for key in keys]


def _passive_fill_event_lifecycle_toxicity_lines(
    summary: dict[str, float | int | str],
) -> list[str]:
    if not summary or int(summary.get("total_events", 0)) == 0:
        return ["- no passive-fill lifecycle toxicity windows"]
    keys = [
        "lifecycle_toxicity_gate_label",
        "total_events",
        "eligible_lifecycle_paths",
        "blocked_lifecycle_paths",
        "worst_lifecycle_path",
        "worst_adverse_post_edge_share",
        "worst_mean_post_minus_pre_realized_edge",
        "weighted_mean_event_fill_probability",
        "weighted_mean_event_adverse_fill_probability",
        "weighted_mean_post_minus_pre_realized_edge",
    ]
    return [f"- {key}: {summary.get(key, 'n/a')}" for key in keys]


def _passive_fill_event_transition_toxicity_lines(
    summary: dict[str, float | int | str],
) -> list[str]:
    if not summary or int(summary.get("total_events", 0)) == 0:
        return ["- no passive-fill transition toxicity windows"]
    keys = [
        "transition_toxicity_label",
        "total_events",
        "eligible_transitions",
        "blocked_transitions",
        "worst_transition",
        "worst_adverse_post_edge_share",
        "worst_mean_post_minus_pre_realized_edge",
        "weighted_mean_event_fill_probability",
        "weighted_mean_event_adverse_fill_probability",
        "weighted_mean_post_minus_pre_realized_edge",
    ]
    return [f"- {key}: {summary.get(key, 'n/a')}" for key in keys]


def _passive_fill_calibration_summary_lines(summary: dict[str, float | int | str]) -> list[str]:
    if not summary or int(summary.get("rows", 0)) == 0:
        return ["- no passive-fill calibration observations"]
    keys = [
        "rows",
        "regimes",
        "weighted_mean_predicted_fill_probability",
        "weighted_realized_fill_rate",
        "expected_calibration_error",
        "weighted_brier_score",
        "worst_regime",
        "worst_absolute_calibration_error",
    ]
    return [f"- {key}: {summary.get(key, 'n/a')}" for key in keys]


def _execution_publishability_release_gate_lines(
    gate: dict[str, float | int | str | bool],
) -> list[str]:
    if not gate:
        return ["- no execution publishability release gate"]
    keys = [
        "decision",
        "passes",
        "release_gate_label",
        "total_rows",
        "weighted_conflict_share",
        "high_priority_conflict_share",
        "quality_gate_label",
        "capacity_stability_label",
        "blocking_reasons",
        "review_reasons",
    ]
    return [f"- {key}: {gate.get(key, 'n/a')}" for key in keys]


def _passive_fill_horizon_sweep_lines(sweep: pd.DataFrame) -> list[str]:
    if sweep.empty:
        return ["- no passive-fill realization horizon rows"]
    rows = []
    for row in sweep.to_dict("records"):
        rows.append(
            "- "
            f"horizon={int(row['horizon'])}: rows={int(row['rows'])}, "
            f"predicted_fill={row['weighted_mean_predicted_fill_probability']:.3f}, "
            f"realized_fill={row['weighted_realized_fill_rate']:.3f}, "
            f"realized_gap={row['realized_fill_rate_gap_vs_shortest']:.3f}, "
            f"brier={row['weighted_brier_score']:.3f}, "
            f"brier_gap={row['brier_score_gap_vs_shortest']:.3f}, "
            f"label={row['horizon_stability_label']}"
        )
    return rows


def _passive_fill_proxy_disagreement_lines(audit: pd.DataFrame) -> list[str]:
    if audit.empty:
        return ["- no passive-fill proxy disagreement observations"]
    rows = []
    for row in audit.to_dict("records"):
        rows.append(
            "- "
            f"{row['side']}: rows={int(row['rows'])}, "
            f"snapshot_fill={row['snapshot_fill_rate']:.3f}, "
            f"event_fill={row['event_fill_rate']:.3f}, "
            f"disagreement={row['disagreement_rate']:.3f}, "
            f"fp={row['false_positive_rate']:.3f}, "
            f"fn={row['false_negative_rate']:.3f}, "
            f"label={row['review_label']}"
        )
    return rows


def _queue_position_fill_surface_lines(surface: pd.DataFrame) -> list[str]:
    if surface.empty:
        return ["- no tradable queue-position fill observations"]
    ranked = surface.sort_values(
        ["absolute_calibration_error", "rows"], ascending=[False, False]
    ).head(5)
    rows = []
    for row in ranked.to_dict("records"):
        rows.append(
            "- "
            f"{row['regime']} qbin={int(row['queue_bin'])} "
            f"pbin={int(row['fill_probability_bin'])}: rows={int(row['rows'])}, "
            f"mean_queue_share={row['mean_queue_share']:.3f}, "
            f"predicted_fill={row['mean_predicted_fill_probability']:.3f}, "
            f"realized_fill={row['realized_fill_rate']:.3f}, "
            f"abs_calibration_error={row['absolute_calibration_error']:.3f}, "
            f"mean_execution_edge={row['mean_execution_adjusted_edge_ticks']:.3f}"
        )
    return rows


def _queue_position_fraction_sweep_lines(sweep: pd.DataFrame) -> list[str]:
    if sweep.empty:
        return ["- no queue-position fraction sweep rows"]
    rows = []
    for row in sweep.to_dict("records"):
        rows.append(
            "- "
            f"fraction={row['queue_position_fraction']:.2f}: "
            f"bid_fill={row['mean_bid_fill_probability']:.3f}, "
            f"ask_fill={row['mean_ask_fill_probability']:.3f}, "
            f"execution_edge={row['mean_execution_adjusted_edge_ticks']:.3f}, "
            f"tradable_share={row['tradable_share']:.3f}, "
            f"dominant_side={row['dominant_execution_side']}"
        )
    return rows


def _queue_position_capacity_frontier_lines(frontier: dict[str, float | int | str]) -> list[str]:
    if not frontier:
        return ["- no queue-position capacity frontier"]
    keys = [
        "capacity_label",
        "max_viable_queue_position_fraction",
        "max_viable_mean_execution_adjusted_edge_ticks",
        "max_viable_tradable_share",
        "edge_decay_to_capacity_ticks",
        "tradable_share_decay_to_capacity",
    ]
    return [f"- {key}: {frontier.get(key, 'n/a')}" for key in keys]


def _queue_position_capacity_stability_lines(
    stability: dict[str, float | int | str | bool],
) -> list[str]:
    if not stability:
        return ["- no queue-position capacity stability"]
    keys = [
        "capacity_stability_label",
        "research_capacity_label",
        "heldout_capacity_label",
        "capacity_fraction_gap",
        "capacity_edge_gap_ticks",
        "capacity_tradable_share_gap",
        "capacity_viable_row_gap",
        "dominant_side_changed",
    ]
    return [f"- {key}: {stability.get(key, 'n/a')}" for key in keys]


def _queue_position_edge_decay_lines(decay: pd.DataFrame) -> list[str]:
    if decay.empty:
        return ["- no queue-position edge-decay regimes"]
    rows = []
    for row in decay.head(5).to_dict("records"):
        rows.append(
            "- "
            f"{row['regime']}: bins={int(row['queue_bins'])}, rows={int(row['rows'])}, "
            f"front_queue={row['front_mean_queue_share']:.3f}, "
            f"back_queue={row['back_mean_queue_share']:.3f}, "
            f"fill_decay={row['fill_rate_decay']:.3f}, "
            f"edge_decay={row['edge_decay_ticks']:.3f}, "
            f"label={row['queue_decay_label']}"
        )
    return rows


def _queue_position_calibration_drift_lines(drift: pd.DataFrame) -> list[str]:
    if drift.empty:
        return ["- no cross-regime queue calibration drift bins"]
    sorted_drift = drift.sort_values(
        ["calibration_error_range", "fill_rate_range", "rows"], ascending=False
    )
    rows = []
    for row in sorted_drift.head(5).to_dict("records"):
        rows.append(
            "- "
            f"{row['best_execution_side']} queue_bin={int(row['queue_share_bin'])}, "
            f"fill_bin={int(row['fill_probability_bin'])}: regimes={int(row['regimes'])}, "
            f"rows={int(row['rows'])}, fill_range={row['fill_rate_range']:.3f}, "
            f"calibration_range={row['calibration_error_range']:.3f}, "
            f"worst_regime={row['worst_regime']}, label={row['drift_label']}"
        )
    return rows


def _queue_position_calibration_stability_summary_lines(
    summary: dict[str, float | int | str],
) -> list[str]:
    if not summary or int(summary.get("cells", 0)) == 0:
        return ["- no train/holdout queue calibration cells"]
    keys = [
        "queue_calibration_stability_label",
        "cells",
        "common_cells",
        "replicated_cells",
        "degraded_cells",
        "lost_cells",
        "gained_cells",
        "degraded_cell_share",
        "mean_absolute_calibration_error_gap",
        "worst_regime",
        "worst_best_execution_side",
        "worst_queue_share_bin",
        "worst_fill_probability_bin",
        "worst_calibration_stability_label",
    ]
    return [f"- {key}: {summary.get(key, 'n/a')}" for key in keys]


def _queue_position_adverse_policy_summary_lines(
    summary: dict[str, float | int | str],
) -> list[str]:
    if not summary or int(summary.get("policies", 0)) == 0:
        return ["- no queue-position adverse-selection policy grid"]
    keys = [
        "policy_summary_label",
        "policies",
        "publishable_policies",
        "best_policy_label",
        "best_fill_threshold",
        "best_adverse_threshold",
        "best_candidate_rows",
        "best_trade_share",
        "best_realized_fill_rate",
        "best_mean_adverse_fill_probability",
        "best_mean_realized_edge_ticks",
        "best_mean_execution_adjusted_edge_ticks",
        "best_toxicity_filtered_share",
        "dominant_side",
    ]
    return [f"- {key}: {summary.get(key, 'n/a')}" for key in keys]


def verify_report(report_dir: Path) -> None:
    manifest_path = report_dir / "artifact_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"missing artifact manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    research_summary_errors = (
        verify_research_summary_sections(report_dir)
        if "research_summary.md" in set(manifest.get("artifacts", []))
        else []
    )
    manifest_artifacts = set(manifest.get("artifacts", []))
    artifact_coverage_errors = (
        verify_artifact_coverage_matrix(report_dir, manifest)
        if "artifact_coverage_matrix.csv" in manifest_artifacts
        else []
    )
    errors = [
        *verify_artifact_manifest(report_dir, manifest),
        *artifact_coverage_errors,
        *verify_artifact_metadata_summary(report_dir, manifest),
        *verify_figure_artifacts(report_dir, manifest),
        *(
            verify_baseline_regime_publishability_summary(report_dir)
            if "baseline_regime_publishability_summary.json" in manifest_artifacts
            else []
        ),
        *(
            verify_baseline_tail_lift_diagnostics(report_dir)
            if "baseline_tail_lift_diagnostics.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_baseline_stress_residual_drift(report_dir)
            if "baseline_stress_residual_drift.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_pressure_memory_decay_summary(report_dir)
            if "pressure_memory_decay_summary.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_pressure_memory_decay_summary(
                report_dir, "heldout_pressure_memory_decay_summary.csv"
            )
            if "heldout_pressure_memory_decay_summary.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_hidden_resiliency_asymmetry_summary(report_dir)
            if "hidden_resiliency_asymmetry_summary.json" in manifest_artifacts
            else []
        ),
        *(
            verify_hidden_resiliency_asymmetry_summary(
                report_dir, "heldout_hidden_resiliency_asymmetry_summary.json"
            )
            if "heldout_hidden_resiliency_asymmetry_summary.json" in manifest_artifacts
            else []
        ),
        *(
            verify_adverse_selection_phase_shift_summary(report_dir)
            if "adverse_selection_phase_shift_summary.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_adverse_selection_phase_shift_summary(
                report_dir, "heldout_adverse_selection_phase_shift_summary.csv"
            )
            if "heldout_adverse_selection_phase_shift_summary.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_phase_shift_artifact_review(report_dir)
            if "phase_shift_artifact_review.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_phase_shift_artifact_review(
                report_dir, "heldout_phase_shift_artifact_review.csv"
            )
            if "heldout_phase_shift_artifact_review.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_execution_publishability_review_artifacts(report_dir)
            if "execution_publishability_review_packet.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_execution_publishability_review_artifacts(
                report_dir, "heldout_execution_publishability_review_packet.csv"
            )
            if "heldout_execution_publishability_review_packet.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_execution_adjusted_edge_component_attribution(report_dir)
            if "execution_adjusted_edge_component_attribution.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_execution_adjusted_edge_component_attribution(
                report_dir, "heldout_execution_adjusted_edge_component_attribution.csv"
            )
            if "heldout_execution_adjusted_edge_component_attribution.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_execution_adjusted_lcri_side_attribution(report_dir)
            if "execution_adjusted_lcri_side_attribution.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_execution_adjusted_lcri_side_attribution(
                report_dir, "heldout_execution_adjusted_lcri_side_attribution.csv"
            )
            if "heldout_execution_adjusted_lcri_side_attribution.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_execution_adjusted_lcri_quantile_diagnostics(report_dir)
            if "execution_adjusted_lcri_quantile_diagnostics.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_execution_adjusted_lcri_quantile_diagnostics(
                report_dir, "heldout_execution_adjusted_lcri_quantile_diagnostics.csv"
            )
            if "heldout_execution_adjusted_lcri_quantile_diagnostics.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_queue_position_lcri_tail_fill_residuals(report_dir)
            if "queue_position_lcri_tail_fill_residuals.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_queue_position_lcri_tail_fill_residuals(
                report_dir, "heldout_queue_position_lcri_tail_fill_residuals.csv"
            )
            if "heldout_queue_position_lcri_tail_fill_residuals.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_queue_position_fill_monotonicity_scorecard(report_dir)
            if "queue_position_fill_monotonicity_scorecard.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_queue_position_fill_monotonicity_scorecard(
                report_dir, "heldout_queue_position_fill_monotonicity_scorecard.csv"
            )
            if "heldout_queue_position_fill_monotonicity_scorecard.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_queue_position_unfilled_opportunity_scorecard(report_dir)
            if "queue_position_unfilled_opportunity_scorecard.json" in manifest_artifacts
            else []
        ),
        *(
            verify_queue_position_unfilled_opportunity_scorecard(
                report_dir, "heldout_queue_position_unfilled_opportunity_scorecard.json"
            )
            if "heldout_queue_position_unfilled_opportunity_scorecard.json" in manifest_artifacts
            else []
        ),
        *(
            verify_execution_publishability_release_gate(report_dir)
            if "execution_publishability_release_gate.json" in manifest_artifacts
            else []
        ),
        *(
            verify_execution_publishability_release_gate(
                report_dir, "heldout_execution_publishability_release_gate.json"
            )
            if "heldout_execution_publishability_release_gate.json" in manifest_artifacts
            else []
        ),
        *(
            verify_execution_adjusted_lcri_event_window_release_scorecard(report_dir)
            if "execution_adjusted_lcri_event_window_release_scorecard.json" in manifest_artifacts
            else []
        ),
        *(
            verify_execution_adjusted_lcri_event_window_release_scorecard(
                report_dir, "heldout_execution_adjusted_lcri_event_window_release_scorecard.json"
            )
            if "heldout_execution_adjusted_lcri_event_window_release_scorecard.json"
            in manifest_artifacts
            else []
        ),
        *(
            verify_passive_fill_event_window_regime_scorecard(report_dir)
            if "passive_fill_event_window_regime_scorecard.json" in manifest_artifacts
            else []
        ),
        *(
            verify_passive_fill_event_window_regime_scorecard(
                report_dir, "heldout_passive_fill_event_window_regime_scorecard.json"
            )
            if "heldout_passive_fill_event_window_regime_scorecard.json" in manifest_artifacts
            else []
        ),
        *(
            verify_trade_confirmed_passive_fill_latency_summary(report_dir)
            if "trade_confirmed_passive_fill_latency_summary.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_trade_confirmed_passive_fill_latency_summary(
                report_dir, "heldout_trade_confirmed_passive_fill_latency_summary.csv"
            )
            if "heldout_trade_confirmed_passive_fill_latency_summary.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_queue_position_trade_confirmation_regime_scorecard(report_dir)
            if "queue_position_trade_confirmation_regime_scorecard.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_queue_position_trade_confirmation_regime_scorecard(
                report_dir, "heldout_queue_position_trade_confirmation_regime_scorecard.csv"
            )
            if "heldout_queue_position_trade_confirmation_regime_scorecard.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_queue_position_trade_confirmation_release_scorecard(report_dir)
            if "queue_position_trade_confirmation_release_scorecard.json" in manifest_artifacts
            else []
        ),
        *(
            verify_queue_position_trade_confirmation_release_scorecard(
                report_dir, "heldout_queue_position_trade_confirmation_release_scorecard.json"
            )
            if "heldout_queue_position_trade_confirmation_release_scorecard.json" in manifest_artifacts
            else []
        ),
        *(
            verify_passive_fill_realization_horizon_sweep(report_dir)
            if "passive_fill_realization_horizon_sweep.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_passive_fill_realization_horizon_sweep(
                report_dir, "heldout_passive_fill_realization_horizon_sweep.csv"
            )
            if "heldout_passive_fill_realization_horizon_sweep.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_passive_fill_threshold_policy_curve(report_dir)
            if "passive_fill_threshold_policy_curve.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_passive_fill_threshold_policy_curve(
                report_dir, "heldout_passive_fill_threshold_policy_curve.csv"
            )
            if "heldout_passive_fill_threshold_policy_curve.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_queue_position_path_drawdown_artifacts(report_dir)
            if "queue_position_path_drawdown_episodes.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_queue_position_path_drawdown_artifacts(
                report_dir,
                "heldout_queue_position_path_drawdown_episodes.csv",
                "heldout_queue_position_path_drawdown_summary.json",
            )
            if "heldout_queue_position_path_drawdown_episodes.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_passive_fill_event_lifecycle_policy_curve(report_dir)
            if "passive_fill_event_lifecycle_policy_curve.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_passive_fill_event_lifecycle_policy_curve(
                report_dir, "heldout_passive_fill_event_lifecycle_policy_curve.csv"
            )
            if "heldout_passive_fill_event_lifecycle_policy_curve.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_passive_fill_event_transition_policy_curve(report_dir)
            if "passive_fill_event_transition_policy_curve.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_passive_fill_event_transition_policy_curve(
                report_dir, "heldout_passive_fill_event_transition_policy_curve.csv"
            )
            if "heldout_passive_fill_event_transition_policy_curve.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_passive_fill_event_policy_stability(report_dir)
            if "passive_fill_event_lifecycle_policy_stability.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_passive_fill_event_policy_stability(
                report_dir,
                "passive_fill_event_transition_policy_stability.csv",
                context_col="regime_transition",
            )
            if "passive_fill_event_transition_policy_stability.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_passive_fill_event_policy_stability_scorecard(
                report_dir,
                "passive_fill_event_lifecycle_policy_stability_scorecard.json",
            )
            if "passive_fill_event_lifecycle_policy_stability_scorecard.json" in manifest_artifacts
            else []
        ),
        *(
            verify_passive_fill_event_policy_stability_scorecard(
                report_dir,
                "passive_fill_event_transition_policy_stability_scorecard.json",
            )
            if "passive_fill_event_transition_policy_stability_scorecard.json" in manifest_artifacts
            else []
        ),
        *(
            verify_alpha_event_review_artifacts(report_dir)
            if {
                "alpha_event_windows.csv",
                "alpha_event_window_summary.json",
                "alpha_event_release_review_packet.csv",
                "alpha_event_drift_gate.json",
                "alpha_event_score_weighted_drift.json",
                "alpha_event_regime_summary.csv",
            }.issubset(manifest_artifacts)
            else []
        ),
        *(
            verify_alpha_event_review_verification_summary(report_dir)
            if "alpha_event_review_verification_summary.json" in manifest_artifacts
            else []
        ),
        *verify_generalization_fragility_diagnostics(report_dir),
        *verify_generalization_fragility_summary(report_dir),
        *verify_generalization_fragility_consistency(report_dir),
        *verify_generalization_stability_confidence_intervals(report_dir),
        *verify_generalization_stability_confidence_summary(report_dir),
        *verify_generalization_stability_confidence_consistency(report_dir),
        *verify_generalization_overview(report_dir),
        *verify_lcri_generalization_gap_leaderboard(report_dir),
        *verify_lcri_generalization_scope_summary(report_dir),
        *verify_lcri_generalization_severity(report_dir),
        *verify_lcri_fragility_gate_alignment(report_dir),
        *verify_lcri_fragility_gate_scorecard(report_dir),
        *(
            verify_lcri_fracture_reversal_gate(report_dir)
            if "lcri_fracture_reversal_gate.json" in manifest_artifacts
            else []
        ),
        *(
            verify_lcri_reversal_transition_gate_consistency(report_dir)
            if "lcri_reversal_transition_gate.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_lcri_ci_gate_contradiction_diagnostics(report_dir)
            if "lcri_ci_gate_contradiction_diagnostics.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_lcri_ci_gate_contradiction_summary(report_dir)
            if "lcri_ci_gate_contradiction_summary.json" in manifest_artifacts
            else []
        ),
        *(
            verify_lcri_ci_gate_contradiction_consistency(report_dir)
            if {
                "lcri_ci_gate_contradiction_diagnostics.csv",
                "lcri_ci_gate_contradiction_summary.json",
            }.issubset(manifest_artifacts)
            else []
        ),
        *(
            verify_lcri_ci_confidence_coverage_scorecard(report_dir)
            if "lcri_ci_confidence_coverage_scorecard.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_lcri_ci_confidence_coverage_summary(report_dir)
            if "lcri_ci_confidence_coverage_summary.json" in manifest_artifacts
            else []
        ),
        *(
            verify_lcri_ci_confidence_coverage_consistency(report_dir)
            if {
                "lcri_ci_confidence_coverage_scorecard.csv",
                "lcri_ci_confidence_coverage_summary.json",
            }.issubset(manifest_artifacts)
            else []
        ),
        *verify_lcri_generalization_severity_by_scope(report_dir),
        *verify_lcri_generalization_severity_consistency(report_dir),
        *verify_lcri_generalization_scope_risk(report_dir),
        *verify_lcri_generalization_scope_gate_decisions(report_dir),
        *verify_lcri_generalization_scope_gate_decision_summary(report_dir),
        *verify_lcri_generalization_scope_gate_consistency(report_dir),
        *verify_lcri_generalization_critical_contexts(report_dir),
        *verify_lcri_generalization_blocker_summary(report_dir),
        *verify_lcri_generalization_severity_summary(report_dir),
        *verify_lcri_generalization_gate_decision(report_dir),
        *verify_lcri_worst_generalization_context(report_dir),
        *verify_lcri_generalization_gate_decision_consistency(report_dir),
        *verify_lcri_generalization_gap_delta(report_dir),
        *verify_lcri_gap_delta_dominant_scopes(report_dir),
        *verify_lcri_gap_delta_flags(report_dir),
        *verify_lcri_gap_delta_improvements(report_dir),
        *verify_lcri_gap_delta_regressions(report_dir),
        *verify_lcri_gap_delta_scorecard(report_dir),
        *verify_lcri_gap_delta_scope_extremes(report_dir),
        *verify_lcri_gap_delta_scope_summary(report_dir),
        *verify_lcri_gap_delta_summary(report_dir),
        *verify_lcri_gap_delta_consistency(report_dir),
        *verify_lcri_scope_stability_contradictions(report_dir),
        *verify_lcri_scope_stability_contradiction_summary(report_dir),
        *verify_lcri_scope_stability_contradictions_consistency(report_dir),
        *verify_lcri_contradiction_review_packet(report_dir),
        *verify_lcri_contradiction_review_packet_summary(report_dir),
        *verify_lcri_uncertainty_weighted_review_priority(report_dir),
        *verify_lcri_uncertainty_weighted_review_priority_summary(report_dir),
        *verify_lcri_uncertainty_weighted_review_priority_consistency(report_dir),
        *(
            verify_lcri_cross_artifact_evidence_index(report_dir)
            if "lcri_cross_artifact_evidence_index.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_lcri_cross_artifact_evidence_index_summary(report_dir)
            if "lcri_cross_artifact_evidence_index_summary.json" in manifest_artifacts
            else []
        ),
        *(
            verify_lcri_cross_artifact_evidence_index_consistency(report_dir)
            if {
                "lcri_cross_artifact_evidence_index.csv",
                "lcri_cross_artifact_evidence_index_summary.json",
            }.issubset(manifest_artifacts)
            else []
        ),
        *(
            verify_lcri_evidence_release_checklist(report_dir)
            if "lcri_evidence_release_checklist.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_lcri_evidence_release_checklist_summary(report_dir)
            if "lcri_evidence_release_checklist_summary.json" in manifest_artifacts
            else []
        ),
        *(
            verify_lcri_evidence_release_checklist_consistency(report_dir)
            if {
                "lcri_cross_artifact_evidence_index.csv",
                "lcri_evidence_release_checklist.csv",
                "lcri_evidence_release_checklist_summary.json",
            }.issubset(manifest_artifacts)
            else []
        ),
        *(
            verify_lcri_owner_handoff_packet(report_dir)
            if "lcri_owner_handoff_packet.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_lcri_owner_handoff_packet_summary(report_dir)
            if "lcri_owner_handoff_packet_summary.json" in manifest_artifacts
            else []
        ),
        *(
            verify_lcri_owner_handoff_packet_consistency(report_dir)
            if {
                "lcri_cross_artifact_evidence_index.csv",
                "lcri_evidence_release_checklist.csv",
                "lcri_owner_handoff_packet.csv",
                "lcri_owner_handoff_packet_summary.json",
            }.issubset(manifest_artifacts)
            else []
        ),
        *(
            verify_lcri_owner_handoff_markdown_packet(report_dir)
            if "lcri_owner_handoff_packet.md" in manifest_artifacts
            else []
        ),
        *(
            verify_lcri_evidence_lineage_map(report_dir)
            if "lcri_evidence_lineage_map.csv" in manifest_artifacts
            else []
        ),
        *(
            verify_lcri_evidence_lineage_map_summary(report_dir)
            if "lcri_evidence_lineage_map_summary.json" in manifest_artifacts
            else []
        ),
        *(
            verify_lcri_evidence_lineage_map_consistency(report_dir)
            if {
                "lcri_cross_artifact_evidence_index.csv",
                "lcri_evidence_release_checklist.csv",
                "lcri_owner_handoff_packet.csv",
                "lcri_evidence_lineage_map.csv",
                "lcri_evidence_lineage_map_summary.json",
            }.issubset(manifest_artifacts)
            else []
        ),
        *research_summary_errors,
    ]
    summary = summarize_verification_errors(errors)
    if errors:
        raise ValueError(f"report verification failed: {summary}: {errors}")
    print(f"Verified report artifacts: {report_dir}")
    print(f"verification summary: {summary}")


def normalize_snapshots(
    input_path: Path,
    output_path: Path,
    tick_size: float,
    levels: int = 5,
    derive_state: bool = False,
) -> None:
    frame = pd.read_csv(input_path)
    normalized = normalize_l2_snapshots(
        frame, tick_size=tick_size, levels=levels, derive_state=derive_state
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output_path, index=False)
    print(f"Wrote normalized snapshots: {output_path}")


def fit_model(
    input_path: Path,
    model_path: Path,
    levels: int,
    ridge: float = 1e-3,
    probability_scale: float = 1.0,
) -> None:
    frame = pd.read_csv(input_path)
    model = LCRIModel(ModelConfig(levels=levels, ridge=ridge, probability_scale=probability_scale)).fit(
        frame
    )
    model.save(model_path)
    print(f"Wrote model: {model_path}")


def describe_model(model_path: Path) -> None:
    model = LCRIModel.load(model_path)
    features = model.baseline.coefficients.size - 1
    regimes = sorted(model.baseline.residual_scale_by_regime or {})
    print(f"schema_version: {model.artifact_version()}")
    print(f"levels: {model.config.levels}")
    print(f"ridge: {model.config.ridge}")
    print(f"probability_scale: {model.config.probability_scale}")
    print(f"features: {features}")
    print(f"regimes: {', '.join(regimes)}")


def score_model(
    input_path: Path,
    model_path: Path,
    output_path: Path,
    columns: list[str] | None = None,
) -> None:
    frame = pd.read_csv(input_path)
    model = LCRIModel.load(model_path)
    scored = model.score_frame(frame)
    if columns is not None:
        columns = [column.strip() for column in columns if column.strip()]
        if not columns:
            raise ValueError("requested score output columns must include at least one column")
        duplicate_columns = sorted({column for column in columns if columns.count(column) > 1})
        if duplicate_columns:
            raise ValueError(f"requested score output columns must be unique: {duplicate_columns}")
        missing = sorted(set(columns) - set(scored.columns))
        if missing:
            raise ValueError(f"requested score output columns are unavailable: {missing}")
        scored = scored.loc[:, columns]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    print(f"Wrote scores: {output_path}")


if __name__ == "__main__":
    main()
