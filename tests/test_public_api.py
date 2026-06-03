import lcri_lab


def test_public_api_exports_calibration_curve() -> None:
    assert callable(lcri_lab.calibration_curve)
    assert "calibration_curve" in lcri_lab.__all__


def test_public_api_exports_calibration_error_gates() -> None:
    assert callable(lcri_lab.calibration_error_summary)
    assert callable(lcri_lab.calibration_gate_decision)
    assert "calibration_error_summary" in lcri_lab.__all__
    assert "calibration_gate_decision" in lcri_lab.__all__


def test_public_api_exports_signal_monotonicity_validation() -> None:
    assert callable(lcri_lab.signal_quantile_monotonicity)
    assert callable(lcri_lab.signal_quantile_monotonicity_summary)
    assert "signal_quantile_monotonicity" in lcri_lab.__all__
    assert "signal_quantile_monotonicity_summary" in lcri_lab.__all__


def test_public_api_exports_artifact_version() -> None:
    assert lcri_lab.ARTIFACT_VERSION == 3
    assert "ARTIFACT_VERSION" in lcri_lab.__all__


def test_public_api_exports_design_feature_names() -> None:
    names = lcri_lab.design_feature_names()
    assert "spread_x_replenishment" in names
    assert "imbalance_fracture" in names
    assert "design_feature_names" in lcri_lab.__all__


def test_public_api_exports_baseline_liquidity_stress_curve() -> None:
    assert callable(lcri_lab.baseline_liquidity_stress_curve)
    assert "baseline_liquidity_stress_curve" in lcri_lab.__all__


def test_public_api_exports_baseline_basis_comparison() -> None:
    assert callable(lcri_lab.baseline_basis_comparison)
    assert "baseline_basis_comparison" in lcri_lab.__all__


def test_public_api_exports_baseline_rolling_basis_comparison() -> None:
    assert callable(lcri_lab.baseline_rolling_basis_comparison)
    assert "baseline_rolling_basis_comparison" in lcri_lab.__all__


def test_public_api_exports_baseline_rolling_basis_summary() -> None:
    assert callable(lcri_lab.baseline_rolling_basis_summary)
    assert "baseline_rolling_basis_summary" in lcri_lab.__all__


def test_public_api_exports_baseline_tail_lift_diagnostics() -> None:
    assert callable(lcri_lab.baseline_tail_lift_diagnostics)
    assert "baseline_tail_lift_diagnostics" in lcri_lab.__all__


def test_public_api_exports_baseline_nonlinear_publishability_summary() -> None:
    assert callable(lcri_lab.baseline_nonlinear_publishability_summary)
    assert "baseline_nonlinear_publishability_summary" in lcri_lab.__all__


def test_public_api_exports_baseline_nonlinear_stress_surface() -> None:
    assert callable(lcri_lab.baseline_nonlinear_stress_surface)
    assert "baseline_nonlinear_stress_surface" in lcri_lab.__all__


def test_public_api_exports_baseline_nonlinear_feature_ablation() -> None:
    assert callable(lcri_lab.baseline_nonlinear_feature_ablation)
    assert "baseline_nonlinear_feature_ablation" in lcri_lab.__all__


def test_public_api_exports_baseline_nonlinear_extrapolation_risk() -> None:
    assert callable(lcri_lab.baseline_nonlinear_extrapolation_risk)
    assert callable(lcri_lab.baseline_nonlinear_extrapolation_risk_summary)
    assert "baseline_nonlinear_extrapolation_risk" in lcri_lab.__all__
    assert "baseline_nonlinear_extrapolation_risk_summary" in lcri_lab.__all__


def test_public_api_exports_baseline_nonlinear_coefficient_stability_summary() -> None:
    assert callable(lcri_lab.baseline_nonlinear_coefficient_stability)
    assert callable(lcri_lab.baseline_nonlinear_coefficient_stability_summary)
    assert "baseline_nonlinear_coefficient_stability" in lcri_lab.__all__
    assert "baseline_nonlinear_coefficient_stability_summary" in lcri_lab.__all__


def test_public_api_exports_baseline_regime_publishability_summary() -> None:
    assert callable(lcri_lab.baseline_regime_publishability_summary)
    assert "baseline_regime_publishability_summary" in lcri_lab.__all__


def test_public_api_exports_baseline_regime_tail_lift_diagnostics() -> None:
    assert callable(lcri_lab.baseline_regime_tail_lift_diagnostics)
    assert callable(lcri_lab.baseline_regime_tail_lift_summary)
    assert "baseline_regime_tail_lift_diagnostics" in lcri_lab.__all__
    assert "baseline_regime_tail_lift_summary" in lcri_lab.__all__


def test_public_api_exports_execution_adjusted_lcri_quantile_diagnostics() -> None:
    assert callable(lcri_lab.execution_adjusted_lcri_quantile_diagnostics)
    assert "execution_adjusted_lcri_quantile_diagnostics" in lcri_lab.__all__


def test_public_api_exports_execution_adjusted_lcri_threshold_curve() -> None:
    assert callable(lcri_lab.execution_adjusted_lcri_threshold_curve)
    assert "execution_adjusted_lcri_threshold_curve" in lcri_lab.__all__


def test_public_api_exports_execution_adjusted_lcri_side_attribution() -> None:
    assert callable(lcri_lab.execution_adjusted_lcri_side_attribution)
    assert "execution_adjusted_lcri_side_attribution" in lcri_lab.__all__


def test_public_api_exports_execution_adjusted_edge_component_attribution() -> None:
    assert callable(lcri_lab.execution_adjusted_edge_component_attribution)
    assert "execution_adjusted_edge_component_attribution" in lcri_lab.__all__


def test_public_api_exports_execution_adjusted_edge_venue_economics_release_scorecard() -> None:
    assert callable(lcri_lab.execution_adjusted_edge_venue_economics_release_scorecard)
    assert "execution_adjusted_edge_venue_economics_release_scorecard" in lcri_lab.__all__


def test_public_api_exports_execution_adjusted_lcri_regime_attribution() -> None:
    assert callable(lcri_lab.execution_adjusted_lcri_regime_attribution)
    assert "execution_adjusted_lcri_regime_attribution" in lcri_lab.__all__


def test_public_api_exports_execution_adjusted_lcri_absorption_attribution() -> None:
    assert callable(lcri_lab.execution_adjusted_lcri_absorption_attribution)
    assert "execution_adjusted_lcri_absorption_attribution" in lcri_lab.__all__


def test_public_api_exports_queue_position_order_size_features() -> None:
    assert callable(lcri_lab.add_queue_position_order_size_features)
    assert "add_queue_position_order_size_features" in lcri_lab.__all__


def test_public_api_exports_queue_position_regime_capacity_frontier() -> None:
    assert callable(lcri_lab.queue_position_regime_capacity_frontier)
    assert "queue_position_regime_capacity_frontier" in lcri_lab.__all__


def test_public_api_exports_queue_position_regime_fraction_sweep() -> None:
    assert callable(lcri_lab.queue_position_regime_fraction_sweep)
    assert "queue_position_regime_fraction_sweep" in lcri_lab.__all__


def test_public_api_exports_queue_position_regime_capacity_concentration() -> None:
    assert callable(lcri_lab.queue_position_regime_capacity_concentration)
    assert "queue_position_regime_capacity_concentration" in lcri_lab.__all__


def test_public_api_exports_queue_position_toxicity_surface() -> None:
    assert callable(lcri_lab.queue_position_toxicity_surface)
    assert "queue_position_toxicity_surface" in lcri_lab.__all__


def test_public_api_exports_queue_position_trade_confirmation_surface() -> None:
    assert callable(lcri_lab.queue_position_trade_confirmation_surface)
    assert "queue_position_trade_confirmation_surface" in lcri_lab.__all__


def test_public_api_exports_queue_position_lcri_execution_interaction_surface() -> None:
    assert callable(lcri_lab.queue_position_lcri_execution_interaction_surface)
    assert "queue_position_lcri_execution_interaction_surface" in lcri_lab.__all__


def test_public_api_exports_queue_position_lcri_execution_interaction_release_scorecard() -> None:
    assert callable(lcri_lab.queue_position_lcri_execution_interaction_release_scorecard)
    assert "queue_position_lcri_execution_interaction_release_scorecard" in lcri_lab.__all__


def test_public_api_exports_queue_position_trade_confirmation_regime_scorecard() -> None:
    assert callable(lcri_lab.queue_position_trade_confirmation_regime_scorecard)
    assert "queue_position_trade_confirmation_regime_scorecard" in lcri_lab.__all__


def test_public_api_exports_queue_position_trade_confirmation_release_scorecard() -> None:
    assert callable(lcri_lab.queue_position_trade_confirmation_release_scorecard)
    assert "queue_position_trade_confirmation_release_scorecard" in lcri_lab.__all__


def test_public_api_exports_queue_position_trade_confirmation_competing_risk_curve() -> None:
    assert callable(lcri_lab.queue_position_trade_confirmation_competing_risk_curve)
    assert "queue_position_trade_confirmation_competing_risk_curve" in lcri_lab.__all__


def test_public_api_exports_queue_position_adverse_selection_policy_frontier() -> None:
    assert callable(lcri_lab.queue_position_adverse_selection_policy_frontier)
    assert callable(lcri_lab.queue_position_adverse_selection_policy_summary)
    assert "queue_position_adverse_selection_policy_frontier" in lcri_lab.__all__
    assert "queue_position_adverse_selection_policy_summary" in lcri_lab.__all__


def test_public_api_exports_queue_position_expected_value_frontier() -> None:
    assert callable(lcri_lab.queue_position_expected_value_frontier)
    assert callable(lcri_lab.queue_position_expected_value_frontier_efficiency)
    assert callable(lcri_lab.queue_position_expected_value_policy_selection)
    assert callable(lcri_lab.queue_position_expected_value_policy_drift)
    assert callable(lcri_lab.queue_position_expected_value_policy_scorecard)
    assert callable(lcri_lab.queue_position_expected_value_stress_table)
    assert callable(lcri_lab.queue_position_expected_value_stress_summary)
    assert "queue_position_expected_value_frontier" in lcri_lab.__all__
    assert "queue_position_expected_value_frontier_efficiency" in lcri_lab.__all__
    assert "queue_position_expected_value_policy_drift" in lcri_lab.__all__
    assert "queue_position_expected_value_policy_selection" in lcri_lab.__all__
    assert "queue_position_expected_value_policy_scorecard" in lcri_lab.__all__
    assert "queue_position_expected_value_stress_table" in lcri_lab.__all__
    assert "queue_position_expected_value_stress_summary" in lcri_lab.__all__


def test_public_api_exports_queue_position_calibration_stability() -> None:
    assert callable(lcri_lab.queue_position_calibration_stability)
    assert callable(lcri_lab.queue_position_calibration_stability_summary)
    assert "queue_position_calibration_stability" in lcri_lab.__all__
    assert "queue_position_calibration_stability_summary" in lcri_lab.__all__


def test_public_api_exports_queue_position_calibration_reliability_scorecard() -> None:
    assert callable(lcri_lab.queue_position_calibration_reliability_scorecard)
    assert "queue_position_calibration_reliability_scorecard" in lcri_lab.__all__


def test_public_api_exports_queue_position_latency_release_scorecard() -> None:
    assert callable(lcri_lab.queue_position_latency_edge_regime_surface)
    assert callable(lcri_lab.queue_position_latency_edge_survival)
    assert callable(lcri_lab.queue_position_latency_edge_survival_scorecard)
    assert callable(lcri_lab.queue_position_latency_release_scorecard)
    assert "queue_position_latency_edge_regime_surface" in lcri_lab.__all__
    assert "queue_position_latency_edge_survival" in lcri_lab.__all__
    assert "queue_position_latency_edge_survival_scorecard" in lcri_lab.__all__
    assert "queue_position_latency_release_scorecard" in lcri_lab.__all__


def test_public_api_exports_trade_confirmed_passive_fill_latency_summary() -> None:
    assert callable(lcri_lab.trade_confirmed_passive_fill_latency_summary)
    assert "trade_confirmed_passive_fill_latency_summary" in lcri_lab.__all__


def test_public_api_exports_passive_fill_event_lead_lag_scorecard() -> None:
    assert callable(lcri_lab.passive_fill_event_lead_lag_scorecard)
    assert "passive_fill_event_lead_lag_scorecard" in lcri_lab.__all__


def test_public_api_exports_passive_fill_event_lifecycle_policy_curve() -> None:
    assert callable(lcri_lab.passive_fill_event_lifecycle_policy_curve)
    assert "passive_fill_event_lifecycle_policy_curve" in lcri_lab.__all__


def test_public_api_exports_passive_fill_event_lifecycle_scorecard() -> None:
    assert callable(lcri_lab.passive_fill_event_lifecycle_scorecard)
    assert "passive_fill_event_lifecycle_scorecard" in lcri_lab.__all__


def test_public_api_exports_signal_lift_summary() -> None:
    assert callable(lcri_lab.summarize_signal_lift)
    assert "summarize_signal_lift" in lcri_lab.__all__


def test_public_api_exports_signal_generalization_gap() -> None:
    assert callable(lcri_lab.signal_generalization_gap)
    assert "signal_generalization_gap" in lcri_lab.__all__


def test_public_api_exports_generalization_overview() -> None:
    assert callable(lcri_lab.generalization_overview)
    assert "generalization_overview" in lcri_lab.__all__


def test_public_api_exports_lcri_generalization_gap_delta() -> None:
    assert callable(lcri_lab.lcri_generalization_gap_delta)
    assert "lcri_generalization_gap_delta" in lcri_lab.__all__


def test_public_api_exports_lcri_generalization_blocker_summary() -> None:
    assert callable(lcri_lab.lcri_generalization_blocker_summary)
    assert "lcri_generalization_blocker_summary" in lcri_lab.__all__


def test_public_api_exports_lcri_generalization_critical_contexts() -> None:
    assert callable(lcri_lab.lcri_generalization_critical_contexts)
    assert "lcri_generalization_critical_contexts" in lcri_lab.__all__


def test_public_api_exports_lcri_generalization_gap_leaderboard() -> None:
    assert callable(lcri_lab.lcri_generalization_gap_leaderboard)
    assert "lcri_generalization_gap_leaderboard" in lcri_lab.__all__


def test_public_api_exports_lcri_generalization_scope_gate_decisions() -> None:
    assert callable(lcri_lab.lcri_generalization_scope_gate_decisions)
    assert "lcri_generalization_scope_gate_decisions" in lcri_lab.__all__


def test_public_api_exports_lcri_generalization_scope_risk() -> None:
    assert callable(lcri_lab.lcri_generalization_scope_risk)
    assert "lcri_generalization_scope_risk" in lcri_lab.__all__


def test_public_api_exports_lcri_generalization_scope_summary() -> None:
    assert callable(lcri_lab.lcri_generalization_scope_summary)
    assert "lcri_generalization_scope_summary" in lcri_lab.__all__


def test_public_api_exports_lcri_generalization_severity() -> None:
    assert callable(lcri_lab.lcri_generalization_severity)
    assert "lcri_generalization_severity" in lcri_lab.__all__


def test_public_api_exports_lcri_generalization_severity_by_scope() -> None:
    assert callable(lcri_lab.lcri_generalization_severity_by_scope)
    assert "lcri_generalization_severity_by_scope" in lcri_lab.__all__


def test_public_api_exports_lcri_generalization_severity_summary() -> None:
    assert callable(lcri_lab.lcri_generalization_severity_summary)
    assert "lcri_generalization_severity_summary" in lcri_lab.__all__


def test_public_api_exports_lcri_worst_generalization_context() -> None:
    assert callable(lcri_lab.lcri_worst_generalization_context)
    assert "lcri_worst_generalization_context" in lcri_lab.__all__


def test_public_api_exports_lcri_generalization_gate_decision() -> None:
    assert callable(lcri_lab.lcri_generalization_gate_decision)
    assert "lcri_generalization_gate_decision" in lcri_lab.__all__


def test_public_api_exports_lcri_gap_delta_summary() -> None:
    assert callable(lcri_lab.lcri_gap_delta_summary)
    assert "lcri_gap_delta_summary" in lcri_lab.__all__


def test_public_api_exports_lcri_scope_gate_decision_summary() -> None:
    assert callable(lcri_lab.lcri_scope_gate_decision_summary)
    assert "lcri_scope_gate_decision_summary" in lcri_lab.__all__


def test_public_api_exports_lcri_gap_delta_dominant_scopes() -> None:
    assert callable(lcri_lab.lcri_gap_delta_dominant_scopes)
    assert "lcri_gap_delta_dominant_scopes" in lcri_lab.__all__


def test_public_api_exports_lcri_gap_delta_improvements() -> None:
    assert callable(lcri_lab.lcri_gap_delta_improvements)
    assert "lcri_gap_delta_improvements" in lcri_lab.__all__


def test_public_api_exports_lcri_gap_delta_regressions() -> None:
    assert callable(lcri_lab.lcri_gap_delta_regressions)
    assert "lcri_gap_delta_regressions" in lcri_lab.__all__


def test_public_api_exports_lcri_gap_delta_scope_summary() -> None:
    assert callable(lcri_lab.lcri_gap_delta_scope_summary)
    assert "lcri_gap_delta_scope_summary" in lcri_lab.__all__


def test_public_api_exports_lcri_gap_delta_scorecard() -> None:
    assert callable(lcri_lab.lcri_gap_delta_scorecard)
    assert "lcri_gap_delta_scorecard" in lcri_lab.__all__


def test_public_api_exports_lcri_gap_delta_flags() -> None:
    assert callable(lcri_lab.lcri_gap_delta_flags)
    assert "lcri_gap_delta_flags" in lcri_lab.__all__


def test_public_api_exports_regime_generalization_gap() -> None:
    assert callable(lcri_lab.regime_generalization_gap)
    assert "regime_generalization_gap" in lcri_lab.__all__


def test_public_api_exports_transition_generalization_gap() -> None:
    assert callable(lcri_lab.transition_generalization_gap)
    assert "transition_generalization_gap" in lcri_lab.__all__


def test_public_api_exports_regime_tagger() -> None:
    assert callable(lcri_lab.tag_liquidity_regimes)
    assert "tag_liquidity_regimes" in lcri_lab.__all__


def test_public_api_exports_cost_aware_labels() -> None:
    assert callable(lcri_lab.add_transaction_cost_labels)
    assert "add_transaction_cost_labels" in lcri_lab.__all__


def test_public_api_exports_cost_aware_evaluation() -> None:
    assert callable(lcri_lab.evaluate_cost_aware_signals)
    assert "evaluate_cost_aware_signals" in lcri_lab.__all__


def test_public_api_exports_phase_shift_artifact_classifier() -> None:
    assert callable(lcri_lab.classify_phase_shift_artifacts)
    assert "classify_phase_shift_artifacts" in lcri_lab.__all__


def test_public_api_exports_transition_conditioned_metrics() -> None:
    assert callable(lcri_lab.transition_conditioned_metrics)
    assert "transition_conditioned_metrics" in lcri_lab.__all__


def test_public_api_exports_transition_signal_lift() -> None:
    assert callable(lcri_lab.transition_signal_lift)
    assert "transition_signal_lift" in lcri_lab.__all__


def test_public_api_exports_transition_robustness_summary() -> None:
    assert callable(lcri_lab.transition_robustness_summary)
    assert "transition_robustness_summary" in lcri_lab.__all__


def test_public_api_exports_reversal_transition_gate_diagnostics() -> None:
    assert callable(lcri_lab.reversal_transition_gate_diagnostics)
    assert "reversal_transition_gate_diagnostics" in lcri_lab.__all__


def test_public_api_exports_alpha_event_score_weighted_drift() -> None:
    assert callable(lcri_lab.alpha_event_score_weighted_drift)
    assert "alpha_event_score_weighted_drift" in lcri_lab.__all__


def test_public_api_exports_alpha_event_lifecycle_summary() -> None:
    assert callable(lcri_lab.alpha_event_window_lifecycle_summary)
    assert "alpha_event_window_lifecycle_summary" in lcri_lab.__all__


def test_public_api_exports_alpha_event_release_review_packet() -> None:
    assert callable(lcri_lab.alpha_event_release_review_packet)
    assert "alpha_event_release_review_packet" in lcri_lab.__all__


def test_public_api_exports_execution_adjusted_queue_fill_tools() -> None:
    assert lcri_lab.FillProbabilityConfig().queue_position_fraction == 0.50
    assert callable(lcri_lab.add_queue_position_features)
    assert callable(lcri_lab.add_passive_fill_probabilities)
    assert callable(lcri_lab.add_execution_adjusted_edge)
    assert callable(lcri_lab.execution_adjusted_edge_summary)
    assert callable(lcri_lab.execution_publishability_review_packet)
    assert callable(lcri_lab.execution_publishability_release_gate)
    assert callable(lcri_lab.passive_fill_calibration_curve)
    assert callable(lcri_lab.passive_fill_calibration_summary)
    assert callable(lcri_lab.passive_fill_event_window_diagnostics)
    assert callable(lcri_lab.passive_fill_event_window_sensitivity)
    assert callable(lcri_lab.passive_fill_event_regime_summary)
    assert callable(lcri_lab.passive_fill_event_transition_policy_curve)
    assert callable(lcri_lab.passive_fill_event_policy_stability_scorecard)
    assert callable(lcri_lab.queue_position_fill_calibration_surface)
    assert callable(lcri_lab.queue_position_fill_surface)
    assert callable(lcri_lab.queue_position_capacity_stability)
    assert callable(lcri_lab.queue_position_calibration_drift)
    assert callable(lcri_lab.queue_position_calibration_residual_summary)
    assert callable(lcri_lab.queue_position_edge_decay)
    assert callable(lcri_lab.queue_position_latency_regime_surface)
    assert callable(lcri_lab.queue_position_execution_readiness_scorecard)
    assert callable(lcri_lab.queue_position_execution_quality_gate)
    assert "FillProbabilityConfig" in lcri_lab.__all__
    assert "add_queue_position_features" in lcri_lab.__all__
    assert "add_passive_fill_probabilities" in lcri_lab.__all__
    assert "add_execution_adjusted_edge" in lcri_lab.__all__
    assert "execution_adjusted_edge_summary" in lcri_lab.__all__
    assert "execution_publishability_review_packet" in lcri_lab.__all__
    assert "execution_publishability_release_gate" in lcri_lab.__all__
    assert "passive_fill_calibration_curve" in lcri_lab.__all__
    assert "passive_fill_calibration_summary" in lcri_lab.__all__
    assert "passive_fill_event_window_diagnostics" in lcri_lab.__all__
    assert "passive_fill_event_window_sensitivity" in lcri_lab.__all__
    assert "passive_fill_event_regime_summary" in lcri_lab.__all__
    assert "passive_fill_event_transition_policy_curve" in lcri_lab.__all__
    assert "passive_fill_event_policy_stability_scorecard" in lcri_lab.__all__
    assert "queue_position_fill_calibration_surface" in lcri_lab.__all__
    assert "queue_position_fill_surface" in lcri_lab.__all__
    assert "queue_position_capacity_stability" in lcri_lab.__all__
    assert "queue_position_calibration_drift" in lcri_lab.__all__
    assert "queue_position_calibration_residual_summary" in lcri_lab.__all__
    assert "queue_position_edge_decay" in lcri_lab.__all__
    assert "queue_position_latency_regime_surface" in lcri_lab.__all__
    assert "queue_position_execution_readiness_scorecard" in lcri_lab.__all__
    assert "queue_position_execution_quality_gate" in lcri_lab.__all__
