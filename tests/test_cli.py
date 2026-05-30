import json
from pathlib import Path

import pandas as pd
import pytest

from lcri_lab.cli import describe_model, fit_model, normalize_snapshots, score_model, verify_report
from lcri_lab.simulator import SimulationConfig, simulate_order_books


def _write_snapshots(path: Path, rows: int = 150) -> None:
    simulate_order_books(SimulationConfig(rows=rows, seed=13)).to_csv(path, index=False)


def test_normalize_snapshots_writes_derived_state(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.csv"
    output_path = tmp_path / "normalized.csv"
    pd.DataFrame(
        {
            "bid_px_1": [99.99, 100.00],
            "ask_px_1": [100.01, 100.03],
            "bid_sz_1": [10.0, 12.0],
            "ask_sz_1": [9.0, 11.0],
        }
    ).to_csv(input_path, index=False)

    normalize_snapshots(input_path, output_path, tick_size=0.01, levels=1, derive_state=True)

    columns = pd.read_csv(output_path).columns
    assert {"mid", "spread_ticks", "volatility", "replenishment_rate"}.issubset(columns)


def test_fit_model_persists_requested_ridge(tmp_path: Path) -> None:
    snapshots = tmp_path / "snapshots.csv"
    model_path = tmp_path / "model.json"
    _write_snapshots(snapshots)

    fit_model(snapshots, model_path, levels=5, ridge=0.25, probability_scale=2.5)

    payload = json.loads(model_path.read_text())
    assert payload["config"]["ridge"] == 0.25
    assert payload["config"]["probability_scale"] == 2.5


def test_describe_model_prints_artifact_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshots = tmp_path / "snapshots.csv"
    model_path = tmp_path / "model.json"
    _write_snapshots(snapshots)
    fit_model(snapshots, model_path, levels=5, ridge=0.25)

    describe_model(model_path)
    output = capsys.readouterr().out

    assert "schema_version: 3" in output
    assert "levels: 5" in output
    assert "features: 18" in output


def test_score_model_writes_selected_columns(tmp_path: Path) -> None:
    snapshots = tmp_path / "snapshots.csv"
    model_path = tmp_path / "model.json"
    output_path = tmp_path / "scores.csv"
    _write_snapshots(snapshots)
    fit_model(snapshots, model_path, levels=5)

    score_model(snapshots, model_path, output_path, columns=["timestamp", "lcri", "lcri_probability"])

    assert list(pd.read_csv(output_path).columns) == ["timestamp", "lcri", "lcri_probability"]


def test_score_model_rejects_empty_column_selection(tmp_path: Path) -> None:
    snapshots = tmp_path / "snapshots.csv"
    model_path = tmp_path / "model.json"
    output_path = tmp_path / "scores.csv"
    _write_snapshots(snapshots)
    fit_model(snapshots, model_path, levels=5)

    with pytest.raises(ValueError, match="at least one column"):
        score_model(snapshots, model_path, output_path, columns=[" ", ""])


def test_score_model_rejects_duplicate_column_selection(tmp_path: Path) -> None:
    snapshots = tmp_path / "snapshots.csv"
    model_path = tmp_path / "model.json"
    output_path = tmp_path / "scores.csv"
    _write_snapshots(snapshots)
    fit_model(snapshots, model_path, levels=5)

    with pytest.raises(ValueError, match="must be unique"):
        score_model(snapshots, model_path, output_path, columns=["timestamp", "lcri", " timestamp "])


def test_verify_report_accepts_intact_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact = tmp_path / "metrics.csv"
    artifact.write_text("signal,value\n")
    overview = tmp_path / "generalization_overview.json"
    overview.write_text(
        json.dumps(
            {
                "signal_rows": 2,
                "regime_rows": 4,
                "transition_rows": 4,
                "max_signal_directional_accuracy_gap": 0.05,
                "max_regime_directional_accuracy_gap": 0.08,
                "max_transition_directional_accuracy_gap": 0.04,
            }
        )
    )
    (tmp_path / "generalization_fragility_diagnostics.csv").write_text(
        "scope,context,signal,full_rows,heldout_rows,full_directional_accuracy,"
        "heldout_directional_accuracy,directional_accuracy_gap,"
        "heldout_directional_accuracy_se,abs_gap_to_se_ratio,fragility_label\n"
        "signal,all,lcri,100,20,0.65,0.55,0.10,0.11,0.9090909090909092,stable\n"
    )
    (tmp_path / "generalization_fragility_summary.json").write_text(
        json.dumps(
            {
                "rows": 1,
                "stable_rows": 1,
                "watch_rows": 0,
                "fragile_rows": 0,
                "max_abs_gap_to_se_ratio": 0.9090909090909092,
                "most_fragile_context": "signal:all:lcri",
            }
        )
    )
    (tmp_path / "generalization_stability_confidence_intervals.csv").write_text(
        "scope,context,signal,heldout_rows,heldout_directional_accuracy,"
        "heldout_directional_accuracy_se,confidence_level,"
        "heldout_directional_accuracy_ci_lower,heldout_directional_accuracy_ci_upper,"
        "heldout_directional_accuracy_ci_width,directional_accuracy_gap,"
        "gap_exceeds_ci_half_width\n"
        "signal,all,lcri,20,0.55,0.11,0.950004209703559,"
        "0.3344,0.7656000000000001,0.43120000000000014,0.10,False\n"
    )
    (tmp_path / "generalization_stability_confidence_summary.json").write_text(
        json.dumps(
            {
                "rows": 1,
                "gap_exceeds_ci_half_width_rows": 0,
                "mean_ci_width": 0.43120000000000014,
                "max_ci_width": 0.43120000000000014,
                "widest_interval_context": "signal:all:lcri",
            }
        )
    )
    leaderboard = tmp_path / "lcri_generalization_gap_leaderboard.csv"
    leaderboard.write_text(
        "scope,context,signal,directional_accuracy_gap\n"
        "signal,all,lcri,0.05\n"
    )
    scope_summary = tmp_path / "lcri_generalization_scope_summary.csv"
    scope_summary.write_text(
        "scope,rows,mean_directional_accuracy_gap,max_directional_accuracy_gap\n"
        "signal,1,0.05,0.05\n"
    )
    severity = tmp_path / "lcri_generalization_severity.csv"
    severity.write_text(
        "scope,context,directional_accuracy_gap,severity\n"
        "signal,all,0.05,critical\n"
    )
    (tmp_path / "lcri_fragility_gate_alignment.csv").write_text(
        "scope,context,directional_accuracy_gap,severity,heldout_rows,"
        "heldout_directional_accuracy_se,abs_gap_to_se_ratio,fragility_label,"
        "alignment_label,review_note\n"
        "signal,all,0.05,critical,20,0.11,0.9090909090909092,stable,"
        "gate_blocks_stable_slice,critical gate blocker exceeds deterministic threshold\n"
    )
    (tmp_path / "lcri_fragility_gate_scorecard.json").write_text(
        json.dumps(
            {
                "rows": 1,
                "aligned_rows": 0,
                "review_required_rows": 1,
                "gate_blocks_stable_slice_rows": 1,
                "uncertainty_fragile_noncritical_rows": 0,
                "uncertainty_watch_stable_gap_rows": 0,
                "critical_rows": 1,
                "critical_stable_slice_share": 1.0,
                "max_abs_gap_to_se_ratio": 0.9090909090909092,
                "worst_review_context": "signal:all:gate_blocks_stable_slice",
            }
        )
    )
    (tmp_path / "lcri_generalization_severity_by_scope.csv").write_text(
        "scope,rows,stable_rows,warning_rows,critical_rows\nsignal,1,0,0,1\n"
    )
    (tmp_path / "lcri_generalization_scope_risk.csv").write_text(
        "scope,rows,warning_or_critical_share,critical_share\nsignal,1,1.0,1.0\n"
    )
    (tmp_path / "lcri_generalization_scope_gate_decisions.csv").write_text(
        "scope,rows,decision,reason\nsignal,1,block,signal blocked\n"
    )
    (tmp_path / "lcri_generalization_critical_contexts.csv").write_text(
        "scope,context,directional_accuracy_gap,severity\nsignal,all,0.05,critical\n"
    )
    blocker_summary = tmp_path / "lcri_generalization_blocker_summary.json"
    blocker_summary.write_text(
        json.dumps(
            {
                "critical_rows": 1,
                "critical_scopes": "signal",
                "max_critical_gap": 0.05,
                "max_critical_context": "signal:all",
            }
        )
    )
    severity_summary = tmp_path / "lcri_generalization_severity_summary.json"
    severity_summary.write_text(
        json.dumps(
            {
                "rows": 1,
                "stable_rows": 0,
                "warning_rows": 0,
                "critical_rows": 1,
                "passes_lcri_generalization_gate": False,
            }
        )
    )
    worst_context = tmp_path / "lcri_worst_generalization_context.json"
    worst_context.write_text(
        json.dumps(
            {
                "scope": "signal",
                "context": "all",
                "directional_accuracy_gap": 0.05,
            }
        )
    )
    gate_decision = tmp_path / "lcri_generalization_gate_decision.json"
    gate_decision.write_text(
        json.dumps(
            {
                "passes": False,
                "decision": "block",
                "rows_evaluated": 1,
                "warning_rows": 0,
                "critical_rows": 1,
                "worst_scope": "signal",
                "worst_context": "all",
                "worst_directional_accuracy_gap": 0.05,
                "reason": "blocked by 1 critical LCRI generalization rows",
            }
        )
    )
    (tmp_path / "lcri_generalization_scope_gate_decision_summary.json").write_text(
        json.dumps(
            {
                "scopes": 1,
                "pass_scopes": 0,
                "warn_scopes": 0,
                "block_scopes": 1,
                "blocked_scope_names": "signal",
                "warn_scope_names": "none",
            }
        )
    )
    delta = tmp_path / "lcri_generalization_gap_delta.csv"
    delta.write_text(
        "scope,context,raw_imbalance_directional_accuracy_gap,"
        "lcri_directional_accuracy_gap,raw_minus_lcri_directional_accuracy_gap\n"
        "signal,all,0.08,0.05,0.03\n"
    )
    (tmp_path / "lcri_gap_delta_dominant_scopes.json").write_text(
        json.dumps(
            {
                "best_scope": "signal",
                "best_mean_raw_minus_lcri_gap": 0.03,
                "worst_scope": "signal",
                "worst_mean_raw_minus_lcri_gap": 0.03,
            }
        )
    )
    flags = tmp_path / "lcri_gap_delta_flags.csv"
    flags.write_text(
        "scope,context,raw_minus_lcri_directional_accuracy_gap,stability_flag\n"
        "signal,all,0.03,lcri_more_stable\n"
    )
    (tmp_path / "lcri_gap_delta_improvements.csv").write_text(
        "scope,context,raw_minus_lcri_directional_accuracy_gap\nsignal,all,0.03\n"
    )
    (tmp_path / "lcri_gap_delta_regressions.csv").write_text(
        "scope,context,raw_minus_lcri_directional_accuracy_gap\n"
    )
    scorecard = tmp_path / "lcri_gap_delta_scorecard.json"
    scorecard.write_text(
        json.dumps(
            {
                "rows": 1,
                "mean_raw_minus_lcri_directional_accuracy_gap": 0.03,
                "median_raw_minus_lcri_directional_accuracy_gap": 0.03,
                "lcri_more_stable_share": 1.0,
                "lcri_less_stable_share": 0.0,
            }
        )
    )
    (tmp_path / "lcri_gap_delta_scope_extremes.csv").write_text(
        "scope,best_context,best_raw_minus_lcri_gap,worst_context,worst_raw_minus_lcri_gap\n"
        "signal,all,0.03,all,0.03\n"
    )
    (tmp_path / "lcri_gap_delta_scope_summary.csv").write_text(
        "scope,rows,mean_raw_minus_lcri_gap,min_raw_minus_lcri_gap,"
        "max_raw_minus_lcri_gap,lcri_more_stable_share,lcri_less_stable_share\n"
        "signal,1,0.03,0.03,0.03,1.0,0.0\n"
    )
    summary = tmp_path / "lcri_gap_delta_summary.json"
    summary.write_text(
        json.dumps(
            {
                "rows": 1,
                "lcri_more_stable_rows": 1,
                "lcri_less_stable_rows": 0,
                "lcri_equal_stability_rows": 0,
                "max_lcri_stability_edge": 0.03,
                "max_lcri_stability_edge_context": "signal:all",
                "max_lcri_instability_edge": 0.03,
                "max_lcri_instability_edge_context": "signal:all",
            }
        )
    )
    (tmp_path / "lcri_scope_stability_contradictions.csv").write_text(
        "scope,decision,rows,lcri_more_stable_share,lcri_less_stable_share,"
        "fragility_review_required_rows,contradiction_label,review_note\n"
        "signal,block,1,1.0,0.0,1,gate_blocks_despite_relative_stability,"
        "absolute LCRI gate blocks while LCRI is usually more stable than raw imbalance in this scope\n"
    )
    (tmp_path / "lcri_scope_stability_contradiction_summary.json").write_text(
        json.dumps(
            {
                "scopes": 1,
                "aligned_scopes": 0,
                "contradiction_scopes": 1,
                "gate_blocks_despite_relative_stability_scopes": 1,
                "pass_scope_with_relative_regressions_scopes": 0,
                "warning_scope_with_broad_relative_regression_scopes": 0,
                "fragility_review_required_rows": 1,
                "worst_contradiction_scope": "signal:gate_blocks_despite_relative_stability",
            }
        )
    )
    (tmp_path / "lcri_contradiction_review_packet.csv").write_text(
        "scope,contradiction_label,decision,scope_rows,lcri_less_stable_share,"
        "fragility_review_required_rows,worst_gate_context,worst_gate_severity,"
        "worst_gate_directional_accuracy_gap,worst_delta_context,"
        "worst_raw_minus_lcri_directional_accuracy_gap,worst_fragility_context,"
        "worst_fragility_alignment_label,worst_fragility_abs_gap_to_se_ratio,review_priority\n"
        "signal,gate_blocks_despite_relative_stability,block,1,0.0,1,all,critical,"
        "0.05,all,0.03,all,gate_blocks_stable_slice,0.9090909090909092,3\n"
    )
    (tmp_path / "lcri_contradiction_review_packet_summary.json").write_text(
        json.dumps(
            {
                "scopes": 1,
                "high_priority_scopes": 1,
                "medium_priority_scopes": 0,
                "low_priority_scopes": 0,
                "fragility_review_required_rows": 1,
                "max_review_priority": 3,
                "worst_review_scope": "signal:gate_blocks_despite_relative_stability",
                "worst_fragility_scope": "signal:0.909091",
            }
        )
    )
    (tmp_path / "lcri_uncertainty_weighted_review_priority.csv").write_text(
        "scope,contradiction_label,base_review_priority,fragility_review_required_rows,"
        "worst_fragility_abs_gap_to_se_ratio,coverage_label,mean_ci_width,max_ci_width,"
        "wide_ci_share,ci_gate_contradiction_rows,high_priority_ci_gate_rows,"
        "uncertainty_weighted_priority,priority_label,review_note\n"
        "signal,gate_blocks_despite_relative_stability,3,1,0.9090909090909092,"
        "missing_ci_coverage,0.0,0.0,0.0,0,0,3.4545454545454546,medium,"
        "schedule review after critical/high scopes; missing_ci_coverage uncertainty evidence is non-trivial\n"
    )
    (tmp_path / "lcri_uncertainty_weighted_review_priority_summary.json").write_text(
        json.dumps(
            {
                "scopes": 1,
                "critical_priority_scopes": 0,
                "high_priority_scopes": 0,
                "medium_priority_scopes": 1,
                "low_priority_scopes": 0,
                "max_uncertainty_weighted_priority": 3.4545454545454546,
                "worst_uncertainty_weighted_scope": "signal:medium",
            }
        )
    )
    manifest = {
        "artifacts": [
            "metrics.csv",
            "generalization_fragility_diagnostics.csv",
            "generalization_fragility_summary.json",
            "generalization_stability_confidence_intervals.csv",
            "generalization_stability_confidence_summary.json",
            "generalization_overview.json",
            "lcri_generalization_gap_leaderboard.csv",
            "lcri_generalization_scope_summary.csv",
            "lcri_generalization_severity.csv",
            "lcri_fragility_gate_alignment.csv",
            "lcri_fragility_gate_scorecard.json",
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
        ],
        "artifact_metadata": {},
    }
    (tmp_path / "artifact_manifest.json").write_text(json.dumps(manifest))

    verify_report(tmp_path)

    captured = capsys.readouterr()
    assert "Verified report artifacts" in captured.out
    assert "verification summary" in captured.out
    assert "passes_verification" in captured.out


def test_verify_report_checks_execution_publishability_packets(tmp_path: Path) -> None:
    (tmp_path / "execution_publishability_review_packet.csv").write_text(
        "publishable_side,best_execution_side,rows,conflict_rows,conflict_share,"
        "mean_execution_adjusted_edge_ticks,mean_best_fill_probability,"
        "mean_best_adverse_fill_probability,mean_publishable_fill_probability,"
        "mean_edge_drag_ticks,review_priority,review_note\n"
        "long,short,3,4,1.2,0.1,0.4,0.2,0.3,0.1,3,bad bounds\n"
    )
    (tmp_path / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "artifacts": ["execution_publishability_review_packet.csv"],
                "artifact_metadata": {},
            }
        )
    )

    with pytest.raises(ValueError, match="execution publishability"):
        verify_report(tmp_path)


def test_verify_report_checks_execution_adjusted_lcri_side_attribution(tmp_path: Path) -> None:
    (tmp_path / "execution_adjusted_lcri_side_attribution.csv").write_text(
        "lcri_side,rows,tradable_rows,execution_conflict_rows,execution_conflict_share,"
        "mean_signal_confidence,mean_execution_adjusted_edge_ticks,"
        "mean_fill_probability_advantage,mean_adverse_fill_probability_advantage,"
        "dominant_execution_side,review_label\n"
        "long,2,3,4,1.4,1.2,0.1,0.2,0.3,sideways,unknown\n"
    )
    (tmp_path / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "artifacts": ["execution_adjusted_lcri_side_attribution.csv"],
                "artifact_metadata": {},
            }
        )
    )

    with pytest.raises(ValueError, match="execution-adjusted LCRI side attribution"):
        verify_report(tmp_path)


def test_verify_report_checks_passive_fill_threshold_policy_curve(tmp_path: Path) -> None:
    (tmp_path / "passive_fill_threshold_policy_curve.csv").write_text(
        "threshold,candidate_rows,trade_share,long_rows,short_rows,"
        "mean_predicted_fill_probability,realized_fill_rate,weighted_brier_score,"
        "mean_realized_edge_ticks,positive_edge_rate,"
        "mean_execution_adjusted_edge_ticks,policy_label\n"
        "1.2,2,1.4,1,1,0.6,0.7,0.2,0.1,0.5,0.08,unknown\n"
    )
    (tmp_path / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "artifacts": ["passive_fill_threshold_policy_curve.csv"],
                "artifact_metadata": {},
            }
        )
    )

    with pytest.raises(ValueError, match="passive fill threshold policy"):
        verify_report(tmp_path)


def test_verify_report_checks_passive_fill_event_transition_policy_curve(tmp_path: Path) -> None:
    (tmp_path / "passive_fill_event_transition_policy_curve.csv").write_text(
        "regime_transition,threshold,total_events,candidate_events,event_share,"
        "mean_event_fill_probability,mean_event_adverse_fill_probability,"
        "mean_event_edge_ticks,mean_post_minus_pre_realized_edge,"
        "adverse_post_edge_share,policy_label\n"
        "calm->thin,1.2,2,-1,1.4,0.6,0.7,0.1,-0.2,1.2,unknown\n"
    )
    (tmp_path / "artifact_manifest.json").write_text(
        json.dumps(
            {
                "artifacts": ["passive_fill_event_transition_policy_curve.csv"],
                "artifact_metadata": {},
            }
        )
    )

    with pytest.raises(ValueError, match="passive fill event transition policy"):
        verify_report(tmp_path)


def test_verify_report_error_includes_summary(tmp_path: Path) -> None:
    (tmp_path / "artifact_manifest.json").write_text(json.dumps({"artifacts": []}))

    with pytest.raises(ValueError, match="passes_verification"):
        verify_report(tmp_path)


def test_verify_report_rejects_changed_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "metrics.csv"
    artifact.write_text("signal,value\n")
    manifest = {
        "artifacts": ["metrics.csv"],
        "artifact_metadata": {
            "metrics.csv": {
                "size_bytes": artifact.stat().st_size,
                "sha256": "0" * 64,
            }
        },
    }
    (tmp_path / "artifact_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="sha256 mismatch"):
        verify_report(tmp_path)


def test_verify_report_rejects_corrupt_figure_artifact(tmp_path: Path) -> None:
    figure = tmp_path / "figures" / "gap.png"
    figure.parent.mkdir()
    figure.write_bytes(b"not a png")
    manifest = {"artifacts": ["figures/gap.png"], "artifact_metadata": {}}
    (tmp_path / "artifact_manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="invalid figure artifact"):
        verify_report(tmp_path)


def test_score_model_rejects_unknown_columns(tmp_path: Path) -> None:
    snapshots = tmp_path / "snapshots.csv"
    model_path = tmp_path / "model.json"
    _write_snapshots(snapshots)
    fit_model(snapshots, model_path, levels=5)

    with pytest.raises(ValueError, match="unavailable"):
        score_model(snapshots, model_path, tmp_path / "scores.csv", columns=["missing"])
