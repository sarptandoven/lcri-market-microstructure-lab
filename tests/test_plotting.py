from pathlib import Path

import pandas as pd
import pytest

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
    assert not (tmp_path / "generalization_fragility_diagnostics.png").exists()
    assert not (tmp_path / "generalization_stability_confidence_intervals.png").exists()
    assert not (tmp_path / "lcri_ci_gate_contradiction_diagnostics.png").exists()
    assert not (tmp_path / "lcri_ci_confidence_coverage_scorecard.png").exists()
    assert not (tmp_path / "lcri_uncertainty_weighted_review_priority.png").exists()
    assert not (tmp_path / "lcri_cross_artifact_evidence_index.png").exists()
    assert not (tmp_path / "lcri_evidence_release_checklist.png").exists()
    assert not (tmp_path / "lcri_owner_handoff_packet.png").exists()
    assert not (tmp_path / "lcri_evidence_lineage_map.png").exists()
    assert not (tmp_path / "lcri_calibration_fracture_pressure.png").exists()
    assert not (tmp_path / "lcri_reversal_transition_gate.png").exists()
    assert not (tmp_path / "pressure_memory_decay_fracture.png").exists()


def test_write_figures_rejects_empty_inputs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty frame"):
        write_figures(pd.DataFrame(), _regime_table(), tmp_path)
    with pytest.raises(ValueError, match="empty regime"):
        write_figures(_scored_frame(), pd.DataFrame(), tmp_path)


def test_write_figures_writes_generalization_fragility_plot(tmp_path: Path) -> None:
    frame = _scored_frame()
    regime_table = _regime_table()

    write_figures(
        frame,
        regime_table,
        tmp_path,
        generalization_fragility_diagnostics=_fragility_diagnostics(),
    )

    assert (tmp_path / "generalization_fragility_diagnostics.png").exists()


def test_write_figures_writes_generalization_stability_confidence_interval_plot(tmp_path: Path) -> None:
    frame = _scored_frame()
    regime_table = _regime_table()

    write_figures(
        frame,
        regime_table,
        tmp_path,
        generalization_stability_confidence_intervals=_stability_confidence_intervals(),
    )

    assert (tmp_path / "generalization_stability_confidence_intervals.png").exists()


def test_write_figures_writes_lcri_gap_delta_scope_extremes_plot(tmp_path: Path) -> None:
    frame = _scored_frame()
    regime_table = _regime_table()

    write_figures(
        frame,
        regime_table,
        tmp_path,
        lcri_gap_delta_scope_extremes=_gap_delta_scope_extremes(),
    )

    assert (tmp_path / "lcri_gap_delta_scope_extremes.png").exists()


def test_write_figures_writes_lcri_ci_gate_contradiction_plot(tmp_path: Path) -> None:
    frame = _scored_frame()
    regime_table = _regime_table()

    write_figures(
        frame,
        regime_table,
        tmp_path,
        lcri_ci_gate_contradiction_diagnostics=_ci_gate_contradiction_diagnostics(),
    )

    assert (tmp_path / "lcri_ci_gate_contradiction_diagnostics.png").exists()


def test_write_figures_writes_lcri_ci_confidence_coverage_plot(tmp_path: Path) -> None:
    frame = _scored_frame()
    regime_table = _regime_table()

    write_figures(
        frame,
        regime_table,
        tmp_path,
        lcri_ci_confidence_coverage_scorecard=_ci_confidence_coverage_scorecard(),
    )

    assert (tmp_path / "lcri_ci_confidence_coverage_scorecard.png").exists()


def test_write_figures_writes_lcri_contradiction_review_packet_plot(tmp_path: Path) -> None:
    frame = _scored_frame()
    regime_table = _regime_table()

    write_figures(
        frame,
        regime_table,
        tmp_path,
        lcri_contradiction_review_packet=_contradiction_review_packet(),
    )

    assert (tmp_path / "lcri_contradiction_review_packet.png").exists()


def test_write_figures_writes_lcri_uncertainty_weighted_priority_plot(tmp_path: Path) -> None:
    frame = _scored_frame()
    regime_table = _regime_table()

    write_figures(
        frame,
        regime_table,
        tmp_path,
        lcri_uncertainty_weighted_review_priority=_uncertainty_weighted_priority(),
    )

    assert (tmp_path / "lcri_uncertainty_weighted_review_priority.png").exists()


def test_write_figures_writes_lcri_cross_artifact_evidence_index_plot(tmp_path: Path) -> None:
    frame = _scored_frame()
    regime_table = _regime_table()

    write_figures(
        frame,
        regime_table,
        tmp_path,
        lcri_cross_artifact_evidence_index=_cross_artifact_evidence_index(),
    )

    assert (tmp_path / "lcri_cross_artifact_evidence_index.png").exists()


def test_write_figures_writes_lcri_evidence_release_checklist_plot(tmp_path: Path) -> None:
    frame = _scored_frame()
    regime_table = _regime_table()

    write_figures(
        frame,
        regime_table,
        tmp_path,
        lcri_evidence_release_checklist=_evidence_release_checklist(),
    )

    assert (tmp_path / "lcri_evidence_release_checklist.png").exists()


def test_write_figures_writes_lcri_owner_handoff_packet_plot(tmp_path: Path) -> None:
    frame = _scored_frame()
    regime_table = _regime_table()

    write_figures(
        frame,
        regime_table,
        tmp_path,
        lcri_owner_handoff_packet=_owner_handoff_packet(),
    )

    assert (tmp_path / "lcri_owner_handoff_packet.png").exists()


def test_write_figures_writes_lcri_evidence_lineage_map_plot(tmp_path: Path) -> None:
    frame = _scored_frame()
    regime_table = _regime_table()

    write_figures(
        frame,
        regime_table,
        tmp_path,
        lcri_evidence_lineage_map=_evidence_lineage_map(),
    )

    assert (tmp_path / "lcri_evidence_lineage_map.png").exists()


def test_write_figures_writes_lcri_calibration_fracture_pressure_plot(tmp_path: Path) -> None:
    frame = _scored_frame()
    regime_table = _regime_table()

    write_figures(
        frame,
        regime_table,
        tmp_path,
        lcri_calibration_fracture_pressure=_calibration_fracture_pressure(),
    )

    assert (tmp_path / "lcri_calibration_fracture_pressure.png").exists()


def test_write_figures_writes_lcri_reversal_transition_gate_plot(tmp_path: Path) -> None:
    frame = _scored_frame()
    regime_table = _regime_table()

    write_figures(
        frame,
        regime_table,
        tmp_path,
        lcri_reversal_transition_gate=_reversal_transition_gate(),
    )

    assert (tmp_path / "lcri_reversal_transition_gate.png").exists()


def test_write_figures_writes_pressure_memory_decay_fracture_plot(tmp_path: Path) -> None:
    write_figures(
        _scored_frame(),
        _regime_table(),
        tmp_path,
        pressure_memory_decay_summary=_pressure_memory_decay_summary(),
    )

    assert (tmp_path / "pressure_memory_decay_fracture.png").exists()


def test_write_figures_writes_lcri_gap_delta_scope_plot(tmp_path: Path) -> None:
    frame = _scored_frame()
    regime_table = _regime_table()

    write_figures(
        frame,
        regime_table,
        tmp_path,
        lcri_gap_delta_scope_summary=_gap_delta_scope_summary(),
    )

    assert (tmp_path / "lcri_gap_delta_scope_summary.png").exists()


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


def _pressure_memory_decay_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pressure_memory_decay_state": ["fast_decay", "persistent"],
            "mean_latent_liquidity_fracture": [2.5, 1.0],
            "mean_release_velocity": [0.5, 0.0],
        }
    )


def _calibration_fracture_pressure() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "quantile": [0, 1, 2],
            "calibration_residual": [0.01, -0.05, -0.2],
            "fracture_pressure": [0.0, 0.03, 0.12],
            "pressure_label": ["aligned", "fractured_shape_only", "fractured_miscalibrated"],
        }
    )


def _reversal_transition_gate() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transition": ["thick->thin", "thin->shock"],
            "transition_stress_share": [0.75, 0.25],
            "transition_gate_decision": ["review", "pass"],
        }
    )


def _gap_delta_scope_extremes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scope": ["signal", "regime"],
            "best_raw_minus_lcri_gap": [0.03, 0.06],
            "worst_raw_minus_lcri_gap": [-0.01, -0.04],
        }
    )


def _gap_delta_scope_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition"],
            "mean_raw_minus_lcri_gap": [0.02, -0.01, 0.04],
        }
    )


def _contradiction_review_packet() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scope": ["signal", "regime"],
            "contradiction_label": [
                "gate_blocks_despite_relative_stability",
                "pass_scope_with_relative_regressions",
            ],
            "review_priority": [3, 2],
            "worst_gate_directional_accuracy_gap": [0.05, 0.01],
            "worst_raw_minus_lcri_directional_accuracy_gap": [0.03, -0.04],
            "worst_fragility_abs_gap_to_se_ratio": [2.2, 0.8],
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


def _ci_confidence_coverage_scorecard() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition"],
            "coverage_label": [
                "blocking_ci_gate_review",
                "ci_gate_contradiction_review",
                "adequate_ci_coverage",
            ],
            "mean_ci_width": [0.16, 0.11, 0.08],
            "max_ci_width": [0.22, 0.18, 0.10],
            "wide_ci_share": [0.75, 0.5, 0.0],
            "gap_exceeds_ci_half_width_share": [0.5, 0.25, 0.0],
            "ci_gate_contradiction_rows": [2, 1, 0],
            "high_priority_ci_gate_rows": [1, 0, 0],
        }
    )


def _uncertainty_weighted_priority() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition"],
            "priority_label": ["critical", "high", "medium"],
            "uncertainty_weighted_priority": [4.2, 3.4, 2.1],
            "base_review_priority": [3, 3, 2],
            "mean_ci_width": [0.16, 0.11, 0.08],
            "wide_ci_share": [0.75, 0.5, 0.25],
            "ci_gate_contradiction_rows": [2, 1, 0],
        }
    )


def _cross_artifact_evidence_index() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition"],
            "evidence_label": ["urgent", "review", "monitor"],
            "evidence_score": [6.4, 4.2, 2.3],
            "critical_rows": [2, 1, 0],
            "gate_decision": ["block", "warn", "pass"],
            "contradiction_label": [
                "gate_blocks_despite_relative_stability",
                "pass_scope_with_relative_regressions",
                "aligned",
            ],
            "uncertainty_weighted_priority": [4.2, 3.4, 2.1],
            "ci_gate_contradiction_rows": [2, 1, 0],
        }
    )


def _evidence_release_checklist() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition"],
            "check_status": ["blocked", "needs_review", "monitor"],
            "evidence_label": ["urgent", "review", "monitor"],
            "evidence_score": [6.4, 4.2, 2.3],
            "required_action": [
                "resolve deterministic blocker before release",
                "owner review required before final go/no-go",
                "monitor in post-release dashboard",
            ],
        }
    )


def _owner_handoff_packet() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition"],
            "handoff_rank": [1, 2, 3],
            "handoff_status": [
                "immediate_owner_decision",
                "owner_review",
                "release_note_monitor",
            ],
            "owner_queue": [
                "owner must decide waive/fix posture for signal before release",
                "owner review queue for regime evidence reconciliation",
                "monitor transition evidence in release notes",
            ],
            "evidence_score": [6.4, 4.2, 2.3],
            "check_status": ["blocked", "needs_review", "monitor"],
            "high_priority_ci_gate_rows": [2, 1, 0],
        }
    )


def _evidence_lineage_map() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition"],
            "lineage_status": ["source_mismatch", "incomplete_lineage", "complete"],
            "evidence_score": [6.4, 4.2, 2.3],
            "evidence_label": ["urgent", "review", "monitor"],
            "check_status": ["blocked", "needs_review", "monitor"],
            "handoff_status": ["immediate_owner_decision", "missing", "release_note_monitor"],
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


def _fragility_diagnostics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition"],
            "context": ["all", "thin", "stable"],
            "signal": ["lcri", "lcri", "raw_imbalance"],
            "abs_gap_to_se_ratio": [2.6, 1.4, 0.5],
            "fragility_label": ["fragile", "watch", "stable"],
        }
    )


def _stability_confidence_intervals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition"],
            "context": ["all", "thin", "stable"],
            "signal": ["lcri", "lcri", "raw_imbalance"],
            "heldout_directional_accuracy": [0.61, 0.54, 0.57],
            "heldout_directional_accuracy_ci_lower": [0.55, 0.45, 0.52],
            "heldout_directional_accuracy_ci_upper": [0.67, 0.63, 0.62],
            "gap_exceeds_ci_half_width": [True, False, False],
        }
    )


def _ci_gate_contradiction_diagnostics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scope": ["signal", "regime", "transition"],
            "context": ["all", "thin", "stable"],
            "ci_gate_label": ["gate_blocks_inside_ci", "stable_gap_outside_ci", "aligned"],
            "review_priority": [3, 2, 0],
            "directional_accuracy_gap": [0.04, 0.03, 0.01],
            "ci_half_width": [0.06, 0.02, 0.03],
        }
    )
