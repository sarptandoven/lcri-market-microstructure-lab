import json

import pandas as pd
import pytest

from lcri_lab.reporting import (
    artifact_coverage_matrix,
    artifact_coverage_summary,
    summarize_artifact_metadata,
    summarize_verification_errors,
    build_artifact_manifest,
    collect_artifact_metadata,
    missing_artifacts,
    verify_artifact_coverage_matrix,
    verify_artifact_manifest,
    verify_artifact_metadata_summary,
    verify_adverse_selection_phase_shift_summary,
    verify_figure_artifacts,
    verify_generalization_fragility_consistency,
    verify_generalization_fragility_diagnostics,
    verify_generalization_fragility_summary,
    verify_generalization_overview,
    verify_generalization_stability_confidence_consistency,
    verify_generalization_stability_confidence_intervals,
    verify_generalization_stability_confidence_summary,
    verify_hidden_resiliency_asymmetry_summary,
    verify_lcri_gap_delta_consistency,
    verify_lcri_gap_delta_dominant_scopes,
    verify_lcri_gap_delta_flags,
    verify_lcri_gap_delta_improvements,
    verify_lcri_gap_delta_regressions,
    verify_lcri_gap_delta_scorecard,
    verify_lcri_gap_delta_scope_extremes,
    verify_lcri_gap_delta_scope_summary,
    verify_lcri_gap_delta_summary,
    verify_lcri_ci_confidence_coverage_consistency,
    verify_lcri_ci_confidence_coverage_scorecard,
    verify_lcri_ci_confidence_coverage_summary,
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
    verify_lcri_fragility_gate_alignment,
    verify_lcri_fragility_gate_scorecard,
    verify_lcri_fracture_reversal_gate,
    verify_lcri_reversal_transition_gate_consistency,
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


def _minimal_png(width: int = 1, height: int = 1) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        b"\x00\x00\x00\rIHDR"
        + width.to_bytes(4, byteorder="big")
        + height.to_bytes(4, byteorder="big")
        + b"\x08\x02\x00\x00\x00"
        b"\x00\x00\x00\x00"
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def test_build_artifact_manifest_records_run_config_and_outputs() -> None:
    manifest = build_artifact_manifest(
        rows=100,
        train_rows=70,
        heldout_rows=30,
        seed=7,
        train_frac=0.7,
        model_artifact_version=2,
        artifacts=["metrics.csv"],
        artifact_metadata={"metrics.csv": {"size_bytes": 10, "sha256": "abc"}},
    )

    assert manifest["run"] == {
        "rows": 100,
        "train_rows": 70,
        "heldout_rows": 30,
        "seed": 7,
        "train_frac": 0.7,
    }
    assert manifest["model"] == {"artifact_version": 2}
    assert manifest["artifacts"] == ["metrics.csv"]
    assert manifest["artifact_metadata"] == {"metrics.csv": {"size_bytes": 10, "sha256": "abc"}}


def test_build_artifact_manifest_rejects_inconsistent_row_counts() -> None:
    with pytest.raises(ValueError, match="sum to rows"):
        build_artifact_manifest(
            rows=100,
            train_rows=80,
            heldout_rows=10,
            seed=7,
            train_frac=0.8,
            model_artifact_version=2,
            artifacts=[],
        )


def test_build_artifact_manifest_rejects_invalid_train_fraction() -> None:
    with pytest.raises(ValueError, match="train_frac"):
        build_artifact_manifest(
            rows=100,
            train_rows=80,
            heldout_rows=20,
            seed=7,
            train_frac=float("nan"),
            model_artifact_version=2,
            artifacts=[],
        )


def test_build_artifact_manifest_rejects_invalid_model_version() -> None:
    with pytest.raises(ValueError, match="model_artifact_version"):
        build_artifact_manifest(
            rows=100,
            train_rows=80,
            heldout_rows=20,
            seed=7,
            train_frac=0.8,
            model_artifact_version=0,
            artifacts=[],
        )


def test_collect_artifact_metadata_records_size_and_digest(tmp_path) -> None:
    (tmp_path / "metrics.csv").write_text("signal,value\n")

    metadata = collect_artifact_metadata(tmp_path, ["metrics.csv", "missing.csv"])

    assert set(metadata) == {"metrics.csv"}
    assert metadata["metrics.csv"]["size_bytes"] == len("signal,value\n")
    assert len(metadata["metrics.csv"]["sha256"]) == 64


def test_verify_hidden_resiliency_asymmetry_summary_checks_schema(tmp_path) -> None:
    write_json(
        tmp_path / "hidden_resiliency_asymmetry_summary.json",
        {
            "fast_decay_mean_fracture": 2.0,
            "slow_or_persistent_mean_fracture": 1.0,
            "fast_minus_slow_fracture": 1.0,
            "fast_minus_slow_velocity": 0.5,
            "hidden_resiliency_asymmetry_score": 1.5,
            "interpretation": "fast_release_masks_fracture",
        },
    )

    assert verify_hidden_resiliency_asymmetry_summary(tmp_path) == []
    write_json(tmp_path / "hidden_resiliency_asymmetry_summary.json", {"interpretation": "x"})
    assert "incomplete hidden resiliency" in verify_hidden_resiliency_asymmetry_summary(tmp_path)[0]


def test_verify_adverse_selection_phase_shift_summary_checks_schema(tmp_path) -> None:
    pd.DataFrame(
        [
            ["fast_decay", 2, 2, 1.0, 0.5, 2.5, 3.75, "fractured_adverse_selection"],
            ["persistent", 1, 1, 0.0, 0.0, 1.0, 0.0, "aligned_pressure_memory"],
        ],
        columns=[
            "pressure_memory_decay_state",
            "observations",
            "active_observations",
            "adverse_selection_phase_shift_rate",
            "mean_release_velocity",
            "mean_latent_liquidity_fracture",
            "adverse_selection_phase_shift_score",
            "phase_shift_interpretation",
        ],
    ).to_csv(tmp_path / "adverse_selection_phase_shift_summary.csv", index=False)

    assert verify_adverse_selection_phase_shift_summary(tmp_path) == []
    pd.DataFrame([{"phase_shift_interpretation": "x"}]).to_csv(
        tmp_path / "adverse_selection_phase_shift_summary.csv", index=False
    )
    assert "incomplete adverse selection" in verify_adverse_selection_phase_shift_summary(tmp_path)[0]


def test_verify_pressure_memory_decay_summary_checks_bounds(tmp_path) -> None:
    columns = [
        "pressure_memory_decay_state",
        "observations",
        "share",
        "decay_events",
        "event_rate",
        "mean_half_life",
        "mean_release_velocity",
    ]
    pd.DataFrame(
        [["fast_decay", 2, 0.4, 2, 1.0, 1.5, 0.3], ["persistent", 3, 0.6, 0, 0.0, 0.0, 0.0]],
        columns=columns,
    ).to_csv(tmp_path / "pressure_memory_decay_summary.csv", index=False)
    pd.DataFrame([["mystery", 1, 0.2, 2, 1.2, -1.0, 0.0]], columns=columns).to_csv(
        tmp_path / "heldout_pressure_memory_decay_summary.csv", index=False
    )

    assert verify_pressure_memory_decay_summary(tmp_path) == []
    errors = verify_pressure_memory_decay_summary(tmp_path, "heldout_pressure_memory_decay_summary.csv")
    assert any("unknown pressure memory decay states" in error for error in errors)
    assert any("bounded pressure memory rates" in error for error in errors)
    assert any("decay events exceed observations" in error for error in errors)


def test_artifact_coverage_matrix_classifies_manifest_artifacts() -> None:
    matrix = artifact_coverage_matrix(
        [
            "metrics.csv",
            "lcri_generalization_severity.csv",
            "lcri_gap_delta_summary.json",
            "lcri_ci_confidence_coverage_scorecard.csv",
            "lcri_calibration_fracture_gate.json",
            "lcri_reversal_stress_concentration_summary.json",
            "lcri_fracture_reversal_gate.json",
            "lcri_reversal_transition_gate.csv",
            "heldout_transition_robustness.json",
            "figures/generalization_gap.png",
            "artifact_manifest.json",
            "lcri_owner_handoff_packet.md",
        ]
    )

    by_artifact = matrix.set_index("artifact")
    assert by_artifact.loc["metrics.csv", "family"] == "metrics"
    assert bool(by_artifact.loc["metrics.csv", "in_research_summary"])
    assert by_artifact.loc["lcri_generalization_severity.csv", "family"] == "lcri_gate"
    assert by_artifact.loc["lcri_gap_delta_summary.json", "family"] == "lcri_gap_delta"
    assert by_artifact.loc["lcri_ci_confidence_coverage_scorecard.csv", "family"] == "lcri_gate"
    assert by_artifact.loc["lcri_calibration_fracture_gate.json", "family"] == "lcri_gate"
    assert by_artifact.loc["lcri_reversal_stress_concentration_summary.json", "family"] == "lcri_gate"
    assert by_artifact.loc["lcri_fracture_reversal_gate.json", "family"] == "lcri_gate"
    assert by_artifact.loc["lcri_reversal_transition_gate.csv", "verification_role"] == (
        "transition_verification"
    )
    assert by_artifact.loc["heldout_transition_robustness.json", "verification_role"] == (
        "transition_verification"
    )
    assert bool(by_artifact.loc["figures/generalization_gap.png", "is_figure"])
    assert by_artifact.loc["figures/generalization_gap.png", "verification_role"] == "visual_evidence"
    assert by_artifact.loc["artifact_manifest.json", "family"] == "audit"
    assert by_artifact.loc["artifact_manifest.json", "verification_role"] == "manifest_audit"
    assert by_artifact.loc["lcri_owner_handoff_packet.md", "family"] == "lcri_gate"
    assert by_artifact.loc["lcri_owner_handoff_packet.md", "verification_role"] == (
        "owner_readiness"
    )
    assert not bool(by_artifact.loc["artifact_manifest.json", "has_manifest_metadata"])


def test_artifact_coverage_summary_empty_matrix_has_stable_role_keys() -> None:
    matrix = artifact_coverage_matrix([])

    assert artifact_coverage_summary(matrix) == {
        "artifacts": 0,
        "research_summary_artifacts": 0,
        "figure_artifacts": 0,
        "metadata_tracked_artifacts": 0,
        "manifest_audit_artifacts": 0,
        "transition_verification_artifacts": 0,
        "lcri_release_evidence_artifacts": 0,
        "owner_readiness_artifacts": 0,
        "visual_evidence_artifacts": 0,
        "supporting_evidence_artifacts": 0,
        "families": 0,
    }


def test_artifact_coverage_summary_counts_audit_surfaces() -> None:
    matrix = artifact_coverage_matrix(
        [
            "metrics.csv",
            "research_summary.md",
            "artifact_coverage_matrix.csv",
            "transition_robustness.json",
            "figures/generalization_gap.png",
            "lcri_gap_delta_summary.json",
            "lcri_owner_handoff_packet.md",
        ]
    )

    assert artifact_coverage_summary(matrix) == {
        "artifacts": 7,
        "research_summary_artifacts": 3,
        "figure_artifacts": 1,
        "metadata_tracked_artifacts": 7,
        "manifest_audit_artifacts": 1,
        "transition_verification_artifacts": 1,
        "lcri_release_evidence_artifacts": 1,
        "owner_readiness_artifacts": 2,
        "visual_evidence_artifacts": 1,
        "supporting_evidence_artifacts": 1,
        "families": 6,
    }


def test_verify_lcri_fracture_reversal_gate_detects_stale_gate(tmp_path) -> None:
    write_json(
        tmp_path / "lcri_reversal_stress_concentration_summary.json",
        {"gate_decision": "review", "max_stress_concentration_ratio": 2.0, "top_regime": "thin"},
    )
    write_json(
        tmp_path / "heldout_lcri_reversal_stress_concentration_summary.json",
        {"gate_decision": "pass", "max_stress_concentration_ratio": 1.0, "top_regime": "thick"},
    )
    write_json(
        tmp_path / "lcri_calibration_fracture_gate.json",
        {
            "decision": "block",
            "passes": False,
            "max_fracture_pressure": 0.12,
            "worst_pressure_quantile": "q7",
        },
    )
    write_json(tmp_path / "lcri_fracture_reversal_gate.json", {"decision": "pass", "passes": True})

    errors = verify_lcri_fracture_reversal_gate(tmp_path)

    assert any("incomplete LCRI fracture reversal gate" in error for error in errors)
    assert any("fracture_reversal_gate.decision" in error for error in errors)


def test_verify_lcri_reversal_transition_gate_consistency_detects_stale_release(tmp_path) -> None:
    write_json(tmp_path / "lcri_fracture_reversal_gate.json", {"decision": "pass", "passes": True})
    pd.DataFrame(
        {
            "transition": ["thin->thick"],
            "total_reversal_coupling": [3.0],
            "transition_stress_share": [0.75],
            "release_gate_decision": ["review"],
            "transition_gate_decision": ["review"],
        }
    ).to_csv(tmp_path / "lcri_reversal_transition_gate.csv", index=False)

    errors = verify_lcri_reversal_transition_gate_consistency(tmp_path)

    assert any("transition gate release mismatch" in error for error in errors)
    assert any("inactive release gate has review transitions" in error for error in errors)


def test_verify_lcri_reversal_transition_gate_consistency_detects_stale_transition_decisions(
    tmp_path,
) -> None:
    write_json(tmp_path / "lcri_fracture_reversal_gate.json", {"decision": "review", "passes": False})
    pd.DataFrame(
        {
            "transition": ["thin->thick", "thick->thin"],
            "total_reversal_coupling": [4.0, 0.0],
            "transition_stress_share": [0.80, 0.60],
            "release_gate_decision": ["review", "review"],
            "transition_gate_decision": ["pass", "review"],
        }
    ).to_csv(tmp_path / "lcri_reversal_transition_gate.csv", index=False)

    errors = verify_lcri_reversal_transition_gate_consistency(tmp_path)

    assert any("active release gate missing high-stress review transition" in error for error in errors)
    assert any(
        "transition gate review decision lacks active high-stress support" in error for error in errors
    )


def test_verify_artifact_coverage_matrix_accepts_matching_audit(tmp_path) -> None:
    artifacts = ["metrics.csv", "research_summary.md", "figures/generalization_gap.png"]
    matrix = artifact_coverage_matrix(artifacts)
    matrix.to_csv(tmp_path / "artifact_coverage_matrix.csv", index=False)
    (tmp_path / "artifact_coverage_summary.json").write_text(
        json.dumps(artifact_coverage_summary(matrix))
    )

    assert verify_artifact_coverage_matrix(tmp_path, {"artifacts": artifacts}) == []


def test_verify_artifact_coverage_matrix_reports_stale_audit(tmp_path) -> None:
    artifacts = ["metrics.csv", "research_summary.md"]
    stale = artifact_coverage_matrix(["metrics.csv"])
    stale.to_csv(tmp_path / "artifact_coverage_matrix.csv", index=False)
    (tmp_path / "artifact_coverage_summary.json").write_text(
        json.dumps(artifact_coverage_summary(stale))
    )

    errors = verify_artifact_coverage_matrix(tmp_path, {"artifacts": artifacts})

    assert "artifact coverage matrix mismatch against manifest artifacts" in errors
    assert any("artifact coverage summary mismatch for artifacts" in error for error in errors)


def test_verify_artifact_coverage_matrix_reports_stale_role_summary(tmp_path) -> None:
    artifacts = [
        "metrics.csv",
        "transition_robustness.json",
        "lcri_owner_handoff_packet.md",
        "figures/generalization_gap.png",
    ]
    matrix = artifact_coverage_matrix(artifacts)
    stale_summary = artifact_coverage_summary(matrix) | {
        "transition_verification_artifacts": 0,
        "owner_readiness_artifacts": 0,
        "visual_evidence_artifacts": 0,
        "supporting_evidence_artifacts": 0,
    }
    matrix.to_csv(tmp_path / "artifact_coverage_matrix.csv", index=False)
    (tmp_path / "artifact_coverage_summary.json").write_text(json.dumps(stale_summary))

    errors = verify_artifact_coverage_matrix(tmp_path, {"artifacts": artifacts})

    expected_mismatches = {
        "transition_verification_artifacts",
        "owner_readiness_artifacts",
        "visual_evidence_artifacts",
        "supporting_evidence_artifacts",
    }
    for key in expected_mismatches:
        assert any(f"artifact coverage summary mismatch for {key}" in error for error in errors)


def test_verify_artifact_manifest_reports_checksum_mismatch(tmp_path) -> None:
    (tmp_path / "metrics.csv").write_text("signal,value\n")
    manifest = {
        "artifacts": ["metrics.csv"],
        "artifact_metadata": {"metrics.csv": {"size_bytes": 1, "sha256": "0" * 64}},
    }

    errors = verify_artifact_manifest(tmp_path, manifest)

    assert "size mismatch: metrics.csv" in errors
    assert "sha256 mismatch: metrics.csv" in errors


def test_verify_artifact_manifest_reports_partial_metadata_coverage(tmp_path) -> None:
    (tmp_path / "metrics.csv").write_text("signal,value\n")
    (tmp_path / "summary.json").write_text("{}\n")
    manifest = {
        "artifacts": ["metrics.csv", "summary.json"],
        "artifact_metadata": collect_artifact_metadata(tmp_path, ["metrics.csv"]),
    }

    errors = verify_artifact_manifest(tmp_path, manifest)

    assert errors == ["missing manifest metadata for artifact: summary.json"]


def test_verify_artifact_manifest_reports_unlisted_and_incomplete_metadata(tmp_path) -> None:
    (tmp_path / "metrics.csv").write_text("signal,value\n")
    manifest = {
        "artifacts": ["metrics.csv"],
        "artifact_metadata": {
            "metrics.csv": {"size_bytes": 13},
            "stale.csv": {"size_bytes": 1, "sha256": "0" * 64},
        },
    }

    errors = verify_artifact_manifest(tmp_path, manifest)

    assert "unexpected manifest metadata for artifact: stale.csv" in errors
    assert "incomplete manifest metadata for artifact metrics.csv: ['sha256']" in errors


def test_verify_artifact_manifest_rejects_unsafe_paths(tmp_path) -> None:
    manifest = {
        "artifacts": ["../outside.csv"],
        "artifact_metadata": {"../outside.csv": {"size_bytes": 1, "sha256": "0" * 64}},
    }

    errors = verify_artifact_manifest(tmp_path, manifest)

    assert errors == [
        "unsafe artifact path: ../outside.csv",
        "unsafe manifest metadata path: ../outside.csv",
    ]


def test_verify_artifact_metadata_summary_accepts_matching_manifest_summary(tmp_path) -> None:
    metrics = tmp_path / "metrics.csv"
    metrics.write_text("signal,value\n")
    summary_metadata = collect_artifact_metadata(tmp_path, ["metrics.csv"])
    (tmp_path / "artifact_metadata_summary.json").write_text(
        json.dumps(summarize_artifact_metadata(summary_metadata))
    )
    manifest = {
        "artifacts": ["metrics.csv", "artifact_metadata_summary.json"],
        "artifact_metadata": collect_artifact_metadata(
            tmp_path, ["metrics.csv", "artifact_metadata_summary.json"]
        ),
    }

    assert verify_artifact_metadata_summary(tmp_path, manifest) == []


def test_verify_artifact_metadata_summary_reports_stale_summary(tmp_path) -> None:
    metrics = tmp_path / "metrics.csv"
    metrics.write_text("signal,value\n")
    (tmp_path / "artifact_metadata_summary.json").write_text(
        json.dumps(
            {
                "artifacts_with_metadata": 0,
                "total_size_bytes": 0,
                "largest_artifact": "none",
                "largest_artifact_size_bytes": 0,
            }
        )
    )
    manifest = {
        "artifacts": ["metrics.csv", "artifact_metadata_summary.json"],
        "artifact_metadata": collect_artifact_metadata(
            tmp_path, ["metrics.csv", "artifact_metadata_summary.json"]
        ),
    }

    errors = verify_artifact_metadata_summary(tmp_path, manifest)

    assert "artifact metadata summary mismatch for artifacts_with_metadata" in errors[0]
    assert any("total_size_bytes" in error for error in errors)
    assert any("largest_artifact" in error for error in errors)


def test_verify_artifact_metadata_summary_reports_manifest_metadata_gaps(tmp_path) -> None:
    (tmp_path / "metrics.csv").write_text("signal,value\n")
    (tmp_path / "artifact_metadata_summary.json").write_text(
        json.dumps(summarize_artifact_metadata({}))
    )
    manifest = {
        "artifacts": ["metrics.csv", "artifact_metadata_summary.json"],
        "artifact_metadata": {},
    }

    errors = verify_artifact_metadata_summary(tmp_path, manifest)

    assert errors == [
        "missing manifest metadata for artifact: artifact_metadata_summary.json",
        "missing manifest metadata for artifact: metrics.csv",
    ]


def test_verify_figure_artifacts_accepts_manifest_png(tmp_path) -> None:
    figure = tmp_path / "figures" / "gap.png"
    figure.parent.mkdir()
    figure.write_bytes(_minimal_png(width=640, height=480))
    manifest = {"artifacts": ["figures/gap.png"]}

    assert verify_figure_artifacts(tmp_path, manifest) == []


def test_verify_figure_artifacts_reports_corrupt_png(tmp_path) -> None:
    figure = tmp_path / "figures" / "gap.png"
    figure.parent.mkdir()
    figure.write_bytes(b"not a png")
    manifest = {"artifacts": ["figures/gap.png"]}

    errors = verify_figure_artifacts(tmp_path, manifest)

    assert errors == ["invalid figure artifact: figures/gap.png: missing PNG signature"]


def test_verify_figure_artifacts_reports_truncated_png(tmp_path) -> None:
    figure = tmp_path / "figures" / "gap.png"
    figure.parent.mkdir()
    figure.write_bytes(b"\x89PNG\r\n\x1a\n")
    manifest = {"artifacts": ["figures/gap.png"]}

    errors = verify_figure_artifacts(tmp_path, manifest)

    assert errors == ["invalid figure artifact: figures/gap.png: truncated PNG header"]


def test_verify_generalization_overview_reports_missing_keys(tmp_path) -> None:
    (tmp_path / "generalization_overview.json").write_text(
        json.dumps({"signal_rows": 2, "regime_rows": 4})
    )

    errors = verify_generalization_overview(tmp_path)

    assert errors == [
        "incomplete generalization overview: "
        "['max_regime_directional_accuracy_gap', "
        "'max_signal_directional_accuracy_gap', "
        "'max_transition_directional_accuracy_gap', 'transition_rows']"
    ]


def test_verify_generalization_fragility_diagnostics_accepts_complete_csv(tmp_path) -> None:
    (tmp_path / "generalization_fragility_diagnostics.csv").write_text(
        "scope,context,signal,full_rows,heldout_rows,full_directional_accuracy,"
        "heldout_directional_accuracy,directional_accuracy_gap,"
        "heldout_directional_accuracy_se,abs_gap_to_se_ratio,fragility_label\n"
        "signal,all,lcri,100,20,0.65,0.55,0.10,0.11,0.91,stable\n"
    )

    assert verify_generalization_fragility_diagnostics(tmp_path) == []


def test_verify_generalization_fragility_diagnostics_reports_missing_columns(tmp_path) -> None:
    (tmp_path / "generalization_fragility_diagnostics.csv").write_text("scope,context\n")

    errors = verify_generalization_fragility_diagnostics(tmp_path)

    assert errors == [
        "incomplete generalization fragility diagnostics: "
        "['abs_gap_to_se_ratio', 'directional_accuracy_gap', 'fragility_label', "
        "'full_rows', 'heldout_directional_accuracy_se', 'heldout_rows', 'signal']"
    ]


def test_verify_generalization_fragility_summary_reports_missing_keys(tmp_path) -> None:
    write_json(tmp_path / "generalization_fragility_summary.json", {"rows": 2})

    errors = verify_generalization_fragility_summary(tmp_path)

    assert errors == [
        "incomplete generalization fragility summary: "
        "['fragile_rows', 'max_abs_gap_to_se_ratio', 'most_fragile_context', "
        "'stable_rows', 'watch_rows']"
    ]


def test_verify_generalization_fragility_consistency_accepts_matching_artifacts(tmp_path) -> None:
    (tmp_path / "generalization_fragility_diagnostics.csv").write_text(
        "scope,context,signal,directional_accuracy_gap,heldout_directional_accuracy_se,"
        "abs_gap_to_se_ratio,fragility_label\n"
        "signal,all,lcri,0.20,0.10,2.0,watch\n"
        "regime,normal,lcri,0.05,0.10,0.5,stable\n"
    )
    write_json(
        tmp_path / "generalization_fragility_summary.json",
        {
            "rows": 2,
            "stable_rows": 1,
            "watch_rows": 1,
            "fragile_rows": 0,
            "max_abs_gap_to_se_ratio": 2.0,
            "most_fragile_context": "signal:all:lcri",
        },
    )

    assert verify_generalization_fragility_consistency(tmp_path) == []


def test_verify_generalization_fragility_consistency_reports_stale_summary(tmp_path) -> None:
    (tmp_path / "generalization_fragility_diagnostics.csv").write_text(
        "scope,context,signal,directional_accuracy_gap,heldout_directional_accuracy_se,"
        "abs_gap_to_se_ratio,fragility_label\n"
        "signal,all,lcri,0.30,0.10,3.0,fragile\n"
    )
    write_json(
        tmp_path / "generalization_fragility_summary.json",
        {
            "rows": 1,
            "stable_rows": 1,
            "watch_rows": 0,
            "fragile_rows": 0,
            "max_abs_gap_to_se_ratio": 1.0,
            "most_fragile_context": "none",
        },
    )

    errors = verify_generalization_fragility_consistency(tmp_path)

    assert "generalization fragility mismatch for summary.stable_rows: expected 0, found 1" in errors
    assert "generalization fragility mismatch for summary.fragile_rows: expected 1, found 0" in errors
    assert any("summary.max_abs_gap_to_se_ratio" in error for error in errors)
    assert any("summary.most_fragile_context" in error for error in errors)


def test_verify_generalization_fragility_consistency_reports_stale_labels_and_ratios(tmp_path) -> None:
    (tmp_path / "generalization_fragility_diagnostics.csv").write_text(
        "scope,context,signal,directional_accuracy_gap,heldout_directional_accuracy_se,"
        "abs_gap_to_se_ratio,fragility_label\n"
        "signal,all,lcri,0.30,0.10,2.0,stable\n"
    )
    write_json(
        tmp_path / "generalization_fragility_summary.json",
        {
            "rows": 1,
            "stable_rows": 1,
            "watch_rows": 0,
            "fragile_rows": 0,
            "max_abs_gap_to_se_ratio": 2.0,
            "most_fragile_context": "signal:all:lcri",
        },
    )

    errors = verify_generalization_fragility_consistency(tmp_path)

    assert any("generalization fragility ratio mismatch for signal:all:lcri" in error for error in errors)
    assert any("generalization fragility label mismatch" in error for error in errors)


def test_verify_generalization_stability_confidence_artifacts_accept_matching_outputs(tmp_path) -> None:
    pd.DataFrame(
        [
            {
                "scope": "signal",
                "context": "all",
                "signal": "lcri",
                "heldout_rows": 100,
                "heldout_directional_accuracy": 0.60,
                "heldout_directional_accuracy_se": 0.05,
                "confidence_level": 0.950004209703559,
                "heldout_directional_accuracy_ci_lower": 0.502,
                "heldout_directional_accuracy_ci_upper": 0.698,
                "heldout_directional_accuracy_ci_width": 0.196,
                "directional_accuracy_gap": 0.12,
                "gap_exceeds_ci_half_width": True,
            }
        ]
    ).to_csv(tmp_path / "generalization_stability_confidence_intervals.csv", index=False)
    write_json(
        tmp_path / "generalization_stability_confidence_summary.json",
        {
            "rows": 1,
            "gap_exceeds_ci_half_width_rows": 1,
            "mean_ci_width": 0.196,
            "max_ci_width": 0.196,
            "widest_interval_context": "signal:all:lcri",
        },
    )

    assert verify_generalization_stability_confidence_intervals(tmp_path) == []
    assert verify_generalization_stability_confidence_summary(tmp_path) == []
    assert verify_generalization_stability_confidence_consistency(tmp_path) == []


def test_verify_generalization_stability_confidence_consistency_reports_stale_width(tmp_path) -> None:
    pd.DataFrame(
        [
            {
                "scope": "signal",
                "context": "all",
                "signal": "lcri",
                "heldout_rows": 100,
                "heldout_directional_accuracy": 0.60,
                "heldout_directional_accuracy_se": 0.05,
                "confidence_level": 0.950004209703559,
                "heldout_directional_accuracy_ci_lower": 0.50,
                "heldout_directional_accuracy_ci_upper": 0.70,
                "heldout_directional_accuracy_ci_width": 0.30,
                "directional_accuracy_gap": 0.01,
                "gap_exceeds_ci_half_width": True,
            }
        ]
    ).to_csv(tmp_path / "generalization_stability_confidence_intervals.csv", index=False)
    write_json(
        tmp_path / "generalization_stability_confidence_summary.json",
        {
            "rows": 1,
            "gap_exceeds_ci_half_width_rows": 1,
            "mean_ci_width": 0.30,
            "max_ci_width": 0.30,
            "widest_interval_context": "signal:all:lcri",
        },
    )

    errors = verify_generalization_stability_confidence_consistency(tmp_path)

    assert any("confidence_interval.signal:all:lcri.width" in error for error in errors)
    assert any("generalization stability confidence gap flag mismatch" in error for error in errors)


def test_summarize_artifact_metadata_reports_totals_and_largest_file() -> None:
    output = summarize_artifact_metadata(
        {
            "metrics.csv": {"size_bytes": 12, "sha256": "abc"},
            "figures/gap.png": {"size_bytes": 40, "sha256": "def"},
        }
    )

    assert output == {
        "artifacts_with_metadata": 2,
        "total_size_bytes": 52,
        "largest_artifact": "figures/gap.png",
        "largest_artifact_size_bytes": 40,
    }


def test_summarize_artifact_metadata_handles_empty_metadata() -> None:
    assert summarize_artifact_metadata({}) == {
        "artifacts_with_metadata": 0,
        "total_size_bytes": 0,
        "largest_artifact": "none",
        "largest_artifact_size_bytes": 0,
    }


def test_summarize_verification_errors_groups_artifact_families() -> None:
    output = summarize_verification_errors(
        [
            "sha256 mismatch: metrics.csv",
            "missing LCRI blocker summary: lcri_generalization_blocker_summary.json",
            "missing generalization overview: generalization_overview.json",
            "missing artifact: figures/lcri_generalization_gap_delta.png",
            "missing metrics.csv",
        ]
    )

    assert output["errors"] == 5
    assert output["manifest"] == 1
    assert output["lcri_gate"] == 1
    assert output["generalization"] == 1
    assert output["figures"] == 1
    assert output["other"] == 1
    assert output["passes_verification"] is False


def test_summarize_verification_errors_passes_without_errors() -> None:
    assert summarize_verification_errors([]) == {
        "errors": 0,
        "manifest": 0,
        "generalization": 0,
        "lcri_gate": 0,
        "figures": 0,
        "other": 0,
        "passes_verification": True,
    }


def test_verify_generalization_overview_accepts_complete_payload(tmp_path) -> None:
    write_json(
        tmp_path / "generalization_overview.json",
        {
            "signal_rows": 2,
            "regime_rows": 4,
            "transition_rows": 4,
            "max_signal_directional_accuracy_gap": 0.05,
            "max_regime_directional_accuracy_gap": 0.08,
            "max_transition_directional_accuracy_gap": 0.04,
        },
    )

    assert verify_generalization_overview(tmp_path) == []


def test_verify_lcri_generalization_gap_leaderboard_accepts_complete_csv(tmp_path) -> None:
    pd.DataFrame(
        [
            {
                "scope": "signal",
                "context": "all",
                "signal": "lcri",
                "directional_accuracy_gap": 0.05,
            }
        ]
    ).to_csv(tmp_path / "lcri_generalization_gap_leaderboard.csv", index=False)

    assert verify_lcri_generalization_gap_leaderboard(tmp_path) == []


def test_verify_lcri_generalization_scope_summary_accepts_complete_csv(tmp_path) -> None:
    pd.DataFrame(
        [
            {
                "scope": "regime",
                "rows": 2,
                "mean_directional_accuracy_gap": 0.07,
                "max_directional_accuracy_gap": 0.09,
            }
        ]
    ).to_csv(tmp_path / "lcri_generalization_scope_summary.csv", index=False)

    assert verify_lcri_generalization_scope_summary(tmp_path) == []


def test_verify_lcri_worst_generalization_context_accepts_complete_payload(tmp_path) -> None:
    write_json(
        tmp_path / "lcri_worst_generalization_context.json",
        {
            "scope": "transition",
            "context": "stable",
            "directional_accuracy_gap": 0.08,
        },
    )

    assert verify_lcri_worst_generalization_context(tmp_path) == []


def test_verify_lcri_generalization_severity_accepts_complete_csv(tmp_path) -> None:
    pd.DataFrame(
        [
            {
                "scope": "transition",
                "context": "stable",
                "directional_accuracy_gap": 0.08,
                "severity": "critical",
            }
        ]
    ).to_csv(tmp_path / "lcri_generalization_severity.csv", index=False)

    assert verify_lcri_generalization_severity(tmp_path) == []


def test_verify_lcri_fragility_gate_alignment_accepts_complete_csv(tmp_path) -> None:
    pd.DataFrame(
        [
            {
                "scope": "transition",
                "context": "stable",
                "directional_accuracy_gap": 0.08,
                "severity": "critical",
                "heldout_rows": 25,
                "heldout_directional_accuracy_se": 0.05,
                "abs_gap_to_se_ratio": 1.6,
                "fragility_label": "stable",
                "alignment_label": "gate_blocks_stable_slice",
                "review_note": "critical gate blocker exceeds deterministic threshold",
            }
        ]
    ).to_csv(tmp_path / "lcri_fragility_gate_alignment.csv", index=False)

    assert verify_lcri_fragility_gate_alignment(tmp_path) == []


def test_verify_lcri_fragility_gate_alignment_reports_stale_label(tmp_path) -> None:
    pd.DataFrame(
        [
            {
                "scope": "signal",
                "context": "all",
                "directional_accuracy_gap": 0.07,
                "severity": "critical",
                "heldout_rows": 80,
                "heldout_directional_accuracy_se": 0.05,
                "abs_gap_to_se_ratio": 0.8,
                "fragility_label": "stable",
                "alignment_label": "aligned",
                "review_note": "stale",
            }
        ]
    ).to_csv(tmp_path / "lcri_fragility_gate_alignment.csv", index=False)

    errors = verify_lcri_fragility_gate_alignment(tmp_path)

    assert errors == [
        "LCRI fragility gate alignment mismatch for signal:all: "
        "expected 'gate_blocks_stable_slice', found 'aligned'"
    ]


def test_verify_lcri_fragility_gate_scorecard_accepts_matching_json(tmp_path) -> None:
    pd.DataFrame(
        [
            {
                "scope": "signal",
                "context": "all",
                "severity": "critical",
                "abs_gap_to_se_ratio": 0.8,
                "alignment_label": "gate_blocks_stable_slice",
            },
            {
                "scope": "regime",
                "context": "thin",
                "severity": "warning",
                "abs_gap_to_se_ratio": 3.4,
                "alignment_label": "uncertainty_fragile_noncritical",
            },
            {
                "scope": "transition",
                "context": "open",
                "severity": "stable",
                "abs_gap_to_se_ratio": 2.1,
                "alignment_label": "uncertainty_watch_stable_gap",
            },
            {
                "scope": "regime",
                "context": "deep",
                "severity": "critical",
                "abs_gap_to_se_ratio": 1.1,
                "alignment_label": "aligned",
            },
        ]
    ).to_csv(tmp_path / "lcri_fragility_gate_alignment.csv", index=False)
    write_json(
        tmp_path / "lcri_fragility_gate_scorecard.json",
        {
            "rows": 4,
            "aligned_rows": 1,
            "review_required_rows": 3,
            "gate_blocks_stable_slice_rows": 1,
            "uncertainty_fragile_noncritical_rows": 1,
            "uncertainty_watch_stable_gap_rows": 1,
            "critical_rows": 2,
            "critical_stable_slice_share": 0.5,
            "max_abs_gap_to_se_ratio": 3.4,
            "worst_review_context": "regime:thin:uncertainty_fragile_noncritical",
        },
    )

    assert verify_lcri_fragility_gate_scorecard(tmp_path) == []


def test_verify_lcri_fragility_gate_scorecard_reports_stale_json(tmp_path) -> None:
    pd.DataFrame(
        [
            {
                "scope": "signal",
                "context": "all",
                "severity": "critical",
                "abs_gap_to_se_ratio": 0.8,
                "alignment_label": "gate_blocks_stable_slice",
            }
        ]
    ).to_csv(tmp_path / "lcri_fragility_gate_alignment.csv", index=False)
    write_json(
        tmp_path / "lcri_fragility_gate_scorecard.json",
        {
            "rows": 1,
            "aligned_rows": 1,
            "review_required_rows": 0,
            "gate_blocks_stable_slice_rows": 0,
            "uncertainty_fragile_noncritical_rows": 0,
            "uncertainty_watch_stable_gap_rows": 0,
            "critical_rows": 1,
            "critical_stable_slice_share": 0.0,
            "max_abs_gap_to_se_ratio": 0.8,
            "worst_review_context": "none",
        },
    )

    errors = verify_lcri_fragility_gate_scorecard(tmp_path)

    assert "LCRI fragility gate scorecard mismatch for aligned_rows: expected 0, found 1" in errors
    assert "LCRI fragility gate scorecard mismatch for review_required_rows: expected 1, found 0" in errors
    assert "LCRI fragility gate scorecard mismatch for worst_review_context: expected 'signal:all:gate_blocks_stable_slice', found 'none'" in errors


def test_verify_lcri_ci_confidence_coverage_accepts_matching_artifacts(tmp_path) -> None:
    pd.DataFrame(
        {
            "scope": ["signal", "regime"],
            "context": ["all", "thin"],
            "signal": ["lcri", "lcri"],
            "heldout_directional_accuracy_ci_width": [0.10, 0.24],
            "gap_exceeds_ci_half_width": [False, True],
        }
    ).to_csv(tmp_path / "generalization_stability_confidence_intervals.csv", index=False)
    pd.DataFrame(
        {
            "scope": ["signal", "regime"],
            "context": ["all", "thin"],
            "ci_gate_label": ["aligned", "gate_warns_inside_ci"],
            "review_priority": [1, 2],
        }
    ).to_csv(tmp_path / "lcri_ci_gate_contradiction_diagnostics.csv", index=False)
    pd.DataFrame(
        {
            "scope": ["regime", "signal"],
            "rows": [1, 1],
            "mean_ci_width": [0.24, 0.10],
            "max_ci_width": [0.24, 0.10],
            "wide_ci_rows": [1, 0],
            "wide_ci_share": [1.0, 0.0],
            "gap_exceeds_ci_half_width_rows": [1, 0],
            "gap_exceeds_ci_half_width_share": [1.0, 0.0],
            "ci_gate_contradiction_rows": [1, 0],
            "high_priority_ci_gate_rows": [0, 0],
            "max_ci_gate_review_priority": [2, 1],
            "coverage_label": ["ci_gate_contradiction_review", "adequate_ci_coverage"],
            "review_note": [
                "scope has CI/gate disagreements; inspect uncertainty-qualified gate rows",
                "scope CI coverage is aligned with current gate evidence",
            ],
        }
    ).to_csv(tmp_path / "lcri_ci_confidence_coverage_scorecard.csv", index=False)
    write_json(
        tmp_path / "lcri_ci_confidence_coverage_summary.json",
        {
            "scopes": 2,
            "review_scopes": 1,
            "blocking_review_scopes": 0,
            "contradiction_review_scopes": 1,
            "wide_ci_review_scopes": 0,
            "total_ci_gate_contradiction_rows": 1,
            "total_wide_ci_rows": 1,
            "worst_ci_confidence_scope": "regime:ci_gate_contradiction_review",
        },
    )

    assert verify_lcri_ci_confidence_coverage_scorecard(tmp_path) == []
    assert verify_lcri_ci_confidence_coverage_summary(tmp_path) == []
    assert verify_lcri_ci_confidence_coverage_consistency(tmp_path) == []


def test_verify_lcri_ci_confidence_coverage_reports_stale_summary(tmp_path) -> None:
    pd.DataFrame(
        {
            "scope": ["signal"],
            "context": ["all"],
            "signal": ["lcri"],
            "heldout_directional_accuracy_ci_width": [0.10],
            "gap_exceeds_ci_half_width": [False],
        }
    ).to_csv(tmp_path / "generalization_stability_confidence_intervals.csv", index=False)
    pd.DataFrame(
        {
            "scope": ["signal"],
            "context": ["all"],
            "ci_gate_label": ["aligned"],
            "review_priority": [1],
        }
    ).to_csv(tmp_path / "lcri_ci_gate_contradiction_diagnostics.csv", index=False)
    pd.DataFrame(
        {
            "scope": ["signal"],
            "rows": [1],
            "mean_ci_width": [0.10],
            "max_ci_width": [0.10],
            "wide_ci_rows": [0],
            "wide_ci_share": [0.0],
            "gap_exceeds_ci_half_width_rows": [0],
            "gap_exceeds_ci_half_width_share": [0.0],
            "ci_gate_contradiction_rows": [0],
            "high_priority_ci_gate_rows": [0],
            "max_ci_gate_review_priority": [1],
            "coverage_label": ["adequate_ci_coverage"],
            "review_note": ["scope CI coverage is aligned with current gate evidence"],
        }
    ).to_csv(tmp_path / "lcri_ci_confidence_coverage_scorecard.csv", index=False)
    write_json(
        tmp_path / "lcri_ci_confidence_coverage_summary.json",
        {
            "scopes": 1,
            "review_scopes": 1,
            "blocking_review_scopes": 0,
            "contradiction_review_scopes": 0,
            "wide_ci_review_scopes": 0,
            "total_ci_gate_contradiction_rows": 0,
            "total_wide_ci_rows": 0,
            "worst_ci_confidence_scope": "none",
        },
    )

    errors = verify_lcri_ci_confidence_coverage_consistency(tmp_path)

    assert any("ci_confidence_coverage_summary.review_scopes" in error for error in errors)
    assert any("ci_confidence_coverage_summary.worst_ci_confidence_scope" in error for error in errors)


def test_verify_lcri_generalization_severity_by_scope_accepts_complete_csv(tmp_path) -> None:
    (tmp_path / "lcri_generalization_severity_by_scope.csv").write_text(
        "scope,rows,stable_rows,warning_rows,critical_rows\n"
        "regime,2,0,1,1\n"
    )

    assert verify_lcri_generalization_severity_by_scope(tmp_path) == []


def test_verify_lcri_generalization_severity_consistency_accepts_matching_rollups(tmp_path) -> None:
    (tmp_path / "lcri_generalization_severity.csv").write_text(
        "scope,context,directional_accuracy_gap,severity\n"
        "regime,thin,0.08,critical\n"
        "regime,deep,0.04,warning\n"
        "transition,open,0.01,stable\n"
    )
    (tmp_path / "lcri_generalization_severity_by_scope.csv").write_text(
        "scope,rows,stable_rows,warning_rows,critical_rows\n"
        "regime,2,0,1,1\n"
        "transition,1,1,0,0\n"
    )
    write_json(
        tmp_path / "lcri_generalization_severity_summary.json",
        {
            "rows": 3,
            "stable_rows": 1,
            "warning_rows": 1,
            "critical_rows": 1,
            "passes_lcri_generalization_gate": False,
        },
    )

    assert verify_lcri_generalization_severity_consistency(tmp_path) == []


def test_verify_lcri_generalization_severity_consistency_reports_stale_rollups(tmp_path) -> None:
    (tmp_path / "lcri_generalization_severity.csv").write_text(
        "scope,context,directional_accuracy_gap,severity\n"
        "regime,thin,0.08,critical\n"
        "regime,deep,0.04,warning\n"
        "transition,open,0.01,stable\n"
    )
    (tmp_path / "lcri_generalization_severity_by_scope.csv").write_text(
        "scope,rows,stable_rows,warning_rows,critical_rows\n"
        "regime,2,0,0,2\n"
        "transition,1,1,0,0\n"
    )
    write_json(
        tmp_path / "lcri_generalization_severity_summary.json",
        {
            "rows": 3,
            "stable_rows": 1,
            "warning_rows": 0,
            "critical_rows": 2,
            "passes_lcri_generalization_gate": False,
        },
    )

    errors = verify_lcri_generalization_severity_consistency(tmp_path)

    assert "LCRI severity summary mismatch for warning_rows" in errors[0]
    assert "LCRI severity summary mismatch for critical_rows" in errors[1]
    assert "LCRI severity scope rollup mismatch for regime.warning_rows" in errors[2]
    assert "LCRI severity scope rollup mismatch for regime.critical_rows" in errors[3]


def test_verify_lcri_generalization_scope_gate_decisions_accepts_complete_csv(tmp_path) -> None:
    (tmp_path / "lcri_generalization_scope_gate_decisions.csv").write_text(
        "scope,rows,decision,reason\n"
        "regime,2,block,regime blocked\n"
    )

    assert verify_lcri_generalization_scope_gate_decisions(tmp_path) == []


def test_verify_lcri_generalization_scope_gate_decision_summary_accepts_payload(tmp_path) -> None:
    write_json(
        tmp_path / "lcri_generalization_scope_gate_decision_summary.json",
        {
            "scopes": 3,
            "pass_scopes": 1,
            "warn_scopes": 1,
            "block_scopes": 1,
            "blocked_scope_names": "regime",
            "warn_scope_names": "transition",
        },
    )

    assert verify_lcri_generalization_scope_gate_decision_summary(tmp_path) == []


def test_verify_lcri_generalization_scope_gate_consistency_accepts_matching_payload(tmp_path) -> None:
    (tmp_path / "lcri_generalization_scope_gate_decisions.csv").write_text(
        "scope,rows,decision,reason\n"
        "regime,2,block,critical regime gap\n"
        "signal,1,pass,stable signal gap\n"
        "transition,3,warn,warning transition gap\n"
    )
    write_json(
        tmp_path / "lcri_generalization_scope_gate_decision_summary.json",
        {
            "scopes": 3,
            "pass_scopes": 1,
            "warn_scopes": 1,
            "block_scopes": 1,
            "blocked_scope_names": "regime",
            "warn_scope_names": "transition",
        },
    )

    assert verify_lcri_generalization_scope_gate_consistency(tmp_path) == []


def test_verify_lcri_generalization_scope_gate_consistency_reports_mismatch(tmp_path) -> None:
    (tmp_path / "lcri_generalization_scope_gate_decisions.csv").write_text(
        "scope,rows,decision,reason\n"
        "regime,2,block,critical regime gap\n"
        "transition,3,warn,warning transition gap\n"
    )
    write_json(
        tmp_path / "lcri_generalization_scope_gate_decision_summary.json",
        {
            "scopes": 2,
            "pass_scopes": 0,
            "warn_scopes": 0,
            "block_scopes": 2,
            "blocked_scope_names": "regime,transition",
            "warn_scope_names": "none",
        },
    )

    errors = verify_lcri_generalization_scope_gate_consistency(tmp_path)

    assert "LCRI scope gate summary mismatch for warn_scopes" in errors[0]
    assert "LCRI scope gate summary mismatch for block_scopes" in errors[1]


def test_verify_lcri_generalization_scope_risk_accepts_complete_csv(tmp_path) -> None:
    (tmp_path / "lcri_generalization_scope_risk.csv").write_text(
        "scope,rows,warning_or_critical_share,critical_share\n"
        "regime,2,1.0,0.5\n"
    )

    assert verify_lcri_generalization_scope_risk(tmp_path) == []


def test_verify_lcri_generalization_critical_contexts_accepts_complete_csv(tmp_path) -> None:
    (tmp_path / "lcri_generalization_critical_contexts.csv").write_text(
        "scope,context,directional_accuracy_gap,severity\n"
        "regime,thin,0.08,critical\n"
    )

    assert verify_lcri_generalization_critical_contexts(tmp_path) == []


def test_verify_lcri_generalization_blocker_summary_accepts_complete_payload(tmp_path) -> None:
    write_json(
        tmp_path / "lcri_generalization_blocker_summary.json",
        {
            "critical_rows": 2,
            "critical_scopes": "regime,transition",
            "max_critical_gap": 0.08,
            "max_critical_context": "regime:thin",
        },
    )

    assert verify_lcri_generalization_blocker_summary(tmp_path) == []


def test_verify_lcri_generalization_severity_summary_accepts_complete_payload(tmp_path) -> None:
    write_json(
        tmp_path / "lcri_generalization_severity_summary.json",
        {
            "rows": 3,
            "stable_rows": 1,
            "warning_rows": 1,
            "critical_rows": 1,
            "passes_lcri_generalization_gate": False,
        },
    )

    assert verify_lcri_generalization_severity_summary(tmp_path) == []


def test_verify_lcri_generalization_gate_decision_accepts_complete_payload(tmp_path) -> None:
    write_json(
        tmp_path / "lcri_generalization_gate_decision.json",
        {
            "passes": False,
            "decision": "block",
            "rows_evaluated": 3,
            "warning_rows": 1,
            "critical_rows": 1,
            "worst_scope": "regime",
            "worst_context": "thin",
            "worst_directional_accuracy_gap": 0.07,
            "reason": "blocked by 1 critical LCRI generalization rows",
        },
    )

    assert verify_lcri_generalization_gate_decision(tmp_path) == []



def test_verify_lcri_generalization_gate_decision_consistency_accepts_matching_payloads(tmp_path) -> None:
    (tmp_path / "lcri_generalization_severity.csv").write_text(
        "scope,context,directional_accuracy_gap,severity\n"
        "regime,thin,0.08,critical\n"
        "transition,open,0.03,warning\n"
    )
    write_json(
        tmp_path / "lcri_generalization_severity_summary.json",
        {
            "rows": 2,
            "stable_rows": 0,
            "warning_rows": 1,
            "critical_rows": 1,
            "passes_lcri_generalization_gate": False,
        },
    )
    write_json(
        tmp_path / "lcri_worst_generalization_context.json",
        {"scope": "regime", "context": "thin", "directional_accuracy_gap": 0.08},
    )
    write_json(
        tmp_path / "lcri_generalization_gate_decision.json",
        {
            "passes": False,
            "decision": "block",
            "rows_evaluated": 2,
            "warning_rows": 1,
            "critical_rows": 1,
            "worst_scope": "regime",
            "worst_context": "thin",
            "worst_directional_accuracy_gap": 0.08,
            "reason": "blocked by 1 critical LCRI generalization rows",
        },
    )
    (tmp_path / "lcri_generalization_critical_contexts.csv").write_text(
        "scope,context,directional_accuracy_gap,severity\n"
        "regime,thin,0.08,critical\n"
    )
    write_json(
        tmp_path / "lcri_generalization_blocker_summary.json",
        {
            "critical_rows": 1,
            "critical_scopes": "regime",
            "max_critical_gap": 0.08,
            "max_critical_context": "regime:thin",
        },
    )

    assert verify_lcri_generalization_gate_decision_consistency(tmp_path) == []


def test_verify_lcri_generalization_gate_decision_consistency_reports_stale_payloads(tmp_path) -> None:
    (tmp_path / "lcri_generalization_severity.csv").write_text(
        "scope,context,directional_accuracy_gap,severity\n"
        "regime,thin,0.08,critical\n"
        "transition,open,0.03,warning\n"
    )
    write_json(
        tmp_path / "lcri_generalization_severity_summary.json",
        {
            "rows": 2,
            "stable_rows": 0,
            "warning_rows": 1,
            "critical_rows": 1,
            "passes_lcri_generalization_gate": False,
        },
    )
    write_json(
        tmp_path / "lcri_worst_generalization_context.json",
        {"scope": "regime", "context": "thin", "directional_accuracy_gap": 0.08},
    )
    write_json(
        tmp_path / "lcri_generalization_gate_decision.json",
        {
            "passes": True,
            "decision": "pass",
            "rows_evaluated": 1,
            "warning_rows": 0,
            "critical_rows": 0,
            "worst_scope": "signal",
            "worst_context": "all",
            "worst_directional_accuracy_gap": 0.01,
            "reason": "stale",
        },
    )
    (tmp_path / "lcri_generalization_critical_contexts.csv").write_text(
        "scope,context,directional_accuracy_gap,severity\n"
        "signal,all,0.01,critical\n"
    )
    write_json(
        tmp_path / "lcri_generalization_blocker_summary.json",
        {
            "critical_rows": 0,
            "critical_scopes": "none",
            "max_critical_gap": 0.0,
            "max_critical_context": "none",
        },
    )

    errors = verify_lcri_generalization_gate_decision_consistency(tmp_path)

    assert "LCRI gate decision mismatch for critical_contexts" in errors[0]
    assert "LCRI gate decision mismatch for gate_decision.passes" in errors[1]
    assert "LCRI gate decision mismatch for gate_decision.decision" in errors[2]
    assert "LCRI gate decision mismatch for blocker_summary.critical_rows" in errors[-4]

def test_verify_lcri_generalization_gap_delta_reports_missing_columns(tmp_path) -> None:
    (tmp_path / "lcri_generalization_gap_delta.csv").write_text("scope,context\n")

    errors = verify_lcri_generalization_gap_delta(tmp_path)

    assert errors == [
        "incomplete LCRI generalization gap delta: "
        "['lcri_directional_accuracy_gap', 'raw_imbalance_directional_accuracy_gap', "
        "'raw_minus_lcri_directional_accuracy_gap']"
    ]


def test_verify_lcri_generalization_gap_delta_accepts_complete_csv(tmp_path) -> None:
    pd.DataFrame(
        [
            {
                "scope": "signal",
                "context": "all",
                "raw_imbalance_directional_accuracy_gap": 0.08,
                "lcri_directional_accuracy_gap": 0.05,
                "raw_minus_lcri_directional_accuracy_gap": 0.03,
            }
        ]
    ).to_csv(tmp_path / "lcri_generalization_gap_delta.csv", index=False)

    assert verify_lcri_generalization_gap_delta(tmp_path) == []


def test_verify_lcri_gap_delta_dominant_scopes_accepts_complete_payload(tmp_path) -> None:
    write_json(
        tmp_path / "lcri_gap_delta_dominant_scopes.json",
        {
            "best_scope": "transition",
            "best_mean_raw_minus_lcri_gap": 0.05,
            "worst_scope": "regime",
            "worst_mean_raw_minus_lcri_gap": -0.03,
        },
    )

    assert verify_lcri_gap_delta_dominant_scopes(tmp_path) == []


def test_verify_lcri_gap_delta_flags_accepts_complete_csv(tmp_path) -> None:
    pd.DataFrame(
        [
            {
                "scope": "signal",
                "context": "all",
                "raw_minus_lcri_directional_accuracy_gap": 0.03,
                "stability_flag": "lcri_more_stable",
            }
        ]
    ).to_csv(tmp_path / "lcri_gap_delta_flags.csv", index=False)

    assert verify_lcri_gap_delta_flags(tmp_path) == []


def test_verify_lcri_gap_delta_summary_reports_missing_keys(tmp_path) -> None:
    write_json(tmp_path / "lcri_gap_delta_summary.json", {"rows": 3})

    errors = verify_lcri_gap_delta_summary(tmp_path)

    assert errors == [
        "incomplete LCRI gap delta summary: "
        "['lcri_equal_stability_rows', 'lcri_less_stable_rows', "
        "'lcri_more_stable_rows', 'max_lcri_instability_edge', "
        "'max_lcri_instability_edge_context', 'max_lcri_stability_edge', "
        "'max_lcri_stability_edge_context']"
    ]


def test_verify_lcri_gap_delta_consistency_accepts_matching_artifacts(tmp_path) -> None:
    _write_matching_gap_delta_artifacts(tmp_path)

    assert verify_lcri_gap_delta_consistency(tmp_path) == []


def test_verify_lcri_gap_delta_consistency_reports_stale_scorecard_and_partitions(tmp_path) -> None:
    _write_matching_gap_delta_artifacts(tmp_path)
    write_json(
        tmp_path / "lcri_gap_delta_scorecard.json",
        {
            "rows": 3,
            "mean_raw_minus_lcri_directional_accuracy_gap": 0.0,
            "median_raw_minus_lcri_directional_accuracy_gap": 0.0,
            "lcri_more_stable_share": 0.0,
            "lcri_less_stable_share": 1.0,
        },
    )
    (tmp_path / "lcri_gap_delta_improvements.csv").write_text(
        "scope,context,raw_minus_lcri_directional_accuracy_gap\n"
        "signal,all,0.0\n"
    )
    write_json(
        tmp_path / "lcri_gap_delta_dominant_scopes.json",
        {
            "best_scope": "regime",
            "best_mean_raw_minus_lcri_gap": -0.02,
            "worst_scope": "transition",
            "worst_mean_raw_minus_lcri_gap": 0.04,
        },
    )

    errors = verify_lcri_gap_delta_consistency(tmp_path)

    assert any("scorecard.mean_raw_minus_lcri_directional_accuracy_gap" in error for error in errors)
    assert any("scorecard.lcri_more_stable_share" in error for error in errors)
    assert any("improvements" in error for error in errors)
    assert any("dominant_scopes.best_scope" in error for error in errors)


def test_verify_lcri_scope_stability_contradictions_accepts_complete_csv(tmp_path) -> None:
    _write_matching_scope_stability_contradiction_artifacts(tmp_path)

    assert verify_lcri_scope_stability_contradictions(tmp_path) == []


def test_verify_lcri_scope_stability_contradiction_summary_reports_missing_keys(tmp_path) -> None:
    write_json(tmp_path / "lcri_scope_stability_contradiction_summary.json", {"scopes": 3})

    errors = verify_lcri_scope_stability_contradiction_summary(tmp_path)

    assert errors == [
        "incomplete LCRI scope stability contradiction summary: "
        "['aligned_scopes', 'contradiction_scopes', 'fragility_review_required_rows', "
        "'gate_blocks_despite_relative_stability_scopes', "
        "'pass_scope_with_relative_regressions_scopes', "
        "'warning_scope_with_broad_relative_regression_scopes', 'worst_contradiction_scope']"
    ]


def test_verify_lcri_scope_stability_contradictions_consistency_accepts_matching_artifacts(tmp_path) -> None:
    _write_matching_scope_stability_contradiction_artifacts(tmp_path)

    assert verify_lcri_scope_stability_contradictions_consistency(tmp_path) == []


def test_verify_lcri_scope_stability_contradictions_consistency_reports_stale_rows(tmp_path) -> None:
    _write_matching_scope_stability_contradiction_artifacts(tmp_path)
    pd.DataFrame(
        [
            {
                "scope": "signal",
                "decision": "block",
                "rows": 1,
                "lcri_more_stable_share": 1.0,
                "lcri_less_stable_share": 0.0,
                "fragility_review_required_rows": 0,
                "contradiction_label": "aligned",
                "review_note": "stale",
            }
        ]
    ).to_csv(tmp_path / "lcri_scope_stability_contradictions.csv", index=False)

    errors = verify_lcri_scope_stability_contradictions_consistency(tmp_path)

    assert any("signal.contradiction_label" in error for error in errors)
    assert any("missing LCRI scope stability contradiction row for scope: regime" in error for error in errors)
    assert any("summary.scopes" in error for error in errors)


def test_verify_lcri_contradiction_review_packet_accepts_matching_sources(tmp_path) -> None:
    _write_matching_contradiction_review_packet_artifacts(tmp_path)

    assert verify_lcri_contradiction_review_packet(tmp_path) == []


def test_verify_lcri_contradiction_review_packet_reports_stale_evidence(tmp_path) -> None:
    _write_matching_contradiction_review_packet_artifacts(tmp_path)
    packet = pd.read_csv(tmp_path / "lcri_contradiction_review_packet.csv")
    packet.loc[packet["scope"] == "regime", "worst_delta_context"] = "stale"
    packet.to_csv(tmp_path / "lcri_contradiction_review_packet.csv", index=False)

    errors = verify_lcri_contradiction_review_packet(tmp_path)

    assert any("contradiction_review_packet.regime.worst_delta_context" in error for error in errors)


def test_verify_lcri_contradiction_review_packet_summary_accepts_matching_packet(tmp_path) -> None:
    _write_matching_contradiction_review_packet_artifacts(tmp_path)
    write_json(
        tmp_path / "lcri_contradiction_review_packet_summary.json",
        {
            "scopes": 2,
            "high_priority_scopes": 1,
            "medium_priority_scopes": 1,
            "low_priority_scopes": 0,
            "fragility_review_required_rows": 1,
            "max_review_priority": 3,
            "worst_review_scope": "regime:pass_scope_with_relative_regressions",
            "worst_fragility_scope": "regime:3.200000",
        },
    )

    assert verify_lcri_contradiction_review_packet_summary(tmp_path) == []


def test_verify_lcri_contradiction_review_packet_summary_reports_stale_counts(tmp_path) -> None:
    _write_matching_contradiction_review_packet_artifacts(tmp_path)
    write_json(
        tmp_path / "lcri_contradiction_review_packet_summary.json",
        {
            "scopes": 2,
            "high_priority_scopes": 0,
            "medium_priority_scopes": 1,
            "low_priority_scopes": 0,
            "fragility_review_required_rows": 1,
            "max_review_priority": 3,
            "worst_review_scope": "regime:pass_scope_with_relative_regressions",
            "worst_fragility_scope": "regime:3.200000",
        },
    )

    errors = verify_lcri_contradiction_review_packet_summary(tmp_path)

    assert any("contradiction_review_packet_summary.high_priority_scopes" in error for error in errors)


def test_verify_lcri_uncertainty_weighted_review_priority_accepts_matching_artifacts(tmp_path) -> None:
    _write_matching_uncertainty_weighted_priority_artifacts(tmp_path)

    assert verify_lcri_uncertainty_weighted_review_priority(tmp_path) == []
    assert verify_lcri_uncertainty_weighted_review_priority_summary(tmp_path) == []
    assert verify_lcri_uncertainty_weighted_review_priority_consistency(tmp_path) == []


def test_verify_lcri_uncertainty_weighted_review_priority_reports_stale_priority(tmp_path) -> None:
    _write_matching_uncertainty_weighted_priority_artifacts(tmp_path)
    priority = pd.read_csv(tmp_path / "lcri_uncertainty_weighted_review_priority.csv")
    priority.loc[priority["scope"] == "regime", "priority_label"] = "low"
    priority.to_csv(tmp_path / "lcri_uncertainty_weighted_review_priority.csv", index=False)

    errors = verify_lcri_uncertainty_weighted_review_priority_consistency(tmp_path)

    assert any("uncertainty_weighted_review_priority.regime.priority_label" in error for error in errors)


def test_verify_lcri_uncertainty_weighted_review_priority_reports_stale_summary(tmp_path) -> None:
    _write_matching_uncertainty_weighted_priority_artifacts(tmp_path)
    write_json(
        tmp_path / "lcri_uncertainty_weighted_review_priority_summary.json",
        {
            "scopes": 2,
            "critical_priority_scopes": 0,
            "high_priority_scopes": 0,
            "medium_priority_scopes": 1,
            "low_priority_scopes": 1,
            "max_uncertainty_weighted_priority": 4.14,
            "worst_uncertainty_weighted_scope": "signal:medium",
        },
    )

    errors = verify_lcri_uncertainty_weighted_review_priority_consistency(tmp_path)

    assert any(
        "uncertainty_weighted_review_priority_summary.critical_priority_scopes" in error
        for error in errors
    )
    assert any(
        "uncertainty_weighted_review_priority_summary.worst_uncertainty_weighted_scope" in error
        for error in errors
    )


def test_verify_lcri_cross_artifact_evidence_index_accepts_matching_artifacts(tmp_path) -> None:
    _write_matching_cross_artifact_evidence_artifacts(tmp_path)

    assert verify_lcri_cross_artifact_evidence_index(tmp_path) == []
    assert verify_lcri_cross_artifact_evidence_index_summary(tmp_path) == []
    assert verify_lcri_cross_artifact_evidence_index_consistency(tmp_path) == []


def test_verify_lcri_cross_artifact_evidence_index_reports_stale_row(tmp_path) -> None:
    _write_matching_cross_artifact_evidence_artifacts(tmp_path)
    evidence = pd.read_csv(tmp_path / "lcri_cross_artifact_evidence_index.csv")
    evidence.loc[evidence["scope"] == "regime", "evidence_label"] = "aligned"
    evidence.to_csv(tmp_path / "lcri_cross_artifact_evidence_index.csv", index=False)

    errors = verify_lcri_cross_artifact_evidence_index_consistency(tmp_path)

    assert any("cross_artifact_evidence_index.regime.evidence_label" in error for error in errors)


def test_verify_lcri_cross_artifact_evidence_index_reports_stale_summary(tmp_path) -> None:
    _write_matching_cross_artifact_evidence_artifacts(tmp_path)
    write_json(
        tmp_path / "lcri_cross_artifact_evidence_index_summary.json",
        {
            "scopes": 2,
            "urgent_scopes": 0,
            "review_scopes": 1,
            "monitor_scopes": 1,
            "aligned_scopes": 0,
            "max_evidence_score": 4.0,
            "worst_evidence_scope": "signal:monitor",
        },
    )

    errors = verify_lcri_cross_artifact_evidence_index_consistency(tmp_path)

    assert any("cross_artifact_evidence_index_summary.urgent_scopes" in error for error in errors)
    assert any("cross_artifact_evidence_index_summary.worst_evidence_scope" in error for error in errors)


def test_verify_lcri_evidence_release_checklist_accepts_matching_artifacts(tmp_path) -> None:
    _write_matching_evidence_release_checklist_artifacts(tmp_path)

    assert verify_lcri_evidence_release_checklist(tmp_path) == []
    assert verify_lcri_evidence_release_checklist_summary(tmp_path) == []
    assert verify_lcri_evidence_release_checklist_consistency(tmp_path) == []


def test_verify_lcri_evidence_release_checklist_reports_stale_row(tmp_path) -> None:
    _write_matching_evidence_release_checklist_artifacts(tmp_path)
    checklist = pd.read_csv(tmp_path / "lcri_evidence_release_checklist.csv")
    checklist.loc[checklist["scope"] == "signal", "check_status"] = "ready"
    checklist.to_csv(tmp_path / "lcri_evidence_release_checklist.csv", index=False)

    errors = verify_lcri_evidence_release_checklist_consistency(tmp_path)

    assert any("evidence_release_checklist.signal.check_status" in error for error in errors)


def test_verify_lcri_evidence_release_checklist_reports_stale_summary(tmp_path) -> None:
    _write_matching_evidence_release_checklist_artifacts(tmp_path)
    write_json(
        tmp_path / "lcri_evidence_release_checklist_summary.json",
        {
            "items": 2,
            "blocked_items": 1,
            "review_items": 1,
            "monitor_items": 0,
            "ready_items": 0,
            "max_evidence_score": 9.38,
            "worst_check_scope": "signal:blocked",
            "release_ready": True,
        },
    )

    errors = verify_lcri_evidence_release_checklist_consistency(tmp_path)

    assert any("evidence_release_checklist_summary.blocked_items" in error for error in errors)
    assert any("evidence_release_checklist_summary.release_ready" in error for error in errors)


def test_verify_lcri_owner_handoff_packet_accepts_matching_artifacts(tmp_path) -> None:
    _write_matching_owner_handoff_packet_artifacts(tmp_path)

    assert verify_lcri_owner_handoff_packet(tmp_path) == []
    assert verify_lcri_owner_handoff_packet_summary(tmp_path) == []
    assert verify_lcri_owner_handoff_packet_consistency(tmp_path) == []
    assert verify_lcri_owner_handoff_markdown_packet(tmp_path) == []


def test_verify_lcri_owner_handoff_packet_reports_stale_row(tmp_path) -> None:
    _write_matching_owner_handoff_packet_artifacts(tmp_path)
    packet = pd.read_csv(tmp_path / "lcri_owner_handoff_packet.csv")
    packet.loc[packet["scope"] == "regime", "handoff_status"] = "signoff_ready"
    packet.to_csv(tmp_path / "lcri_owner_handoff_packet.csv", index=False)

    errors = verify_lcri_owner_handoff_packet_consistency(tmp_path)

    assert any("owner_handoff_packet.regime.handoff_status" in error for error in errors)


def test_verify_lcri_owner_handoff_packet_reports_stale_summary(tmp_path) -> None:
    _write_matching_owner_handoff_packet_artifacts(tmp_path)
    write_json(
        tmp_path / "lcri_owner_handoff_packet_summary.json",
        {
            "items": 2,
            "immediate_items": 0,
            "review_items": 0,
            "monitor_items": 0,
            "signoff_items": 2,
            "max_evidence_score": 11.25,
            "top_handoff_scope": "regime:signoff_ready",
            "handoff_clear": True,
        },
    )

    errors = verify_lcri_owner_handoff_packet_consistency(tmp_path)

    assert any("owner_handoff_packet_summary.immediate_items" in error for error in errors)
    assert any("owner_handoff_packet_summary.handoff_clear" in error for error in errors)


def test_verify_lcri_owner_handoff_markdown_packet_reports_stale_summary(tmp_path) -> None:
    _write_matching_owner_handoff_packet_artifacts(tmp_path)
    text = (tmp_path / "lcri_owner_handoff_packet.md").read_text()
    (tmp_path / "lcri_owner_handoff_packet.md").write_text(
        text.replace("- immediate_items: 2", "- immediate_items: 0")
    )

    errors = verify_lcri_owner_handoff_markdown_packet(tmp_path)

    assert "stale LCRI owner handoff markdown summary field: immediate_items" in errors


def test_verify_lcri_evidence_lineage_map_accepts_matching_artifacts(tmp_path) -> None:
    _write_matching_evidence_lineage_map_artifacts(tmp_path)

    assert verify_lcri_evidence_lineage_map(tmp_path) == []
    assert verify_lcri_evidence_lineage_map_summary(tmp_path) == []
    assert verify_lcri_evidence_lineage_map_consistency(tmp_path) == []


def test_verify_lcri_evidence_lineage_map_reports_stale_row(tmp_path) -> None:
    _write_matching_evidence_lineage_map_artifacts(tmp_path)
    lineage = pd.read_csv(tmp_path / "lcri_evidence_lineage_map.csv")
    lineage.loc[lineage["scope"] == "signal", "lineage_status"] = "incomplete_lineage"
    lineage.to_csv(tmp_path / "lcri_evidence_lineage_map.csv", index=False)

    errors = verify_lcri_evidence_lineage_map_consistency(tmp_path)

    assert any("evidence_lineage_map.signal.lineage_status" in error for error in errors)


def test_verify_lcri_evidence_lineage_map_reports_stale_summary(tmp_path) -> None:
    _write_matching_evidence_lineage_map_artifacts(tmp_path)
    write_json(
        tmp_path / "lcri_evidence_lineage_map_summary.json",
        {
            "scopes": 2,
            "complete_scopes": 1,
            "source_mismatch_scopes": 1,
            "incomplete_scopes": 0,
            "max_evidence_score": 11.25,
            "worst_lineage_scope": "signal:source_mismatch",
            "lineage_clear": False,
        },
    )

    errors = verify_lcri_evidence_lineage_map_consistency(tmp_path)

    assert any("evidence_lineage_map_summary.complete_scopes" in error for error in errors)
    assert any("evidence_lineage_map_summary.lineage_clear" in error for error in errors)


def test_verify_lcri_gap_delta_improvements_accepts_complete_csv(tmp_path) -> None:
    (tmp_path / "lcri_gap_delta_improvements.csv").write_text(
        "scope,context,raw_minus_lcri_directional_accuracy_gap\n"
        "transition,transition,0.06\n"
    )

    assert verify_lcri_gap_delta_improvements(tmp_path) == []


def test_verify_lcri_gap_delta_regressions_accepts_complete_csv(tmp_path) -> None:
    (tmp_path / "lcri_gap_delta_regressions.csv").write_text(
        "scope,context,raw_minus_lcri_directional_accuracy_gap\n"
        "regime,thin,-0.04\n"
    )

    assert verify_lcri_gap_delta_regressions(tmp_path) == []


def test_verify_lcri_gap_delta_scorecard_accepts_complete_payload(tmp_path) -> None:
    write_json(
        tmp_path / "lcri_gap_delta_scorecard.json",
        {
            "rows": 3,
            "mean_raw_minus_lcri_directional_accuracy_gap": 0.02,
            "median_raw_minus_lcri_directional_accuracy_gap": 0.01,
            "lcri_more_stable_share": 0.67,
            "lcri_less_stable_share": 0.33,
        },
    )

    assert verify_lcri_gap_delta_scorecard(tmp_path) == []


def test_verify_lcri_gap_delta_scope_extremes_accepts_complete_csv(tmp_path) -> None:
    (tmp_path / "lcri_gap_delta_scope_extremes.csv").write_text(
        "scope,best_context,best_raw_minus_lcri_gap,worst_context,worst_raw_minus_lcri_gap\n"
        "regime,deep,0.06,thin,-0.04\n"
    )

    assert verify_lcri_gap_delta_scope_extremes(tmp_path) == []


def test_verify_lcri_gap_delta_scope_summary_accepts_complete_csv(tmp_path) -> None:
    (tmp_path / "lcri_gap_delta_scope_summary.csv").write_text(
        "scope,rows,mean_raw_minus_lcri_gap,min_raw_minus_lcri_gap,"
        "max_raw_minus_lcri_gap,lcri_more_stable_share,lcri_less_stable_share\n"
        "regime,2,0.01,-0.04,0.06,0.5,0.5\n"
    )

    assert verify_lcri_gap_delta_scope_summary(tmp_path) == []


def test_verify_lcri_gap_delta_scope_summary_rejects_missing_share_columns(tmp_path) -> None:
    (tmp_path / "lcri_gap_delta_scope_summary.csv").write_text(
        "scope,rows,mean_raw_minus_lcri_gap,min_raw_minus_lcri_gap,max_raw_minus_lcri_gap\n"
        "regime,2,0.01,-0.04,0.06\n"
    )

    errors = verify_lcri_gap_delta_scope_summary(tmp_path)

    assert errors
    assert "lcri_more_stable_share" in errors[0]
    assert "lcri_less_stable_share" in errors[0]


def test_verify_lcri_gap_delta_summary_accepts_complete_payload(tmp_path) -> None:
    write_json(
        tmp_path / "lcri_gap_delta_summary.json",
        {
            "rows": 3,
            "lcri_more_stable_rows": 2,
            "lcri_less_stable_rows": 1,
            "lcri_equal_stability_rows": 0,
            "max_lcri_stability_edge": 0.03,
            "max_lcri_stability_edge_context": "signal:all",
            "max_lcri_instability_edge": -0.04,
            "max_lcri_instability_edge_context": "regime:thin",
        },
    )

    assert verify_lcri_gap_delta_summary(tmp_path) == []


def test_missing_artifacts_reports_absent_paths(tmp_path) -> None:
    (tmp_path / "metrics.csv").write_text("signal\n")
    (tmp_path / "figures").mkdir()
    (tmp_path / "figures" / "plot.png").write_text("png")

    missing = missing_artifacts(
        tmp_path,
        ["metrics.csv", "transition_lift.csv", "figures/plot.png"],
    )

    assert missing == ["transition_lift.csv"]


def test_verify_research_summary_sections_accepts_matching_artifacts(tmp_path) -> None:
    pd.DataFrame(
        [
            {
                "signal": "lcri",
                "directional_accuracy": 0.62,
                "brier_score": 0.21,
            }
        ]
    ).to_csv(tmp_path / "metrics.csv", index=False)
    write_json(
        tmp_path / "generalization_overview.json",
        {
            "signal_rows": 2,
            "max_signal_directional_accuracy_gap": 0.03,
        },
    )
    (tmp_path / "research_summary.md").write_text(
        "\n".join(
            [
                "# LCRI Research Summary",
                "",
                "## Signal quality",
                "",
                "| signal | directional_accuracy | brier_score |",
                "| --- | --- | --- |",
                "| lcri | 0.620000 | 0.210000 |",
                "",
                "## Generalization overview",
                "",
                "- signal_rows: 2",
                "- max_signal_directional_accuracy_gap: 0.030000",
                "",
            ]
        )
    )

    assert verify_research_summary_sections(tmp_path) == []


def test_verify_research_summary_sections_reports_stale_placeholder(tmp_path) -> None:
    pd.DataFrame([{"signal": "lcri", "directional_accuracy": 0.62}]).to_csv(
        tmp_path / "metrics.csv",
        index=False,
    )
    (tmp_path / "research_summary.md").write_text(
        "# LCRI Research Summary\n\n## Signal quality\n\n_Not generated._\n"
    )

    errors = verify_research_summary_sections(tmp_path)

    assert errors == ["stale research summary section for generated artifact: Signal quality"]


def test_verify_research_summary_sections_reports_missing_json_key(tmp_path) -> None:
    write_json(
        tmp_path / "lcri_gap_delta_summary.json",
        {
            "rows": 2,
            "lcri_more_stable_rows": 1,
        },
    )
    (tmp_path / "research_summary.md").write_text(
        "# LCRI Research Summary\n\n## LCRI gap delta summary\n\n- rows: 2\n"
    )

    errors = verify_research_summary_sections(tmp_path)

    assert errors == [
        "research summary section missing JSON keys for LCRI gap delta summary: "
        "['lcri_more_stable_rows']"
    ]


def test_verify_research_summary_sections_reports_stale_json_value(tmp_path) -> None:
    write_json(
        tmp_path / "generalization_fragility_summary.json",
        {
            "rows": 3,
            "max_abs_gap_to_se_ratio": 2.5,
        },
    )
    (tmp_path / "research_summary.md").write_text(
        "\n".join(
            [
                "# LCRI Research Summary",
                "",
                "## Generalization fragility summary",
                "",
                "- rows: 2",
                "- max_abs_gap_to_se_ratio: 2.500000",
                "",
            ]
        )
    )

    errors = verify_research_summary_sections(tmp_path)

    assert errors == [
        "research summary section JSON values mismatch for Generalization fragility summary: "
        "['rows']"
    ]


def test_verify_research_summary_sections_reports_missing_csv_column(tmp_path) -> None:
    pd.DataFrame([{"scope": "signal", "context": "all"}]).to_csv(
        tmp_path / "lcri_generalization_gap_leaderboard.csv",
        index=False,
    )
    (tmp_path / "research_summary.md").write_text(
        "# LCRI Research Summary\n\n## LCRI generalization gap leaderboard\n\n| scope |\n| --- |\n| signal |\n"
    )

    errors = verify_research_summary_sections(tmp_path)

    assert errors == [
        "research summary section missing CSV columns for LCRI generalization gap leaderboard: "
        "['context']"
    ]


def test_verify_research_summary_sections_reports_stale_csv_value(tmp_path) -> None:
    pd.DataFrame(
        [
            {
                "scope": "transition",
                "context": "post_absorption",
                "directional_accuracy_gap": 0.071,
            }
        ]
    ).to_csv(tmp_path / "lcri_generalization_severity.csv", index=False)
    (tmp_path / "research_summary.md").write_text(
        "\n".join(
            [
                "# LCRI Research Summary",
                "",
                "## LCRI generalization severity",
                "",
                "| scope | context | directional_accuracy_gap |",
                "| --- | --- | --- |",
                "| transition | post_absorption | 0.070000 |",
                "",
            ]
        )
    )

    errors = verify_research_summary_sections(tmp_path)

    assert errors == ["research summary section CSV values mismatch for LCRI generalization severity"]


def test_write_research_summary_marks_missing_generalization_gap(tmp_path) -> None:
    path = tmp_path / "summary.md"
    metrics = pd.DataFrame(
        [
            {
                "signal": "raw_imbalance",
                "directional_accuracy": 0.55,
                "brier_score": 0.30,
                "rank_correlation": 0.10,
            }
        ]
    )
    transition_lift = pd.DataFrame(
        [{"segment": "stable", "rows": 3, "directional_accuracy_lift": 0.10}]
    )

    write_research_summary(
        path,
        rows=10,
        train_rows=7,
        heldout_rows=3,
        seed=4,
        train_frac=0.7,
        metrics=metrics,
        transition_lift=transition_lift,
        transition_robustness={},
    )

    text = path.read_text()
    assert "## Signal generalization gap" in text
    assert "_Not generated._" in text


def test_write_research_summary_marks_missing_regime_generalization_gap(tmp_path) -> None:
    path = tmp_path / "summary.md"
    metrics = pd.DataFrame(
        [
            {
                "signal": "raw_imbalance",
                "directional_accuracy": 0.55,
                "brier_score": 0.30,
                "rank_correlation": 0.10,
            }
        ]
    )
    transition_lift = pd.DataFrame(
        [{"segment": "stable", "rows": 3, "directional_accuracy_lift": 0.10}]
    )

    write_research_summary(
        path,
        rows=10,
        train_rows=7,
        heldout_rows=3,
        seed=4,
        train_frac=0.7,
        metrics=metrics,
        transition_lift=transition_lift,
        transition_robustness={},
    )

    text = path.read_text()
    assert "## Regime generalization gap" in text
    assert text.count("_Not generated._") >= 2


def test_write_research_summary_marks_missing_generalization_leaderboard(tmp_path) -> None:
    path = tmp_path / "summary.md"
    metrics = pd.DataFrame(
        [
            {
                "signal": "raw_imbalance",
                "directional_accuracy": 0.55,
                "brier_score": 0.30,
                "rank_correlation": 0.10,
            }
        ]
    )
    transition_lift = pd.DataFrame(
        [{"segment": "stable", "rows": 3, "directional_accuracy_lift": 0.10}]
    )

    write_research_summary(
        path,
        rows=10,
        train_rows=7,
        heldout_rows=3,
        seed=4,
        train_frac=0.7,
        metrics=metrics,
        transition_lift=transition_lift,
        transition_robustness={},
    )

    text = path.read_text()
    assert "## Generalization gap leaderboard" in text
    assert text.count("_Not generated._") >= 5


def test_write_research_summary_marks_missing_generalization_overview(tmp_path) -> None:
    path = tmp_path / "summary.md"
    metrics = pd.DataFrame(
        [
            {
                "signal": "raw_imbalance",
                "directional_accuracy": 0.55,
                "brier_score": 0.30,
                "rank_correlation": 0.10,
            }
        ]
    )
    transition_lift = pd.DataFrame(
        [{"segment": "stable", "rows": 3, "directional_accuracy_lift": 0.10}]
    )

    write_research_summary(
        path,
        rows=10,
        train_rows=7,
        heldout_rows=3,
        seed=4,
        train_frac=0.7,
        metrics=metrics,
        transition_lift=transition_lift,
        transition_robustness={},
    )

    text = path.read_text()
    assert "## Generalization overview" in text
    assert text.count("_Not generated._") >= 4


def test_write_research_summary_marks_missing_transition_generalization_gap(tmp_path) -> None:
    path = tmp_path / "summary.md"
    metrics = pd.DataFrame(
        [
            {
                "signal": "raw_imbalance",
                "directional_accuracy": 0.55,
                "brier_score": 0.30,
                "rank_correlation": 0.10,
            }
        ]
    )
    transition_lift = pd.DataFrame(
        [{"segment": "stable", "rows": 3, "directional_accuracy_lift": 0.10}]
    )

    write_research_summary(
        path,
        rows=10,
        train_rows=7,
        heldout_rows=3,
        seed=4,
        train_frac=0.7,
        metrics=metrics,
        transition_lift=transition_lift,
        transition_robustness={},
    )

    text = path.read_text()
    assert "## Transition generalization gap" in text
    assert text.count("_Not generated._") >= 3


def test_write_json_writes_sorted_pretty_payload(tmp_path) -> None:
    path = tmp_path / "payload.json"

    write_json(path, {"b": 2, "a": True})

    assert json.loads(path.read_text()) == {"a": True, "b": 2}
    assert path.read_text().startswith('{\n  "a"')


def _write_matching_gap_delta_artifacts(tmp_path) -> None:
    pd.DataFrame(
        [
            {
                "scope": "transition",
                "context": "news",
                "raw_imbalance_directional_accuracy_gap": 0.08,
                "lcri_directional_accuracy_gap": 0.04,
                "raw_minus_lcri_directional_accuracy_gap": 0.04,
            },
            {
                "scope": "regime",
                "context": "thin",
                "raw_imbalance_directional_accuracy_gap": 0.03,
                "lcri_directional_accuracy_gap": 0.05,
                "raw_minus_lcri_directional_accuracy_gap": -0.02,
            },
            {
                "scope": "signal",
                "context": "all",
                "raw_imbalance_directional_accuracy_gap": 0.02,
                "lcri_directional_accuracy_gap": 0.02,
                "raw_minus_lcri_directional_accuracy_gap": 0.0,
            },
        ]
    ).to_csv(tmp_path / "lcri_generalization_gap_delta.csv", index=False)
    pd.DataFrame(
        [
            {
                "scope": "transition",
                "context": "news",
                "raw_imbalance_directional_accuracy_gap": 0.08,
                "lcri_directional_accuracy_gap": 0.04,
                "raw_minus_lcri_directional_accuracy_gap": 0.04,
                "stability_flag": "lcri_more_stable",
            },
            {
                "scope": "regime",
                "context": "thin",
                "raw_imbalance_directional_accuracy_gap": 0.03,
                "lcri_directional_accuracy_gap": 0.05,
                "raw_minus_lcri_directional_accuracy_gap": -0.02,
                "stability_flag": "lcri_less_stable",
            },
            {
                "scope": "signal",
                "context": "all",
                "raw_imbalance_directional_accuracy_gap": 0.02,
                "lcri_directional_accuracy_gap": 0.02,
                "raw_minus_lcri_directional_accuracy_gap": 0.0,
                "stability_flag": "lcri_equal_stability",
            },
        ]
    ).to_csv(tmp_path / "lcri_gap_delta_flags.csv", index=False)
    pd.DataFrame(
        [
            {
                "scope": "transition",
                "context": "news",
                "raw_minus_lcri_directional_accuracy_gap": 0.04,
            }
        ]
    ).to_csv(tmp_path / "lcri_gap_delta_improvements.csv", index=False)
    pd.DataFrame(
        [
            {
                "scope": "regime",
                "context": "thin",
                "raw_minus_lcri_directional_accuracy_gap": -0.02,
            }
        ]
    ).to_csv(tmp_path / "lcri_gap_delta_regressions.csv", index=False)
    pd.DataFrame(
        [
            {
                "scope": "regime",
                "rows": 1,
                "mean_raw_minus_lcri_gap": -0.02,
                "min_raw_minus_lcri_gap": -0.02,
                "max_raw_minus_lcri_gap": -0.02,
                "lcri_more_stable_share": 0.0,
                "lcri_less_stable_share": 1.0,
            },
            {
                "scope": "signal",
                "rows": 1,
                "mean_raw_minus_lcri_gap": 0.0,
                "min_raw_minus_lcri_gap": 0.0,
                "max_raw_minus_lcri_gap": 0.0,
                "lcri_more_stable_share": 0.0,
                "lcri_less_stable_share": 0.0,
            },
            {
                "scope": "transition",
                "rows": 1,
                "mean_raw_minus_lcri_gap": 0.04,
                "min_raw_minus_lcri_gap": 0.04,
                "max_raw_minus_lcri_gap": 0.04,
                "lcri_more_stable_share": 1.0,
                "lcri_less_stable_share": 0.0,
            },
        ]
    ).to_csv(tmp_path / "lcri_gap_delta_scope_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scope": "regime",
                "best_context": "thin",
                "best_raw_minus_lcri_gap": -0.02,
                "worst_context": "thin",
                "worst_raw_minus_lcri_gap": -0.02,
            },
            {
                "scope": "signal",
                "best_context": "all",
                "best_raw_minus_lcri_gap": 0.0,
                "worst_context": "all",
                "worst_raw_minus_lcri_gap": 0.0,
            },
            {
                "scope": "transition",
                "best_context": "news",
                "best_raw_minus_lcri_gap": 0.04,
                "worst_context": "news",
                "worst_raw_minus_lcri_gap": 0.04,
            },
        ]
    ).to_csv(tmp_path / "lcri_gap_delta_scope_extremes.csv", index=False)
    write_json(
        tmp_path / "lcri_gap_delta_scorecard.json",
        {
            "rows": 3,
            "mean_raw_minus_lcri_directional_accuracy_gap": 0.006666666666666667,
            "median_raw_minus_lcri_directional_accuracy_gap": 0.0,
            "lcri_more_stable_share": 1 / 3,
            "lcri_less_stable_share": 1 / 3,
        },
    )
    write_json(
        tmp_path / "lcri_gap_delta_summary.json",
        {
            "rows": 3,
            "lcri_more_stable_rows": 1,
            "lcri_less_stable_rows": 1,
            "lcri_equal_stability_rows": 1,
            "max_lcri_stability_edge": 0.04,
            "max_lcri_stability_edge_context": "transition:news",
            "max_lcri_instability_edge": -0.02,
            "max_lcri_instability_edge_context": "regime:thin",
        },
    )
    write_json(
        tmp_path / "lcri_gap_delta_dominant_scopes.json",
        {
            "best_scope": "transition",
            "best_mean_raw_minus_lcri_gap": 0.04,
            "worst_scope": "regime",
            "worst_mean_raw_minus_lcri_gap": -0.02,
        },
    )


def _write_matching_scope_stability_contradiction_artifacts(tmp_path) -> None:
    pd.DataFrame(
        [
            {"scope": "signal", "rows": 1, "decision": "block"},
            {"scope": "regime", "rows": 2, "decision": "pass"},
            {"scope": "transition", "rows": 2, "decision": "warn"},
        ]
    ).to_csv(tmp_path / "lcri_generalization_scope_gate_decisions.csv", index=False)
    pd.DataFrame(
        [
            {
                "scope": "signal",
                "rows": 1,
                "mean_raw_minus_lcri_gap": 0.04,
                "min_raw_minus_lcri_gap": 0.04,
                "max_raw_minus_lcri_gap": 0.04,
                "lcri_more_stable_share": 1.0,
                "lcri_less_stable_share": 0.0,
            },
            {
                "scope": "regime",
                "rows": 2,
                "mean_raw_minus_lcri_gap": 0.0,
                "min_raw_minus_lcri_gap": -0.02,
                "max_raw_minus_lcri_gap": 0.02,
                "lcri_more_stable_share": 0.5,
                "lcri_less_stable_share": 0.5,
            },
            {
                "scope": "transition",
                "rows": 2,
                "mean_raw_minus_lcri_gap": -0.03,
                "min_raw_minus_lcri_gap": -0.04,
                "max_raw_minus_lcri_gap": -0.02,
                "lcri_more_stable_share": 0.0,
                "lcri_less_stable_share": 1.0,
            },
        ]
    ).to_csv(tmp_path / "lcri_gap_delta_scope_summary.csv", index=False)
    pd.DataFrame(
        [
            {"scope": "signal", "alignment_label": "aligned"},
            {"scope": "regime", "alignment_label": "uncertainty_fragile_noncritical"},
            {"scope": "regime", "alignment_label": "aligned"},
            {"scope": "transition", "alignment_label": "gate_blocks_stable_slice"},
        ]
    ).to_csv(tmp_path / "lcri_fragility_gate_alignment.csv", index=False)
    pd.DataFrame(
        [
            {
                "scope": "signal",
                "decision": "block",
                "rows": 1,
                "lcri_more_stable_share": 1.0,
                "lcri_less_stable_share": 0.0,
                "fragility_review_required_rows": 0,
                "contradiction_label": "gate_blocks_despite_relative_stability",
                "review_note": "absolute LCRI gate blocks while LCRI is usually more stable than raw imbalance in this scope",
            },
            {
                "scope": "regime",
                "decision": "pass",
                "rows": 2,
                "lcri_more_stable_share": 0.5,
                "lcri_less_stable_share": 0.5,
                "fragility_review_required_rows": 1,
                "contradiction_label": "pass_scope_with_relative_regressions",
                "review_note": "absolute LCRI gate passes but at least one context is less stable than raw imbalance",
            },
            {
                "scope": "transition",
                "decision": "warn",
                "rows": 2,
                "lcri_more_stable_share": 0.0,
                "lcri_less_stable_share": 1.0,
                "fragility_review_required_rows": 1,
                "contradiction_label": "warning_scope_with_broad_relative_regression",
                "review_note": "warning scope also shows broad relative regression versus raw imbalance",
            },
        ]
    ).to_csv(tmp_path / "lcri_scope_stability_contradictions.csv", index=False)
    write_json(
        tmp_path / "lcri_scope_stability_contradiction_summary.json",
        {
            "scopes": 3,
            "aligned_scopes": 0,
            "contradiction_scopes": 3,
            "gate_blocks_despite_relative_stability_scopes": 1,
            "pass_scope_with_relative_regressions_scopes": 1,
            "warning_scope_with_broad_relative_regression_scopes": 1,
            "fragility_review_required_rows": 2,
            "worst_contradiction_scope": "transition:warning_scope_with_broad_relative_regression",
        },
    )


def _write_matching_contradiction_review_packet_artifacts(tmp_path) -> None:
    pd.DataFrame(
        [
            {
                "scope": "signal",
                "decision": "block",
                "rows": 1,
                "lcri_more_stable_share": 1.0,
                "lcri_less_stable_share": 0.0,
                "fragility_review_required_rows": 0,
                "contradiction_label": "gate_blocks_despite_relative_stability",
                "review_note": "absolute LCRI gate blocks while LCRI is usually more stable than raw imbalance in this scope",
            },
            {
                "scope": "regime",
                "decision": "pass",
                "rows": 2,
                "lcri_more_stable_share": 0.5,
                "lcri_less_stable_share": 0.5,
                "fragility_review_required_rows": 1,
                "contradiction_label": "pass_scope_with_relative_regressions",
                "review_note": "absolute LCRI gate passes but at least one context is less stable than raw imbalance",
            },
        ]
    ).to_csv(tmp_path / "lcri_scope_stability_contradictions.csv", index=False)
    pd.DataFrame(
        [
            {
                "scope": "signal",
                "context": "all",
                "directional_accuracy_gap": 0.07,
                "severity": "critical",
            },
            {
                "scope": "regime",
                "context": "thin",
                "directional_accuracy_gap": 0.03,
                "severity": "warning",
            },
            {
                "scope": "regime",
                "context": "deep",
                "directional_accuracy_gap": 0.05,
                "severity": "critical",
            },
        ]
    ).to_csv(tmp_path / "lcri_generalization_severity.csv", index=False)
    pd.DataFrame(
        [
            {
                "scope": "signal",
                "context": "all",
                "raw_minus_lcri_directional_accuracy_gap": 0.02,
            },
            {
                "scope": "regime",
                "context": "thin",
                "raw_minus_lcri_directional_accuracy_gap": -0.04,
            },
            {
                "scope": "regime",
                "context": "deep",
                "raw_minus_lcri_directional_accuracy_gap": 0.01,
            },
        ]
    ).to_csv(tmp_path / "lcri_generalization_gap_delta.csv", index=False)
    pd.DataFrame(
        [
            {
                "scope": "signal",
                "context": "all",
                "alignment_label": "aligned",
                "abs_gap_to_se_ratio": 0.8,
            },
            {
                "scope": "regime",
                "context": "thin",
                "alignment_label": "uncertainty_fragile_noncritical",
                "abs_gap_to_se_ratio": 3.2,
            },
            {
                "scope": "regime",
                "context": "deep",
                "alignment_label": "aligned",
                "abs_gap_to_se_ratio": 1.1,
            },
        ]
    ).to_csv(tmp_path / "lcri_fragility_gate_alignment.csv", index=False)
    pd.DataFrame(
        [
            {
                "scope": "regime",
                "contradiction_label": "pass_scope_with_relative_regressions",
                "decision": "pass",
                "scope_rows": 2,
                "lcri_less_stable_share": 0.5,
                "fragility_review_required_rows": 1,
                "worst_gate_context": "deep",
                "worst_gate_severity": "critical",
                "worst_gate_directional_accuracy_gap": 0.05,
                "worst_delta_context": "thin",
                "worst_raw_minus_lcri_directional_accuracy_gap": -0.04,
                "worst_fragility_context": "thin",
                "worst_fragility_alignment_label": "uncertainty_fragile_noncritical",
                "worst_fragility_abs_gap_to_se_ratio": 3.2,
                "review_priority": 3,
            },
            {
                "scope": "signal",
                "contradiction_label": "gate_blocks_despite_relative_stability",
                "decision": "block",
                "scope_rows": 1,
                "lcri_less_stable_share": 0.0,
                "fragility_review_required_rows": 0,
                "worst_gate_context": "all",
                "worst_gate_severity": "critical",
                "worst_gate_directional_accuracy_gap": 0.07,
                "worst_delta_context": "all",
                "worst_raw_minus_lcri_directional_accuracy_gap": 0.02,
                "worst_fragility_context": "all",
                "worst_fragility_alignment_label": "aligned",
                "worst_fragility_abs_gap_to_se_ratio": 0.8,
                "review_priority": 2,
            },
        ]
    ).to_csv(tmp_path / "lcri_contradiction_review_packet.csv", index=False)


def _write_matching_uncertainty_weighted_priority_artifacts(tmp_path) -> None:
    _write_matching_contradiction_review_packet_artifacts(tmp_path)
    pd.DataFrame(
        [
            {
                "scope": "signal",
                "rows": 1,
                "mean_ci_width": 0.10,
                "max_ci_width": 0.12,
                "wide_ci_rows": 0,
                "wide_ci_share": 0.0,
                "gap_exceeds_ci_half_width_rows": 0,
                "gap_exceeds_ci_half_width_share": 0.0,
                "ci_gate_contradiction_rows": 1,
                "high_priority_ci_gate_rows": 1,
                "max_ci_gate_review_priority": 3,
                "coverage_label": "blocking_ci_gate_review",
                "review_note": "scope has high-priority CI/gate disagreement; review before accepting gate owner decision",
            },
            {
                "scope": "regime",
                "rows": 2,
                "mean_ci_width": 0.24,
                "max_ci_width": 0.30,
                "wide_ci_rows": 2,
                "wide_ci_share": 1.0,
                "gap_exceeds_ci_half_width_rows": 1,
                "gap_exceeds_ci_half_width_share": 0.5,
                "ci_gate_contradiction_rows": 2,
                "high_priority_ci_gate_rows": 0,
                "max_ci_gate_review_priority": 2,
                "coverage_label": "ci_gate_contradiction_review",
                "review_note": "scope has CI/gate disagreements; inspect uncertainty-qualified gate rows",
            },
        ]
    ).to_csv(tmp_path / "lcri_ci_confidence_coverage_scorecard.csv", index=False)
    pd.DataFrame(
        [
            {
                "scope": "regime",
                "contradiction_label": "pass_scope_with_relative_regressions",
                "base_review_priority": 3,
                "fragility_review_required_rows": 1,
                "worst_fragility_abs_gap_to_se_ratio": 3.2,
                "coverage_label": "ci_gate_contradiction_review",
                "mean_ci_width": 0.24,
                "max_ci_width": 0.30,
                "wide_ci_share": 1.0,
                "ci_gate_contradiction_rows": 2,
                "high_priority_ci_gate_rows": 0,
                "uncertainty_weighted_priority": 7.2,
                "priority_label": "critical",
                "review_note": "review first: contradiction evidence is amplified by ci_gate_contradiction_review and heldout uncertainty",
            },
            {
                "scope": "signal",
                "contradiction_label": "gate_blocks_despite_relative_stability",
                "base_review_priority": 2,
                "fragility_review_required_rows": 0,
                "worst_fragility_abs_gap_to_se_ratio": 0.8,
                "coverage_label": "blocking_ci_gate_review",
                "mean_ci_width": 0.10,
                "max_ci_width": 0.12,
                "wide_ci_share": 0.0,
                "ci_gate_contradiction_rows": 1,
                "high_priority_ci_gate_rows": 1,
                "uncertainty_weighted_priority": 4.14,
                "priority_label": "medium",
                "review_note": "schedule review after critical/high scopes; blocking_ci_gate_review uncertainty evidence is non-trivial",
            },
        ]
    ).to_csv(tmp_path / "lcri_uncertainty_weighted_review_priority.csv", index=False)
    write_json(
        tmp_path / "lcri_uncertainty_weighted_review_priority_summary.json",
        {
            "scopes": 2,
            "critical_priority_scopes": 1,
            "high_priority_scopes": 0,
            "medium_priority_scopes": 1,
            "low_priority_scopes": 0,
            "max_uncertainty_weighted_priority": 7.2,
            "worst_uncertainty_weighted_scope": "regime:critical",
        },
    )


def _write_matching_cross_artifact_evidence_artifacts(tmp_path) -> None:
    _write_matching_uncertainty_weighted_priority_artifacts(tmp_path)
    pd.DataFrame(
        [
            {
                "scope": "signal",
                "rows": 1,
                "stable_rows": 0,
                "warning_rows": 0,
                "critical_rows": 1,
            },
            {
                "scope": "regime",
                "rows": 2,
                "stable_rows": 0,
                "warning_rows": 1,
                "critical_rows": 1,
            },
        ]
    ).to_csv(tmp_path / "lcri_generalization_severity_by_scope.csv", index=False)
    pd.DataFrame(
        [
            {"scope": "signal", "rows": 1, "decision": "block", "reason": "critical"},
            {"scope": "regime", "rows": 2, "decision": "pass", "reason": "ok"},
        ]
    ).to_csv(tmp_path / "lcri_generalization_scope_gate_decisions.csv", index=False)
    pd.DataFrame(
        [
            {
                "scope": "signal",
                "rows": 1,
                "mean_raw_minus_lcri_gap": 0.02,
                "min_raw_minus_lcri_gap": 0.02,
                "max_raw_minus_lcri_gap": 0.02,
                "lcri_more_stable_share": 1.0,
                "lcri_less_stable_share": 0.0,
            },
            {
                "scope": "regime",
                "rows": 2,
                "mean_raw_minus_lcri_gap": -0.015,
                "min_raw_minus_lcri_gap": -0.04,
                "max_raw_minus_lcri_gap": 0.01,
                "lcri_more_stable_share": 0.5,
                "lcri_less_stable_share": 0.5,
            },
        ]
    ).to_csv(tmp_path / "lcri_gap_delta_scope_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "scope": "regime",
                "gate_decision": "pass",
                "severity_rows": 2,
                "warning_rows": 1,
                "critical_rows": 1,
                "lcri_more_stable_share": 0.5,
                "lcri_less_stable_share": 0.5,
                "fragility_review_required_rows": 1,
                "ci_gate_contradiction_rows": 2,
                "high_priority_ci_gate_rows": 0,
                "contradiction_label": "pass_scope_with_relative_regressions",
                "priority_label": "critical",
                "uncertainty_weighted_priority": 7.2,
                "evidence_score": 11.25,
                "evidence_label": "urgent",
                "review_note": "owner review first: pass gate with pass_scope_with_relative_regressions cross-artifact evidence",
            },
            {
                "scope": "signal",
                "gate_decision": "block",
                "severity_rows": 1,
                "warning_rows": 0,
                "critical_rows": 1,
                "lcri_more_stable_share": 1.0,
                "lcri_less_stable_share": 0.0,
                "fragility_review_required_rows": 0,
                "ci_gate_contradiction_rows": 1,
                "high_priority_ci_gate_rows": 1,
                "contradiction_label": "gate_blocks_despite_relative_stability",
                "priority_label": "medium",
                "uncertainty_weighted_priority": 4.14,
                "evidence_score": 9.38,
                "evidence_label": "urgent",
                "review_note": "owner review first: block gate with gate_blocks_despite_relative_stability cross-artifact evidence",
            },
        ]
    ).to_csv(tmp_path / "lcri_cross_artifact_evidence_index.csv", index=False)
    write_json(
        tmp_path / "lcri_cross_artifact_evidence_index_summary.json",
        {
            "scopes": 2,
            "urgent_scopes": 2,
            "review_scopes": 0,
            "monitor_scopes": 0,
            "aligned_scopes": 0,
            "max_evidence_score": 11.25,
            "worst_evidence_scope": "regime:urgent",
        },
    )


def _write_matching_evidence_release_checklist_artifacts(tmp_path) -> None:
    _write_matching_cross_artifact_evidence_artifacts(tmp_path)
    pd.DataFrame(
        [
            {
                "scope": "regime",
                "check_status": "blocked",
                "checklist_item": "regime release evidence reconciliation",
                "gate_decision": "pass",
                "evidence_label": "urgent",
                "evidence_score": 11.25,
                "priority_label": "critical",
                "required_action": "resolve or explicitly waive regime evidence before release sign-off",
                "source_artifact": "lcri_cross_artifact_evidence_index.csv",
            },
            {
                "scope": "signal",
                "check_status": "blocked",
                "checklist_item": "signal release evidence reconciliation",
                "gate_decision": "block",
                "evidence_label": "urgent",
                "evidence_score": 9.38,
                "priority_label": "medium",
                "required_action": "resolve or explicitly waive signal evidence before release sign-off",
                "source_artifact": "lcri_cross_artifact_evidence_index.csv",
            },
        ]
    ).to_csv(tmp_path / "lcri_evidence_release_checklist.csv", index=False)
    write_json(
        tmp_path / "lcri_evidence_release_checklist_summary.json",
        {
            "items": 2,
            "blocked_items": 2,
            "review_items": 0,
            "monitor_items": 0,
            "ready_items": 0,
            "max_evidence_score": 11.25,
            "worst_check_scope": "regime:blocked",
            "release_ready": False,
        },
    )


def _write_matching_owner_handoff_packet_artifacts(tmp_path) -> None:
    _write_matching_evidence_release_checklist_artifacts(tmp_path)
    packet = pd.DataFrame(
        [
            {
                "scope": "regime",
                "handoff_rank": 1,
                "handoff_status": "immediate_owner_decision",
                "owner_queue": "owner must decide waive/fix posture for regime before release",
                "check_status": "blocked",
                "gate_decision": "pass",
                "evidence_label": "urgent",
                "evidence_score": 11.25,
                "priority_label": "critical",
                "critical_rows": 1,
                "warning_rows": 1,
                "fragility_review_required_rows": 1,
                "ci_gate_contradiction_rows": 2,
                "high_priority_ci_gate_rows": 0,
                "lcri_less_stable_share": 0.5,
                "required_action": "resolve or explicitly waive regime evidence before release sign-off",
                "evidence_source_artifact": "lcri_cross_artifact_evidence_index.csv",
                "checklist_source_artifact": "lcri_evidence_release_checklist.csv",
            },
            {
                "scope": "signal",
                "handoff_rank": 2,
                "handoff_status": "immediate_owner_decision",
                "owner_queue": "owner must decide waive/fix posture for signal before release",
                "check_status": "blocked",
                "gate_decision": "block",
                "evidence_label": "urgent",
                "evidence_score": 9.38,
                "priority_label": "medium",
                "critical_rows": 1,
                "warning_rows": 0,
                "fragility_review_required_rows": 0,
                "ci_gate_contradiction_rows": 1,
                "high_priority_ci_gate_rows": 1,
                "lcri_less_stable_share": 0.0,
                "required_action": "resolve or explicitly waive signal evidence before release sign-off",
                "evidence_source_artifact": "lcri_cross_artifact_evidence_index.csv",
                "checklist_source_artifact": "lcri_evidence_release_checklist.csv",
            },
        ]
    )
    packet.to_csv(tmp_path / "lcri_owner_handoff_packet.csv", index=False)
    summary = {
        "items": 2,
        "immediate_items": 2,
        "review_items": 0,
        "monitor_items": 0,
        "signoff_items": 0,
        "max_evidence_score": 11.25,
        "top_handoff_scope": "regime:immediate_owner_decision",
        "handoff_clear": False,
    }
    write_json(tmp_path / "lcri_owner_handoff_packet_summary.json", summary)
    write_lcri_owner_handoff_markdown_packet(
        tmp_path / "lcri_owner_handoff_packet.md",
        packet=packet,
        summary=summary,
    )


def _write_matching_evidence_lineage_map_artifacts(tmp_path) -> None:
    _write_matching_owner_handoff_packet_artifacts(tmp_path)
    pd.DataFrame(
        [
            {
                "scope": "regime",
                "evidence_label": "urgent",
                "check_status": "blocked",
                "handoff_status": "immediate_owner_decision",
                "evidence_score": 11.25,
                "evidence_source_artifact": "lcri_cross_artifact_evidence_index.csv",
                "checklist_source_artifact": "lcri_cross_artifact_evidence_index.csv",
                "handoff_source_artifact": "lcri_cross_artifact_evidence_index.csv",
                "lineage_status": "complete",
                "lineage_note": "regime evidence chain is complete from index to release checklist to handoff",
            },
            {
                "scope": "signal",
                "evidence_label": "urgent",
                "check_status": "blocked",
                "handoff_status": "immediate_owner_decision",
                "evidence_score": 9.38,
                "evidence_source_artifact": "lcri_cross_artifact_evidence_index.csv",
                "checklist_source_artifact": "lcri_cross_artifact_evidence_index.csv",
                "handoff_source_artifact": "lcri_cross_artifact_evidence_index.csv",
                "lineage_status": "complete",
                "lineage_note": "signal evidence chain is complete from index to release checklist to handoff",
            },
        ]
    ).to_csv(tmp_path / "lcri_evidence_lineage_map.csv", index=False)
    write_json(
        tmp_path / "lcri_evidence_lineage_map_summary.json",
        {
            "scopes": 2,
            "complete_scopes": 2,
            "source_mismatch_scopes": 0,
            "incomplete_scopes": 0,
            "max_evidence_score": 11.25,
            "worst_lineage_scope": "regime:complete",
            "lineage_clear": True,
        },
    )


def test_write_research_summary_includes_metrics_and_robustness(tmp_path) -> None:
    path = tmp_path / "summary.md"
    metrics = pd.DataFrame(
        [
            {
                "signal": "lcri",
                "directional_accuracy": 0.61,
                "brier_score": 0.22,
                "rank_correlation": 0.18,
            }
        ]
    )
    generalization_gap = pd.DataFrame(
        [
            {
                "signal": "lcri",
                "directional_accuracy_gap": 0.03,
                "brier_score_gap": 0.01,
                "rank_correlation_gap": 0.02,
            }
        ]
    )
    regime_generalization_gap = pd.DataFrame(
        [
            {
                "regime": "thin",
                "signal": "lcri",
                "directional_accuracy_gap": 0.04,
                "brier_score_gap": 0.02,
                "rank_correlation_gap": 0.03,
            }
        ]
    )
    transition_generalization_gap = pd.DataFrame(
        [
            {
                "segment": "transition",
                "signal": "lcri",
                "directional_accuracy_gap": 0.05,
                "brier_score_gap": 0.03,
                "rank_correlation_gap": 0.04,
            }
        ]
    )
    generalization_overview = {
        "signal_rows": 2,
        "max_signal_directional_accuracy_gap": 0.05,
    }
    generalization_gap_leaderboard = pd.DataFrame(
        [
            {
                "scope": "transition",
                "context": "transition",
                "signal": "lcri",
                "directional_accuracy_gap": 0.05,
            }
        ]
    )
    lcri_generalization_gap_leaderboard = pd.DataFrame(
        [
            {
                "scope": "signal",
                "context": "all",
                "signal": "lcri",
                "directional_accuracy_gap": 0.03,
            }
        ]
    )
    lcri_generalization_severity = pd.DataFrame(
        [
            {
                "scope": "signal",
                "context": "all",
                "directional_accuracy_gap": 0.03,
                "severity": "warning",
            }
        ]
    )
    lcri_generalization_severity_summary = {
        "rows": 1,
        "stable_rows": 0,
        "warning_rows": 1,
        "critical_rows": 0,
        "passes_lcri_generalization_gate": True,
    }
    lcri_gap_delta_flags = pd.DataFrame(
        [
            {
                "scope": "signal",
                "context": "all",
                "stability_flag": "lcri_more_stable",
            }
        ]
    )
    lcri_gap_delta_summary = {
        "rows": 3,
        "lcri_more_stable_rows": 2,
        "lcri_less_stable_rows": 1,
        "lcri_equal_stability_rows": 0,
        "max_lcri_stability_edge_context": "signal:all",
    }
    transition_lift = pd.DataFrame(
        [
            {
                "segment": "transition",
                "rows": 4,
                "directional_accuracy_lift": 0.25,
            }
        ]
    )

    write_research_summary(
        path,
        rows=100,
        train_rows=70,
        heldout_rows=30,
        seed=7,
        train_frac=0.7,
        metrics=metrics,
        heldout_metrics=metrics,
        generalization_gap=generalization_gap,
        regime_generalization_gap=regime_generalization_gap,
        transition_generalization_gap=transition_generalization_gap,
        generalization_overview=generalization_overview,
        generalization_gap_leaderboard=generalization_gap_leaderboard,
        lcri_generalization_gap_leaderboard=lcri_generalization_gap_leaderboard,
        lcri_generalization_severity=lcri_generalization_severity,
        lcri_generalization_severity_by_scope=pd.DataFrame(
            {"scope": ["regime"], "rows": [2], "stable_rows": [0], "warning_rows": [1], "critical_rows": [1]}
        ),
        lcri_generalization_severity_summary=lcri_generalization_severity_summary,
        lcri_gap_delta_flags=lcri_gap_delta_flags,
        lcri_gap_delta_summary=lcri_gap_delta_summary,
        transition_lift=transition_lift,
        transition_robustness={"passes_transition_robustness": True},
        heldout_transition_lift=transition_lift,
        heldout_transition_robustness={"passes_transition_robustness": True},
        lcri_reversal_stress_summary={
            "gate_decision": "review",
            "top_regime": "thin",
            "max_stress_concentration_ratio": 2.4,
        },
        heldout_lcri_reversal_stress_summary={
            "gate_decision": "pass",
            "top_regime": "thick",
            "max_stress_concentration_ratio": 1.1,
        },
        lcri_fracture_reversal_gate={"decision": "review", "passes": False},
        lcri_reversal_transition_gate=pd.DataFrame(
            {
                "transition": ["thick->thin"],
                "transition_stress_share": [0.8],
                "transition_gate_decision": ["review"],
            }
        ),
        heldout_lcri_reversal_transition_gate=pd.DataFrame(
            {
                "transition": ["thin->shock"],
                "transition_stress_share": [1.0],
                "transition_gate_decision": ["pass"],
            }
        ),
    )

    text = path.read_text()
    assert "# LCRI Research Summary" in text
    assert "- seed: 7" in text
    assert "## Heldout signal quality" in text
    assert "## Signal generalization gap" in text
    assert "## Regime generalization gap" in text
    assert "## Transition generalization gap" in text
    assert "## Generalization overview" in text
    assert "## Generalization gap leaderboard" in text
    assert "| signal | directional_accuracy | brier_score | rank_correlation |" in text
    assert "| lcri | 0.610000 | 0.220000 | 0.180000 |" in text
    assert "| lcri | 0.030000 | 0.010000 | 0.020000 |" in text
    assert "| thin | lcri | 0.040000 | 0.020000 | 0.030000 |" in text
    assert "| transition | lcri | 0.050000 | 0.030000 | 0.040000 |" in text
    assert "- signal_rows: 2" in text
    assert "- max_signal_directional_accuracy_gap: 0.050000" in text
    assert "| transition | transition | lcri | 0.050000 |" in text
    assert "## LCRI generalization gap leaderboard" in text
    assert "| signal | all | lcri | 0.030000 |" in text
    assert "## LCRI generalization severity" in text
    assert "| signal | all | 0.030000 | warning |" in text
    assert "## LCRI generalization severity summary" in text
    assert "- warning_rows: 1" in text
    assert "## LCRI gap delta flags" in text
    assert "| signal | all | lcri_more_stable |" in text
    assert "## LCRI gap delta summary" in text
    assert "- max_lcri_stability_edge_context: signal:all" in text
    assert "## Heldout transition lift" in text
    assert "## Heldout transition robustness" in text
    assert "- passes_transition_robustness: true" in text
    assert "## LCRI reversal stress concentration summary" in text
    assert "- max_stress_concentration_ratio: 2.400000" in text
    assert "## Heldout LCRI reversal stress concentration summary" in text
    assert "## LCRI fracture reversal gate" in text
    assert "- decision: review" in text
    assert "## LCRI reversal transition gate" in text
    assert "| thick->thin | 0.800000 | review |" in text
    assert "## Heldout LCRI reversal transition gate" in text
