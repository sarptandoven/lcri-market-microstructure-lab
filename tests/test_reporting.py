import json

import pandas as pd

from lcri_lab.reporting import (
    build_artifact_manifest,
    collect_artifact_metadata,
    missing_artifacts,
    verify_artifact_manifest,
    verify_generalization_overview,
    verify_lcri_gap_delta_flags,
    verify_lcri_gap_delta_scorecard,
    verify_lcri_gap_delta_summary,
    verify_lcri_generalization_critical_contexts,
    verify_lcri_generalization_gate_decision,
    verify_lcri_generalization_gap_delta,
    verify_lcri_generalization_gap_leaderboard,
    verify_lcri_generalization_scope_risk,
    verify_lcri_generalization_scope_summary,
    verify_lcri_generalization_severity,
    verify_lcri_generalization_severity_by_scope,
    verify_lcri_generalization_severity_summary,
    verify_lcri_worst_generalization_context,
    write_json,
    write_research_summary,
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


def test_collect_artifact_metadata_records_size_and_digest(tmp_path) -> None:
    (tmp_path / "metrics.csv").write_text("signal,value\n")

    metadata = collect_artifact_metadata(tmp_path, ["metrics.csv", "missing.csv"])

    assert set(metadata) == {"metrics.csv"}
    assert metadata["metrics.csv"]["size_bytes"] == len("signal,value\n")
    assert len(metadata["metrics.csv"]["sha256"]) == 64


def test_verify_artifact_manifest_reports_checksum_mismatch(tmp_path) -> None:
    (tmp_path / "metrics.csv").write_text("signal,value\n")
    manifest = {
        "artifacts": ["metrics.csv"],
        "artifact_metadata": {"metrics.csv": {"size_bytes": 1, "sha256": "0" * 64}},
    }

    errors = verify_artifact_manifest(tmp_path, manifest)

    assert "size mismatch: metrics.csv" in errors
    assert "sha256 mismatch: metrics.csv" in errors


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


def test_verify_lcri_generalization_severity_by_scope_accepts_complete_csv(tmp_path) -> None:
    (tmp_path / "lcri_generalization_severity_by_scope.csv").write_text(
        "scope,rows,stable_rows,warning_rows,critical_rows\n"
        "regime,2,0,1,1\n"
    )

    assert verify_lcri_generalization_severity_by_scope(tmp_path) == []


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
