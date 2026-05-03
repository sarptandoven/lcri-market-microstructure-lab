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

    assert "schema_version: 2" in output
    assert "levels: 5" in output
    assert "features: 14" in output


def test_score_model_writes_selected_columns(tmp_path: Path) -> None:
    snapshots = tmp_path / "snapshots.csv"
    model_path = tmp_path / "model.json"
    output_path = tmp_path / "scores.csv"
    _write_snapshots(snapshots)
    fit_model(snapshots, model_path, levels=5)

    score_model(snapshots, model_path, output_path, columns=["timestamp", "lcri", "lcri_probability"])

    assert list(pd.read_csv(output_path).columns) == ["timestamp", "lcri", "lcri_probability"]


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
    (tmp_path / "lcri_generalization_severity_by_scope.csv").write_text(
        "scope,rows,stable_rows,warning_rows,critical_rows\nregime,1,0,0,1\n"
    )
    (tmp_path / "lcri_generalization_scope_risk.csv").write_text(
        "scope,rows,warning_or_critical_share,critical_share\nregime,1,1.0,1.0\n"
    )
    (tmp_path / "lcri_generalization_scope_gate_decisions.csv").write_text(
        "scope,rows,decision,reason\nregime,1,block,regime blocked\n"
    )
    (tmp_path / "lcri_generalization_critical_contexts.csv").write_text(
        "scope,context,directional_accuracy_gap,severity\nregime,thin,0.08,critical\n"
    )
    blocker_summary = tmp_path / "lcri_generalization_blocker_summary.json"
    blocker_summary.write_text(
        json.dumps(
            {
                "critical_rows": 1,
                "critical_scopes": "regime",
                "max_critical_gap": 0.08,
                "max_critical_context": "regime:thin",
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
    delta = tmp_path / "lcri_generalization_gap_delta.csv"
    delta.write_text(
        "scope,context,raw_imbalance_directional_accuracy_gap,"
        "lcri_directional_accuracy_gap,raw_minus_lcri_directional_accuracy_gap\n"
        "signal,all,0.08,0.05,0.03\n"
    )
    flags = tmp_path / "lcri_gap_delta_flags.csv"
    flags.write_text(
        "scope,context,raw_minus_lcri_directional_accuracy_gap,stability_flag\n"
        "signal,all,0.03,lcri_more_stable\n"
    )
    (tmp_path / "lcri_gap_delta_regressions.csv").write_text(
        "scope,context,raw_minus_lcri_directional_accuracy_gap\nregime,thin,-0.04\n"
    )
    scorecard = tmp_path / "lcri_gap_delta_scorecard.json"
    scorecard.write_text(
        json.dumps(
            {
                "rows": 3,
                "mean_raw_minus_lcri_directional_accuracy_gap": 0.02,
                "median_raw_minus_lcri_directional_accuracy_gap": 0.01,
                "lcri_more_stable_share": 0.67,
                "lcri_less_stable_share": 0.33,
            }
        )
    )
    (tmp_path / "lcri_gap_delta_scope_summary.csv").write_text(
        "scope,rows,mean_raw_minus_lcri_gap,min_raw_minus_lcri_gap,max_raw_minus_lcri_gap\n"
        "regime,2,0.01,-0.04,0.06\n"
    )
    summary = tmp_path / "lcri_gap_delta_summary.json"
    summary.write_text(
        json.dumps(
            {
                "rows": 3,
                "lcri_more_stable_rows": 2,
                "lcri_less_stable_rows": 1,
                "lcri_equal_stability_rows": 0,
                "max_lcri_stability_edge": 0.03,
                "max_lcri_stability_edge_context": "signal:all",
                "max_lcri_instability_edge": -0.04,
                "max_lcri_instability_edge_context": "regime:thin",
            }
        )
    )
    manifest = {
        "artifacts": [
            "metrics.csv",
            "generalization_overview.json",
            "lcri_generalization_gap_leaderboard.csv",
            "lcri_generalization_scope_summary.csv",
            "lcri_generalization_severity.csv",
            "lcri_generalization_severity_by_scope.csv",
            "lcri_generalization_scope_risk.csv",
            "lcri_generalization_scope_gate_decisions.csv",
            "lcri_generalization_critical_contexts.csv",
            "lcri_generalization_blocker_summary.json",
            "lcri_generalization_severity_summary.json",
            "lcri_worst_generalization_context.json",
            "lcri_generalization_gate_decision.json",
            "lcri_generalization_gap_delta.csv",
            "lcri_gap_delta_flags.csv",
            "lcri_gap_delta_regressions.csv",
            "lcri_gap_delta_scorecard.json",
            "lcri_gap_delta_scope_summary.csv",
            "lcri_gap_delta_summary.json",
        ],
        "artifact_metadata": {},
    }
    (tmp_path / "artifact_manifest.json").write_text(json.dumps(manifest))

    verify_report(tmp_path)

    captured = capsys.readouterr()
    assert "Verified report artifacts" in captured.out
    assert "verification summary" in captured.out
    assert "passes_verification" in captured.out


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


def test_score_model_rejects_unknown_columns(tmp_path: Path) -> None:
    snapshots = tmp_path / "snapshots.csv"
    model_path = tmp_path / "model.json"
    _write_snapshots(snapshots)
    fit_model(snapshots, model_path, levels=5)

    with pytest.raises(ValueError, match="unavailable"):
        score_model(snapshots, model_path, tmp_path / "scores.csv", columns=["missing"])
