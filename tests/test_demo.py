import json
from pathlib import Path

import pytest

from lcri_lab.cli import run_demo


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
    assert (tmp_path / "lcri_generalization_critical_contexts.csv").exists()
    assert (tmp_path / "lcri_generalization_blocker_summary.json").exists()
    assert (tmp_path / "lcri_generalization_severity_summary.json").exists()
    assert (tmp_path / "lcri_worst_generalization_context.json").exists()
    assert (tmp_path / "lcri_generalization_gate_decision.json").exists()
    assert (tmp_path / "lcri_generalization_gap_delta.csv").exists()
    assert (tmp_path / "lcri_gap_delta_flags.csv").exists()
    assert (tmp_path / "lcri_gap_delta_improvements.csv").exists()
    assert (tmp_path / "lcri_gap_delta_regressions.csv").exists()
    assert (tmp_path / "lcri_gap_delta_scorecard.json").exists()
    assert (tmp_path / "lcri_gap_delta_scope_extremes.csv").exists()
    assert (tmp_path / "lcri_gap_delta_scope_summary.csv").exists()
    assert (tmp_path / "lcri_gap_delta_summary.json").exists()
    assert (tmp_path / "transition_lift.csv").exists()
    assert (tmp_path / "heldout_transition_lift.csv").exists()
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
    assert (tmp_path / "figures" / "lcri_generalization_gap_delta.png").exists()
    assert (tmp_path / "figures" / "lcri_generalization_severity_by_scope.png").exists()
    assert (tmp_path / "figures" / "lcri_gap_delta_scope_summary.png").exists()

    robustness = json.loads((tmp_path / "transition_robustness.json").read_text())
    assert "passes_transition_robustness" in robustness
    metadata_summary = json.loads((tmp_path / "artifact_metadata_summary.json").read_text())
    assert metadata_summary["artifacts_with_metadata"] > 0
    assert metadata_summary["total_size_bytes"] > 0
    assert metadata_summary["largest_artifact"] != "none"
    manifest = json.loads((tmp_path / "artifact_manifest.json").read_text())
    assert manifest["run"]["seed"] == 3
    assert manifest["model"]["artifact_version"] == 2
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
    assert manifest["artifact_metadata"]["figures/lcri_generalization_severity_by_scope.png"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["figures/lcri_gap_delta_scope_summary.png"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["lcri_gap_delta_flags.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["lcri_gap_delta_improvements.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["lcri_gap_delta_regressions.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["lcri_gap_delta_scorecard.json"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["lcri_gap_delta_scope_extremes.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["lcri_gap_delta_scope_summary.csv"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["artifact_metadata_summary.json"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["lcri_gap_delta_summary.json"]["size_bytes"] > 0
    assert manifest["artifact_metadata"]["heldout_transition_lift.csv"]["size_bytes"] > 0
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
    assert "## LCRI generalization severity by scope" in summary
    assert "## LCRI generalization severity summary" in summary
    assert "## LCRI worst generalization context" in summary
    assert "## LCRI generalization gate decision" in summary
    assert "## LCRI generalization gap delta" in summary
    assert "## LCRI gap delta flags" in summary
    assert "## LCRI gap delta scorecard" in summary
    assert "## LCRI gap delta summary" in summary
    assert "## Transition robustness" in summary
    assert "## Heldout transition lift" in summary


def test_run_demo_rejects_invalid_train_fraction(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="train_frac"):
        run_demo(rows=100, seed=3, train_frac=1.0, output=tmp_path)
