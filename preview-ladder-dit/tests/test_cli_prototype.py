from __future__ import annotations

import json

from preview_ladder_dit.cli import main


def test_cli_workflow_generate_evaluate_validate_compare_figures_inspect(tmp_path, capsys):
    fixtures = tmp_path / "fixtures"
    reports_a = tmp_path / "reports_a"
    reports_b = tmp_path / "reports_b"
    figures = tmp_path / "figures"
    tables = tmp_path / "tables"
    comparison = tmp_path / "comparison.json"

    assert main(["generate-fixtures", "--out", str(fixtures), "--cases", "clean", "boundary_halo", "--frames", "4", "--height", "12", "--width", "12"]) == 0
    assert (fixtures / "manifest.json").exists()

    assert main(["evaluate", "--fixtures", str(fixtures), "--out", str(reports_a)]) == 0
    assert (reports_a / "summary.json").exists()
    assert (reports_a / "metrics.csv").exists()

    assert main(["validate-submission", str(reports_a)]) == 0
    validation = json.loads((reports_a / "validation.json").read_text(encoding="utf-8"))
    assert validation["ok"] is True
    assert validation["report_count"] == 2

    assert main(["evaluate", "--fixtures", str(fixtures), "--out", str(reports_b)]) == 0
    assert main(["compare", "--left", str(reports_a), "--right", str(reports_b), "--out", str(comparison)]) == 0
    assert json.loads(comparison.read_text(encoding="utf-8"))["task_count"] == 2

    assert main(["paper-figures", "--reports", str(reports_a), "--out", str(figures)]) == 0
    assert (figures / "figure_metrics_by_task.csv").exists()
    assert (figures / "vega_preview_final_score.json").exists()

    assert main(["export-paper-tables", "--reports", str(reports_a), "--out", str(tables)]) == 0
    assert (tables / "table_aggregate_results.csv").exists()
    assert (tables / "table_metric_summary.csv").exists()
    assert (tables / "table_per_task_metrics.csv").exists()
    aggregate_csv = (tables / "table_aggregate_results.csv").read_text(encoding="utf-8")
    assert "primary_score_mean" in aggregate_csv
    assert "aggregate_consistency_score_mean" in aggregate_csv

    assert main(["inspect-report", str(reports_a / "report-clean.json")]) == 0
    captured = capsys.readouterr().out
    assert "preview_final_l1" in captured
