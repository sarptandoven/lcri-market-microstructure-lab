import json

from preview_ladder_dit.fixtures import FIXTURE_CASES, make_fixture, write_fixtures
from preview_ladder_dit.harness import run_synthetic
from preview_ladder_dit.schema import load_and_validate_run_report


def test_fixture_cases_produce_expected_metric_ordering():
    clean = make_fixture("clean")
    bad = make_fixture("temporal_flicker")

    from preview_ladder_dit.metrics import preview_final_consistency_report

    clean_report = preview_final_consistency_report(source=clean.source, preview=clean.preview, final=clean.final, mask=clean.mask)
    bad_report = preview_final_consistency_report(source=bad.source, preview=bad.preview, final=bad.final, mask=bad.mask)

    assert clean_report.preview_final_l1 == 0.0
    assert bad_report.preview_final_l1 > clean_report.preview_final_l1
    assert bad_report.temporal_flicker_delta > clean_report.temporal_flicker_delta


def test_write_fixtures_and_run_synthetic_reports(tmp_path):
    paths = write_fixtures(tmp_path / "fixtures", cases=["clean", "background_leak"], frames=4, height=16, width=16)
    assert [p.name for p in paths] == ["clean.json", "background_leak.json"]
    assert json.loads(paths[0].read_text())["case"] == "clean"

    report_paths = run_synthetic(tmp_path / "run", cases=["clean"], frames=4, height=16, width=16)
    assert len(report_paths) == 1
    report = load_and_validate_run_report(report_paths[0])
    assert report["schema_version"] == "preview-ladder-report/v0.3"
    assert set(report["metrics"]).issuperset({"preview_final_l1", "background_preservation_error", "commitment_weighted_error", "commitment_packet_loss"})
    assert report["artifacts"]["commitments"]["schema_version"] == "preview-commitment/v0.2"


def test_all_fixture_cases_registered():
    assert "identity_drift" in FIXTURE_CASES
    assert "mask_instability" in FIXTURE_CASES
