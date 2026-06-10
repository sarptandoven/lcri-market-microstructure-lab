import json

from preview_ladder_dit.api import PreviewLadderConfig, evaluate_fixture_dict, evaluate_preview_final
from preview_ladder_dit.fixtures import make_fixture
from preview_ladder_dit.schema import load_and_validate_run_report


def test_evaluate_preview_final_returns_valid_report_and_scorecard(tmp_path):
    fixture = make_fixture("clean", frames=4, height=16, width=16)
    result = evaluate_preview_final(
        source=fixture.source,
        preview=fixture.preview,
        final=fixture.final,
        mask=fixture.mask,
        config=PreviewLadderConfig(
            run_id="api-clean",
            task_id="task-clean",
            task_type="synthetic_masked_replacement",
            prompt="clean accepted preview",
            seed=7,
            model={"backend": "unit-test", "preview_backend": "synthetic", "final_backend": "synthetic"},
            environment={"ci": True},
        ),
    )

    assert result.metrics.preview_final_l1 == 0.0
    assert result.commitments.schema_version == "preview-commitment/v0.2"
    assert result.scorecard.passed is True
    assert result.report.to_dict()["metrics"]["commitment_packet_loss"] == 0.0

    report_path = tmp_path / "report.json"
    result.write_report_json(report_path)
    loaded = load_and_validate_run_report(report_path)
    assert loaded["run_id"] == "api-clean"
    assert loaded["task"]["edit_spec"]["seed"] == 7


def test_evaluate_fixture_dict_flags_known_failure():
    fixture = make_fixture("background_leak", frames=4, height=16, width=16)
    result = evaluate_fixture_dict(fixture.to_dict())

    assert result.scorecard.passed is False
    assert result.report.to_dict()["task"]["task_type"] == "synthetic_masked_replacement"
    assert any("background" in message for message in result.scorecard.diagnosis)
    json.dumps(result.to_dict())
