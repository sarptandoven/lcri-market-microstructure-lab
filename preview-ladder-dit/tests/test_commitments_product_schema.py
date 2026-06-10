from preview_ladder_dit.commitments import commitment_loss, commitment_weight_map, extract_commitments
from preview_ladder_dit.fixtures import make_fixture
from preview_ladder_dit.product import build_preview_scorecard
import pytest

from preview_ladder_dit.schema import REPORT_SCHEMA_VERSION, SubmissionIdentity, validate_run_report


def test_extract_commitments_emits_stable_packet_for_clean_fixture():
    fixture = make_fixture("clean", frames=4, height=16, width=16)
    packet = extract_commitments(fixture.preview, fixture.mask)

    assert packet.schema_version == "preview-commitment/v0.2"
    assert len(packet.frames) == 4
    assert 0.0 <= packet.lock_budget <= 1.0
    assert all(0.0 <= w <= 1.0 for w in commitment_weight_map(packet))
    assert commitment_loss(fixture.preview, fixture.final, fixture.mask, packet) == 0.0


def test_scorecard_passes_clean_and_fails_boundary_fixture():
    clean = make_fixture("clean", frames=4, height=16, width=16)
    bad = make_fixture("boundary_halo", frames=4, height=16, width=16)

    from preview_ladder_dit.metrics import preview_final_consistency_report

    clean_metrics = preview_final_consistency_report(source=clean.source, preview=clean.preview, final=clean.final, mask=clean.mask).to_dict()
    bad_metrics = preview_final_consistency_report(source=bad.source, preview=bad.preview, final=bad.final, mask=bad.mask).to_dict()
    clean_card = build_preview_scorecard({"run_id": "clean", "task_id": "clean", "metrics": clean_metrics, "latency_log": {"events": []}})
    bad_card = build_preview_scorecard({"run_id": "bad", "task_id": "bad", "metrics": bad_metrics, "latency_log": {"events": []}})

    assert clean_card.passed is True
    assert clean_card.trust_score > bad_card.trust_score
    assert bad_card.passed is False


def test_v02_schema_requires_commitment_metric():
    report = {
        "schema_version": "preview-ladder-report/v0.2",
        "run_id": "r",
        "task_id": "t",
        "task": {},
        "artifacts": {"preview": None, "final": None, "commitments": {}, "uncertainty": {}},
        "metrics": {"preview_final_l1": 0.0, "commitment_weighted_error": 0.0},
        "latency_log": {"events": []},
        "model": {},
        "environment": {},
    }
    validate_run_report(report)


def test_schema_validates_submission_identity_hashes():
    digest = "a" * 64
    report = {
        "schema_version": "preview-ladder-report/v0.2",
        "run_id": "r",
        "task_id": "t",
        "submission": SubmissionIdentity(
            method_name="m",
            method_version="1",
            backend_family="synthetic",
            preview_mode="fast",
            final_mode="committed",
            artifact_hashes={"preview": digest},
        ).to_dict(),
        "task": {},
        "artifacts": {"preview": None, "final": None, "commitments": {}, "uncertainty": {}},
        "metrics": {"preview_final_l1": 0.0, "commitment_weighted_error": 0.0},
        "latency_log": {"events": []},
        "model": {},
        "environment": {},
    }
    validate_run_report(report)

    report["submission"]["artifact_hashes"]["preview"] = "not-a-sha"
    with pytest.raises(ValueError, match="sha256"):
        validate_run_report(report)


def test_current_schema_requires_submission_identity():
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": "r",
        "task_id": "t",
        "task": {},
        "artifacts": {"preview": None, "final": None, "commitments": {}, "uncertainty": {}},
        "metrics": {"preview_final_l1": 0.0, "commitment_weighted_error": 0.0},
        "latency_log": {"events": []},
        "model": {},
        "environment": {},
    }
    with pytest.raises(ValueError, match="submission identity"):
        validate_run_report(report)


def test_current_schema_requires_preview_final_latency_milestones_when_events_present():
    digest = "b" * 64
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": "r",
        "task_id": "t",
        "submission": SubmissionIdentity(
            method_name="m",
            method_version="1",
            backend_family="synthetic",
            preview_mode="deterministic_preview",
            final_mode="deterministic_final",
            artifact_hashes={"preview_array": digest, "final_array": digest},
        ).to_dict(),
        "task": {},
        "artifacts": {"preview": None, "final": None, "commitments": {}, "uncertainty": {}},
        "metrics": {"preview_final_l1": 0.0, "commitment_weighted_error": 0.0},
        "latency_log": {
            "events": [
                {"name": "preview_start", "role": "preview_generate", "started_at_s": 0.0, "ended_at_s": 0.0},
                {"name": "preview_end", "role": "preview_generate", "started_at_s": 1.0, "ended_at_s": 1.0},
                {"name": "final_start", "role": "final_generate", "started_at_s": 2.0, "ended_at_s": 2.0},
            ]
        },
        "model": {},
        "environment": {},
    }
    with pytest.raises(ValueError, match="final_end"):
        validate_run_report(report)

    report["latency_log"]["events"].append(
        {"name": "final_end", "role": "final_generate", "started_at_s": 3.0, "ended_at_s": 3.0}
    )
    validate_run_report(report)
