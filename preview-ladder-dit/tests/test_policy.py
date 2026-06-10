from preview_ladder_dit.commitments import extract_commitments
from preview_ladder_dit.fixtures import FixtureBundle, make_fixture
from preview_ladder_dit.ltx_adapter import LTXPreviewFinalAdapter
from preview_ladder_dit.metrics import preview_final_consistency_report
from preview_ladder_dit.policy import (
    build_conditioning_hints,
    project_mask_to_latent_grid,
    select_commitment_policy,
)


def _metrics_for(case: str) -> tuple[dict[str, float], FixtureBundle]:
    fixture = make_fixture(case, frames=4, height=16, width=16)
    metrics = preview_final_consistency_report(
        source=fixture.source,
        preview=fixture.preview,
        final=fixture.final,
        mask=fixture.mask,
    ).to_dict()
    return metrics, fixture


def test_policy_locks_clean_preview_more_than_boundary_halo():
    clean_metrics, clean_fixture = _metrics_for("clean")
    bad_metrics, bad_fixture = _metrics_for("boundary_halo")
    clean_packet = extract_commitments(clean_fixture.preview, clean_fixture.mask)
    bad_packet = extract_commitments(bad_fixture.preview, bad_fixture.mask)

    clean_policy = select_commitment_policy(clean_metrics, clean_packet)
    bad_policy = select_commitment_policy(bad_metrics, bad_packet)

    assert clean_policy.schema_version == "preview-commitment-policy/v0.1"
    assert clean_policy.boundary_lock > bad_policy.boundary_lock
    assert clean_policy.release_rate < bad_policy.release_rate
    assert bad_policy.background_lock >= 0.8
    assert any("boundary" in reason for reason in bad_policy.reasons)


def test_policy_releases_motion_when_trajectory_proxy_is_uncertain():
    metrics, fixture = _metrics_for("clean")
    packet = extract_commitments(fixture.preview, fixture.mask)

    stable = select_commitment_policy(metrics, packet, uncertainty={"motion_uncertainty": 0.0})
    unstable = select_commitment_policy(metrics, packet, uncertainty={"motion_uncertainty": 0.9})

    assert unstable.trajectory_lock < stable.trajectory_lock
    assert unstable.release_rate > stable.release_rate
    assert any("motion" in reason for reason in unstable.reasons)


def test_project_mask_to_latent_grid_conservatively_marks_covered_tokens():
    mask = [
        [[True, False, False, False], [False, False, False, False], [False, False, True, True], [False, False, False, False]],
        [[False, False, False, False], [False, True, False, False], [False, False, False, False], [False, False, False, False]],
    ]

    latent = project_mask_to_latent_grid(mask, latent_frames=1, latent_height=2, latent_width=2)

    assert latent == [[[True, False], [False, True]]]


def test_conditioning_hints_are_backend_agnostic_and_weighted():
    metrics, fixture = _metrics_for("clean")
    packet = extract_commitments(fixture.preview, fixture.mask)
    policy = select_commitment_policy(metrics, packet)
    latent_mask = project_mask_to_latent_grid(fixture.mask, latent_frames=1, latent_height=2, latent_width=2)

    hints = build_conditioning_hints(policy=policy, packet=packet, latent_mask=latent_mask)

    assert hints["schema_version"] == "preview-conditioning-hints/v0.1"
    assert hints["latent_mask_shape"] == (1, 2, 2)
    assert 0.0 < hints["active_token_fraction"] <= 1.0
    assert len(hints["frame_weights"]) == len(packet.frames)
    assert set(hints["denoising_hooks"]) == {"background", "boundary", "trajectory", "occupancy", "appearance"}


def test_ltx_adapter_exposes_conditioning_hints_without_ltx_dependency():
    metrics, fixture = _metrics_for("clean")
    packet = extract_commitments(fixture.preview, fixture.mask)
    policy = select_commitment_policy(metrics, packet)
    adapter = LTXPreviewFinalAdapter()

    hints = adapter.conditioning_hints_from_policy(mask=fixture.mask, policy=policy, packet=packet)

    assert hints["schema_version"] == "preview-conditioning-hints/v0.1"
    assert "policy" in hints
