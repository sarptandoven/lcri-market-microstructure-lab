import pytest

from preview_ladder_dit.metrics import (
    background_temporal_leak,
    background_preservation_error,
    boundary_consistency_error,
    boundary_signed_bias,
    mask_occupancy_delta,
    preview_confidence_release_rate,
    preview_final_consistency_report,
    temporal_edge_jitter,
    temporal_flicker_delta,
    trajectory_acceleration_drift,
    trajectory_center_error,
)


def test_preview_final_report_penalizes_boundary_and_temporal_mismatch():
    source = [
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ]
    preview = [
        [[0.0, 0.5, 0.0], [0.0, 0.5, 0.0]],
        [[0.0, 0.7, 0.0], [0.0, 0.7, 0.0]],
    ]
    final = [
        [[0.1, 0.6, 0.0], [0.0, 0.4, 0.0]],
        [[0.2, 0.9, 0.0], [0.0, 0.8, 0.0]],
    ]
    mask = [
        [[False, True, False], [False, True, False]],
        [[False, True, False], [False, True, False]],
    ]

    report = preview_final_consistency_report(source=source, preview=preview, final=final, mask=mask)

    assert report.preview_final_l1 > 0
    assert report.boundary_consistency_error > 0
    assert report.temporal_flicker_delta > 0
    assert report.background_preservation_error > 0
    assert set(report.to_dict()).issuperset({
        "preview_final_l1",
        "boundary_consistency_error",
        "boundary_signed_bias",
        "temporal_flicker_delta",
        "background_preservation_error",
        "background_temporal_leak",
        "trajectory_center_error",
        "trajectory_acceleration_drift",
        "mask_occupancy_delta",
        "temporal_edge_jitter",
        "confidence_weighted_l1",
        "preview_confidence_release_rate",
        "occupancy_iou_error",
        "low_frequency_drift",
        "local_temporal_residual_flicker",
        "commitment_weighted_error",
    })


def test_background_preservation_ignores_masked_region():
    source = [[[0.0, 0.0]]]
    final = [[[1.0, 0.25]]]
    mask = [[[True, False]]]

    assert background_preservation_error(source, final, mask) == 0.25


def test_boundary_consistency_uses_dilated_mask_band():
    preview = [[[0.0, 1.0, 0.0]]]
    final = [[[0.5, 1.0, 0.5]]]
    mask = [[[False, True, False]]]

    assert boundary_consistency_error(preview, final, mask, boundary_radius=1) == 1.0 / 3.0


def test_temporal_flicker_delta_compares_change_magnitude_inside_mask():
    preview = [[[0.0, 0.2]], [[0.0, 0.6]]]
    final = [[[0.0, 0.2]], [[0.0, 1.0]]]
    mask = [[[False, True]], [[False, True]]]

    assert temporal_flicker_delta(preview, final, mask) == pytest.approx(0.4)


def test_deeper_metrics_detect_trajectory_occupancy_and_edge_jitter():
    preview = [
        [[0.0, 0.9, 0.0, 0.0]],
        [[0.0, 0.9, 0.0, 0.0]],
    ]
    final = [
        [[0.0, 0.0, 0.9, 0.0]],
        [[0.0, 0.0, 0.0, 0.0]],
    ]
    mask = [[[False, True, True, False]], [[False, True, True, False]]]

    assert trajectory_center_error(preview, final, mask) > 0
    assert mask_occupancy_delta(preview, final, mask) > 0
    assert temporal_edge_jitter(preview, final, mask) > 0


def test_boundary_signed_bias_preserves_bleed_direction():
    preview = [[[0.0, 1.0, 0.0]]]
    outward_bleed = [[[0.5, 1.0, 0.5]]]
    inward_erosion = [[[0.0, 0.5, 0.0]]]
    mask = [[[False, True, False]]]

    assert boundary_signed_bias(preview, outward_bleed, mask, boundary_radius=1) > 0
    assert boundary_signed_bias(preview, inward_erosion, mask, boundary_radius=1) < 0


def test_background_temporal_leak_detects_outside_mask_flicker():
    source = [[[0.0, 0.0]], [[0.0, 0.0]], [[0.0, 0.0]]]
    final = [[[0.0, 0.0]], [[0.0, 0.4]], [[0.0, 0.0]]]
    mask = [[[True, False]], [[True, False]], [[True, False]]]

    assert background_temporal_leak(source, final, mask) == pytest.approx(0.4)


def test_trajectory_acceleration_drift_detects_motion_curvature_change():
    preview = [
        [[0.9, 0.0, 0.0, 0.0]],
        [[0.0, 0.9, 0.0, 0.0]],
        [[0.0, 0.0, 0.9, 0.0]],
    ]
    final = [
        [[0.9, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.9, 0.0]],
        [[0.0, 0.0, 0.0, 0.9]],
    ]
    mask = [[[True, True, True, True]], [[True, True, True, True]], [[True, True, True, True]]]

    assert trajectory_acceleration_drift(preview, final, mask) > 0


def test_preview_confidence_release_rate_reports_boundary_release_budget():
    preview = [[[0.0, 1.0, 1.0, 0.0]]]
    mask = [[[False, True, True, False]]]

    assert 0.0 < preview_confidence_release_rate(preview, mask, boundary_radius=1) <= 0.65
