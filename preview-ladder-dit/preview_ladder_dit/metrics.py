from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

Frame = Sequence[Sequence[float]]
Video = Sequence[Frame]
Mask = Sequence[Sequence[Sequence[bool]]]


@dataclass(frozen=True)
class PreviewFinalReport:
    """Report for preview-to-final replacement consistency.

    All metrics are lower-is-better. Inputs use frame-major videos shaped
    `(frames, height, width)` with scalar values for the deterministic harness.
    Real experiments can swap these arrays for embeddings, masks, or perceptual
    features while preserving this public report contract.
    """

    preview_final_l1: float
    boundary_consistency_error: float
    boundary_signed_bias: float
    temporal_flicker_delta: float
    background_preservation_error: float
    background_temporal_leak: float
    trajectory_center_error: float
    trajectory_acceleration_drift: float
    mask_occupancy_delta: float
    temporal_edge_jitter: float
    confidence_weighted_l1: float
    preview_confidence_release_rate: float
    occupancy_iou_error: float
    low_frequency_drift: float
    local_temporal_residual_flicker: float
    commitment_weighted_error: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def preview_final_consistency_report(
    *,
    source: Video,
    preview: Video,
    final: Video,
    mask: Mask,
    boundary_radius: int = 1,
) -> PreviewFinalReport:
    _validate_same_video_shape(source, preview, "source", "preview")
    _validate_same_video_shape(source, final, "source", "final")
    _validate_mask_shape(mask, source)
    return PreviewFinalReport(
        preview_final_l1=_mean_abs_diff(preview, final, mask),
        boundary_consistency_error=boundary_consistency_error(preview, final, mask, boundary_radius=boundary_radius),
        boundary_signed_bias=boundary_signed_bias(preview, final, mask, boundary_radius=boundary_radius),
        temporal_flicker_delta=temporal_flicker_delta(preview, final, mask),
        background_preservation_error=background_preservation_error(source, final, mask),
        background_temporal_leak=background_temporal_leak(source, final, mask),
        trajectory_center_error=trajectory_center_error(preview, final, mask),
        trajectory_acceleration_drift=trajectory_acceleration_drift(preview, final, mask),
        mask_occupancy_delta=mask_occupancy_delta(preview, final, mask),
        temporal_edge_jitter=temporal_edge_jitter(preview, final, mask, boundary_radius=boundary_radius),
        confidence_weighted_l1=confidence_weighted_l1(preview, final, mask, boundary_radius=boundary_radius),
        preview_confidence_release_rate=preview_confidence_release_rate(preview, mask, boundary_radius=boundary_radius),
        occupancy_iou_error=occupancy_iou_error(preview, final, mask),
        low_frequency_drift=low_frequency_drift(preview, final, mask),
        local_temporal_residual_flicker=local_temporal_residual_flicker(preview, final, mask),
        commitment_weighted_error=commitment_weighted_error(preview, final, mask, boundary_radius=boundary_radius),
    )


def background_preservation_error(source: Video, final: Video, mask: Mask) -> float:
    """Mean absolute change outside the edited mask."""

    _validate_same_video_shape(source, final, "source", "final")
    _validate_mask_shape(mask, source)
    total = 0.0
    count = 0
    for t, frame in enumerate(source):
        for y, row in enumerate(frame):
            for x, value in enumerate(row):
                if not mask[t][y][x]:
                    total += abs(value - final[t][y][x])
                    count += 1
    return total / count if count else 0.0


def background_temporal_leak(source: Video, final: Video, mask: Mask) -> float:
    """Mean temporal-delta change outside the edited mask."""

    _validate_same_video_shape(source, final, "source", "final")
    _validate_mask_shape(mask, source)
    if len(source) < 2:
        return 0.0
    total = 0.0
    count = 0
    for t in range(1, len(source)):
        for y, row in enumerate(source[t]):
            for x, value in enumerate(row):
                if not mask[t][y][x] and not mask[t - 1][y][x]:
                    source_delta = value - source[t - 1][y][x]
                    final_delta = final[t][y][x] - final[t - 1][y][x]
                    total += abs(final_delta - source_delta)
                    count += 1
    return total / count if count else 0.0


def boundary_consistency_error(preview: Video, final: Video, mask: Mask, *, boundary_radius: int = 1) -> float:
    """Mean preview/final difference on the dilated mask boundary band."""

    _validate_same_video_shape(preview, final, "preview", "final")
    _validate_mask_shape(mask, preview)
    boundary = _boundary_band(mask, boundary_radius)
    return _mean_abs_diff(preview, final, boundary)


def boundary_signed_bias(preview: Video, final: Video, mask: Mask, *, boundary_radius: int = 1) -> float:
    """Signed outward-minus-inward residual at the mask boundary."""

    _validate_same_video_shape(preview, final, "preview", "final")
    _validate_mask_shape(mask, preview)
    inside = _inner_boundary_band(mask, boundary_radius)
    outside = _outer_boundary_band(mask, boundary_radius)
    return _mean_signed_diff(preview, final, outside) + _mean_signed_diff(preview, final, inside)


def temporal_flicker_delta(preview: Video, final: Video, mask: Mask) -> float:
    """Difference between preview and final temporal change magnitudes inside the mask."""

    _validate_same_video_shape(preview, final, "preview", "final")
    _validate_mask_shape(mask, preview)
    if len(preview) < 2:
        return 0.0
    total = 0.0
    count = 0
    for t in range(1, len(preview)):
        for y, row in enumerate(preview[t]):
            for x, _ in enumerate(row):
                if mask[t][y][x] or mask[t - 1][y][x]:
                    preview_delta = abs(preview[t][y][x] - preview[t - 1][y][x])
                    final_delta = abs(final[t][y][x] - final[t - 1][y][x])
                    total += abs(preview_delta - final_delta)
                    count += 1
    return total / count if count else 0.0


def trajectory_center_error(preview: Video, final: Video, mask: Mask, *, threshold: float = 0.5) -> float:
    """Mean normalized centroid drift for salient replacement pixels inside the mask."""

    _validate_same_video_shape(preview, final, "preview", "final")
    _validate_mask_shape(mask, preview)
    if not preview:
        return 0.0
    height = len(preview[0])
    width = len(preview[0][0])
    diag = (height * height + width * width) ** 0.5 or 1.0
    total = 0.0
    count = 0
    for t in range(len(preview)):
        pc = _threshold_centroid(preview[t], mask[t], threshold)
        fc = _threshold_centroid(final[t], mask[t], threshold)
        if pc is None or fc is None:
            continue
        total += (((pc[0] - fc[0]) ** 2 + (pc[1] - fc[1]) ** 2) ** 0.5) / diag
        count += 1
    return total / count if count else 0.0


def trajectory_acceleration_drift(preview: Video, final: Video, mask: Mask, *, threshold: float = 0.5) -> float:
    """Mean normalized second-derivative drift of masked salient centroids."""

    _validate_same_video_shape(preview, final, "preview", "final")
    _validate_mask_shape(mask, preview)
    if len(preview) < 3:
        return 0.0
    height = len(preview[0])
    width = len(preview[0][0])
    diag = (height * height + width * width) ** 0.5 or 1.0
    preview_centroids = [_threshold_centroid(preview[t], mask[t], threshold) for t in range(len(preview))]
    final_centroids = [_threshold_centroid(final[t], mask[t], threshold) for t in range(len(final))]
    total = 0.0
    count = 0
    for t in range(2, len(preview)):
        points = (preview_centroids[t], preview_centroids[t - 1], preview_centroids[t - 2], final_centroids[t], final_centroids[t - 1], final_centroids[t - 2])
        if any(item is None for item in points):
            continue
        p2, p1, p0, f2, f1, f0 = points
        assert p0 is not None and p1 is not None and p2 is not None
        assert f0 is not None and f1 is not None and f2 is not None
        preview_ax = p2[0] - 2.0 * p1[0] + p0[0]
        preview_ay = p2[1] - 2.0 * p1[1] + p0[1]
        final_ax = f2[0] - 2.0 * f1[0] + f0[0]
        final_ay = f2[1] - 2.0 * f1[1] + f0[1]
        total += (((preview_ax - final_ax) ** 2 + (preview_ay - final_ay) ** 2) ** 0.5) / diag
        count += 1
    return total / count if count else 0.0


def mask_occupancy_delta(preview: Video, final: Video, mask: Mask, *, threshold: float = 0.5) -> float:
    """Mean difference in salient replacement occupancy inside the approved mask."""

    _validate_same_video_shape(preview, final, "preview", "final")
    _validate_mask_shape(mask, preview)
    total = 0.0
    count = 0
    for t, frame in enumerate(preview):
        p_on = 0
        f_on = 0
        eligible = 0
        for y, row in enumerate(frame):
            for x, value in enumerate(row):
                if mask[t][y][x]:
                    eligible += 1
                    p_on += 1 if value >= threshold else 0
                    f_on += 1 if final[t][y][x] >= threshold else 0
        if eligible:
            total += abs(p_on - f_on) / eligible
            count += 1
    return total / count if count else 0.0


def temporal_edge_jitter(preview: Video, final: Video, mask: Mask, *, boundary_radius: int = 1) -> float:
    """Temporal instability of preview/final boundary error across adjacent frames."""

    _validate_same_video_shape(preview, final, "preview", "final")
    _validate_mask_shape(mask, preview)
    if len(preview) < 2:
        return 0.0
    boundary = _boundary_band(mask, boundary_radius)
    frame_errors = [_mean_abs_diff([preview[t]], [final[t]], [boundary[t]]) for t in range(len(preview))]
    return sum(abs(frame_errors[t] - frame_errors[t - 1]) for t in range(1, len(frame_errors))) / (len(frame_errors) - 1)


def confidence_weighted_l1(preview: Video, final: Video, mask: Mask, *, boundary_radius: int = 1) -> float:
    """Preview/final L1 weighted by a deterministic confidence proxy."""

    _validate_same_video_shape(preview, final, "preview", "final")
    _validate_mask_shape(mask, preview)
    boundary = _boundary_band(mask, boundary_radius)
    total = 0.0
    weight_total = 0.0
    for t, frame in enumerate(preview):
        for y, row in enumerate(frame):
            for x, value in enumerate(row):
                if mask[t][y][x]:
                    weight = 0.35 if boundary[t][y][x] else 1.0
                    total += weight * abs(value - final[t][y][x])
                    weight_total += weight
    return total / weight_total if weight_total else 0.0


def preview_confidence_release_rate(preview: Video, mask: Mask, *, boundary_radius: int = 1) -> float:
    """Fraction of masked preview area intentionally left weakly committed."""

    _validate_mask_shape(mask, preview)
    boundary = _inner_boundary_band(mask, boundary_radius)
    released = 0.0
    count = 0
    for t, frame in enumerate(preview):
        for y, row in enumerate(frame):
            for x, _ in enumerate(row):
                if mask[t][y][x]:
                    released += 0.65 if boundary[t][y][x] else 0.0
                    count += 1
    return released / count if count else 0.0


def occupancy_iou_error(preview: Video, final: Video, mask: Mask, *, threshold: float = 0.5) -> float:
    """One minus masked salient-region IoU between preview and final."""

    _validate_same_video_shape(preview, final, "preview", "final")
    _validate_mask_shape(mask, preview)
    intersection = 0
    union = 0
    for t, frame in enumerate(preview):
        for y, row in enumerate(frame):
            for x, value in enumerate(row):
                if not mask[t][y][x]:
                    continue
                p = value >= threshold
                f = final[t][y][x] >= threshold
                intersection += 1 if p and f else 0
                union += 1 if p or f else 0
    return 1.0 - intersection / union if union else 0.0


def low_frequency_drift(preview: Video, final: Video, mask: Mask) -> float:
    """Difference in per-frame masked mean appearance."""

    _validate_same_video_shape(preview, final, "preview", "final")
    _validate_mask_shape(mask, preview)
    total = 0.0
    count = 0
    for t in range(len(preview)):
        pv = [preview[t][y][x] for y, row in enumerate(preview[t]) for x, _ in enumerate(row) if mask[t][y][x]]
        fv = [final[t][y][x] for y, row in enumerate(final[t]) for x, _ in enumerate(row) if mask[t][y][x]]
        if pv and fv:
            total += abs(sum(pv) / len(pv) - sum(fv) / len(fv))
            count += 1
    return total / count if count else 0.0


def local_temporal_residual_flicker(preview: Video, final: Video, mask: Mask) -> float:
    """Temporal variation of the preview-final residual inside the mask."""

    _validate_same_video_shape(preview, final, "preview", "final")
    _validate_mask_shape(mask, preview)
    if len(preview) < 2:
        return 0.0
    total = 0.0
    count = 0
    for t in range(1, len(preview)):
        for y, row in enumerate(preview[t]):
            for x, value in enumerate(row):
                if mask[t][y][x] or mask[t - 1][y][x]:
                    prev_residual = final[t - 1][y][x] - preview[t - 1][y][x]
                    residual = final[t][y][x] - value
                    total += abs(residual - prev_residual)
                    count += 1
    return total / count if count else 0.0


def commitment_weighted_error(preview: Video, final: Video, mask: Mask, *, boundary_radius: int = 1) -> float:
    """No-dependency approximation of commitment-packet preservation error."""

    return (
        0.5 * confidence_weighted_l1(preview, final, mask, boundary_radius=boundary_radius)
        + 0.3 * low_frequency_drift(preview, final, mask)
        + 0.2 * occupancy_iou_error(preview, final, mask)
    )


def _mean_abs_diff(left: Video, right: Video, include: Mask) -> float:
    _validate_same_video_shape(left, right, "left", "right")
    _validate_mask_shape(include, left)
    total = 0.0
    count = 0
    for t, frame in enumerate(left):
        for y, row in enumerate(frame):
            for x, value in enumerate(row):
                if include[t][y][x]:
                    total += abs(value - right[t][y][x])
                    count += 1
    return total / count if count else 0.0


def _mean_signed_diff(left: Video, right: Video, include: Mask) -> float:
    _validate_same_video_shape(left, right, "left", "right")
    _validate_mask_shape(include, left)
    total = 0.0
    count = 0
    for t, frame in enumerate(left):
        for y, row in enumerate(frame):
            for x, value in enumerate(row):
                if include[t][y][x]:
                    total += right[t][y][x] - value
                    count += 1
    return total / count if count else 0.0


def _threshold_centroid(frame: Frame, mask_frame: Sequence[Sequence[bool]], threshold: float) -> tuple[float, float] | None:
    total_x = 0.0
    total_y = 0.0
    count = 0
    for y, row in enumerate(frame):
        for x, value in enumerate(row):
            if mask_frame[y][x] and value >= threshold:
                total_x += x
                total_y += y
                count += 1
    if not count:
        return None
    return total_x / count, total_y / count


def _boundary_band(mask: Mask, radius: int) -> list[list[list[bool]]]:
    if radius < 0:
        raise ValueError("boundary_radius must be non-negative")
    frames = len(mask)
    height = len(mask[0])
    width = len(mask[0][0])
    out = [[[False for _ in range(width)] for _ in range(height)] for _ in range(frames)]
    for t in range(frames):
        for y in range(height):
            for x in range(width):
                if not mask[t][y][x]:
                    continue
                for yy in range(max(0, y - radius), min(height, y + radius + 1)):
                    for xx in range(max(0, x - radius), min(width, x + radius + 1)):
                        out[t][yy][xx] = True
    return out


def _inner_boundary_band(mask: Mask, radius: int) -> list[list[list[bool]]]:
    if radius < 0:
        raise ValueError("boundary_radius must be non-negative")
    frames = len(mask)
    height = len(mask[0])
    width = len(mask[0][0])
    out = [[[False for _ in range(width)] for _ in range(height)] for _ in range(frames)]
    for t in range(frames):
        for y in range(height):
            for x in range(width):
                if mask[t][y][x] and _has_neighbor_value(mask[t], y, x, radius, False):
                    out[t][y][x] = True
    return out


def _outer_boundary_band(mask: Mask, radius: int) -> list[list[list[bool]]]:
    if radius < 0:
        raise ValueError("boundary_radius must be non-negative")
    frames = len(mask)
    height = len(mask[0])
    width = len(mask[0][0])
    out = [[[False for _ in range(width)] for _ in range(height)] for _ in range(frames)]
    for t in range(frames):
        for y in range(height):
            for x in range(width):
                if (not mask[t][y][x]) and _has_neighbor_value(mask[t], y, x, radius, True):
                    out[t][y][x] = True
    return out


def _has_neighbor_value(frame_mask: Sequence[Sequence[bool]], y: int, x: int, radius: int, value: bool) -> bool:
    height = len(frame_mask)
    width = len(frame_mask[0])
    for yy in range(max(0, y - radius), min(height, y + radius + 1)):
        for xx in range(max(0, x - radius), min(width, x + radius + 1)):
            if bool(frame_mask[yy][xx]) == value:
                return True
    return False


def _validate_same_video_shape(left: Sequence[Sequence[Sequence[object]]], right: Sequence[Sequence[Sequence[object]]], left_name: str, right_name: str) -> None:
    if len(left) != len(right):
        raise ValueError(f"{left_name} and {right_name} must have the same frame count")
    for t, (lf, rf) in enumerate(zip(left, right)):
        if len(lf) != len(rf):
            raise ValueError(f"{left_name} and {right_name} frame {t} must have the same height")
        for y, (lr, rr) in enumerate(zip(lf, rf)):
            if len(lr) != len(rr):
                raise ValueError(f"{left_name} and {right_name} frame {t}, row {y} must have the same width")


def _validate_mask_shape(mask: Mask, video: Video) -> None:
    _validate_same_video_shape(mask, video, "mask", "video")
