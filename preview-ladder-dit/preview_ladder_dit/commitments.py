from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

Frame = Sequence[Sequence[float]]
Video = Sequence[Frame]
Mask = Sequence[Sequence[Sequence[bool]]]


@dataclass(frozen=True)
class CommitmentFrame:
    frame_index: int
    bbox_xyxy: tuple[int, int, int, int] | None
    centroid_xy: tuple[float, float] | None
    occupancy: float
    mean_value: float
    low_frequency_value: float
    boundary_uncertainty: float
    release_fraction: float


@dataclass(frozen=True)
class CommitmentPacket:
    """Dependency-light preview commitment packet.

    The packet is intentionally scalar-array based so it can be generated in CI,
    serialized into benchmark reports, and later mirrored by RGB, latent, flow,
    or embedding backends. High lock_budget means the preview is internally
    stable enough to constrain final rendering. High release_fraction means the
    final pass should retain freedom around boundary, occlusion, or uncertain
    regions.
    """

    schema_version: str
    threshold: float
    boundary_radius: int
    frames: tuple[CommitmentFrame, ...]
    global_trajectory_jitter: float
    mean_boundary_uncertainty: float
    lock_budget: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def extract_commitments(
    preview: Video,
    mask: Mask,
    *,
    threshold: float = 0.5,
    boundary_radius: int = 1,
) -> CommitmentPacket:
    _validate_video_mask(preview, mask)
    boundary = _boundary_band(mask, boundary_radius)
    frames: list[CommitmentFrame] = []
    centroids: list[tuple[float, float]] = []
    for t, frame in enumerate(preview):
        values: list[float] = []
        salient_x: list[int] = []
        salient_y: list[int] = []
        boundary_values: list[float] = []
        for y, row in enumerate(frame):
            for x, value in enumerate(row):
                if mask[t][y][x]:
                    v = float(value)
                    values.append(v)
                    if v >= threshold:
                        salient_x.append(x)
                        salient_y.append(y)
                if boundary[t][y][x]:
                    boundary_values.append(float(value))
        occupancy = len(salient_x) / len(values) if values else 0.0
        mean_value = sum(values) / len(values) if values else 0.0
        low_frequency = _local_mean(frame, mask[t])
        centroid = None
        bbox = None
        if salient_x:
            centroid = (sum(salient_x) / len(salient_x), sum(salient_y) / len(salient_y))
            bbox = (min(salient_x), min(salient_y), max(salient_x) + 1, max(salient_y) + 1)
            centroids.append(centroid)
        boundary_uncertainty = _threshold_uncertainty(boundary_values, threshold)
        release_fraction = max(0.0, min(1.0, 0.5 * boundary_uncertainty + 0.5 * (1.0 - occupancy)))
        frames.append(
            CommitmentFrame(
                frame_index=t,
                bbox_xyxy=bbox,
                centroid_xy=centroid,
                occupancy=occupancy,
                mean_value=mean_value,
                low_frequency_value=low_frequency,
                boundary_uncertainty=boundary_uncertainty,
                release_fraction=release_fraction,
            )
        )
    jitter = _trajectory_jitter(centroids)
    mean_uncertainty = sum(f.boundary_uncertainty for f in frames) / len(frames) if frames else 0.0
    mean_release = sum(f.release_fraction for f in frames) / len(frames) if frames else 1.0
    lock_budget = max(0.0, min(1.0, 1.0 - 0.5 * jitter - 0.3 * mean_uncertainty - 0.2 * mean_release))
    return CommitmentPacket(
        schema_version="preview-commitment/v0.2",
        threshold=threshold,
        boundary_radius=boundary_radius,
        frames=tuple(frames),
        global_trajectory_jitter=jitter,
        mean_boundary_uncertainty=mean_uncertainty,
        lock_budget=lock_budget,
    )


def commitment_weight_map(packet: CommitmentPacket) -> list[float]:
    """Return per-frame preservation weights for final denoising wrappers."""

    weights: list[float] = []
    for frame in packet.frames:
        weights.append(max(0.0, min(1.0, packet.lock_budget * (1.0 - frame.release_fraction))))
    return weights


def commitment_loss(preview: Video, final: Video, mask: Mask, packet: CommitmentPacket) -> float:
    """Commitment-weighted preview/final L1 inside the edit mask."""

    _validate_video_mask(preview, mask)
    _validate_video_mask(final, mask)
    weights = commitment_weight_map(packet)
    total = 0.0
    weight_total = 0.0
    for t, frame in enumerate(preview):
        weight = weights[t] if t < len(weights) else packet.lock_budget
        for y, row in enumerate(frame):
            for x, value in enumerate(row):
                if mask[t][y][x]:
                    total += weight * abs(float(value) - float(final[t][y][x]))
                    weight_total += weight
    return total / weight_total if weight_total else 0.0


def _threshold_uncertainty(values: Sequence[float], threshold: float) -> float:
    if not values:
        return 0.0
    # Values close to the decision threshold are ownership-uncertain.
    uncertainty = [max(0.0, 1.0 - min(1.0, abs(v - threshold) / max(threshold, 1.0 - threshold, 1e-9))) for v in values]
    return sum(uncertainty) / len(uncertainty)


def _local_mean(frame: Frame, mask_frame: Sequence[Sequence[bool]]) -> float:
    values = [float(value) for y, row in enumerate(frame) for x, value in enumerate(row) if mask_frame[y][x]]
    return sum(values) / len(values) if values else 0.0


def _trajectory_jitter(centroids: Sequence[tuple[float, float]]) -> float:
    if len(centroids) < 3:
        return 0.0
    speeds = []
    for idx in range(1, len(centroids)):
        ax, ay = centroids[idx - 1]
        bx, by = centroids[idx]
        speeds.append(((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5)
    mean_speed = sum(speeds) / len(speeds) if speeds else 0.0
    if not mean_speed:
        return 0.0
    return min(1.0, sum(abs(s - mean_speed) for s in speeds) / (len(speeds) * mean_speed))


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
                here = mask[t][y][x]
                for yy in range(max(0, y - radius), min(height, y + radius + 1)):
                    for xx in range(max(0, x - radius), min(width, x + radius + 1)):
                        if mask[t][yy][xx] != here:
                            out[t][y][x] = True
                            out[t][yy][xx] = True
    return out


def _validate_video_mask(video: Sequence[Sequence[Sequence[object]]], mask: Sequence[Sequence[Sequence[object]]]) -> None:
    if len(video) != len(mask):
        raise ValueError("video and mask must have the same frame count")
    if not video:
        raise ValueError("video must contain at least one frame")
    for t, (frame, mask_frame) in enumerate(zip(video, mask)):
        if len(frame) != len(mask_frame):
            raise ValueError(f"frame {t} height mismatch")
        if not frame:
            raise ValueError(f"frame {t} is empty")
        for y, (row, mask_row) in enumerate(zip(frame, mask_frame)):
            if len(row) != len(mask_row):
                raise ValueError(f"frame {t} row {y} width mismatch")
