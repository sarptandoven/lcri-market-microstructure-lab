"""Prototype contract for preview commitments in Preview Ladder DiT.

This file is intentionally dependency-light. It turns scalar video, mask, and optional
uncertainty arrays into a JSON-serializable commitment map that a final renderer can
consume. The target repo can later replace scalar arrays with latent tensors,
embeddings, optical flow, SAM2 masks, and DiT token masks while preserving the same
high-level fields.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

Video = Sequence[Sequence[Sequence[float]]]
Mask = Sequence[Sequence[Sequence[bool]]]
Uncertainty = Sequence[Sequence[Sequence[float]]]


@dataclass(frozen=True)
class FrameCommitment:
    frame_index: int
    mask_occupancy: float
    centroid_x: float | None
    centroid_y: float | None
    mean_inside_value: float | None
    boundary_mean_value: float | None
    lock_ratio: float
    refine_ratio: float


@dataclass(frozen=True)
class PreviewCommitment:
    schema_version: str
    frames: int
    height: int
    width: int
    lock_threshold: float
    refine_threshold: float
    frame_commitments: list[FrameCommitment]

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["frame_commitments"] = [asdict(f) for f in self.frame_commitments]
        return data


def extract_preview_commitment(
    preview: Video,
    mask: Mask,
    uncertainty: Uncertainty | None = None,
    *,
    lock_threshold: float = 0.25,
    refine_threshold: float = 0.65,
) -> PreviewCommitment:
    """Extract a final-render conditioning contract from a preview.

    lock_ratio is the fraction of edited pixels whose uncertainty is low enough to
    freeze or strongly condition during final denoising. refine_ratio is the fraction
    whose uncertainty is high enough to receive extra denoising or relaxed guidance.
    """
    _validate_video_and_mask(preview, mask)
    if uncertainty is not None:
        _validate_video_and_mask(uncertainty, mask)
    frames = len(preview)
    height = len(preview[0])
    width = len(preview[0][0])
    frame_commitments: list[FrameCommitment] = []
    for t in range(frames):
        inside = [(x, y, preview[t][y][x]) for y in range(height) for x in range(width) if mask[t][y][x]]
        boundary = _boundary_pixels(preview, mask, t)
        eligible = len(inside)
        if eligible:
            centroid_x = sum(x for x, _, _ in inside) / eligible
            centroid_y = sum(y for _, y, _ in inside) / eligible
            mean_inside = sum(v for _, _, v in inside) / eligible
        else:
            centroid_x = centroid_y = mean_inside = None
        boundary_mean = sum(boundary) / len(boundary) if boundary else None
        if uncertainty is None or eligible == 0:
            lock_ratio = 1.0 if eligible else 0.0
            refine_ratio = 0.0
        else:
            uvals = [uncertainty[t][y][x] for y in range(height) for x in range(width) if mask[t][y][x]]
            lock_ratio = sum(1 for u in uvals if u <= lock_threshold) / eligible
            refine_ratio = sum(1 for u in uvals if u >= refine_threshold) / eligible
        frame_commitments.append(
            FrameCommitment(
                frame_index=t,
                mask_occupancy=eligible / (height * width),
                centroid_x=centroid_x,
                centroid_y=centroid_y,
                mean_inside_value=mean_inside,
                boundary_mean_value=boundary_mean,
                lock_ratio=lock_ratio,
                refine_ratio=refine_ratio,
            )
        )
    return PreviewCommitment(
        schema_version="preview-commitment/v0.1-prototype",
        frames=frames,
        height=height,
        width=width,
        lock_threshold=lock_threshold,
        refine_threshold=refine_threshold,
        frame_commitments=frame_commitments,
    )


def _boundary_pixels(video: Video, mask: Mask, t: int) -> list[float]:
    height = len(video[t])
    width = len(video[t][0])
    vals: list[float] = []
    for y in range(height):
        for x in range(width):
            here = mask[t][y][x]
            edge = False
            for yy in range(max(0, y - 1), min(height, y + 2)):
                for xx in range(max(0, x - 1), min(width, x + 2)):
                    if mask[t][yy][xx] != here:
                        edge = True
            if edge:
                vals.append(video[t][y][x])
    return vals


def _validate_video_and_mask(video: Video, mask: Mask) -> None:
    if not video or not video[0] or not video[0][0]:
        raise ValueError("video must be non-empty frame-major data")
    if len(video) != len(mask):
        raise ValueError("video and mask frame counts differ")
    for t, frame in enumerate(video):
        if len(frame) != len(mask[t]):
            raise ValueError(f"frame {t} height differs")
        for y, row in enumerate(frame):
            if len(row) != len(mask[t][y]):
                raise ValueError(f"frame {t} row {y} width differs")
