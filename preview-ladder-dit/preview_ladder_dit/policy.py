from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .commitments import CommitmentPacket

Mask = Sequence[Sequence[Sequence[bool]]]


@dataclass(frozen=True)
class CommitmentPolicy:
    """Uncertainty-aware release policy for preview-to-final commitments.

    Lock strengths are normalized to [0, 1]. High values mean the final pass
    should preserve that approved preview attribute. ``release_rate`` records
    deliberate freedom left for uncertain regions, commonly mask boundaries and
    unstable motion.
    """

    schema_version: str
    background_lock: float
    boundary_lock: float
    trajectory_lock: float
    occupancy_lock: float
    appearance_lock: float
    release_rate: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_commitment_policy(
    metrics: Mapping[str, float],
    packet: CommitmentPacket | None = None,
    *,
    uncertainty: Mapping[str, float] | None = None,
) -> CommitmentPolicy:
    """Select deterministic lock and release strengths from scalar signals."""

    proxies = dict(uncertainty or {})
    mean_release = _mean_frame_attr(packet, "release_fraction", default=0.0)
    boundary_uncertainty = max(
        _coalesce(proxies.get("boundary_uncertainty"), packet.mean_boundary_uncertainty if packet is not None else None),
        _normalize(metrics.get("boundary_consistency_error", 0.0), 0.20),
    )
    motion_uncertainty = max(
        _coalesce(proxies.get("motion_uncertainty"), packet.global_trajectory_jitter if packet is not None else None),
        _normalize(metrics.get("trajectory_center_error", 0.0), 0.16),
        _normalize(metrics.get("trajectory_acceleration_drift", 0.0), 0.16),
    )
    occupancy_uncertainty = max(
        _coalesce(proxies.get("occupancy_uncertainty"), mean_release),
        _normalize(metrics.get("mask_occupancy_delta", metrics.get("occupancy_iou_error", 0.0)), 0.30),
    )
    appearance_uncertainty = max(
        _coalesce(proxies.get("appearance_uncertainty")),
        _normalize(metrics.get("low_frequency_drift", metrics.get("preview_final_l1", 0.0)), 0.18),
    )
    background_error = max(
        _normalize(metrics.get("background_preservation_error", 0.0), 0.08),
        _normalize(metrics.get("background_temporal_leak", 0.0), 0.08),
    )
    flicker_error = _normalize(
        max(metrics.get("temporal_flicker_delta", 0.0), metrics.get("local_temporal_residual_flicker", 0.0)),
        0.20,
    )
    packet_lock = packet.lock_budget if packet is not None else 1.0

    background_lock = _clamp(0.80 + 0.20 * background_error)
    boundary_lock = _clamp(packet_lock * (1.0 - 0.75 * boundary_uncertainty))
    trajectory_lock = _clamp(packet_lock * (1.0 - 0.65 * motion_uncertainty - 0.20 * flicker_error))
    occupancy_lock = _clamp(packet_lock * (1.0 - 0.65 * occupancy_uncertainty))
    appearance_lock = _clamp(packet_lock * (1.0 - 0.55 * appearance_uncertainty - 0.15 * flicker_error))
    release_rate = _clamp(
        0.15 * (1.0 - packet_lock)
        + 0.30 * boundary_uncertainty
        + 0.25 * motion_uncertainty
        + 0.20 * occupancy_uncertainty
        + 0.10 * appearance_uncertainty
    )

    reasons: list[str] = []
    if background_error > 0.25:
        reasons.append("background preservation error detected, keep background strongly locked")
    else:
        reasons.append("background outside the mask should remain locked by default")
    if boundary_uncertainty > 0.35:
        reasons.append("boundary is uncertain, release boundary pixels instead of hard-locking halos")
    if motion_uncertainty > 0.35 or flicker_error > 0.35:
        reasons.append("motion or temporal residual is unstable, reduce trajectory lock")
    if occupancy_uncertainty > 0.35:
        reasons.append("occupancy is unstable, leave shape growth or shrinkage freedom for final pass")
    if appearance_uncertainty > 0.35:
        reasons.append("appearance proxy is unstable, soften appearance commitment")

    return CommitmentPolicy(
        schema_version="preview-commitment-policy/v0.1",
        background_lock=background_lock,
        boundary_lock=boundary_lock,
        trajectory_lock=trajectory_lock,
        occupancy_lock=occupancy_lock,
        appearance_lock=appearance_lock,
        release_rate=release_rate,
        reasons=tuple(reasons),
    )


def project_mask_to_latent_grid(
    mask: Mask,
    *,
    latent_frames: int,
    latent_height: int,
    latent_width: int,
) -> list[list[list[bool]]]:
    """Project a pixel mask to latent tokens by conservative max pooling."""

    if latent_frames <= 0 or latent_height <= 0 or latent_width <= 0:
        raise ValueError("latent dimensions must be positive")
    _validate_mask(mask)
    frames = len(mask)
    height = len(mask[0])
    width = len(mask[0][0])
    projected = [[[False for _ in range(latent_width)] for _ in range(latent_height)] for _ in range(latent_frames)]
    for t in range(frames):
        lt = min(latent_frames - 1, (t * latent_frames) // frames)
        for y in range(height):
            ly = min(latent_height - 1, (y * latent_height) // height)
            for x in range(width):
                if mask[t][y][x]:
                    lx = min(latent_width - 1, (x * latent_width) // width)
                    projected[lt][ly][lx] = True
    return projected


def build_conditioning_hints(
    *,
    policy: CommitmentPolicy,
    packet: CommitmentPacket | None,
    latent_mask: Sequence[Sequence[Sequence[bool]]],
) -> dict[str, Any]:
    """Build backend-agnostic hints for LTX-style denoising adapters."""

    active_tokens = sum(1 for frame in latent_mask for row in frame for value in row if value)
    total_tokens = sum(1 for frame in latent_mask for row in frame for _ in row)
    return {
        "schema_version": "preview-conditioning-hints/v0.1",
        "policy": policy.to_dict(),
        "latent_mask_shape": (
            len(latent_mask),
            len(latent_mask[0]) if latent_mask else 0,
            len(latent_mask[0][0]) if latent_mask and latent_mask[0] else 0,
        ),
        "active_token_fraction": active_tokens / total_tokens if total_tokens else 0.0,
        "frame_weights": _frame_weights(policy, packet),
        "denoising_hooks": {
            "background": "preserve source latents outside projected mask",
            "boundary": "blend preview conditioning according to boundary_lock and release_rate",
            "trajectory": "condition centroid/keyframe tokens according to trajectory_lock",
            "occupancy": "condition projected edit-token occupancy according to occupancy_lock",
            "appearance": "condition low-frequency preview appearance according to appearance_lock",
        },
    }


def _frame_weights(policy: CommitmentPolicy, packet: CommitmentPacket | None) -> list[float]:
    if packet is None:
        return []
    base = (policy.boundary_lock + policy.trajectory_lock + policy.occupancy_lock + policy.appearance_lock) / 4.0
    return [_clamp(base * (1.0 - frame.release_fraction)) for frame in packet.frames]


def _mean_frame_attr(packet: CommitmentPacket | None, attr: str, *, default: float) -> float:
    if packet is None or not packet.frames:
        return default
    return _clamp(sum(float(getattr(frame, attr)) for frame in packet.frames) / len(packet.frames))


def _coalesce(*values: float | None) -> float:
    for value in values:
        if value is not None:
            return _clamp(float(value))
    return 0.0


def _normalize(value: float | None, scale: float) -> float:
    return _clamp(float(value or 0.0) / scale if scale > 0 else 0.0)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _validate_mask(mask: Mask) -> None:
    if not mask:
        raise ValueError("mask must contain at least one frame")
    height = len(mask[0])
    width = len(mask[0][0]) if height else 0
    if height == 0 or width == 0:
        raise ValueError("mask frames must be non-empty")
    for t, frame in enumerate(mask):
        if len(frame) != height:
            raise ValueError(f"mask frame {t} height mismatch")
        for y, row in enumerate(frame):
            if len(row) != width:
                raise ValueError(f"mask frame {t} row {y} width mismatch")
