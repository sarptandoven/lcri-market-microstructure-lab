from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .metrics import PreviewFinalReport, preview_final_consistency_report
from .policy import CommitmentPolicy, build_conditioning_hints, project_mask_to_latent_grid


@dataclass(frozen=True)
class LTXVideoSpec:
    width: int
    height: int
    frames: int
    fps: float

    def validate(self) -> None:
        if self.width % 32 or self.height % 32:
            raise ValueError("LTX width and height must be multiples of 32")
        if (self.frames - 1) % 8:
            raise ValueError("LTX video frame count should satisfy 8k + 1")
        if self.fps <= 0:
            raise ValueError("fps must be positive")


@dataclass(frozen=True)
class LTXReplacementRequest:
    prompt: str
    spec: LTXVideoSpec
    seed: int = 0
    source_video_path: str | None = None
    mask_path: str | None = None
    negative_prompt: str = ""
    preview_steps: int = 8
    final_steps: int = 32
    scheduler: str = "ltx-default"
    mask_policy: str = "preserve_background_commit_interior_release_boundary"
    commitment_strength: float = 0.75
    backend_options: dict[str, Any] | None = None


@dataclass(frozen=True)
class LTXStageLog:
    name: str
    latency_ms: float
    width: int
    height: int
    frames: int
    fps: float
    latent_shape: tuple[int, int, int, int, int]
    step_count: int | None = None
    sigma_count: int | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if data["metadata"] is None:
            data["metadata"] = {}
        return data


@dataclass(frozen=True)
class LTXAdapterResult:
    preview: Any
    final: Any | None
    report: PreviewFinalReport | None
    logs: list[LTXStageLog]


def ltx_video_latent_shape(frames: int, height: int, width: int, channels: int = 128) -> tuple[int, int, int, int, int]:
    return (1, channels, (frames - 1) // 8 + 1, height // 32, width // 32)


class LTXPreviewFinalAdapter:
    """Dependency-light LTX adapter contract.

    Real LTX imports intentionally happen in a future backend subclass so the
    benchmark package remains installable in CI without checkpoints or torch.
    This skeleton fixes request/result shapes and the metric handoff.
    """

    backend_name = "dry-run-contract"

    def preview(self, request: LTXReplacementRequest) -> LTXAdapterResult:
        request.spec.validate()
        shape = ltx_video_latent_shape(request.spec.frames, request.spec.height, request.spec.width)
        log = LTXStageLog(
            name="preview",
            latency_ms=0.0,
            width=request.spec.width,
            height=request.spec.height,
            frames=request.spec.frames,
            fps=request.spec.fps,
            latent_shape=shape,
            metadata={
                "backend": self.backend_name,
                "prompt": request.prompt,
                "scheduler": request.scheduler,
                "mask_policy": request.mask_policy,
                "commitment_strength": request.commitment_strength,
                "backend_options": request.backend_options or {},
            },
        )
        return LTXAdapterResult(preview=None, final=None, report=None, logs=[log])

    def final(self, request: LTXReplacementRequest, preview: LTXAdapterResult) -> LTXAdapterResult:
        request.spec.validate()
        shape = ltx_video_latent_shape(request.spec.frames, request.spec.height, request.spec.width)
        log = LTXStageLog(
            name="final",
            latency_ms=0.0,
            width=request.spec.width,
            height=request.spec.height,
            frames=request.spec.frames,
            fps=request.spec.fps,
            latent_shape=shape,
            metadata={
                "backend": self.backend_name,
                "scheduler": request.scheduler,
                "mask_policy": request.mask_policy,
                "commitment_strength": request.commitment_strength,
                "backend_options": request.backend_options or {},
            },
        )
        return LTXAdapterResult(preview=preview.preview, final=None, report=None, logs=[*preview.logs, log])

    def preview_then_final(self, request: LTXReplacementRequest) -> LTXAdapterResult:
        return self.final(request, self.preview(request))

    def report_from_arrays(self, *, source, preview, final, mask) -> PreviewFinalReport:
        return preview_final_consistency_report(source=source, preview=preview, final=final, mask=mask)

    def conditioning_hints_from_policy(self, *, mask, policy: CommitmentPolicy, packet=None) -> dict[str, Any]:
        """Return backend-agnostic conditioning hints for a committed final pass."""

        latent_shape = ltx_video_latent_shape(_policy_frames(packet, mask), len(mask[0]), len(mask[0][0]))
        _, _, latent_frames, latent_height, latent_width = latent_shape
        latent_mask = project_mask_to_latent_grid(
            mask,
            latent_frames=max(1, latent_frames),
            latent_height=max(1, latent_height),
            latent_width=max(1, latent_width),
        )
        return build_conditioning_hints(policy=policy, packet=packet, latent_mask=latent_mask)


def _policy_frames(packet, mask) -> int:
    if packet is not None:
        return len(packet.frames)
    return len(mask)
