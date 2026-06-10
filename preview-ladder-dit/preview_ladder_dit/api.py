from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .commitments import CommitmentPacket, commitment_loss, extract_commitments
from .metrics import Mask, PreviewFinalReport, Video, preview_final_consistency_report
from .product import PreviewScorecard, build_preview_scorecard
from .schema import LatencyEvent, RunReport, SubmissionIdentity, VideoArtifact, validate_run_report


@dataclass(frozen=True)
class PreviewLadderConfig:
    """Model-agnostic public configuration for preview-final evaluation.

    This is deliberately dependency-free. LTX, ComfyUI, hosted APIs, or custom
    video-DiT backends should generate source/preview/final/mask artifacts, then
    call this API to emit the same report contract.
    """

    run_id: str
    task_id: str
    task_type: str = "masked_video_replacement"
    prompt: str = ""
    mask_id: str = "primary_edit_mask"
    seed: int | None = None
    model: Mapping[str, Any] | None = None
    environment: Mapping[str, Any] | None = None
    method_name: str = "external_preview_ladder_method"
    method_version: str = "unspecified"
    backend_family: str = "external_or_synthetic"
    preview_mode: str = "accepted_preview"
    final_mode: str = "final_render"
    boundary_radius: int = 1


@dataclass(frozen=True)
class EvaluationResult:
    """In-memory result returned by the public preview-final API."""

    metrics: PreviewFinalReport
    commitments: CommitmentPacket
    scorecard: PreviewScorecard
    report: RunReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics.to_dict(),
            "commitments": self.commitments.to_dict(),
            "scorecard": self.scorecard.to_dict(),
            "report": self.report.to_dict(),
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    def write_report_json(self, path: str | Path) -> None:
        self.report.write_json(path)


def evaluate_preview_final(
    *,
    source: Video,
    preview: Video,
    final: Video,
    mask: Mask,
    config: PreviewLadderConfig,
    preview_artifact: VideoArtifact | None = None,
    final_artifact: VideoArtifact | None = None,
    latency_events: list[LatencyEvent] | None = None,
    gates: Mapping[str, float] | None = None,
) -> EvaluationResult:
    """Evaluate one accepted preview against one final render.

    The function computes dependency-light metrics, extracts a serializable
    preview commitment packet, builds a product scorecard, and returns a
    validated v0.3 `RunReport`. Heavy backends should keep their imports outside
    this function and pass only decoded scalar features or arrays here.
    """

    metrics = preview_final_consistency_report(
        source=source,
        preview=preview,
        final=final,
        mask=mask,
        boundary_radius=config.boundary_radius,
    )
    packet = extract_commitments(preview, mask, boundary_radius=config.boundary_radius)
    metric_dict = metrics.to_dict()
    metric_dict["commitment_packet_loss"] = commitment_loss(preview, final, mask, packet)
    task = {
        "task_type": config.task_type,
        "edit_spec": {"prompt": config.prompt, "mask_id": config.mask_id, "seed": config.seed},
    }
    report = RunReport(
        run_id=config.run_id,
        task_id=config.task_id,
        metrics=metric_dict,
        preview=preview_artifact,
        final=final_artifact,
        latency_events=latency_events or [],
        model=dict(config.model or {"backend": "external_or_synthetic", "ltx_optional": True}),
        environment=dict(config.environment or {}),
        submission=SubmissionIdentity(
            method_name=config.method_name,
            method_version=config.method_version,
            backend_family=config.backend_family,
            preview_mode=config.preview_mode,
            final_mode=config.final_mode,
            artifact_hashes={
                "source_array": _sha256_json(source),
                "preview_array": _sha256_json(preview),
                "final_array": _sha256_json(final),
                "mask_array": _sha256_json(mask),
            },
        ),
        commitments=packet.to_dict(),
        uncertainty={
            "source": "preview_commitment_boundary_threshold_proxy",
            "mean_boundary_uncertainty": packet.mean_boundary_uncertainty,
            "lock_budget": packet.lock_budget,
        },
        task=task,
    )
    report_data = report.to_dict()
    validate_run_report(report_data)
    scorecard = build_preview_scorecard(report_data, gates=gates)
    return EvaluationResult(metrics=metrics, commitments=packet, scorecard=scorecard, report=report)


def evaluate_fixture_dict(data: Mapping[str, Any], *, run_id: str | None = None, task_id: str | None = None) -> EvaluationResult:
    """Evaluate a fixture JSON object produced by `preview_ladder_dit.fixtures`.

    This convenience API is useful for notebooks, demos, and integration tests.
    """

    case = str(data.get("case", "fixture"))
    config = PreviewLadderConfig(
        run_id=run_id or f"fixture-{case}",
        task_id=task_id or f"fixture-{case}",
        task_type="synthetic_masked_replacement",
        prompt=str(data.get("metadata", {}).get("failure_description", case)) if isinstance(data.get("metadata", {}), Mapping) else case,
        model={"backend": "synthetic_fixture", "ltx_optional": True},
        environment={"case": case},
    )
    return evaluate_preview_final(
        source=data["source"],
        preview=data["preview"],
        final=data["final"],
        mask=data["mask"],
        config=config,
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


__all__ = [
    "EvaluationResult",
    "PreviewLadderConfig",
    "evaluate_fixture_dict",
    "evaluate_preview_final",
]
