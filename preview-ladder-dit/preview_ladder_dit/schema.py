from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
REPORT_SCHEMA_VERSION = "preview-ladder-report/v0.3"
ALLOWED_REPORT_SCHEMA_VERSIONS = {REPORT_SCHEMA_VERSION, "preview-ladder-report/v0.2", "preview-ladder-report/v0.1"}
PREVIEW_FINAL_LATENCY_MILESTONES = ("preview_start", "preview_end", "final_start", "final_end")
STANDARD_LATENCY_ROLES = {
    "io",
    "metrics",
    "preview_generate",
    "preview_decode",
    "commitment_extract",
    "final_generate",
    "final_decode",
    "report_write",
}


@dataclass(frozen=True)
class SubmissionIdentity:
    method_name: str
    method_version: str
    backend_family: str
    preview_mode: str
    final_mode: str
    artifact_hashes: Mapping[str, str]
    report_schema_version: str = REPORT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VideoArtifact:
    uri: str
    sha256: str
    width: int
    height: int
    frame_count: int
    fps: float
    duration_s: float
    media_type: str = "video/mp4"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LatencyEvent:
    name: str
    role: str
    started_at_s: float
    ended_at_s: float
    device: str = "unknown"
    metadata: Mapping[str, Any] | None = None

    @property
    def duration_s(self) -> float:
        return self.ended_at_s - self.started_at_s

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["duration_s"] = self.duration_s
        if data["metadata"] is None:
            data["metadata"] = {}
        return data


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    task_type: str
    source: VideoArtifact
    masks: Sequence[VideoArtifact]
    prompt: str
    mask_ids: Sequence[str]
    seed: int
    frame_range: tuple[int, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "source": self.source.to_dict(),
            "masks": [m.to_dict() for m in self.masks],
            "edit_spec": {"prompt": self.prompt, "mask_ids": list(self.mask_ids), "seed": self.seed},
            "evaluation": {"frame_range": {"start": self.frame_range[0], "end_exclusive": self.frame_range[1]}},
        }


@dataclass(frozen=True)
class RunReport:
    run_id: str
    task_id: str
    metrics: Mapping[str, float]
    preview: VideoArtifact | None
    final: VideoArtifact | None
    latency_events: Sequence[LatencyEvent]
    model: Mapping[str, Any]
    environment: Mapping[str, Any]
    submission: SubmissionIdentity | Mapping[str, Any] | None = None
    commitments: Mapping[str, Any] | None = None
    uncertainty: Mapping[str, Any] | None = None
    task: Mapping[str, Any] | None = None
    schema_version: str = REPORT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "task": dict(self.task or {}),
            "submission": _submission_to_dict(self.submission),
            "artifacts": {
                "preview": self.preview.to_dict() if self.preview else None,
                "final": self.final.to_dict() if self.final else None,
                "commitments": dict(self.commitments or {}),
                "uncertainty": dict(self.uncertainty or {}),
            },
            "metrics": dict(self.metrics),
            "latency_log": {"events": [e.to_dict() for e in self.latency_events]},
            "model": dict(self.model),
            "environment": dict(self.environment),
        }

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def validate_run_report(report: Mapping[str, Any]) -> None:
    schema_version = report.get("schema_version")
    if schema_version not in ALLOWED_REPORT_SCHEMA_VERSIONS:
        raise ValueError(f"schema_version must be one of {sorted(ALLOWED_REPORT_SCHEMA_VERSIONS)}")
    _require_str(report, "run_id")
    _require_str(report, "task_id")
    submission = report.get("submission")
    if schema_version == REPORT_SCHEMA_VERSION and not submission:
        raise ValueError("v0.3 report requires submission identity")
    if submission:
        _validate_submission_identity(submission)
    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping) or not metrics:
        raise ValueError("metrics must be a non-empty object")
    for key, value in metrics.items():
        if not isinstance(key, str) or not isinstance(value, (int, float)):
            raise ValueError("metrics values must be numeric")
    if schema_version == REPORT_SCHEMA_VERSION:
        for required in ("preview_final_l1", "commitment_weighted_error"):
            if required not in metrics:
                raise ValueError(f"v0.2 report metrics missing {required}")
    events = report.get("latency_log", {}).get("events", [])
    if not isinstance(events, list):
        raise ValueError("latency_log.events must be a list")
    prev_start = -1.0
    seen_roles: set[str] = set()
    for event in events:
        for key in ("name", "role", "started_at_s", "ended_at_s"):
            if key not in event:
                raise ValueError(f"latency event missing {key}")
        if event["started_at_s"] < prev_start:
            raise ValueError("latency events must be monotonic by started_at_s")
        if event["ended_at_s"] < event["started_at_s"]:
            raise ValueError(f"latency event {event['name']} ends before it starts")
        if not isinstance(event["role"], str) or not event["role"]:
            raise ValueError("latency event role must be a non-empty string")
        seen_roles.add(event["role"])
        prev_start = event["started_at_s"]
    if schema_version in {REPORT_SCHEMA_VERSION, "preview-ladder-report/v0.2"} and events and not seen_roles.intersection(STANDARD_LATENCY_ROLES):
        raise ValueError("v0.2/v0.3 report should use at least one standard latency role")
    if schema_version == REPORT_SCHEMA_VERSION and events:
        _validate_preview_final_milestones(events)
    artifacts = report.get("artifacts", {})
    if not isinstance(artifacts, Mapping):
        raise ValueError("artifacts must be an object")
    for label in ("preview", "final"):
        artifact = artifacts.get(label)
        if artifact is not None:
            validate_video_artifact(artifact, label)
    if schema_version in {REPORT_SCHEMA_VERSION, "preview-ladder-report/v0.2"}:
        _validate_optional_object(artifacts.get("commitments", {}), "artifacts.commitments")
        _validate_optional_object(artifacts.get("uncertainty", {}), "artifacts.uncertainty")
        _validate_optional_object(report.get("task", {}), "task")
        _validate_optional_object(report.get("model", {}), "model")
        _validate_optional_object(report.get("environment", {}), "environment")


def validate_video_artifact(artifact: Mapping[str, Any], label: str = "artifact") -> None:
    for key in ("uri", "sha256", "width", "height", "frame_count", "fps", "duration_s", "media_type"):
        if key not in artifact:
            raise ValueError(f"{label} missing {key}")
    if not SHA256_RE.match(str(artifact["sha256"])):
        raise ValueError(f"{label}.sha256 must be lowercase 64-char sha256")
    for key in ("width", "height", "frame_count"):
        if int(artifact[key]) <= 0:
            raise ValueError(f"{label}.{key} must be positive")
    if float(artifact["fps"]) <= 0 or float(artifact["duration_s"]) <= 0:
        raise ValueError(f"{label} fps and duration_s must be positive")


def load_and_validate_run_report(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_run_report(data)
    return data


def _validate_optional_object(value: Any, label: str) -> None:
    if value is not None and not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")


def _submission_to_dict(value: SubmissionIdentity | Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, SubmissionIdentity):
        return value.to_dict()
    return dict(value)


def _validate_submission_identity(value: Any) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("submission must be an object")
    for key in ("method_name", "method_version", "backend_family", "preview_mode", "final_mode", "report_schema_version"):
        _require_str(value, key)
    if value["report_schema_version"] not in ALLOWED_REPORT_SCHEMA_VERSIONS:
        raise ValueError(f"submission.report_schema_version must be one of {sorted(ALLOWED_REPORT_SCHEMA_VERSIONS)}")
    hashes = value.get("artifact_hashes")
    if not isinstance(hashes, Mapping) or not hashes:
        raise ValueError("submission.artifact_hashes must be a non-empty object")
    for key, digest in hashes.items():
        if not isinstance(key, str) or not key:
            raise ValueError("submission.artifact_hashes keys must be non-empty strings")
        if not SHA256_RE.match(str(digest)):
            raise ValueError(f"submission.artifact_hashes.{key} must be lowercase 64-char sha256")


def _validate_preview_final_milestones(events: Sequence[Mapping[str, Any]]) -> None:
    names = {str(event.get("name", "")) for event in events}
    missing = [name for name in PREVIEW_FINAL_LATENCY_MILESTONES if name not in names]
    if missing:
        raise ValueError(f"v0.3 latency_log.events missing preview/final milestones: {', '.join(missing)}")
    by_name = {str(event.get("name")): event for event in events}
    ordered_times = [float(by_name[name]["started_at_s"]) for name in PREVIEW_FINAL_LATENCY_MILESTONES]
    if ordered_times != sorted(ordered_times):
        raise ValueError("v0.3 preview/final latency milestones must be ordered preview_start <= preview_end <= final_start <= final_end")


def _require_str(mapping: Mapping[str, Any], key: str) -> None:
    if not isinstance(mapping.get(key), str) or not mapping[key]:
        raise ValueError(f"{key} must be a non-empty string")
