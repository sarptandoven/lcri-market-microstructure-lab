from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .fixtures import FIXTURE_CASES, write_fixtures
from .commitments import commitment_loss, extract_commitments
from .metrics import preview_final_consistency_report
from .schema import LatencyEvent, RunReport, SubmissionIdentity, validate_run_report


@dataclass
class Timer:
    events: list[LatencyEvent] = field(default_factory=list)

    def mark(self, name: str, role: str) -> None:
        now = time.perf_counter()
        self.events.append(LatencyEvent(name=name, role=role, started_at_s=now, ended_at_s=now))

    @contextmanager
    def span(self, name: str, role: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            end = time.perf_counter()
            self.events.append(LatencyEvent(name=name, role=role, started_at_s=start, ended_at_s=end))


def run_synthetic(out_dir: str | Path, *, cases: list[str] | None = None, frames: int = 8, height: int = 32, width: int = 32) -> list[Path]:
    out = Path(out_dir)
    selected = cases or list(FIXTURE_CASES)
    paths = write_fixtures(out / "fixtures", cases=selected, frames=frames, height=height, width=width)
    report_paths: list[Path] = []
    for fixture_path in paths:
        timer = Timer()
        with timer.span("fixture.load", "io"):
            data = json.loads(fixture_path.read_text(encoding="utf-8"))
        timer.mark("preview_start", "preview_generate")
        timer.mark("preview_end", "preview_generate")
        timer.mark("final_start", "final_generate")
        timer.mark("final_end", "final_generate")
        with timer.span("metrics.preview_final_consistency_report", "metrics"):
            metrics = preview_final_consistency_report(
                source=data["source"], preview=data["preview"], final=data["final"], mask=data["mask"]
            ).to_dict()
        with timer.span("commitments.extract", "commitment_extract"):
            commitment_packet = extract_commitments(data["preview"], data["mask"])
            metrics["commitment_packet_loss"] = commitment_loss(data["preview"], data["final"], data["mask"], commitment_packet)
        artifact_hashes = {
            "fixture_json": _sha256_bytes(fixture_path.read_bytes()),
            "preview_array": _sha256_json(data["preview"]),
            "final_array": _sha256_json(data["final"]),
            "mask_array": _sha256_json(data["mask"]),
        }
        report = RunReport(
            run_id=f"synthetic-{data['case']}",
            task_id=f"synthetic-{data['case']}",
            metrics=metrics,
            preview=None,
            final=None,
            latency_events=timer.events,
            model={"backend": "synthetic_fixture_generator", "preview_steps": 0, "final_steps": 0, "scheduler": "deterministic_fixture"},
            environment={"fixture": fixture_path.name},
            submission=SubmissionIdentity(
                method_name="synthetic_fixture_baseline",
                method_version="0.1.0",
                backend_family="synthetic",
                preview_mode="deterministic_fixture_preview",
                final_mode="deterministic_fixture_final",
                artifact_hashes=artifact_hashes,
            ),
            commitments=commitment_packet.to_dict(),
            uncertainty={"source": "deterministic_boundary_threshold_proxy", "mean_boundary_uncertainty": commitment_packet.mean_boundary_uncertainty},
            task={"task_type": "synthetic_masked_replacement", "case": data["case"], "seed": 0},
        )
        report_data = report.to_dict()
        validate_run_report(report_data)
        report_path = out / f"report-{data['case']}.json"
        report_path.write_text(json.dumps(report_data, indent=2, sort_keys=True), encoding="utf-8")
        report_paths.append(report_path)
    return report_paths


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))
