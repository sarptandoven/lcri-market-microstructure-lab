from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

try:
    from preview_ladder_dit.fixtures import make_fixture
    from preview_ladder_dit.metrics import preview_final_consistency_report
    from preview_ladder_dit.schema import LatencyEvent, RunReport, VideoArtifact, validate_run_report
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "Run from the repo/workdir root so preview_ladder_dit is importable, or set PYTHONPATH."
    ) from exc

Video = list[list[list[float]]]
Mask = list[list[list[bool]]]


@dataclass
class Timer:
    events: list[LatencyEvent] = field(default_factory=list)

    def record(self, name: str, role: str, start: float, end: float, metadata: Mapping[str, Any] | None = None) -> None:
        self.events.append(LatencyEvent(name=name, role=role, started_at_s=start, ended_at_s=end, metadata=metadata or {}))

    def measure(self, name: str, role: str, fn: Callable[[], Any], metadata: Mapping[str, Any] | None = None) -> Any:
        start = time.perf_counter()
        try:
            return fn()
        finally:
            self.record(name, role, start, time.perf_counter(), metadata)


def _sha256_json(data: Mapping[str, Any]) -> str:
    blob = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _artifact_from_arrays(uri: str, data: Mapping[str, Any], fps: float = 8.0) -> VideoArtifact:
    frames = data["video"]
    frame_count = len(frames)
    height = len(frames[0]) if frame_count else 0
    width = len(frames[0][0]) if height else 0
    return VideoArtifact(
        uri=uri,
        sha256=_sha256_json(data),
        width=width,
        height=height,
        frame_count=frame_count,
        fps=fps,
        duration_s=frame_count / fps,
        media_type="application/vnd.preview-ladder.array-video+json",
    )


class AdapterError(RuntimeError):
    pass


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, Callable[[Mapping[str, Any], Path, Timer], dict[str, Any]]] = {
            "fixture": fixture_adapter,
            "json_file": json_file_adapter,
            "command_json": command_json_adapter,
        }

    def run(self, spec: Mapping[str, Any], out_dir: Path, timer: Timer) -> dict[str, Any]:
        adapter_name = str(spec.get("adapter", ""))
        if adapter_name not in self._adapters:
            raise AdapterError(f"unknown adapter {adapter_name!r}; expected one of {sorted(self._adapters)}")
        return self._adapters[adapter_name](spec, out_dir, timer)


def fixture_adapter(spec: Mapping[str, Any], out_dir: Path, timer: Timer) -> dict[str, Any]:
    """Generate deterministic synthetic source/preview/final/mask arrays.

    Input spec:
      {"adapter":"fixture", "case":"clean", "frames":8, "height":32, "width":32}
    """
    def build() -> dict[str, Any]:
        bundle = make_fixture(
            str(spec.get("case", "clean")),
            frames=int(spec.get("frames", 8)),
            height=int(spec.get("height", 32)),
            width=int(spec.get("width", 32)),
        )
        return bundle.to_dict()

    result = timer.measure("adapter.fixture.generate", "adapter", build, {"case": spec.get("case", "clean")})
    artifact_path = out_dir / f"fixture-{result['case']}.json"
    artifact_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    result["artifact_uri"] = str(artifact_path)
    return result


def json_file_adapter(spec: Mapping[str, Any], out_dir: Path, timer: Timer) -> dict[str, Any]:
    """Load arrays from a JSON file with source, preview, final, and mask keys."""
    path = Path(str(spec.get("path", ""))).expanduser()
    if not path.exists():
        raise AdapterError(f"json_file adapter path does not exist: {path}")

    def load() -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    result = timer.measure("adapter.json_file.load", "io", load, {"path": str(path)})
    result["artifact_uri"] = str(path)
    return result


def command_json_adapter(spec: Mapping[str, Any], out_dir: Path, timer: Timer) -> dict[str, Any]:
    """Run an external command that prints the same JSON shape as json_file.

    This makes the harness useful for real preview/final engines without importing
    model-specific code. The command receives no implicit stdin. Environment and
    model setup should be explicit in the command array.
    """
    command = spec.get("command")
    if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
        raise AdapterError("command_json adapter requires command: list[str]")

    def run_command() -> dict[str, Any]:
        completed = subprocess.run(command, cwd=str(out_dir), text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            raise AdapterError(
                f"command_json failed with exit {completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
        try:
            data = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise AdapterError(f"command_json stdout is not JSON: {exc}") from exc
        data.setdefault("command_stderr", completed.stderr)
        return data

    return timer.measure("adapter.command_json.run", "adapter", run_command, {"command": command})


def _require_bundle(data: Mapping[str, Any], label: str) -> None:
    for key in ("source", "preview", "final", "mask"):
        if key not in data:
            raise AdapterError(f"{label} output missing {key}")


def run_task(task: Mapping[str, Any], out_root: Path, registry: AdapterRegistry) -> Path:
    task_id = str(task.get("task_id") or task.get("id") or "task")
    run_id = str(task.get("run_id") or f"{task_id}-{int(time.time() * 1000)}")
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    timer = Timer()

    adapter_specs = task.get("adapters")
    if not isinstance(adapter_specs, list) or not adapter_specs:
        raise AdapterError(f"task {task_id} requires non-empty adapters list")

    # Current contract: first adapter that returns all arrays is evaluated. Later
    # adapters may be used for sidecar preprocessing/postprocessing in follow-up tranches.
    bundle: dict[str, Any] | None = None
    adapter_outputs: list[dict[str, Any]] = []
    for idx, spec in enumerate(adapter_specs):
        if not isinstance(spec, Mapping):
            raise AdapterError(f"task {task_id} adapter {idx} must be an object")
        output = registry.run(spec, out_dir, timer)
        adapter_outputs.append({"adapter": spec.get("adapter"), "keys": sorted(output.keys()), "artifact_uri": output.get("artifact_uri")})
        if all(k in output for k in ("source", "preview", "final", "mask")) and bundle is None:
            bundle = output

    if bundle is None:
        raise AdapterError(f"task {task_id} had no adapter output with source/preview/final/mask")
    _require_bundle(bundle, task_id)

    metrics = timer.measure(
        "metrics.preview_final_consistency_report",
        "metrics",
        lambda: preview_final_consistency_report(
            source=bundle["source"],
            preview=bundle["preview"],
            final=bundle["final"],
            mask=bundle["mask"],
            boundary_radius=int(task.get("boundary_radius", 1)),
        ).to_dict(),
        {"boundary_radius": int(task.get("boundary_radius", 1))},
    )
    metrics.update(_latency_metrics(timer.events))

    preview_artifact = _artifact_from_arrays(f"memory://{run_id}/preview", {"video": bundle["preview"]}, fps=float(task.get("fps", 8.0)))
    final_artifact = _artifact_from_arrays(f"memory://{run_id}/final", {"video": bundle["final"]}, fps=float(task.get("fps", 8.0)))
    report = RunReport(
        run_id=run_id,
        task_id=task_id,
        metrics=metrics,
        preview=preview_artifact,
        final=final_artifact,
        latency_events=timer.events,
        model={"adapters": adapter_outputs, "hypothesis": task.get("hypothesis", "")},
        environment={
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "runner": "prototypes/harness_runner/runner.py",
        },
    )
    report_data = report.to_dict()
    report_data["task"] = {k: v for k, v in task.items() if k != "adapters"}
    validate_run_report(report_data)
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report_data, indent=2, sort_keys=True), encoding="utf-8")
    return report_path


def _latency_metrics(events: list[LatencyEvent]) -> dict[str, float]:
    total = sum(event.duration_s for event in events)
    by_role: dict[str, float] = {}
    for event in events:
        by_role[event.role] = by_role.get(event.role, 0.0) + event.duration_s
    metrics = {"latency_total_s": total}
    metrics.update({f"latency_{role}_s": duration for role, duration in sorted(by_role.items())})
    return metrics


def run_task_file(task_json: Path, out_root: Path) -> dict[str, Any]:
    spec = json.loads(task_json.read_text(encoding="utf-8"))
    tasks = spec.get("tasks", [spec]) if isinstance(spec, Mapping) else spec
    if not isinstance(tasks, list):
        raise AdapterError("task JSON must be an object, an object with tasks list, or a task list")
    out_root.mkdir(parents=True, exist_ok=True)
    registry = AdapterRegistry()
    report_paths = [run_task(task, out_root, registry) for task in tasks]
    aggregate = {"reports": [str(path) for path in report_paths], "count": len(report_paths)}
    aggregate_path = out_root / "aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True), encoding="utf-8")
    return aggregate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preview Ladder DiT prototype experiment harness runner")
    parser.add_argument("task_json", help="task JSON file or object with tasks[]")
    parser.add_argument("--out", required=True, help="output directory for reports")
    args = parser.parse_args(argv)
    aggregate = run_task_file(Path(args.task_json), Path(args.out))
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
