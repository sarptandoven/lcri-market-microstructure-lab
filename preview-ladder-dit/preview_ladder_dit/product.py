from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

DEFAULT_GATES = {
    "preview_final_l1": 0.08,
    "boundary_consistency_error": 0.12,
    "boundary_signed_bias": 0.04,
    "temporal_flicker_delta": 0.10,
    "background_preservation_error": 0.03,
    "background_temporal_leak": 0.04,
    "trajectory_center_error": 0.08,
    "trajectory_acceleration_drift": 0.05,
    "preview_confidence_release_rate": 0.70,
    "commitment_weighted_error": 0.10,
}

_ABSOLUTE_GATED_METRICS = {"boundary_signed_bias"}


@dataclass(frozen=True)
class GateResult:
    metric: str
    value: float
    threshold: float
    passed: bool
    observed: bool = True
    mode: str = "max"


@dataclass(frozen=True)
class PreviewScorecard:
    run_id: str
    task_id: str
    trust_score: float
    passed: bool
    gates: tuple[GateResult, ...]
    diagnosis: tuple[str, ...]
    latency: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_preview_scorecard(report: Mapping[str, Any], gates: Mapping[str, float] | None = None) -> PreviewScorecard:
    thresholds = dict(DEFAULT_GATES)
    if gates:
        thresholds.update({str(k): float(v) for k, v in gates.items()})
    metrics = report.get("metrics", {})
    gate_results: list[GateResult] = []
    for metric, threshold in thresholds.items():
        observed = metric in metrics
        value = float(metrics.get(metric, 0.0)) if observed else 0.0
        mode = "abs_max" if metric in _ABSOLUTE_GATED_METRICS else "max"
        comparable = abs(value) if mode == "abs_max" else value
        gate_results.append(
            GateResult(metric=metric, value=value, threshold=threshold, passed=observed and comparable <= threshold, observed=observed, mode=mode)
        )
    trust_score = _trust_score(gate_results)
    diagnosis = _diagnose(metrics, gate_results)
    return PreviewScorecard(
        run_id=str(report.get("run_id", "")),
        task_id=str(report.get("task_id", "")),
        trust_score=trust_score,
        passed=all(g.passed for g in gate_results),
        gates=tuple(gate_results),
        diagnosis=tuple(diagnosis),
        latency=_latency_summary(report),
    )


def _trust_score(gates: list[GateResult]) -> float:
    if not gates:
        return 0.0
    scores = []
    for gate in gates:
        if not gate.observed:
            scores.append(0.0)
            continue
        value = abs(gate.value) if gate.mode == "abs_max" else gate.value
        if gate.threshold <= 0:
            scores.append(1.0 if value <= gate.threshold else 0.0)
        else:
            scores.append(max(0.0, min(1.0, 1.0 - value / (2.0 * gate.threshold))))
    return sum(scores) / len(scores)


def _diagnose(metrics: Mapping[str, Any], gates: list[GateResult]) -> list[str]:
    failed = [gate.metric for gate in gates if not gate.passed]
    missing = [gate.metric for gate in gates if not gate.observed]
    diagnosis: list[str] = []
    if not failed:
        diagnosis.append("preview commitments are internally consistent with final output under configured gates")
    if missing:
        diagnosis.append("scorecard is incomplete because required gate metrics are missing: " + ", ".join(sorted(missing)))
    if "background_preservation_error" in failed:
        diagnosis.append("background changed outside the mask, tighten source preservation or retake mask")
    if "background_temporal_leak" in failed:
        diagnosis.append("background temporal leak detected outside the mask, preserve source-frame temporal deltas in unedited regions")
    if "boundary_consistency_error" in failed:
        diagnosis.append("boundary drift or halo dominates, release boundary uncertainty and preserve mask interior")
    if "boundary_signed_bias" in failed:
        signed = float(metrics.get("boundary_signed_bias", 0.0))
        if signed > 0:
            diagnosis.append("boundary signed bias indicates outward bleed beyond the accepted mask boundary")
        elif signed < 0:
            diagnosis.append("boundary signed bias indicates inward erosion or underfill near the accepted mask boundary")
        else:
            diagnosis.append("boundary signed bias is missing or exceeds configured absolute gate")
    if "temporal_flicker_delta" in failed or "commitment_weighted_error" in failed:
        diagnosis.append("temporal or commitment drift is high, increase final conditioning on accepted preview structure")
    if "trajectory_center_error" in failed:
        diagnosis.append("replacement trajectory diverged, add keyframe or tracker commitments")
    if "trajectory_acceleration_drift" in failed:
        diagnosis.append("replacement motion curvature changed, add velocity or acceleration commitments across frames")
    if "preview_confidence_release_rate" in failed:
        diagnosis.append("preview released too much committed area, require tighter preview confidence or split uncertain regions before final render")
    for metric in failed:
        if metric in missing:
            continue
        if not any(metric in item for item in diagnosis):
            diagnosis.append(f"{metric} exceeded gate")
    if float(metrics.get("occupancy_iou_error", 0.0)) > 0.25:
        diagnosis.append("salient replacement occupancy changed substantially")
    return diagnosis


def _latency_summary(report: Mapping[str, Any]) -> dict[str, float]:
    events = report.get("latency_log", {}).get("events", [])
    by_role: dict[str, float] = {}
    for event in events:
        role = str(event.get("role", "unknown"))
        duration = float(event.get("duration_s", float(event.get("ended_at_s", 0.0)) - float(event.get("started_at_s", 0.0))))
        by_role[role] = by_role.get(role, 0.0) + max(0.0, duration)
    by_role["total_s"] = sum(by_role.values())
    return by_role
