from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lcri_lab.alpha import (
    alpha_event_drift_gate,
    alpha_event_release_review_packet,
    alpha_event_score_weighted_drift,
    alpha_event_window_summary,
)
from lcri_lab.evaluation import (
    lcri_ci_confidence_coverage_scorecard,
    lcri_ci_confidence_coverage_summary,
    lcri_ci_gate_contradiction_diagnostics,
    lcri_ci_gate_contradiction_summary,
    lcri_contradiction_review_packet_summary,
    lcri_cross_artifact_evidence_index,
    lcri_cross_artifact_evidence_index_summary,
    lcri_evidence_release_checklist,
    lcri_evidence_release_checklist_summary,
    lcri_evidence_lineage_map,
    lcri_evidence_lineage_map_summary,
    lcri_owner_handoff_packet,
    lcri_owner_handoff_packet_summary,
    lcri_uncertainty_weighted_review_priority,
    lcri_uncertainty_weighted_review_priority_summary,
)
from lcri_lab.reversal import fracture_reversal_release_gate


def build_artifact_manifest(
    *,
    rows: int,
    train_rows: int,
    heldout_rows: int,
    seed: int,
    train_frac: float,
    model_artifact_version: int,
    artifacts: list[str],
    artifact_metadata: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a reproducibility manifest for a demo run."""
    if min(rows, train_rows, heldout_rows) < 0:
        raise ValueError("manifest row counts must be non-negative")
    if train_rows + heldout_rows != rows:
        raise ValueError("train_rows and heldout_rows must sum to rows")
    if not math.isfinite(train_frac) or not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be finite and between 0 and 1")
    if model_artifact_version < 1:
        raise ValueError("model_artifact_version must be at least 1")
    return {
        "run": {
            "rows": rows,
            "train_rows": train_rows,
            "heldout_rows": heldout_rows,
            "seed": seed,
            "train_frac": train_frac,
        },
        "model": {
            "artifact_version": model_artifact_version,
        },
        "artifacts": artifacts,
        "artifact_metadata": artifact_metadata or {},
    }


def collect_artifact_metadata(output_dir: Path, artifacts: list[str]) -> dict[str, dict[str, Any]]:
    """Collect size and SHA-256 metadata for generated artifacts."""
    metadata = {}
    for artifact in artifacts:
        if not _is_safe_artifact_path(artifact):
            continue
        path = output_dir / artifact
        if not path.exists():
            continue
        metadata[artifact] = {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    return metadata


def missing_artifacts(output_dir: Path, artifacts: list[str]) -> list[str]:
    """Return expected artifact paths that are absent from a report directory."""
    return [
        artifact
        for artifact in artifacts
        if _is_safe_artifact_path(artifact) and not (output_dir / artifact).exists()
    ]


def artifact_coverage_matrix(artifacts: list[str]) -> pd.DataFrame:
    """Build a planned report artifact coverage matrix.

    The matrix is intentionally based on planned manifest paths rather than file
    presence so it can be generated before the final manifest and later
    recomputed by verification without circular dependencies.
    """
    rows = []
    summary_artifacts = set(_RESEARCH_SUMMARY_ARTIFACT_SECTIONS.values())
    for artifact in artifacts:
        rows.append(
            {
                "artifact": artifact,
                "family": _artifact_family(artifact),
                "verification_role": _artifact_verification_role(artifact),
                "extension": Path(artifact).suffix.lstrip(".") or "none",
                "in_research_summary": artifact in summary_artifacts,
                "is_figure": artifact.startswith("figures/") and artifact.endswith(".png"),
                "has_manifest_metadata": artifact != "artifact_manifest.json",
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "artifact",
            "family",
            "verification_role",
            "extension",
            "in_research_summary",
            "is_figure",
            "has_manifest_metadata",
        ],
    )


def artifact_coverage_summary(matrix: pd.DataFrame) -> dict[str, int]:
    """Summarize planned artifact coverage for quick release audits."""
    if matrix.empty:
        return {
            "artifacts": 0,
            "research_summary_artifacts": 0,
            "figure_artifacts": 0,
            "metadata_tracked_artifacts": 0,
            "manifest_audit_artifacts": 0,
            "transition_verification_artifacts": 0,
            "lcri_release_evidence_artifacts": 0,
            "owner_readiness_artifacts": 0,
            "visual_evidence_artifacts": 0,
            "supporting_evidence_artifacts": 0,
            "families": 0,
        }
    role = matrix["verification_role"]
    return {
        "artifacts": len(matrix),
        "research_summary_artifacts": int(matrix["in_research_summary"].astype(bool).sum()),
        "figure_artifacts": int(matrix["is_figure"].astype(bool).sum()),
        "metadata_tracked_artifacts": int(matrix["has_manifest_metadata"].astype(bool).sum()),
        "manifest_audit_artifacts": int((role == "manifest_audit").sum()),
        "transition_verification_artifacts": int((role == "transition_verification").sum()),
        "lcri_release_evidence_artifacts": int((role == "lcri_release_evidence").sum()),
        "owner_readiness_artifacts": int((role == "owner_readiness").sum()),
        "visual_evidence_artifacts": int((role == "visual_evidence").sum()),
        "supporting_evidence_artifacts": int((role == "supporting_evidence").sum()),
        "families": int(matrix["family"].nunique()),
    }


def verify_artifact_coverage_matrix(output_dir: Path, manifest: dict[str, Any]) -> list[str]:
    """Return errors when artifact coverage audit files are missing or stale."""
    matrix_path = output_dir / "artifact_coverage_matrix.csv"
    summary_path = output_dir / "artifact_coverage_summary.json"
    if not matrix_path.exists():
        return ["missing artifact coverage matrix: artifact_coverage_matrix.csv"]
    if not summary_path.exists():
        return ["missing artifact coverage summary: artifact_coverage_summary.json"]

    artifacts = list(manifest.get("artifacts", []))
    expected_matrix = artifact_coverage_matrix(artifacts)
    found_matrix = pd.read_csv(matrix_path)
    errors: list[str] = []
    _compare_artifact_coverage_records(errors, expected=expected_matrix, found=found_matrix)

    expected_summary = artifact_coverage_summary(expected_matrix)
    found_summary = json.loads(summary_path.read_text())
    missing_summary_keys = sorted(set(expected_summary) - set(found_summary))
    if missing_summary_keys:
        errors.append(f"incomplete artifact coverage summary: {missing_summary_keys}")
    for key, expected_value in expected_summary.items():
        if found_summary.get(key) != expected_value:
            errors.append(
                f"artifact coverage summary mismatch for {key}: "
                f"expected {expected_value!r}, found {found_summary.get(key)!r}"
            )
    return errors


def verify_artifact_manifest(output_dir: Path, manifest: dict[str, Any]) -> list[str]:
    """Return manifest verification errors for missing or changed artifacts."""
    raw_artifacts = manifest.get("artifacts", [])
    if not isinstance(raw_artifacts, list):
        return ["manifest artifacts must be a list"]
    artifacts = raw_artifacts
    artifact_set = set(artifacts)
    metadata = manifest.get("artifact_metadata", {})
    if not isinstance(metadata, dict):
        return ["manifest artifact_metadata must be an object"]
    metadata_artifacts = set(metadata)
    duplicate_artifacts = sorted({artifact for artifact in artifacts if artifacts.count(artifact) > 1})
    errors = [f"duplicate manifest artifact: {artifact}" for artifact in duplicate_artifacts]
    errors.extend(
        f"unsafe artifact path: {artifact}"
        for artifact in artifacts
        if not _is_safe_artifact_path(artifact)
    )
    errors.extend(f"missing artifact: {artifact}" for artifact in missing_artifacts(output_dir, artifacts))

    if metadata:
        for artifact in sorted(artifact_set - metadata_artifacts):
            errors.append(f"missing manifest metadata for artifact: {artifact}")
        for artifact in sorted(metadata_artifacts - artifact_set):
            errors.append(f"unexpected manifest metadata for artifact: {artifact}")

    current_metadata = collect_artifact_metadata(output_dir, artifacts)
    for artifact, expected in metadata.items():
        if not _is_safe_artifact_path(artifact):
            errors.append(f"unsafe manifest metadata path: {artifact}")
            continue
        missing_keys = sorted({"size_bytes", "sha256"} - set(expected))
        if missing_keys:
            errors.append(f"incomplete manifest metadata for artifact {artifact}: {missing_keys}")
            continue
        current = current_metadata.get(artifact)
        if current is None:
            continue
        if current.get("size_bytes") != expected.get("size_bytes"):
            errors.append(f"size mismatch: {artifact}")
        if current.get("sha256") != expected.get("sha256"):
            errors.append(f"sha256 mismatch: {artifact}")
    return errors


def verify_artifact_metadata_summary(output_dir: Path, manifest: dict[str, Any]) -> list[str]:
    """Return errors when the compact metadata summary is stale.

    Generated reports write ``artifact_metadata_summary.json`` before the final
    manifest, so the summary intentionally excludes itself. This check uses the
    manifest's artifact metadata as the source of truth and recomputes the compact
    audit payload reviewers see in release logs.
    """
    summary_name = "artifact_metadata_summary.json"
    artifacts = list(manifest.get("artifacts", []))
    if summary_name not in artifacts:
        return []

    summary_path = output_dir / summary_name
    if not summary_path.exists():
        return [f"missing artifact metadata summary: {summary_name}"]

    metadata = manifest.get("artifact_metadata", {})
    metadata_artifacts = {
        artifact: metadata[artifact]
        for artifact in artifacts
        if artifact != summary_name and artifact in metadata
    }
    missing_metadata = sorted(artifact for artifact in artifacts if artifact not in metadata)
    errors = [f"missing manifest metadata for artifact: {artifact}" for artifact in missing_metadata]

    try:
        summary = json.loads(summary_path.read_text())
    except json.JSONDecodeError as exc:
        return [f"invalid artifact metadata summary JSON: {exc.msg}"]

    expected_summary = summarize_artifact_metadata(metadata_artifacts)
    missing_summary_keys = sorted(set(expected_summary) - set(summary))
    if missing_summary_keys:
        errors.append(f"incomplete artifact metadata summary: {missing_summary_keys}")

    for key, expected_value in expected_summary.items():
        if summary.get(key) != expected_value:
            errors.append(
                f"artifact metadata summary mismatch for {key}: "
                f"expected {expected_value!r}, found {summary.get(key)!r}"
            )
    return errors


def verify_research_summary_sections(output_dir: Path) -> list[str]:
    """Return errors when the markdown summary is stale against artifacts.

    The generated markdown is the owner-facing research digest. Structural CSV
    and JSON checks can pass while the digest still carries stale ``_Not
    generated._`` placeholders or omits newly generated payload keys. This
    verifier cross-checks the summary sections against the artifacts they render.
    """
    summary_path = output_dir / "research_summary.md"
    if not summary_path.exists():
        return ["missing research summary: research_summary.md"]

    text = summary_path.read_text()
    errors = []
    for section, artifact in _RESEARCH_SUMMARY_ARTIFACT_SECTIONS.items():
        artifact_path = output_dir / artifact
        if not artifact_path.exists():
            continue

        body = _markdown_section_body(text, section)
        if body is None:
            errors.append(f"missing research summary section: {section}")
            continue

        if "_Not generated._" in body:
            errors.append(f"stale research summary section for generated artifact: {section}")
            continue

        if artifact.endswith(".csv"):
            errors.extend(_verify_summary_csv_section(section, artifact_path, body))
        elif artifact.endswith(".json"):
            errors.extend(_verify_summary_json_section(section, artifact_path, body))
    return errors


def verify_pressure_memory_decay_summary(
    output_dir: Path, artifact: str = "pressure_memory_decay_summary.csv"
) -> list[str]:
    """Return errors for incomplete pressure-memory decay diagnostics."""
    path = output_dir / artifact
    if not path.exists():
        return [f"missing pressure memory decay summary: {artifact}"]
    frame = pd.read_csv(path)
    required = {
        "pressure_memory_decay_state",
        "observations",
        "share",
        "decay_events",
        "event_rate",
        "mean_half_life",
        "mean_release_velocity",
    }
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        return [f"incomplete pressure memory decay summary {artifact}: {missing}"]
    numeric = frame[list(required - {"pressure_memory_decay_state"})].astype(float)
    errors: list[str] = []
    unknown = sorted(set(frame["pressure_memory_decay_state"].astype(str)) - {"inactive", "fast_decay", "slow_decay", "persistent"})
    if unknown:
        errors.append(f"unknown pressure memory decay states in {artifact}: {unknown}")
    if not np.isfinite(numeric.to_numpy()).all():
        errors.append(f"non-finite pressure memory decay values in {artifact}")
    if not numeric[["share", "event_rate"]].apply(lambda col: col.between(0.0, 1.0).all()).all():
        errors.append(f"bounded pressure memory rates violated in {artifact}")
    if not numeric[["observations", "decay_events", "mean_half_life", "mean_release_velocity"]].ge(0.0).all().all():
        errors.append(f"negative pressure memory decay values in {artifact}")
    if (numeric["decay_events"] > numeric["observations"]).any():
        errors.append(f"pressure memory decay events exceed observations in {artifact}")
    if numeric["observations"].sum() > 0 and not math.isclose(numeric["share"].sum(), 1.0, abs_tol=1e-6):
        errors.append(f"pressure memory decay shares do not sum to one in {artifact}")
    return errors


def verify_hidden_resiliency_asymmetry_summary(
    output_dir: Path, artifact: str = "hidden_resiliency_asymmetry_summary.json"
) -> list[str]:
    """Return errors for incomplete hidden-resiliency asymmetry diagnostics."""
    path = output_dir / artifact
    if not path.exists():
        return [f"missing hidden resiliency asymmetry summary: {artifact}"]
    payload = json.loads(path.read_text())
    required = {
        "fast_decay_mean_fracture",
        "slow_or_persistent_mean_fracture",
        "fast_minus_slow_fracture",
        "fast_minus_slow_velocity",
        "hidden_resiliency_asymmetry_score",
        "interpretation",
    }
    missing = sorted(required - set(payload))
    if missing:
        return [f"incomplete hidden resiliency asymmetry summary {artifact}: {missing}"]
    numeric = [payload[key] for key in required - {"interpretation"}]
    errors: list[str] = []
    if not np.isfinite(numeric).all():
        errors.append(f"non-finite hidden resiliency asymmetry values in {artifact}")
    if payload["interpretation"] not in {
        "fast_release_masks_fracture",
        "slow_memory_carries_fracture",
        "balanced_resiliency",
    }:
        errors.append(f"unknown hidden resiliency interpretation in {artifact}")
    return errors


def verify_adverse_selection_phase_shift_summary(
    output_dir: Path, artifact: str = "adverse_selection_phase_shift_summary.csv"
) -> list[str]:
    """Return errors for incomplete adverse-selection phase-shift diagnostics."""
    path = output_dir / artifact
    if not path.exists():
        return [f"missing adverse selection phase-shift summary: {artifact}"]
    frame = pd.read_csv(path)
    required = {
        "pressure_memory_decay_state",
        "observations",
        "active_pressure_observations",
        "adverse_selection_phase_shift_rate",
        "mean_release_velocity",
        "mean_latent_liquidity_fracture",
        "adverse_selection_phase_shift_score",
        "phase_shift_interpretation",
    }
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        return [f"incomplete adverse selection phase-shift summary {artifact}: {missing}"]

    numeric_columns = list(required - {"pressure_memory_decay_state", "phase_shift_interpretation"})
    numeric = frame[numeric_columns].astype(float)
    errors: list[str] = []
    if not np.isfinite(numeric.to_numpy()).all():
        errors.append(f"non-finite adverse selection phase-shift values in {artifact}")
    if not numeric[["adverse_selection_phase_shift_rate"]].apply(lambda col: col.between(0.0, 1.0).all()).all():
        errors.append(f"bounded adverse selection phase-shift rates violated in {artifact}")
    if not numeric.ge(0.0).all().all():
        errors.append(f"negative adverse selection phase-shift values in {artifact}")
    unknown = sorted(
        set(frame["phase_shift_interpretation"].astype(str))
        - {"fractured_adverse_selection", "localized_phase_shift", "aligned_pressure_memory"}
    )
    if unknown:
        errors.append(f"unknown adverse selection phase-shift interpretations in {artifact}: {unknown}")
    return errors


def verify_phase_shift_artifact_review(
    output_dir: Path, artifact: str = "phase_shift_artifact_review.csv"
) -> list[str]:
    """Return errors for stale phase-shift artifact triage tables."""
    path = output_dir / artifact
    if not path.exists():
        return [f"missing phase-shift artifact review: {artifact}"]
    frame = pd.read_csv(path)
    required = {
        "pressure_memory_decay_state",
        "adverse_selection_phase_shift_rate",
        "mean_latent_liquidity_fracture",
        "adverse_selection_phase_shift_score",
        "phase_shift_artifact",
        "phase_shift_review_priority",
    }
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        return [f"incomplete phase-shift artifact review {artifact}: {missing}"]

    numeric = frame[list(required - {"pressure_memory_decay_state", "phase_shift_artifact"})].astype(float)
    errors: list[str] = []
    if not np.isfinite(numeric.to_numpy()).all():
        errors.append(f"non-finite phase-shift artifact review values in {artifact}")
    if not numeric["adverse_selection_phase_shift_rate"].between(0.0, 1.0).all():
        errors.append(f"bounded phase-shift artifact rates violated in {artifact}")
    if not numeric.ge(0.0).all().all():
        errors.append(f"negative phase-shift artifact review values in {artifact}")
    known = {
        "fractured_adverse_selection",
        "return_sign_flip",
        "localized_fracture_shift",
        "thin_phase_shift",
        "aligned_pressure_memory",
    }
    unknown = sorted(set(frame["phase_shift_artifact"].astype(str)) - known)
    if unknown:
        errors.append(f"unknown phase-shift artifact labels in {artifact}: {unknown}")
    return errors


_ALPHA_EVENT_REVIEW_ARTIFACTS = {
    "events": "alpha_event_windows.csv",
    "summary": "alpha_event_window_summary.json",
    "packet": "alpha_event_release_review_packet.csv",
    "gate": "alpha_event_drift_gate.json",
    "weighted": "alpha_event_score_weighted_drift.json",
    "regimes": "alpha_event_regime_summary.csv",
}
_ALPHA_EVENT_BLOCKER_COLUMNS = ["severity", "artifact", "check", "message", "owner_action"]
_ALPHA_EVENT_ERROR_ARTIFACT_HINTS = {
    "window summary": "alpha_event_window_summary.json",
    "score-weighted drift": "alpha_event_score_weighted_drift.json",
    "drift gate": "alpha_event_drift_gate.json",
    "release review packet": "alpha_event_release_review_packet.csv",
    "regime": "alpha_event_regime_summary.csv",
}


def verify_trade_confirmed_passive_fill_latency_summary(
    output_dir: Path, artifact: str = "trade_confirmed_passive_fill_latency_summary.csv"
) -> list[str]:
    """Return errors for trade-confirmed passive-fill latency review artifacts."""
    path = output_dir / artifact
    if not path.exists():
        return [f"missing trade-confirmed passive fill latency summary: {artifact}"]
    frame = pd.read_csv(path)
    required = {
        "side",
        "rows",
        "trade_confirmed_fill_rate",
        "cancel_only_clear_rate",
        "mean_fill_latency",
        "p95_fill_latency",
        "mean_trade_depletion",
        "mean_cancel_depletion",
        "review_label",
    }
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        return [f"incomplete trade-confirmed passive fill latency summary {artifact}: {missing}"]

    numeric_columns = list(required - {"side", "review_label"})
    numeric = frame[numeric_columns].astype(float)
    errors: list[str] = []
    known_sides = {"bid", "ask", "all"}
    unknown_sides = sorted(set(frame["side"].astype(str)) - known_sides)
    if unknown_sides:
        errors.append(f"unknown trade-confirmed passive fill sides in {artifact}: {unknown_sides}")
    if not np.isfinite(numeric.to_numpy()).all():
        errors.append(f"non-finite trade-confirmed passive fill latency values in {artifact}")
    if not numeric["rows"].ge(0.0).all():
        errors.append(f"negative trade-confirmed passive fill latency row counts in {artifact}")
    rate_columns = ["trade_confirmed_fill_rate", "cancel_only_clear_rate"]
    if not numeric[rate_columns].apply(lambda col: col.between(0.0, 1.0).all()).all():
        errors.append(f"bounded trade-confirmed passive fill rates violated in {artifact}")
    nonnegative_columns = [
        "mean_fill_latency",
        "p95_fill_latency",
        "mean_trade_depletion",
        "mean_cancel_depletion",
    ]
    if not numeric[nonnegative_columns].ge(0.0).all().all():
        errors.append(f"negative trade-confirmed passive fill latency/depletion values in {artifact}")
    if (numeric["p95_fill_latency"] + 1e-12 < numeric["mean_fill_latency"]).any():
        errors.append(f"trade-confirmed passive fill p95 latency below mean in {artifact}")
    labels = {
        "trade_confirmed_execution_ok",
        "latency_risk",
        "cancel_only_clear_risk",
        "cancel_only_and_latency_risk",
    }
    unknown_labels = sorted(set(frame["review_label"].astype(str)) - labels)
    if unknown_labels:
        errors.append(f"invalid trade-confirmed passive fill review labels in {artifact}: {unknown_labels}")
    return errors


def verify_queue_position_trade_confirmation_release_scorecard(
    output_dir: Path,
    artifact: str = "queue_position_trade_confirmation_release_scorecard.json",
) -> list[str]:
    """Return errors for queue-position trade-confirmation release scorecards."""
    path = output_dir / artifact
    if not path.exists():
        return [f"missing queue-position trade confirmation release scorecard: {artifact}"]
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [f"invalid queue-position trade confirmation release scorecard JSON {artifact}: {exc}"]

    required = {
        "evaluated_cells",
        "supported_cells",
        "total_rows",
        "supported_rows",
        "unsupported_rows",
        "weighted_confirmation_shortfall",
        "weighted_cancel_only_clear_rate",
        "weighted_stale_trade_confirmed_fill_share",
        "max_confirmation_shortfall",
        "max_cancel_only_clear_rate",
        "max_stale_trade_confirmed_fill_share",
        "worst_confirmation_cell",
        "worst_confirmation_cell_rows",
        "worst_confirmation_cell_label",
        "trade_confirmation_release_label",
        "publishable",
        "blocking_reasons",
        "review_reasons",
    }
    missing = sorted(required - set(payload))
    if missing:
        return [f"incomplete queue-position trade confirmation release scorecard {artifact}: {missing}"]

    errors: list[str] = []
    numeric_keys = [
        "evaluated_cells",
        "supported_cells",
        "total_rows",
        "supported_rows",
        "unsupported_rows",
        "weighted_confirmation_shortfall",
        "weighted_cancel_only_clear_rate",
        "weighted_stale_trade_confirmed_fill_share",
        "max_confirmation_shortfall",
        "max_cancel_only_clear_rate",
        "max_stale_trade_confirmed_fill_share",
        "worst_confirmation_cell_rows",
    ]
    numeric = {key: float(payload[key]) for key in numeric_keys}
    if not all(math.isfinite(value) for value in numeric.values()):
        errors.append(f"non-finite queue-position trade confirmation scorecard values in {artifact}")
    count_columns = [
        "evaluated_cells",
        "supported_cells",
        "total_rows",
        "supported_rows",
        "unsupported_rows",
        "worst_confirmation_cell_rows",
    ]
    if any(numeric[key] < 0.0 for key in count_columns):
        errors.append(f"negative queue-position trade confirmation scorecard counts in {artifact}")
    if any(not math.isclose(numeric[key], round(numeric[key]), abs_tol=1e-9) for key in count_columns):
        errors.append(f"non-integer queue-position trade confirmation scorecard counts in {artifact}")
    if numeric["supported_cells"] > numeric["evaluated_cells"]:
        errors.append(f"supported queue-position trade confirmation cells exceed evaluated cells in {artifact}")
    if not math.isclose(
        numeric["supported_rows"] + numeric["unsupported_rows"],
        numeric["total_rows"],
        abs_tol=1e-9,
    ):
        errors.append(f"inconsistent queue-position trade confirmation scorecard row totals in {artifact}")
    rate_columns = [
        "weighted_cancel_only_clear_rate",
        "weighted_stale_trade_confirmed_fill_share",
        "max_cancel_only_clear_rate",
        "max_stale_trade_confirmed_fill_share",
    ]
    if any(not 0.0 <= numeric[key] <= 1.0 for key in rate_columns):
        errors.append(f"bounded queue-position trade confirmation scorecard rates violated in {artifact}")
    signed_shortfall_columns = ["weighted_confirmation_shortfall", "max_confirmation_shortfall"]
    if any(not -1.0 <= numeric[key] <= 1.0 for key in signed_shortfall_columns):
        errors.append(f"bounded queue-position trade confirmation scorecard shortfall violated in {artifact}")
    decision = str(payload["trade_confirmation_release_label"])
    if decision not in {"pass", "review", "block"}:
        errors.append(f"invalid queue-position trade confirmation release decision in {artifact}: {decision}")
    if not isinstance(payload["publishable"], bool):
        errors.append(f"non-boolean queue-position trade confirmation publishable flag in {artifact}")
    elif payload["publishable"] != (decision == "pass"):
        errors.append(f"queue-position trade confirmation publishable flag contradicts decision in {artifact}")
    for key in [
        "worst_confirmation_cell",
        "worst_confirmation_cell_label",
        "blocking_reasons",
        "review_reasons",
    ]:
        if not str(payload[key]):
            errors.append(f"blank queue-position trade confirmation scorecard text field {key} in {artifact}")
    return errors


def verify_queue_position_unfilled_opportunity_scorecard(
    output_dir: Path,
    artifact: str = "queue_position_unfilled_opportunity_scorecard.json",
) -> list[str]:
    """Return errors for queue-position unfilled-opportunity release scorecards."""
    path = output_dir / artifact
    if not path.exists():
        return [f"missing queue-position unfilled opportunity scorecard: {artifact}"]
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [f"invalid queue-position unfilled opportunity scorecard JSON {artifact}: {exc}"]

    required = {
        "evaluated_cells",
        "tail_cells",
        "tail_rows",
        "max_tail_unfilled_opportunity_share",
        "min_tail_edge_capture_rate",
        "weighted_tail_unfilled_opportunity_share",
        "weighted_tail_edge_capture_rate",
        "worst_tail_cell",
        "unfilled_opportunity_release_label",
        "publishable",
        "blocking_reasons",
        "review_reasons",
    }
    missing = sorted(required - set(payload))
    if missing:
        return [f"incomplete queue-position unfilled opportunity scorecard {artifact}: {missing}"]

    errors: list[str] = []
    numeric_keys = [
        "evaluated_cells",
        "tail_cells",
        "tail_rows",
        "max_tail_unfilled_opportunity_share",
        "min_tail_edge_capture_rate",
        "weighted_tail_unfilled_opportunity_share",
        "weighted_tail_edge_capture_rate",
    ]
    numeric = {key: float(payload[key]) for key in numeric_keys}
    if not all(math.isfinite(value) for value in numeric.values()):
        errors.append(f"non-finite queue-position unfilled opportunity scorecard values in {artifact}")
    count_columns = ["evaluated_cells", "tail_cells", "tail_rows"]
    if any(numeric[key] < 0.0 for key in count_columns):
        errors.append(f"negative queue-position unfilled opportunity scorecard counts in {artifact}")
    if any(not math.isclose(numeric[key], round(numeric[key]), abs_tol=1e-9) for key in count_columns):
        errors.append(f"non-integer queue-position unfilled opportunity scorecard counts in {artifact}")
    if numeric["tail_cells"] > numeric["evaluated_cells"]:
        errors.append(f"tail queue-position unfilled opportunity cells exceed evaluated cells in {artifact}")
    rate_columns = [
        "max_tail_unfilled_opportunity_share",
        "min_tail_edge_capture_rate",
        "weighted_tail_unfilled_opportunity_share",
        "weighted_tail_edge_capture_rate",
    ]
    if any(not 0.0 <= numeric[key] <= 1.0 for key in rate_columns):
        errors.append(f"bounded queue-position unfilled opportunity scorecard rates violated in {artifact}")
    decision = str(payload["unfilled_opportunity_release_label"])
    if decision not in {"pass", "review", "block"}:
        errors.append(f"invalid queue-position unfilled opportunity release decision in {artifact}: {decision}")
    if not isinstance(payload["publishable"], bool):
        errors.append(f"non-boolean queue-position unfilled opportunity publishable flag in {artifact}")
    elif payload["publishable"] != (decision == "pass"):
        errors.append(f"queue-position unfilled opportunity publishable flag contradicts decision in {artifact}")
    for key in ["worst_tail_cell", "blocking_reasons", "review_reasons"]:
        if not str(payload[key]):
            errors.append(f"blank queue-position unfilled opportunity scorecard text field {key} in {artifact}")
    return errors


def verify_queue_position_trade_confirmation_regime_scorecard(
    output_dir: Path,
    artifact: str = "queue_position_trade_confirmation_regime_scorecard.csv",
) -> list[str]:
    """Return errors for per-regime queue-position trade-confirmation scorecards."""
    path = output_dir / artifact
    if not path.exists():
        return [f"missing queue-position trade confirmation regime scorecard: {artifact}"]
    frame = pd.read_csv(path)
    required = {
        "regime",
        "cells",
        "supported_cells",
        "rows",
        "supported_rows",
        "unsupported_rows",
        "weighted_predicted_fill_probability",
        "weighted_trade_confirmed_fill_rate",
        "weighted_confirmation_shortfall",
        "weighted_cancel_only_clear_rate",
        "weighted_stale_trade_confirmed_fill_share",
        "max_confirmation_shortfall",
        "max_cancel_only_clear_rate",
        "max_stale_trade_confirmed_fill_share",
        "worst_confirmation_cell",
        "worst_confirmation_cell_rows",
        "worst_confirmation_cell_label",
        "trade_confirmation_regime_label",
        "publishable",
        "blocking_reasons",
        "review_reasons",
        "regime_priority_rank",
    }
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        return [f"incomplete queue-position trade confirmation regime scorecard {artifact}: {missing}"]

    numeric_columns = list(required - {"regime", "worst_confirmation_cell", "worst_confirmation_cell_label", "trade_confirmation_regime_label", "publishable", "blocking_reasons", "review_reasons"})
    numeric = frame[numeric_columns].astype(float)
    errors: list[str] = []
    if not np.isfinite(numeric.to_numpy()).all():
        errors.append(f"non-finite queue-position trade confirmation regime scorecard values in {artifact}")
    count_columns = [
        "cells",
        "supported_cells",
        "rows",
        "supported_rows",
        "unsupported_rows",
        "worst_confirmation_cell_rows",
        "regime_priority_rank",
    ]
    if not numeric[count_columns].ge(0.0).all().all():
        errors.append(f"negative queue-position trade confirmation regime counts in {artifact}")
    if not numeric[count_columns].apply(lambda col: np.isclose(col, np.round(col), atol=1e-9).all()).all():
        errors.append(f"non-integer queue-position trade confirmation regime counts in {artifact}")
    if (numeric["supported_cells"] > numeric["cells"]).any():
        errors.append(f"supported queue-position trade confirmation regime cells exceed cells in {artifact}")
    if not np.isclose(
        numeric["supported_rows"] + numeric["unsupported_rows"],
        numeric["rows"],
        atol=1e-9,
    ).all():
        errors.append(f"inconsistent queue-position trade confirmation regime row totals in {artifact}")
    rate_columns = [
        "weighted_predicted_fill_probability",
        "weighted_trade_confirmed_fill_rate",
        "weighted_cancel_only_clear_rate",
        "weighted_stale_trade_confirmed_fill_share",
        "max_cancel_only_clear_rate",
        "max_stale_trade_confirmed_fill_share",
    ]
    if not numeric[rate_columns].apply(lambda col: col.between(0.0, 1.0).all()).all():
        errors.append(f"bounded queue-position trade confirmation regime rates violated in {artifact}")
    shortfall_columns = ["weighted_confirmation_shortfall", "max_confirmation_shortfall"]
    if not numeric[shortfall_columns].apply(lambda col: col.between(-1.0, 1.0).all()).all():
        errors.append(f"bounded queue-position trade confirmation regime shortfall violated in {artifact}")
    labels = frame["trade_confirmation_regime_label"].astype(str)
    unknown_labels = sorted(set(labels) - {"pass", "review", "block"})
    if unknown_labels:
        errors.append(f"invalid queue-position trade confirmation regime labels in {artifact}: {unknown_labels}")
    publishable = frame["publishable"]
    if not publishable.map(lambda value: isinstance(value, bool)).all():
        errors.append(f"non-boolean queue-position trade confirmation regime publishable flag in {artifact}")
    elif not (publishable == labels.eq("pass")).all():
        errors.append(f"queue-position trade confirmation regime publishable flag contradicts decision in {artifact}")
    text_columns = [
        "regime",
        "worst_confirmation_cell",
        "worst_confirmation_cell_label",
        "blocking_reasons",
        "review_reasons",
    ]
    for column in text_columns:
        if frame[column].astype(str).str.len().eq(0).any():
            errors.append(f"blank queue-position trade confirmation regime text field {column} in {artifact}")
    return errors



def verify_execution_publishability_review_artifacts(
    output_dir: Path, artifact: str = "execution_publishability_review_packet.csv"
) -> list[str]:
    """Return errors for incomplete execution-aware publishability review packets."""
    path = output_dir / artifact
    if not path.exists():
        return [f"missing execution publishability review packet: {artifact}"]
    frame = pd.read_csv(path)
    required = {
        "publishable_side",
        "best_execution_side",
        "rows",
        "conflict_rows",
        "conflict_share",
        "mean_execution_adjusted_edge_ticks",
        "mean_best_fill_probability",
        "mean_best_adverse_fill_probability",
        "mean_publishable_fill_probability",
        "mean_edge_drag_ticks",
        "review_priority",
        "review_note",
    }
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        return [f"incomplete execution publishability review packet {artifact}: {missing}"]

    numeric_columns = list(required - {"publishable_side", "best_execution_side", "review_note"})
    numeric = frame[numeric_columns].astype(float)
    errors: list[str] = []
    valid_sides = {"long", "short", "abstain"}
    unknown_publishable = sorted(set(frame["publishable_side"].astype(str)) - valid_sides)
    unknown_best = sorted(set(frame["best_execution_side"].astype(str)) - valid_sides)
    if unknown_publishable:
        errors.append(f"unknown publishable sides in {artifact}: {unknown_publishable}")
    if unknown_best:
        errors.append(f"unknown best execution sides in {artifact}: {unknown_best}")
    if not np.isfinite(numeric.to_numpy()).all():
        errors.append(f"non-finite execution publishability values in {artifact}")
    count_columns = ["rows", "conflict_rows", "review_priority"]
    if not numeric[count_columns].ge(0.0).all().all():
        errors.append(f"negative execution publishability counts in {artifact}")
    probability_columns = [
        "conflict_share",
        "mean_best_fill_probability",
        "mean_best_adverse_fill_probability",
        "mean_publishable_fill_probability",
    ]
    if not numeric[probability_columns].apply(lambda col: col.between(0.0, 1.0).all()).all():
        errors.append(f"bounded execution publishability probabilities violated in {artifact}")
    if (numeric["conflict_rows"] > numeric["rows"]).any():
        errors.append(f"execution publishability conflict rows exceed rows in {artifact}")
    expected_conflict = frame["publishable_side"].astype(str) != frame["best_execution_side"].astype(str)
    if not (numeric["conflict_rows"].eq(0.0) == ~expected_conflict).all():
        errors.append(f"execution publishability conflict flags inconsistent with sides in {artifact}")
    if frame["review_note"].astype(str).str.len().eq(0).any():
        errors.append(f"blank execution publishability review notes in {artifact}")
    return errors


def verify_queue_position_fill_monotonicity_scorecard(
    output_dir: Path,
    artifact: str = "queue_position_fill_monotonicity_scorecard.csv",
    *,
    regime_col: str = "pressure_memory_decay_state",
) -> list[str]:
    """Return errors for queue-depth passive-fill monotonicity scorecards."""
    path = output_dir / artifact
    if not path.exists():
        return [f"missing queue-position fill monotonicity scorecard: {artifact}"]
    frame = pd.read_csv(path)
    required = {
        regime_col,
        "best_execution_side",
        "queue_bins",
        "queue_steps",
        "rows",
        "predicted_fill_inversions",
        "realized_fill_inversions",
        "max_predicted_fill_inversion",
        "max_realized_fill_inversion",
        "monotonicity_label",
    }
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        return [f"incomplete queue-position fill monotonicity scorecard {artifact}: {missing}"]

    numeric_columns = list(required - {regime_col, "best_execution_side", "monotonicity_label"})
    numeric = frame[numeric_columns].astype(float)
    errors: list[str] = []
    if frame[regime_col].astype(str).str.len().eq(0).any():
        errors.append(f"blank queue-position fill monotonicity regimes in {artifact}")
    valid_sides = {"long", "short", "abstain"}
    unknown_sides = sorted(set(frame["best_execution_side"].astype(str)) - valid_sides)
    if unknown_sides:
        errors.append(f"unknown queue-position fill monotonicity sides in {artifact}: {unknown_sides}")
    valid_labels = {
        "queue_fill_monotonicity_pass",
        "queue_fill_monotonicity_review",
        "queue_fill_monotonicity_block",
    }
    unknown_labels = sorted(set(frame["monotonicity_label"].astype(str)) - valid_labels)
    if unknown_labels:
        errors.append(
            f"unknown queue-position fill monotonicity labels in {artifact}: {unknown_labels}"
        )
    if not np.isfinite(numeric.to_numpy()).all():
        errors.append(f"non-finite queue-position fill monotonicity values in {artifact}")
    count_columns = [
        "queue_bins",
        "queue_steps",
        "rows",
        "predicted_fill_inversions",
        "realized_fill_inversions",
    ]
    if not numeric[count_columns].ge(0.0).all().all():
        errors.append(f"negative queue-position fill monotonicity counts in {artifact}")
    expected_steps = (numeric["queue_bins"] - 1.0).clip(lower=0.0)
    if not numeric["queue_steps"].eq(expected_steps).all():
        errors.append(f"queue-position fill monotonicity step counts contradict bins in {artifact}")
    if (numeric["predicted_fill_inversions"] > numeric["queue_steps"]).any() or (
        numeric["realized_fill_inversions"] > numeric["queue_steps"]
    ).any():
        errors.append(f"queue-position fill monotonicity inversions exceed queue steps in {artifact}")
    if (numeric["max_predicted_fill_inversion"] > 1.0).any() or (
        numeric["max_realized_fill_inversion"] > 1.0
    ).any():
        errors.append(
            f"queue-position fill monotonicity inversion magnitudes exceed probability bounds in {artifact}"
        )
    return errors


def verify_execution_adjusted_edge_component_attribution(
    output_dir: Path, artifact: str = "execution_adjusted_edge_component_attribution.csv"
) -> list[str]:
    """Return errors for raw/fill/toxicity execution edge attribution artifacts."""
    path = output_dir / artifact
    if not path.exists():
        return [f"missing execution edge component attribution: {artifact}"]
    frame = pd.read_csv(path)
    required = {
        "best_execution_side",
        "rows",
        "mean_raw_edge_ticks",
        "mean_fill_captured_edge_ticks",
        "mean_adverse_selection_cost_ticks",
        "mean_execution_adjusted_edge_ticks",
        "mean_fill_shortfall_ticks",
        "fill_capture_ratio",
        "adverse_drag_ratio",
    }
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        return [f"incomplete execution edge component attribution {artifact}: {missing}"]

    numeric_columns = list(required - {"best_execution_side"})
    numeric = frame[numeric_columns].astype(float)
    errors: list[str] = []
    valid_sides = {"long", "short", "abstain"}
    unknown_sides = sorted(set(frame["best_execution_side"].astype(str)) - valid_sides)
    if unknown_sides:
        errors.append(f"unknown execution sides in {artifact}: {unknown_sides}")
    if not np.isfinite(numeric.to_numpy()).all():
        errors.append(f"non-finite execution edge component attribution values in {artifact}")
    if not numeric["rows"].ge(0.0).all():
        errors.append(f"negative execution edge component counts in {artifact}")
    nonnegative_columns = [
        "mean_adverse_selection_cost_ticks",
        "adverse_drag_ratio",
    ]
    if not numeric[nonnegative_columns].ge(0.0).all().all():
        errors.append(f"negative execution edge component drag values in {artifact}")
    ratio_columns = ["fill_capture_ratio", "adverse_drag_ratio"]
    if not numeric[ratio_columns].apply(lambda col: col.between(0.0, 1.0).all()).all():
        errors.append(f"bounded execution edge component ratios violated in {artifact}")
    return errors


def verify_execution_adjusted_lcri_side_attribution(
    output_dir: Path, artifact: str = "execution_adjusted_lcri_side_attribution.csv"
) -> list[str]:
    """Return errors for malformed execution-adjusted LCRI side attribution artifacts."""
    path = output_dir / artifact
    if not path.exists():
        return [f"missing execution-adjusted LCRI side attribution: {artifact}"]
    frame = pd.read_csv(path)
    required = {
        "lcri_side",
        "rows",
        "tradable_rows",
        "execution_conflict_rows",
        "execution_conflict_share",
        "mean_signal_confidence",
        "mean_execution_adjusted_edge_ticks",
        "mean_fill_probability_advantage",
        "mean_adverse_fill_probability_advantage",
        "dominant_execution_side",
        "review_label",
    }
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        return [f"incomplete execution-adjusted LCRI side attribution {artifact}: {missing}"]

    numeric_columns = list(required - {"lcri_side", "dominant_execution_side", "review_label"})
    numeric = frame[numeric_columns].astype(float)
    errors: list[str] = []
    valid_lcri_sides = {"long", "short", "neutral"}
    valid_execution_sides = {"long", "short", "none"}
    valid_labels = {
        "execution_side_preserved",
        "execution_friction_review",
        "execution_side_inversion_review",
        "neutral_signal",
    }
    unknown_lcri_sides = sorted(set(frame["lcri_side"].astype(str)) - valid_lcri_sides)
    unknown_execution_sides = sorted(
        set(frame["dominant_execution_side"].astype(str)) - valid_execution_sides
    )
    unknown_labels = sorted(set(frame["review_label"].astype(str)) - valid_labels)
    if unknown_lcri_sides:
        errors.append(f"unknown LCRI sides in {artifact}: {unknown_lcri_sides}")
    if unknown_execution_sides:
        errors.append(f"unknown dominant execution sides in {artifact}: {unknown_execution_sides}")
    if unknown_labels:
        errors.append(f"unknown execution-adjusted LCRI side attribution labels in {artifact}: {unknown_labels}")
    if not np.isfinite(numeric.to_numpy()).all():
        errors.append(f"non-finite execution-adjusted LCRI side attribution values in {artifact}")
    count_columns = ["rows", "tradable_rows", "execution_conflict_rows"]
    if not numeric[count_columns].ge(0.0).all().all():
        errors.append(f"negative execution-adjusted LCRI side attribution counts in {artifact}")
    probability_columns = [
        "execution_conflict_share",
        "mean_signal_confidence",
        "mean_adverse_fill_probability_advantage",
    ]
    if not numeric[probability_columns].apply(lambda col: col.between(0.0, 1.0).all()).all():
        errors.append(f"bounded execution-adjusted LCRI side attribution probabilities violated in {artifact}")
    if (numeric["tradable_rows"] > numeric["rows"]).any():
        errors.append(f"execution-adjusted LCRI tradable rows exceed rows in {artifact}")
    if (numeric["execution_conflict_rows"] > numeric["rows"]).any():
        errors.append(f"execution-adjusted LCRI conflict rows exceed rows in {artifact}")
    neutral = frame["lcri_side"].astype(str) == "neutral"
    if neutral.any():
        neutral_numeric = numeric.loc[neutral]
        if not neutral_numeric["execution_conflict_rows"].eq(0.0).all():
            errors.append(f"neutral execution-adjusted LCRI rows report conflicts in {artifact}")
        if not (frame.loc[neutral, "review_label"].astype(str) == "neutral_signal").all():
            errors.append(f"neutral execution-adjusted LCRI rows have non-neutral labels in {artifact}")
    return errors


def verify_execution_adjusted_lcri_quantile_diagnostics(
    output_dir: Path, artifact: str = "execution_adjusted_lcri_quantile_diagnostics.csv"
) -> list[str]:
    """Return errors for execution-adjusted LCRI quantile survival artifacts."""
    path = output_dir / artifact
    if not path.exists():
        return [f"missing execution-adjusted LCRI quantile diagnostics: {artifact}"]
    frame = pd.read_csv(path)
    required = {
        "bucket",
        "rows",
        "mean_abs_lcri",
        "mean_abs_execution_adjusted_lcri_score",
        "signal_survival_ratio",
        "tradable_share",
        "mean_selected_fill_probability",
        "mean_selected_adverse_fill_probability",
        "fill_minus_adverse_probability_spread",
        "mean_execution_adjusted_edge_ticks",
        "edge_drag_vs_raw_abs_lcri",
    }
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        return [f"incomplete execution-adjusted LCRI quantile diagnostics {artifact}: {missing}"]

    numeric_columns = list(required - {"bucket"})
    numeric = frame[numeric_columns].astype(float)
    errors: list[str] = []
    if not np.isfinite(numeric.to_numpy()).all():
        errors.append(f"non-finite execution-adjusted LCRI quantile values in {artifact}")
    if not numeric["rows"].ge(1.0).all():
        errors.append(f"non-positive execution-adjusted LCRI quantile rows in {artifact}")
    nonnegative_columns = [
        "mean_abs_lcri",
        "mean_abs_execution_adjusted_lcri_score",
        "signal_survival_ratio",
    ]
    if not numeric[nonnegative_columns].ge(0.0).all().all():
        errors.append(f"negative execution-adjusted LCRI quantile magnitudes in {artifact}")
    probability_columns = [
        "tradable_share",
        "signal_survival_ratio",
        "mean_selected_fill_probability",
        "mean_selected_adverse_fill_probability",
    ]
    if not numeric[probability_columns].apply(lambda col: col.between(0.0, 1.0).all()).all():
        errors.append(f"bounded execution-adjusted LCRI quantile probabilities violated in {artifact}")
    if frame["bucket"].astype(str).str.len().eq(0).any():
        errors.append(f"blank execution-adjusted LCRI quantile buckets in {artifact}")
    if frame["bucket"].astype(str).duplicated().any():
        errors.append(f"duplicate execution-adjusted LCRI quantile buckets in {artifact}")
    if (numeric["mean_abs_execution_adjusted_lcri_score"] - numeric["mean_abs_lcri"] > 1e-9).any():
        errors.append(f"execution-adjusted LCRI quantile survival exceeds raw signal in {artifact}")
    return errors


def verify_execution_adjusted_lcri_event_window_release_scorecard(
    output_dir: Path,
    artifact: str = "execution_adjusted_lcri_event_window_release_scorecard.json",
) -> list[str]:
    """Return errors for execution-adjusted LCRI event-window release scorecards."""
    path = output_dir / artifact
    if not path.exists():
        return [f"missing execution-adjusted LCRI event-window release scorecard: {artifact}"]
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [f"invalid execution-adjusted LCRI event-window release scorecard JSON {artifact}: {exc}"]
    required = {
        "high_lcri_rows",
        "toxic_high_lcri_rows",
        "toxic_high_lcri_row_share",
        "event_high_lcri_rows",
        "event_toxic_high_lcri_rows",
        "event_toxic_high_lcri_row_share",
        "weighted_high_lcri_signal_survival_ratio",
        "weighted_high_lcri_fill_adverse_spread",
        "weighted_high_lcri_negative_edge_share",
        "worst_event_window_regime",
        "worst_event_window_bucket",
        "worst_event_window_label",
        "release_decision",
        "release_label",
        "blocking_reasons",
        "review_reasons",
    }
    missing = sorted(required - set(payload))
    if missing:
        return [f"incomplete execution-adjusted LCRI event-window release scorecard {artifact}: {missing}"]

    errors: list[str] = []
    numeric_columns = [
        "high_lcri_rows",
        "toxic_high_lcri_rows",
        "toxic_high_lcri_row_share",
        "event_high_lcri_rows",
        "event_toxic_high_lcri_rows",
        "event_toxic_high_lcri_row_share",
        "weighted_high_lcri_signal_survival_ratio",
        "weighted_high_lcri_fill_adverse_spread",
        "weighted_high_lcri_negative_edge_share",
    ]
    numeric = {column: float(payload[column]) for column in numeric_columns}
    if not all(math.isfinite(value) for value in numeric.values()):
        errors.append(f"non-finite execution-adjusted LCRI event-window scorecard values in {artifact}")
    count_columns = [
        "high_lcri_rows",
        "toxic_high_lcri_rows",
        "event_high_lcri_rows",
        "event_toxic_high_lcri_rows",
    ]
    if not all(numeric[column] >= 0.0 and numeric[column].is_integer() for column in count_columns):
        errors.append(f"negative execution-adjusted LCRI event-window scorecard row counts in {artifact}")
    if (
        numeric["toxic_high_lcri_rows"] > numeric["high_lcri_rows"]
        or numeric["event_toxic_high_lcri_rows"] > numeric["event_high_lcri_rows"]
        or numeric["event_high_lcri_rows"] > numeric["high_lcri_rows"]
    ):
        errors.append(
            f"impossible execution-adjusted LCRI event-window scorecard row counts in {artifact}"
        )
    bounded_columns = [
        "toxic_high_lcri_row_share",
        "event_toxic_high_lcri_row_share",
        "weighted_high_lcri_signal_survival_ratio",
        "weighted_high_lcri_negative_edge_share",
    ]
    if not all(0.0 <= numeric[column] <= 1.0 for column in bounded_columns):
        errors.append(f"bounded execution-adjusted LCRI event-window scorecard shares violated in {artifact}")
    expected_toxic_share = (
        numeric["toxic_high_lcri_rows"] / numeric["high_lcri_rows"]
        if numeric["high_lcri_rows"] > 0.0
        else 0.0
    )
    expected_event_toxic_share = (
        numeric["event_toxic_high_lcri_rows"] / numeric["event_high_lcri_rows"]
        if numeric["event_high_lcri_rows"] > 0.0
        else 0.0
    )
    if abs(numeric["toxic_high_lcri_row_share"] - expected_toxic_share) > 1e-9 or abs(
        numeric["event_toxic_high_lcri_row_share"] - expected_event_toxic_share
    ) > 1e-9:
        errors.append(f"inconsistent execution-adjusted LCRI event-window scorecard shares in {artifact}")
    valid_decisions = {"pass", "review", "block"}
    decision = str(payload["release_decision"])
    if decision not in valid_decisions:
        errors.append(f"invalid execution-adjusted LCRI event-window scorecard decision in {artifact}")
    expected_label = f"execution_lcri_event_window_{'blocked' if decision == 'block' else decision}"
    if str(payload["release_label"]) != expected_label:
        errors.append(f"inconsistent execution-adjusted LCRI event-window scorecard label in {artifact}")
    text_columns = [
        "worst_event_window_regime",
        "worst_event_window_bucket",
        "worst_event_window_label",
        "blocking_reasons",
        "review_reasons",
    ]
    if any(not str(payload[column]) for column in text_columns):
        errors.append(f"blank execution-adjusted LCRI event-window scorecard text fields in {artifact}")
    return errors


def verify_passive_fill_event_window_regime_scorecard(
    output_dir: Path,
    artifact: str = "passive_fill_event_window_regime_scorecard.json",
) -> list[str]:
    """Return errors for passive-fill event-window regime release scorecards."""
    path = output_dir / artifact
    if not path.exists():
        return [f"missing passive-fill event-window regime scorecard: {artifact}"]
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [f"invalid passive-fill event-window regime scorecard JSON {artifact}: {exc}"]

    required = {
        "regimes",
        "total_rows",
        "event_rows",
        "event_row_share",
        "post_event_rows",
        "post_event_negative_edge_share",
        "post_event_mean_toxicity_probability",
        "event_mean_execution_adjusted_edge_ticks",
        "worst_regime_by_toxicity",
        "worst_regime_toxicity_probability",
        "worst_regime_negative_edge_share",
        "event_window_release_label",
    }
    missing = sorted(required - set(payload))
    if missing:
        return [f"incomplete passive-fill event-window regime scorecard {artifact}: {missing}"]

    errors: list[str] = []
    numeric_columns = [
        "regimes",
        "total_rows",
        "event_rows",
        "event_row_share",
        "post_event_rows",
        "post_event_negative_edge_share",
        "post_event_mean_toxicity_probability",
        "event_mean_execution_adjusted_edge_ticks",
        "worst_regime_toxicity_probability",
        "worst_regime_negative_edge_share",
    ]
    numeric = {column: float(payload[column]) for column in numeric_columns}
    if not all(math.isfinite(value) for value in numeric.values()):
        errors.append(f"non-finite passive-fill event-window scorecard values in {artifact}")
    count_columns = ["regimes", "total_rows", "event_rows", "post_event_rows"]
    if not all(numeric[column] >= 0.0 and numeric[column].is_integer() for column in count_columns):
        errors.append(f"negative passive-fill event-window scorecard row counts in {artifact}")
    if (
        numeric["event_rows"] > numeric["total_rows"]
        or numeric["post_event_rows"] > numeric["total_rows"]
    ):
        errors.append(f"impossible passive-fill event-window scorecard row counts in {artifact}")
    bounded_columns = [
        "event_row_share",
        "post_event_negative_edge_share",
        "post_event_mean_toxicity_probability",
        "worst_regime_toxicity_probability",
        "worst_regime_negative_edge_share",
    ]
    if not all(0.0 <= numeric[column] <= 1.0 for column in bounded_columns):
        errors.append(f"bounded passive-fill event-window scorecard shares violated in {artifact}")
    expected_event_share = (
        numeric["event_rows"] / numeric["total_rows"] if numeric["total_rows"] > 0.0 else 0.0
    )
    if abs(numeric["event_row_share"] - expected_event_share) > 1e-9:
        errors.append(f"inconsistent passive-fill event-window scorecard event share in {artifact}")
    valid_labels = {
        "insufficient_event_window_evidence",
        "toxic_post_event_reversal",
        "nonpositive_event_edge",
        "event_window_execution_ready",
    }
    if str(payload["event_window_release_label"]) not in valid_labels:
        errors.append(f"invalid passive-fill event-window scorecard release label in {artifact}")
    if not str(payload["worst_regime_by_toxicity"]):
        errors.append(f"blank passive-fill event-window scorecard worst regime in {artifact}")
    return errors


def verify_queue_position_lcri_tail_fill_residuals(
    output_dir: Path, artifact: str = "queue_position_lcri_tail_fill_residuals.csv"
) -> list[str]:
    """Return errors for queue-position LCRI tail fill calibration residual artifacts."""
    path = output_dir / artifact
    if not path.exists():
        return [f"missing queue-position LCRI tail fill residuals: {artifact}"]
    frame = pd.read_csv(path)
    required = {
        "regime",
        "best_execution_side",
        "lcri_tail_bin",
        "rows",
        "mean_abs_lcri",
        "mean_predicted_fill_probability",
        "realized_fill_rate",
        "fill_residual",
        "absolute_fill_residual",
        "mean_execution_adjusted_edge_ticks",
        "residual_edge_drag_ticks",
        "tail_fill_residual_label",
    }
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        return [f"incomplete queue-position LCRI tail fill residuals {artifact}: {missing}"]

    numeric_columns = list(required - {"regime", "best_execution_side", "tail_fill_residual_label"})
    numeric = frame[numeric_columns].astype(float)
    errors: list[str] = []
    if not np.isfinite(numeric.to_numpy()).all():
        errors.append(f"non-finite queue-position LCRI tail residual values in {artifact}")
    if not numeric["rows"].ge(1.0).all():
        errors.append(f"non-positive queue-position LCRI tail residual rows in {artifact}")
    nonnegative_columns = [
        "lcri_tail_bin",
        "mean_abs_lcri",
        "absolute_fill_residual",
        "residual_edge_drag_ticks",
    ]
    if not numeric[nonnegative_columns].ge(0.0).all().all():
        errors.append(f"negative queue-position LCRI tail residual magnitudes in {artifact}")
    probability_columns = ["mean_predicted_fill_probability", "realized_fill_rate"]
    if not numeric[probability_columns].apply(lambda col: col.between(0.0, 1.0).all()).all():
        errors.append(f"bounded queue-position LCRI tail residual probabilities violated in {artifact}")
    if not (numeric["absolute_fill_residual"] - numeric["fill_residual"].abs()).abs().le(1e-9).all():
        errors.append(f"inconsistent queue-position LCRI tail residual magnitudes in {artifact}")
    if not numeric["residual_edge_drag_ticks"].ge(0.0).all():
        errors.append(f"negative queue-position LCRI tail residual drag in {artifact}")
    if frame[["regime", "best_execution_side"]].astype(str).apply(lambda col: col.str.len().eq(0)).any().any():
        errors.append(f"blank queue-position LCRI tail residual keys in {artifact}")
    valid_labels = {"tail_fill_calibrated", "tail_fill_overstated", "tail_fill_understated"}
    if not frame["tail_fill_residual_label"].astype(str).isin(valid_labels).all():
        errors.append(f"invalid queue-position LCRI tail residual labels in {artifact}")
    return errors


def verify_queue_position_path_drawdown_artifacts(
    output_dir: Path,
    episodes_artifact: str = "queue_position_path_drawdown_episodes.csv",
    summary_artifact: str = "queue_position_path_drawdown_summary.json",
) -> list[str]:
    """Return errors for queue-position path drawdown release artifacts."""
    errors: list[str] = []
    episodes_path = output_dir / episodes_artifact
    summary_path = output_dir / summary_artifact
    if not episodes_path.exists():
        errors.append(f"missing queue-position path drawdown episodes: {episodes_artifact}")
    if not summary_path.exists():
        errors.append(f"missing queue-position path drawdown summary: {summary_artifact}")
    if errors:
        return errors

    episodes = pd.read_csv(episodes_path)
    required_episode_columns = {
        "path_id",
        "max_drawdown_ticks",
        "recovery_edge_ticks",
        "dominant_event_window_regime",
        "episode_risk_label",
    }
    missing_episode_columns = sorted(required_episode_columns - set(episodes.columns))
    if missing_episode_columns:
        errors.append(
            f"incomplete queue-position path drawdown episodes {episodes_artifact}: "
            f"{missing_episode_columns}"
        )
    else:
        numeric = episodes[["max_drawdown_ticks", "recovery_edge_ticks"]].astype(float)
        if not np.isfinite(numeric.to_numpy()).all():
            errors.append(f"non-finite queue-position path drawdown values in {episodes_artifact}")
        if not numeric.ge(0.0).all().all():
            errors.append(f"negative queue-position path drawdown magnitudes in {episodes_artifact}")
        key_columns = ["path_id", "dominant_event_window_regime"]
        keys = episodes[key_columns]
        if keys.isna().any().any() or keys.astype(str).apply(lambda col: col.str.strip().eq("")).any().any():
            errors.append(f"blank queue-position path drawdown keys in {episodes_artifact}")
        valid_episode_labels = {"path_drawdown_recovered", "path_drawdown_open"}
        if not episodes["episode_risk_label"].astype(str).isin(valid_episode_labels).all():
            errors.append(f"invalid queue-position path drawdown labels in {episodes_artifact}")

    summary = json.loads(summary_path.read_text())
    required_summary_keys = {
        "episodes",
        "paths_with_drawdown",
        "open_episodes",
        "open_episode_share",
        "severe_episodes",
        "severe_episode_share",
        "mean_drawdown_ticks",
        "max_drawdown_ticks",
        "total_drawdown_ticks",
        "total_recovery_edge_ticks",
        "recovery_coverage_ratio",
        "dominant_drawdown_regime",
        "dominant_regime_drawdown_share",
        "top_path_id",
        "top_path_drawdown_share",
        "drawdown_summary_label",
        "blocking_reasons",
        "review_reasons",
    }
    missing_summary_keys = sorted(required_summary_keys - set(summary))
    if missing_summary_keys:
        errors.append(
            f"incomplete queue-position path drawdown summary {summary_artifact}: "
            f"{missing_summary_keys}"
        )
    valid_summary_labels = {"queue_drawdown_pass", "queue_drawdown_review", "queue_drawdown_blocked"}
    if summary.get("drawdown_summary_label") not in valid_summary_labels:
        errors.append(f"invalid queue-position path drawdown summary label in {summary_artifact}")
    numeric_summary_keys = sorted(
        required_summary_keys
        - {
            "dominant_drawdown_regime",
            "top_path_id",
            "drawdown_summary_label",
            "blocking_reasons",
            "review_reasons",
        }
    )
    present_numeric = [key for key in numeric_summary_keys if key in summary]
    if present_numeric:
        values = np.array([float(summary[key]) for key in present_numeric], dtype=float)
        if not np.isfinite(values).all():
            errors.append(f"non-finite queue-position path drawdown summary values in {summary_artifact}")
        if (values < 0.0).any():
            errors.append(f"negative queue-position path drawdown summary values in {summary_artifact}")
    return errors


def verify_baseline_tail_lift_diagnostics(
    output_dir: Path, artifact: str = "baseline_tail_lift_diagnostics.csv"
) -> list[str]:
    """Return errors for nonlinear-baseline holdout tail lift diagnostics."""
    path = output_dir / artifact
    if not path.exists():
        return [f"missing baseline tail lift diagnostics: {artifact}"]
    frame = pd.read_csv(path)
    required = {
        "tail_bucket",
        "feature",
        "test_rows",
        "feature_min",
        "feature_max",
        "core_rmse",
        "nonlinear_rmse",
        "nonlinear_rmse_lift_vs_core",
        "core_residual_mean",
        "nonlinear_residual_mean",
        "tail_publishability_note",
    }
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        return [f"incomplete baseline tail lift diagnostics {artifact}: {missing}"]

    numeric_columns = list(required - {"tail_bucket", "feature", "tail_publishability_note"})
    numeric = frame[numeric_columns].astype(float)
    errors: list[str] = []
    if not np.isfinite(numeric.to_numpy()).all():
        errors.append(f"non-finite baseline tail lift values in {artifact}")
    if not numeric["test_rows"].ge(1.0).all():
        errors.append(f"non-positive baseline tail lift rows in {artifact}")
    if not numeric[["core_rmse", "nonlinear_rmse"]].ge(0.0).all().all():
        errors.append(f"negative baseline tail lift RMSE in {artifact}")
    if not numeric["nonlinear_rmse_lift_vs_core"].between(-1.0, 1.0).all():
        errors.append(f"bounded baseline tail lift values violated in {artifact}")
    if (numeric["feature_max"] < numeric["feature_min"]).any():
        errors.append(f"baseline tail lift feature bounds inverted in {artifact}")
    if frame["tail_bucket"].astype(str).str.len().eq(0).any():
        errors.append(f"blank baseline tail lift buckets in {artifact}")
    if frame["tail_bucket"].astype(str).duplicated().any():
        errors.append(f"duplicate baseline tail lift buckets in {artifact}")
    if frame["feature"].astype(str).str.len().eq(0).any():
        errors.append(f"blank baseline tail lift features in {artifact}")
    valid_notes = {"nonlinear_tail_lift_supported", "nonlinear_tail_lift_fragile"}
    if not frame["tail_publishability_note"].astype(str).isin(valid_notes).all():
        errors.append(f"invalid baseline tail lift publishability notes in {artifact}")
    return errors


def verify_baseline_stress_residual_drift(
    output_dir: Path, artifact: str = "baseline_stress_residual_drift.csv"
) -> list[str]:
    """Return errors for stress-bucket residual drift neutralization diagnostics."""
    path = output_dir / artifact
    if not path.exists():
        return [f"missing baseline stress residual drift diagnostics: {artifact}"]
    frame = pd.read_csv(path)
    required = {
        "stress_bucket",
        "feature",
        "test_rows",
        "feature_min",
        "feature_max",
        "core_residual_mean",
        "nonlinear_residual_mean",
        "residual_mean_abs_reduction",
        "core_residual_drift_vs_low_bucket",
        "nonlinear_residual_drift_vs_low_bucket",
        "drift_publishability_note",
    }
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        return [f"incomplete baseline stress residual drift diagnostics {artifact}: {missing}"]

    numeric_columns = list(required - {"stress_bucket", "feature", "drift_publishability_note"})
    numeric = frame[numeric_columns].astype(float)
    errors: list[str] = []
    if not np.isfinite(numeric.to_numpy()).all():
        errors.append(f"non-finite baseline stress residual drift values in {artifact}")
    if not numeric["test_rows"].ge(1.0).all():
        errors.append(f"non-positive baseline stress residual drift rows in {artifact}")
    if (numeric["feature_max"] < numeric["feature_min"]).any():
        errors.append(f"baseline stress residual drift feature bounds inverted in {artifact}")
    neutralized_note = frame["drift_publishability_note"].astype(str).eq(
        "nonlinear_residual_drift_neutralized"
    )
    if (numeric.loc[neutralized_note, "residual_mean_abs_reduction"] < 0.0).any():
        errors.append(f"negative baseline stress residual drift reductions in {artifact}")
    core_drift = numeric["core_residual_drift_vs_low_bucket"].abs()
    nonlinear_drift = numeric["nonlinear_residual_drift_vs_low_bucket"].abs()
    if ((nonlinear_drift - core_drift > 1e-9) & neutralized_note).any():
        errors.append(f"baseline stress residual drift not neutralized in {artifact}")
    if frame["stress_bucket"].astype(str).str.len().eq(0).any():
        errors.append(f"blank baseline stress residual drift buckets in {artifact}")
    if frame["stress_bucket"].astype(str).duplicated().any():
        errors.append(f"duplicate baseline stress residual drift buckets in {artifact}")
    if frame["feature"].astype(str).str.len().eq(0).any():
        errors.append(f"blank baseline stress residual drift features in {artifact}")
    valid_notes = {"nonlinear_residual_drift_neutralized", "nonlinear_residual_drift_fragile"}
    if not frame["drift_publishability_note"].astype(str).isin(valid_notes).all():
        errors.append(f"invalid baseline stress residual drift publishability notes in {artifact}")
    return errors


def verify_baseline_regime_publishability_summary(
    output_dir: Path, artifact: str = "baseline_regime_publishability_summary.json"
) -> list[str]:
    """Return errors for malformed nonlinear-baseline regime release gates."""
    path = output_dir / artifact
    if not path.exists():
        return [f"missing baseline regime publishability summary: {artifact}"]
    summary = json.loads(path.read_text())
    required = {
        "regimes",
        "supported_regimes",
        "unsupported_regimes",
        "min_regime_mean_lift",
        "min_regime_worst_fold_lift",
        "min_regime_winner_rate",
        "weakest_regime",
        "publishable",
        "review_note",
    }
    missing = sorted(required - set(summary))
    if missing:
        return [f"incomplete baseline regime publishability summary {artifact}: {missing}"]

    errors: list[str] = []
    int_keys = ["regimes", "supported_regimes", "unsupported_regimes"]
    for key in int_keys:
        value = summary[key]
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(f"non-integer baseline regime publishability {key} in {artifact}")
    regimes = int(summary["regimes"])
    supported = int(summary["supported_regimes"])
    unsupported = int(summary["unsupported_regimes"])
    if regimes < 1:
        errors.append(f"non-positive baseline regime publishability regime count in {artifact}")
    if supported < 0 or unsupported < 0 or supported + unsupported != regimes:
        errors.append(f"inconsistent baseline regime publishability support counts in {artifact}")

    numeric_keys = [
        "min_regime_mean_lift",
        "min_regime_worst_fold_lift",
        "min_regime_winner_rate",
    ]
    numeric = np.array([float(summary[key]) for key in numeric_keys], dtype=float)
    if not np.isfinite(numeric).all():
        errors.append(f"non-finite baseline regime publishability metrics in {artifact}")
    if not -1.0 <= float(summary["min_regime_mean_lift"]) <= 1.0:
        errors.append(f"bounded baseline regime mean lift violated in {artifact}")
    if not -1.0 <= float(summary["min_regime_worst_fold_lift"]) <= 1.0:
        errors.append(f"bounded baseline regime worst lift violated in {artifact}")
    if not 0.0 <= float(summary["min_regime_winner_rate"]) <= 1.0:
        errors.append(f"bounded baseline regime winner rate violated in {artifact}")
    if not isinstance(summary["publishable"], bool):
        errors.append(f"non-boolean baseline regime publishability decision in {artifact}")
    if not str(summary["weakest_regime"]).strip():
        errors.append(f"blank baseline regime publishability weakest regime in {artifact}")
    valid_notes = {"nonlinear_lift_regime_robust", "nonlinear_lift_regime_fragile"}
    if str(summary["review_note"]) not in valid_notes:
        errors.append(f"invalid baseline regime publishability review note in {artifact}")
    return errors


def verify_execution_publishability_release_gate(
    output_dir: Path, artifact: str = "execution_publishability_release_gate.json"
) -> list[str]:
    """Return errors for malformed execution publishability release gates."""
    path = output_dir / artifact
    if not path.exists():
        return [f"missing execution publishability release gate: {artifact}"]
    gate = json.loads(path.read_text())
    required = {
        "total_rows",
        "conflict_rows",
        "weighted_conflict_share",
        "high_priority_conflict_rows",
        "high_priority_conflict_share",
        "quality_gate_label",
        "capacity_stability_label",
        "regime_capacity_stability_label",
        "lost_capacity_regimes",
        "stable_regime_share",
        "worst_capacity_regime",
        "lcri_regime_survival_label",
        "weak_lcri_regime_sides",
        "worst_lcri_regime",
        "worst_lcri_side",
        "min_lcri_execution_survival_share",
        "max_lcri_execution_conflict_share",
        "blocking_reasons",
        "review_reasons",
        "decision",
        "passes",
        "release_gate_label",
    }
    missing = sorted(required - set(gate))
    if missing:
        return [f"incomplete execution publishability release gate {artifact}: {missing}"]

    errors: list[str] = []
    numeric_keys = [
        "total_rows",
        "conflict_rows",
        "weighted_conflict_share",
        "high_priority_conflict_rows",
        "high_priority_conflict_share",
        "lost_capacity_regimes",
        "stable_regime_share",
        "weak_lcri_regime_sides",
        "min_lcri_execution_survival_share",
        "max_lcri_execution_conflict_share",
    ]
    numeric = {key: float(gate[key]) for key in numeric_keys}
    if not all(math.isfinite(value) for value in numeric.values()):
        errors.append(f"non-finite execution publishability release gate values in {artifact}")
    if any(numeric[key] < 0.0 for key in ["total_rows", "conflict_rows", "high_priority_conflict_rows", "lost_capacity_regimes"]):
        errors.append(f"negative execution publishability release gate counts in {artifact}")
    if not 0.0 <= numeric["weighted_conflict_share"] <= 1.0:
        errors.append(f"bounded execution release conflict share violated in {artifact}")
    if not 0.0 <= numeric["high_priority_conflict_share"] <= 1.0:
        errors.append(f"bounded execution release high-priority conflict share violated in {artifact}")
    if not 0.0 <= numeric["stable_regime_share"] <= 1.0:
        errors.append(f"bounded execution release regime stability share violated in {artifact}")
    if not 0.0 <= numeric["min_lcri_execution_survival_share"] <= 1.0:
        errors.append(f"bounded execution release LCRI survival share violated in {artifact}")
    if not 0.0 <= numeric["max_lcri_execution_conflict_share"] <= 1.0:
        errors.append(f"bounded execution release LCRI conflict share violated in {artifact}")
    if numeric["conflict_rows"] > numeric["total_rows"]:
        errors.append(f"execution release conflict rows exceed total rows in {artifact}")
    if numeric["high_priority_conflict_rows"] > numeric["total_rows"]:
        errors.append(f"execution release high-priority conflict rows exceed total rows in {artifact}")
    if str(gate["decision"]) not in {"pass", "review", "block"}:
        errors.append(f"invalid execution release decision in {artifact}: {gate['decision']}")
    if not isinstance(gate["passes"], bool):
        errors.append(f"non-boolean execution release passes flag in {artifact}")
    elif gate["passes"] != (str(gate["decision"]) == "pass"):
        errors.append(f"execution release passes flag contradicts decision in {artifact}")
    expected_label = {
        "pass": "execution_release_publishable",
        "review": "execution_release_review",
        "block": "execution_release_blocked",
    }.get(str(gate["decision"]))
    if expected_label is not None and str(gate["release_gate_label"]) != expected_label:
        errors.append(f"execution release label contradicts decision in {artifact}")
    if not str(gate["quality_gate_label"]):
        errors.append(f"blank execution release quality gate label in {artifact}")
    if not str(gate["capacity_stability_label"]):
        errors.append(f"blank execution release capacity stability label in {artifact}")
    if not str(gate["regime_capacity_stability_label"]):
        errors.append(f"blank execution release regime capacity stability label in {artifact}")
    if not str(gate["worst_capacity_regime"]):
        errors.append(f"blank execution release worst capacity regime in {artifact}")
    if not str(gate["lcri_regime_survival_label"]):
        errors.append(f"blank execution release LCRI regime survival label in {artifact}")
    if not str(gate["worst_lcri_regime"]):
        errors.append(f"blank execution release worst LCRI regime in {artifact}")
    if not str(gate["worst_lcri_side"]):
        errors.append(f"blank execution release worst LCRI side in {artifact}")
    return errors


def verify_passive_fill_realization_horizon_sweep(
    output_dir: Path, artifact: str = "passive_fill_realization_horizon_sweep.csv"
) -> list[str]:
    """Return errors for horizon-sensitivity passive-fill calibration sweeps."""
    path = output_dir / artifact
    if not path.exists():
        return [f"missing passive-fill realization horizon sweep: {artifact}"]
    frame = pd.read_csv(path)
    required = {
        "horizon",
        "rows",
        "bins",
        "regimes",
        "weighted_mean_predicted_fill_probability",
        "weighted_realized_fill_rate",
        "weighted_calibration_error",
        "expected_calibration_error",
        "weighted_brier_score",
        "worst_absolute_calibration_error",
        "realized_fill_rate_gap_vs_shortest",
        "brier_score_gap_vs_shortest",
        "horizon_stability_label",
    }
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        return [f"incomplete passive-fill realization horizon sweep {artifact}: {missing}"]

    numeric_columns = list(required - {"horizon_stability_label"})
    numeric = frame[numeric_columns].astype(float)
    errors: list[str] = []
    if not np.isfinite(numeric.to_numpy()).all():
        errors.append(f"non-finite passive-fill horizon sweep values in {artifact}")
    if not numeric[["horizon", "rows", "bins", "regimes"]].ge(1.0).all().all():
        errors.append(f"non-positive passive-fill horizon sweep counts in {artifact}")
    if not numeric["horizon"].is_monotonic_increasing:
        errors.append(f"unsorted passive-fill realization horizons in {artifact}")
    if numeric["horizon"].duplicated().any():
        errors.append(f"duplicate passive-fill realization horizons in {artifact}")
    probability_columns = [
        "weighted_mean_predicted_fill_probability",
        "weighted_realized_fill_rate",
        "expected_calibration_error",
        "weighted_brier_score",
        "worst_absolute_calibration_error",
    ]
    if not numeric[probability_columns].apply(lambda col: col.between(0.0, 1.0).all()).all():
        errors.append(f"bounded passive-fill horizon sweep probabilities violated in {artifact}")
    signed_columns = [
        "weighted_calibration_error",
        "realized_fill_rate_gap_vs_shortest",
        "brier_score_gap_vs_shortest",
    ]
    if not numeric[signed_columns].apply(lambda col: col.between(-1.0, 1.0).all()).all():
        errors.append(f"bounded passive-fill horizon sweep signed gaps violated in {artifact}")
    if abs(float(numeric.iloc[0]["realized_fill_rate_gap_vs_shortest"])) > 1e-12:
        errors.append(f"first passive-fill realization horizon is not anchored in {artifact}")
    if abs(float(numeric.iloc[0]["brier_score_gap_vs_shortest"])) > 1e-12:
        errors.append(f"first passive-fill brier horizon is not anchored in {artifact}")
    allowed_labels = {
        "anchor_horizon",
        "later_fill_realization",
        "horizon_fragile",
        "horizon_stable",
    }
    unknown_labels = sorted(set(frame["horizon_stability_label"].astype(str)) - allowed_labels)
    if unknown_labels:
        errors.append(f"unknown passive-fill horizon stability labels in {artifact}: {unknown_labels}")
    if str(frame.iloc[0]["horizon_stability_label"]) != "anchor_horizon":
        errors.append(f"first passive-fill realization horizon is not labeled anchor_horizon in {artifact}")
    return errors


def verify_passive_fill_threshold_policy_curve(
    output_dir: Path, artifact: str = "passive_fill_threshold_policy_curve.csv"
) -> list[str]:
    """Return errors for executable passive-fill threshold policy curves."""
    path = output_dir / artifact
    if not path.exists():
        return [f"missing passive fill threshold policy curve: {artifact}"]
    frame = pd.read_csv(path)
    required = {
        "threshold",
        "candidate_rows",
        "trade_share",
        "long_rows",
        "short_rows",
        "mean_predicted_fill_probability",
        "realized_fill_rate",
        "weighted_brier_score",
        "mean_realized_edge_ticks",
        "positive_edge_rate",
        "mean_execution_adjusted_edge_ticks",
        "policy_label",
    }
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        return [f"incomplete passive fill threshold policy curve {artifact}: {missing}"]

    numeric_columns = list(required - {"policy_label"})
    numeric = frame[numeric_columns].astype(float)
    errors: list[str] = []
    if not np.isfinite(numeric.to_numpy()).all():
        errors.append(f"non-finite passive fill threshold policy values in {artifact}")
    count_columns = ["candidate_rows", "long_rows", "short_rows"]
    if not numeric[count_columns].ge(0.0).all().all():
        errors.append(f"negative passive fill threshold policy counts in {artifact}")
    probability_columns = [
        "threshold",
        "trade_share",
        "mean_predicted_fill_probability",
        "realized_fill_rate",
        "weighted_brier_score",
        "positive_edge_rate",
    ]
    if not numeric[probability_columns].apply(lambda col: col.between(0.0, 1.0).all()).all():
        errors.append(f"bounded passive fill threshold policy probabilities violated in {artifact}")
    if not numeric["threshold"].is_monotonic_increasing:
        errors.append(f"unsorted passive fill threshold policy thresholds in {artifact}")
    if numeric["threshold"].duplicated().any():
        errors.append(f"duplicate passive fill threshold policy thresholds in {artifact}")
    if (numeric["long_rows"] + numeric["short_rows"] > numeric["candidate_rows"]).any():
        errors.append(f"passive fill threshold side counts exceed candidates in {artifact}")
    allowed_labels = {
        "no_executable_policy",
        "broad_execution_policy",
        "selective_high_quality_policy",
        "edge_positive_fill_uncertain_policy",
        "execution_policy_rejected",
    }
    unknown_labels = sorted(set(frame["policy_label"].astype(str)) - allowed_labels)
    if unknown_labels:
        errors.append(f"unknown passive fill threshold policy labels in {artifact}: {unknown_labels}")
    return errors


def verify_passive_fill_event_lifecycle_policy_curve(
    output_dir: Path, artifact: str = "passive_fill_event_lifecycle_policy_curve.csv"
) -> list[str]:
    """Return errors for lifecycle-conditioned passive-fill event policy curves."""
    path = output_dir / artifact
    if not path.exists():
        return [f"missing passive fill event lifecycle policy curve: {artifact}"]
    frame = pd.read_csv(path)
    required = {
        "lifecycle_path",
        "pre_window_regime",
        "event_regime",
        "post_window_regime",
        "threshold",
        "total_events",
        "candidate_events",
        "event_share",
        "mean_event_fill_probability",
        "mean_event_adverse_fill_probability",
        "mean_event_edge_ticks",
        "mean_pre_realized_edge_sum",
        "mean_post_realized_edge_sum",
        "mean_post_minus_pre_realized_edge",
        "adverse_post_edge_share",
        "policy_label",
    }
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        return [f"incomplete passive fill event lifecycle policy curve {artifact}: {missing}"]

    numeric_columns = list(
        required
        - {
            "lifecycle_path",
            "pre_window_regime",
            "event_regime",
            "post_window_regime",
            "policy_label",
        }
    )
    numeric = frame[numeric_columns].astype(float)
    errors: list[str] = []
    if not np.isfinite(numeric.to_numpy()).all():
        errors.append(f"non-finite passive fill event lifecycle policy values in {artifact}")
    count_columns = ["total_events", "candidate_events"]
    if not numeric[count_columns].ge(0.0).all().all():
        errors.append(f"negative passive fill event lifecycle policy counts in {artifact}")
    probability_columns = [
        "threshold",
        "event_share",
        "mean_event_fill_probability",
        "mean_event_adverse_fill_probability",
        "adverse_post_edge_share",
    ]
    if not numeric[probability_columns].apply(lambda col: col.between(0.0, 1.0).all()).all():
        errors.append(
            f"bounded passive fill event lifecycle policy probabilities violated in {artifact}"
        )
    if (numeric["candidate_events"] > numeric["total_events"]).any():
        errors.append(f"passive fill event lifecycle policy candidates exceed totals in {artifact}")
    threshold_order = frame.groupby("lifecycle_path", sort=False)["threshold"].apply(
        lambda col: col.astype(float).is_monotonic_increasing
    )
    if not threshold_order.all():
        errors.append(f"unsorted passive fill event lifecycle policy thresholds in {artifact}")
    duplicate_thresholds = frame.duplicated(["lifecycle_path", "threshold"]).any()
    if duplicate_thresholds:
        errors.append(f"duplicate passive fill event lifecycle policy thresholds in {artifact}")
    allowed_labels = {
        "no_lifecycle_policy_events",
        "broad_lifecycle_policy",
        "selective_lifecycle_policy",
        "lifecycle_policy_review",
        "lifecycle_policy_blocked",
    }
    unknown_labels = sorted(set(frame["policy_label"].astype(str)) - allowed_labels)
    if unknown_labels:
        errors.append(
            f"unknown passive fill event lifecycle policy labels in {artifact}: {unknown_labels}"
        )
    expected_path = (
        frame["pre_window_regime"].astype(str)
        + "|"
        + frame["event_regime"].astype(str)
        + "|"
        + frame["post_window_regime"].astype(str)
    )
    if not frame["lifecycle_path"].astype(str).eq(expected_path).all():
        errors.append(f"passive fill event lifecycle policy paths are inconsistent in {artifact}")
    return errors



def verify_passive_fill_event_transition_policy_curve(
    output_dir: Path, artifact: str = "passive_fill_event_transition_policy_curve.csv"
) -> list[str]:
    """Return errors for transition-conditioned passive-fill event policy curves."""
    path = output_dir / artifact
    if not path.exists():
        return [f"missing passive fill event transition policy curve: {artifact}"]
    frame = pd.read_csv(path)
    required = {
        "regime_transition",
        "threshold",
        "total_events",
        "candidate_events",
        "event_share",
        "mean_event_fill_probability",
        "mean_event_adverse_fill_probability",
        "mean_event_edge_ticks",
        "mean_post_minus_pre_realized_edge",
        "adverse_post_edge_share",
        "policy_label",
    }
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        return [f"incomplete passive fill event transition policy curve {artifact}: {missing}"]

    numeric_columns = list(required - {"regime_transition", "policy_label"})
    numeric = frame[numeric_columns].astype(float)
    errors: list[str] = []
    if not np.isfinite(numeric.to_numpy()).all():
        errors.append(f"non-finite passive fill event transition policy values in {artifact}")
    count_columns = ["total_events", "candidate_events"]
    if not numeric[count_columns].ge(0.0).all().all():
        errors.append(f"negative passive fill event transition policy counts in {artifact}")
    probability_columns = [
        "threshold",
        "event_share",
        "mean_event_fill_probability",
        "mean_event_adverse_fill_probability",
        "adverse_post_edge_share",
    ]
    if not numeric[probability_columns].apply(lambda col: col.between(0.0, 1.0).all()).all():
        errors.append(
            f"bounded passive fill event transition policy probabilities violated in {artifact}"
        )
    if (numeric["candidate_events"] > numeric["total_events"]).any():
        errors.append(f"passive fill event transition policy candidates exceed totals in {artifact}")
    threshold_order = frame.groupby("regime_transition", sort=False)["threshold"].apply(
        lambda col: col.astype(float).is_monotonic_increasing
    )
    if not threshold_order.all():
        errors.append(f"unsorted passive fill event transition policy thresholds in {artifact}")
    duplicate_thresholds = frame.duplicated(["regime_transition", "threshold"]).any()
    if duplicate_thresholds:
        errors.append(f"duplicate passive fill event transition policy thresholds in {artifact}")
    allowed_labels = {
        "no_transition_policy_events",
        "broad_transition_policy",
        "selective_transition_policy",
        "transition_policy_review",
        "transition_policy_blocked",
    }
    unknown_labels = sorted(set(frame["policy_label"].astype(str)) - allowed_labels)
    if unknown_labels:
        errors.append(
            f"unknown passive fill event transition policy labels in {artifact}: {unknown_labels}"
        )
    return errors


def verify_passive_fill_event_policy_stability(
    output_dir: Path,
    artifact: str = "passive_fill_event_lifecycle_policy_stability.csv",
    *,
    context_col: str = "lifecycle_path",
) -> list[str]:
    """Return errors for train-vs-heldout passive-fill event policy stability artifacts."""
    path = output_dir / artifact
    if not path.exists():
        return [f"missing passive fill event policy stability: {artifact}"]
    frame = pd.read_csv(path)
    required = {
        context_col,
        "threshold",
        "train_total_events",
        "heldout_total_events",
        "train_candidate_events",
        "heldout_candidate_events",
        "candidate_event_retention",
        "train_event_share",
        "heldout_event_share",
        "event_share_delta",
        "train_mean_event_fill_probability",
        "heldout_mean_event_fill_probability",
        "mean_event_fill_probability_delta",
        "train_mean_event_adverse_fill_probability",
        "heldout_mean_event_adverse_fill_probability",
        "mean_event_adverse_fill_probability_delta",
        "train_mean_post_minus_pre_realized_edge",
        "heldout_mean_post_minus_pre_realized_edge",
        "mean_post_minus_pre_realized_edge_delta",
        "train_adverse_post_edge_share",
        "heldout_adverse_post_edge_share",
        "adverse_post_edge_share_delta",
        "train_policy_label",
        "heldout_policy_label",
        "heldout_stability_label",
    }
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        return [f"incomplete passive fill event policy stability {artifact}: {missing}"]

    label_columns = {context_col, "train_policy_label", "heldout_policy_label", "heldout_stability_label"}
    numeric_columns = list(required - label_columns)
    numeric = frame[numeric_columns].astype(float)
    errors: list[str] = []
    if not np.isfinite(numeric.to_numpy()).all():
        errors.append(f"non-finite passive fill event policy stability values in {artifact}")
    count_columns = [
        "train_total_events",
        "heldout_total_events",
        "train_candidate_events",
        "heldout_candidate_events",
    ]
    if not numeric[count_columns].ge(0.0).all().all():
        errors.append(f"negative passive fill event policy stability counts in {artifact}")
    probability_columns = [
        "threshold",
        "train_event_share",
        "heldout_event_share",
        "train_mean_event_fill_probability",
        "heldout_mean_event_fill_probability",
        "train_mean_event_adverse_fill_probability",
        "heldout_mean_event_adverse_fill_probability",
        "train_adverse_post_edge_share",
        "heldout_adverse_post_edge_share",
    ]
    if not numeric[probability_columns].apply(lambda col: col.between(0.0, 1.0).all()).all():
        errors.append(f"bounded passive fill event policy stability probabilities violated in {artifact}")
    if (numeric["train_candidate_events"] > numeric["train_total_events"]).any() or (
        numeric["heldout_candidate_events"] > numeric["heldout_total_events"]
    ).any():
        errors.append(f"passive fill event policy stability candidates exceed totals in {artifact}")
    expected_retention = np.divide(
        numeric["heldout_candidate_events"],
        numeric["train_candidate_events"].replace(0.0, np.nan),
    ).fillna(0.0)
    if not np.allclose(numeric["candidate_event_retention"], expected_retention, atol=1e-9):
        errors.append(f"inconsistent passive fill event policy stability retention in {artifact}")
    delta_pairs = [
        ("event_share_delta", "heldout_event_share", "train_event_share"),
        (
            "mean_event_fill_probability_delta",
            "heldout_mean_event_fill_probability",
            "train_mean_event_fill_probability",
        ),
        (
            "mean_event_adverse_fill_probability_delta",
            "heldout_mean_event_adverse_fill_probability",
            "train_mean_event_adverse_fill_probability",
        ),
        (
            "mean_post_minus_pre_realized_edge_delta",
            "heldout_mean_post_minus_pre_realized_edge",
            "train_mean_post_minus_pre_realized_edge",
        ),
        ("adverse_post_edge_share_delta", "heldout_adverse_post_edge_share", "train_adverse_post_edge_share"),
    ]
    inconsistent_deltas = [
        delta
        for delta, heldout, train in delta_pairs
        if not np.allclose(numeric[delta], numeric[heldout] - numeric[train], atol=1e-9)
    ]
    if inconsistent_deltas:
        errors.append(
            f"inconsistent passive fill event policy stability deltas in {artifact}: {inconsistent_deltas}"
        )
    allowed_stability_labels = {
        "heldout_policy_blocker",
        "heldout_policy_review",
        "heldout_policy_no_events",
        "heldout_policy_missing",
        "heldout_policy_stable",
    }
    unknown_stability_labels = sorted(
        set(frame["heldout_stability_label"].astype(str)) - allowed_stability_labels
    )
    if unknown_stability_labels:
        errors.append(
            f"unknown passive fill event policy stability labels in {artifact}: {unknown_stability_labels}"
        )
    return errors


def verify_passive_fill_event_policy_stability_scorecard(
    output_dir: Path,
    artifact: str = "passive_fill_event_policy_stability_scorecard.json",
) -> list[str]:
    """Return errors for candidate-weighted passive-fill policy stability gates."""
    path = output_dir / artifact
    if not path.exists():
        return [f"missing passive fill event policy stability scorecard: {artifact}"]
    payload = json.loads(path.read_text())
    required = {
        "rows",
        "policy_paths",
        "total_train_candidate_events",
        "total_heldout_candidate_events",
        "candidate_event_retention",
        "blocker_rows",
        "review_rows",
        "no_event_rows",
        "missing_rows",
        "blocker_train_candidate_share",
        "review_train_candidate_share",
        "no_event_train_candidate_share",
        "missing_train_candidate_share",
        "weighted_mean_post_minus_pre_realized_edge_delta",
        "weighted_adverse_post_edge_share_delta",
        "worst_policy_path",
        "worst_threshold",
        "worst_heldout_stability_label",
        "policy_stability_decision",
        "policy_stability_label",
        "blocking_reasons",
        "review_reasons",
    }
    missing = sorted(required - set(payload))
    if missing:
        return [f"incomplete passive fill event policy stability scorecard {artifact}: {missing}"]

    count_columns = [
        "rows",
        "policy_paths",
        "total_train_candidate_events",
        "total_heldout_candidate_events",
        "blocker_rows",
        "review_rows",
        "no_event_rows",
        "missing_rows",
    ]
    share_columns = [
        "blocker_train_candidate_share",
        "review_train_candidate_share",
        "no_event_train_candidate_share",
        "missing_train_candidate_share",
    ]
    numeric_columns = [
        *count_columns,
        "candidate_event_retention",
        *share_columns,
        "weighted_mean_post_minus_pre_realized_edge_delta",
        "weighted_adverse_post_edge_share_delta",
        "worst_threshold",
    ]
    numeric = {key: float(payload[key]) for key in numeric_columns}
    errors: list[str] = []
    if not np.isfinite(list(numeric.values())).all():
        errors.append(f"non-finite passive fill event policy stability scorecard values in {artifact}")
    if any(numeric[key] < 0.0 for key in count_columns):
        errors.append(f"negative passive fill event policy stability scorecard counts in {artifact}")
    if any(not math.isclose(numeric[key], round(numeric[key]), abs_tol=1e-9) for key in count_columns):
        errors.append(f"non-integer passive fill event policy stability scorecard counts in {artifact}")
    if any(not 0.0 <= numeric[key] <= 1.0 for key in [*share_columns, "worst_threshold"]):
        errors.append(f"bounded passive fill event policy stability scorecard shares violated in {artifact}")
    if numeric["weighted_adverse_post_edge_share_delta"] < -1.0 or numeric[
        "weighted_adverse_post_edge_share_delta"
    ] > 1.0:
        errors.append(f"bounded passive fill event policy stability scorecard deltas violated in {artifact}")

    expected_retention = (
        numeric["total_heldout_candidate_events"] / numeric["total_train_candidate_events"]
        if numeric["total_train_candidate_events"] > 0.0
        else 0.0
    )
    if not math.isclose(numeric["candidate_event_retention"], expected_retention, abs_tol=1e-9):
        errors.append(f"inconsistent passive fill event policy stability scorecard retention in {artifact}")
    if numeric["policy_paths"] > numeric["rows"] or (
        numeric["blocker_rows"]
        + numeric["review_rows"]
        + numeric["no_event_rows"]
        + numeric["missing_rows"]
        > numeric["rows"]
    ):
        errors.append(f"impossible passive fill event policy stability scorecard row counts in {artifact}")

    allowed_labels = {
        "heldout_policy_blocker",
        "heldout_policy_review",
        "heldout_policy_no_events",
        "heldout_policy_missing",
        "heldout_policy_stable",
        "none",
    }
    allowed_decisions = {"pass", "review", "block"}
    expected_gate_labels = {
        "pass": "passive_fill_policy_stability_pass",
        "review": "passive_fill_policy_stability_review",
        "block": "passive_fill_policy_stability_blocked",
    }
    unknown_labels: list[str] = []
    if str(payload["worst_heldout_stability_label"]) not in allowed_labels:
        unknown_labels.append(str(payload["worst_heldout_stability_label"]))
    decision = str(payload["policy_stability_decision"])
    if decision not in allowed_decisions:
        unknown_labels.append(decision)
    if unknown_labels:
        errors.append(
            f"unknown passive fill event policy stability scorecard labels in {artifact}: {unknown_labels}"
        )
    expected_label = expected_gate_labels.get(decision)
    if expected_label is not None and str(payload["policy_stability_label"]) != expected_label:
        errors.append(f"inconsistent passive fill event policy stability scorecard decision in {artifact}")
    if str(payload["blocking_reasons"]) != "none" and decision != "block":
        errors.append(f"inconsistent passive fill event policy stability scorecard decision in {artifact}")
    if str(payload["review_reasons"]) != "none" and decision == "pass":
        errors.append(f"inconsistent passive fill event policy stability scorecard decision in {artifact}")
    if numeric["rows"] == 0.0 and str(payload["worst_policy_path"]) != "none":
        errors.append(f"inconsistent passive fill event policy stability scorecard worst path in {artifact}")
    return errors


def verify_event_level_passive_fill_horizon_sweep(
    output_dir: Path, artifact: str = "event_level_passive_fill_horizon_sweep.csv"
) -> list[str]:
    """Return errors for event-message passive-fill calibration horizon sweeps."""
    path = output_dir / artifact
    if not path.exists():
        return [f"missing event-level passive-fill horizon sweep: {artifact}"]
    frame = pd.read_csv(path)
    required = {
        "horizon",
        "event_depletion_source",
        "rows",
        "bins",
        "regimes",
        "weighted_mean_predicted_fill_probability",
        "weighted_realized_fill_rate",
        "weighted_calibration_error",
        "expected_calibration_error",
        "weighted_brier_score",
        "worst_absolute_calibration_error",
        "realized_fill_rate_gap_vs_shortest",
        "brier_score_gap_vs_shortest",
        "horizon_stability_label",
    }
    missing = sorted(required - set(frame.columns))
    if missing or frame.empty:
        return [f"incomplete event-level passive-fill horizon sweep {artifact}: {missing}"]

    numeric_columns = list(required - {"event_depletion_source", "horizon_stability_label"})
    numeric = frame[numeric_columns].astype(float)
    errors: list[str] = []
    if not np.isfinite(numeric.to_numpy()).all():
        errors.append(f"non-finite event-level passive-fill horizon sweep values in {artifact}")
    if not numeric["horizon"].gt(0.0).all():
        errors.append(f"non-positive event-level passive-fill horizons in {artifact}")
    if not numeric[["rows", "bins", "regimes"]].ge(1.0).all().all():
        errors.append(f"non-positive event-level passive-fill horizon sweep counts in {artifact}")
    if not numeric["horizon"].is_monotonic_increasing:
        errors.append(f"unsorted event-level passive-fill horizons in {artifact}")
    if numeric["horizon"].duplicated().any():
        errors.append(f"duplicate event-level passive-fill horizons in {artifact}")
    probability_columns = [
        "weighted_mean_predicted_fill_probability",
        "weighted_realized_fill_rate",
        "expected_calibration_error",
        "weighted_brier_score",
        "worst_absolute_calibration_error",
    ]
    if not numeric[probability_columns].apply(lambda col: col.between(0.0, 1.0).all()).all():
        errors.append(f"bounded event-level passive-fill horizon probabilities violated in {artifact}")
    signed_columns = [
        "weighted_calibration_error",
        "realized_fill_rate_gap_vs_shortest",
        "brier_score_gap_vs_shortest",
    ]
    if not numeric[signed_columns].apply(lambda col: col.between(-1.0, 1.0).all()).all():
        errors.append(f"bounded event-level passive-fill horizon signed gaps violated in {artifact}")
    if abs(float(numeric.iloc[0]["realized_fill_rate_gap_vs_shortest"])) > 1e-12:
        errors.append(f"first event-level passive-fill horizon is not anchored in {artifact}")
    if abs(float(numeric.iloc[0]["brier_score_gap_vs_shortest"])) > 1e-12:
        errors.append(f"first event-level passive-fill brier horizon is not anchored in {artifact}")
    allowed_labels = {
        "anchor_horizon",
        "later_fill_realization",
        "horizon_fragile",
        "horizon_stable",
    }
    unknown_labels = sorted(set(frame["horizon_stability_label"].astype(str)) - allowed_labels)
    if unknown_labels:
        errors.append(f"unknown event-level passive-fill horizon stability labels in {artifact}: {unknown_labels}")
    if str(frame.iloc[0]["horizon_stability_label"]) != "anchor_horizon":
        errors.append(f"first event-level passive-fill horizon is not labeled anchor_horizon in {artifact}")
    sources = set(frame["event_depletion_source"].astype(str))
    if sources != {"events"}:
        errors.append("non-event passive-fill horizon depletion sources")
    return errors


def verify_alpha_event_review_artifacts(output_dir: Path) -> list[str]:
    """Return errors when alpha event review artifacts are incomplete or stale."""
    paths = {
        label: output_dir / artifact
        for label, artifact in _ALPHA_EVENT_REVIEW_ARTIFACTS.items()
    }
    missing = [path.name for path in paths.values() if not path.exists()]
    if missing:
        return [f"missing alpha event review artifacts: {missing}"]

    events = pd.read_csv(paths["events"])
    summary = json.loads(paths["summary"].read_text())
    packet = pd.read_csv(paths["packet"])
    gate = json.loads(paths["gate"].read_text())
    weighted = json.loads(paths["weighted"].read_text())
    regimes = pd.read_csv(paths["regimes"])
    required_packet = {
        "decision",
        "passes",
        "review_priority",
        "events",
        "adverse_post_drift_share",
        "score_weighted_adverse_share",
        "score_weighted_post_minus_pre_return",
        "top_weighted_event_index",
        "worst_event_regime",
        "release_note",
        "gate_reason",
    }
    missing_packet = sorted(required_packet - set(packet.columns))
    if missing_packet or len(packet) != 1:
        errors = []
        if missing_packet:
            errors.append(f"incomplete alpha event release review packet: {missing_packet}")
        if len(packet) != 1:
            errors.append("alpha event release review packet must contain exactly one row")
        return errors

    expected_summary = alpha_event_window_summary(events)
    expected_weighted = alpha_event_score_weighted_drift(events)
    expected_gate = alpha_event_drift_gate(expected_summary)
    errors: list[str] = []
    _compare_alpha_event_mapping(
        errors,
        label="alpha event window summary",
        expected=expected_summary,
        found=summary,
    )
    _compare_alpha_event_mapping(
        errors,
        label="alpha event score-weighted drift",
        expected=expected_weighted,
        found=weighted,
    )
    _compare_alpha_event_mapping(
        errors,
        label="alpha event drift gate",
        expected=expected_gate,
        found=gate,
    )

    expected = alpha_event_release_review_packet(expected_gate, expected_weighted, regimes).iloc[0]
    found = packet.iloc[0]
    for column in required_packet:
        expected_value = expected[column]
        found_value = found[column]
        if isinstance(expected_value, bool):
            mismatch = bool(found_value) != expected_value
        elif isinstance(expected_value, (int, float, np.integer, np.floating)):
            mismatch = not math.isclose(float(found_value), float(expected_value), rel_tol=1e-9, abs_tol=1e-9)
        else:
            mismatch = str(found_value) != str(expected_value)
        if mismatch:
            errors.append(f"alpha event release review packet mismatch for {column}")
    return errors


def alpha_event_review_verification_summary(output_dir: Path) -> dict[str, Any]:
    """Summarize alpha-event artifact verification for release-owner triage.

    The detailed verifier returns actionable error strings. This companion payload
    gives dashboards a stable, compact health row without needing to parse those
    messages or reopen every source artifact.
    """
    artifact_paths = {
        label: output_dir / artifact
        for label, artifact in _ALPHA_EVENT_REVIEW_ARTIFACTS.items()
    }
    missing = [path.name for path in artifact_paths.values() if not path.exists()]
    errors = verify_alpha_event_review_artifacts(output_dir)
    error_summary = summarize_verification_errors(errors)
    blocker_diagnostics = alpha_event_review_blocker_diagnostics(output_dir)
    packet_payload = _read_alpha_event_packet_payload(artifact_paths["packet"])
    return {
        "artifacts_expected": len(artifact_paths),
        "artifacts_present": len(artifact_paths) - len(missing),
        "missing_artifacts": missing,
        "errors": len(errors),
        "error_families": {
            family: count
            for family, count in error_summary.items()
            if family not in {"errors", "passes_verification"} and count
        },
        "blocking_artifacts": sorted(set(blocker_diagnostics["artifact"].astype(str))),
        "blocking_errors": errors[:3],
        "passes_verification": len(errors) == 0,
        "decision": packet_payload.get("decision", "unknown"),
        "review_priority": packet_payload.get("review_priority", "unknown"),
        "owner_action": _alpha_event_owner_action(errors, missing),
    }


def alpha_event_review_blocker_diagnostics(output_dir: Path) -> pd.DataFrame:
    """Return one row per alpha-event review blocker for owner triage.

    The verifier intentionally returns concise strings for CLI checks. This
    diagnostic keeps the same source of truth but projects those strings into a
    stable table that dashboards and markdown summaries can group by artifact.
    """
    artifact_paths = {
        label: output_dir / artifact
        for label, artifact in _ALPHA_EVENT_REVIEW_ARTIFACTS.items()
    }
    rows: list[dict[str, str]] = []
    missing = [path.name for path in artifact_paths.values() if not path.exists()]
    for artifact in missing:
        rows.append(
            {
                "severity": "block",
                "artifact": artifact,
                "check": "missing_artifact",
                "message": f"missing alpha event review artifact: {artifact}",
                "owner_action": "regenerate missing alpha event review artifacts",
            }
        )
    if missing:
        return pd.DataFrame(rows, columns=_ALPHA_EVENT_BLOCKER_COLUMNS)

    for error in verify_alpha_event_review_artifacts(output_dir):
        rows.append(
            {
                "severity": "block",
                "artifact": _alpha_event_error_artifact(error),
                "check": _alpha_event_error_check(error),
                "message": error,
                "owner_action": "rerun alpha event review artifact generation before owner review",
            }
        )
    return pd.DataFrame(rows, columns=_ALPHA_EVENT_BLOCKER_COLUMNS)


def alpha_event_review_blocker_summary(output_dir: Path) -> dict[str, Any]:
    """Summarize alpha-event blocker diagnostics for release dashboards."""
    blockers = alpha_event_review_blocker_diagnostics(output_dir)
    if blockers.empty:
        return {
            "blocker_rows": 0,
            "severity_counts": {},
            "affected_artifacts": [],
            "check_counts": {},
            "top_artifact": "none",
            "owner_action": "alpha event review artifacts have no verification blockers",
        }

    top_artifact = blockers["artifact"].astype(str).iloc[0]
    return {
        "blocker_rows": int(len(blockers)),
        "severity_counts": {
            str(key): int(value)
            for key, value in blockers["severity"].astype(str).value_counts().sort_index().items()
        },
        "affected_artifacts": sorted(set(blockers["artifact"].astype(str))),
        "check_counts": {
            str(key): int(value)
            for key, value in blockers["check"].astype(str).value_counts().sort_index().items()
        },
        "top_artifact": str(top_artifact),
        "owner_action": str(blockers["owner_action"].astype(str).iloc[0]),
    }


def alpha_event_review_owner_readiness(output_dir: Path) -> dict[str, Any]:
    """Condense alpha-event verification into the next release-owner step."""
    verification = alpha_event_review_verification_summary(output_dir)
    blockers = alpha_event_review_blocker_summary(output_dir)
    decision = str(verification.get("decision", "unknown"))
    passes = bool(verification.get("passes_verification", False))
    next_step = _alpha_event_next_review_step(
        passes=passes,
        decision=decision,
        owner_action=str(verification.get("owner_action", "review alpha event artifacts")),
    )
    return {
        "ready_for_owner_review": bool(passes and decision in {"pass", "review", "block"}),
        "passes_verification": passes,
        "decision": decision,
        "review_priority": verification.get("review_priority", "unknown"),
        "blocker_rows": int(blockers["blocker_rows"]),
        "top_blocking_artifact": str(blockers["top_artifact"]),
        "owner_action": str(blockers["owner_action"]),
        "next_review_step": next_step,
    }


def verify_alpha_event_review_verification_summary(output_dir: Path) -> list[str]:
    """Return errors when the compact alpha-event verification summary is stale."""
    path = output_dir / "alpha_event_review_verification_summary.json"
    if not path.exists():
        return ["missing alpha event review verification summary: alpha_event_review_verification_summary.json"]
    try:
        found = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [f"invalid alpha event review verification summary JSON: {exc.msg}"]

    expected = alpha_event_review_verification_summary(output_dir)
    errors: list[str] = []
    missing_keys = sorted(set(expected) - set(found))
    if missing_keys:
        errors.append(f"incomplete alpha event review verification summary: {missing_keys}")
        return errors
    for key, expected_value in expected.items():
        found_value = found[key]
        if isinstance(expected_value, bool):
            mismatch = bool(found_value) != expected_value
        elif isinstance(expected_value, (int, float)):
            mismatch = not math.isclose(float(found_value), float(expected_value), rel_tol=1e-9, abs_tol=1e-9)
        else:
            mismatch = found_value != expected_value
        if mismatch:
            errors.append(f"alpha event review verification summary mismatch for {key}")
    return errors


def _read_alpha_event_packet_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        packet = pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return {}
    if len(packet) != 1:
        return {}
    return packet.iloc[0].to_dict()


def _alpha_event_owner_action(errors: list[str], missing: list[str]) -> str:
    if missing:
        return "regenerate missing alpha event review artifacts"
    if errors:
        return "rerun alpha event review artifact generation before owner review"
    return "alpha event review artifacts are ready for owner review"


def _alpha_event_next_review_step(*, passes: bool, decision: str, owner_action: str) -> str:
    if not passes:
        return owner_action
    if decision == "pass":
        return "release owner can accept alpha event review evidence"
    if decision == "review":
        return "release owner should inspect high-priority alpha event drift before acceptance"
    if decision == "block":
        return "keep release blocked until alpha event drift improves or thresholds are waived"
    return "review alpha event packet decision before acceptance"


def _alpha_event_error_artifact(error: str) -> str:
    lower = error.lower()
    for hint, artifact in _ALPHA_EVENT_ERROR_ARTIFACT_HINTS.items():
        if hint in lower:
            return artifact
    return "alpha_event_review"


def _alpha_event_error_check(error: str) -> str:
    lower = error.lower()
    if "mismatch" in lower:
        return "stale_artifact"
    if "incomplete" in lower:
        return "incomplete_artifact"
    if "must contain exactly one row" in lower:
        return "invalid_row_count"
    return "verification_error"


def _compare_alpha_event_mapping(
    errors: list[str],
    *,
    label: str,
    expected: dict[str, float | int | str | bool],
    found: dict[str, Any],
) -> None:
    missing = sorted(set(expected) - set(found))
    if missing:
        errors.append(f"incomplete {label}: {missing}")
        return
    for key, expected_value in expected.items():
        found_value = found[key]
        if isinstance(expected_value, bool):
            mismatch = bool(found_value) != expected_value
        elif isinstance(expected_value, (int, float, np.integer, np.floating)):
            mismatch = not math.isclose(float(found_value), float(expected_value), rel_tol=1e-9, abs_tol=1e-9)
        else:
            mismatch = str(found_value) != str(expected_value)
        if mismatch:
            errors.append(f"{label} mismatch for {key}")


def verify_figure_artifacts(output_dir: Path, manifest: dict[str, Any]) -> list[str]:
    """Return errors for manifest-listed PNG figures that are unreadable.

    Size and hash checks prove a figure has not changed, but they do not prove it
    is a PNG image that downstream report readers can render. This verifier keeps
    the check lightweight by validating the PNG signature, IHDR dimensions, and
    IEND trailer for manifest-listed figure artifacts.
    """
    errors = []
    for artifact in manifest.get("artifacts", []):
        if not _is_safe_artifact_path(artifact) or not str(artifact).endswith(".png"):
            continue
        path = output_dir / artifact
        if not path.exists():
            continue
        try:
            data = path.read_bytes()
        except OSError as exc:
            errors.append(f"unreadable figure artifact: {artifact}: {exc}")
            continue
        png_error = _validate_png_bytes(data)
        if png_error is not None:
            errors.append(f"invalid figure artifact: {artifact}: {png_error}")
    return errors


def verify_generalization_overview(output_dir: Path) -> list[str]:
    """Return errors for a missing or incomplete generalization overview."""
    path = output_dir / "generalization_overview.json"
    if not path.exists():
        return ["missing generalization overview: generalization_overview.json"]

    payload = json.loads(path.read_text())
    required = {
        "signal_rows",
        "regime_rows",
        "transition_rows",
        "max_signal_directional_accuracy_gap",
        "max_regime_directional_accuracy_gap",
        "max_transition_directional_accuracy_gap",
    }
    missing = sorted(required - set(payload))
    if missing:
        return [f"incomplete generalization overview: {missing}"]
    return []


def verify_generalization_fragility_diagnostics(output_dir: Path) -> list[str]:
    """Return errors for a missing or incomplete heldout fragility table."""
    path = output_dir / "generalization_fragility_diagnostics.csv"
    if not path.exists():
        return ["missing generalization fragility diagnostics: generalization_fragility_diagnostics.csv"]

    columns = set(pd.read_csv(path, nrows=1).columns)
    required = {
        "scope",
        "context",
        "signal",
        "full_rows",
        "heldout_rows",
        "directional_accuracy_gap",
        "heldout_directional_accuracy_se",
        "abs_gap_to_se_ratio",
        "fragility_label",
    }
    missing = sorted(required - columns)
    if missing:
        return [f"incomplete generalization fragility diagnostics: {missing}"]
    return []


def verify_generalization_fragility_summary(output_dir: Path) -> list[str]:
    """Return errors for a missing or incomplete heldout fragility summary."""
    path = output_dir / "generalization_fragility_summary.json"
    if not path.exists():
        return ["missing generalization fragility summary: generalization_fragility_summary.json"]

    payload = json.loads(path.read_text())
    required = {
        "rows",
        "stable_rows",
        "watch_rows",
        "fragile_rows",
        "max_abs_gap_to_se_ratio",
        "most_fragile_context",
    }
    missing = sorted(required - set(payload))
    if missing:
        return [f"incomplete generalization fragility summary: {missing}"]
    return []


def verify_generalization_fragility_consistency(output_dir: Path) -> list[str]:
    """Return errors when fragility diagnostics and summary disagree.

    Heldout fragility is an uncertainty diagnostic rather than a release gate,
    but the summary is still redundant with the row-level CSV. Recompute label
    thresholds, row counts, and the worst context so stale dashboards cannot
    overstate or understate heldout fragility.
    """
    diagnostics_path = output_dir / "generalization_fragility_diagnostics.csv"
    summary_path = output_dir / "generalization_fragility_summary.json"
    if not diagnostics_path.exists() or not summary_path.exists():
        return []

    diagnostics = pd.read_csv(diagnostics_path)
    required_columns = {
        "scope",
        "context",
        "signal",
        "directional_accuracy_gap",
        "heldout_directional_accuracy_se",
        "abs_gap_to_se_ratio",
        "fragility_label",
    }
    missing_columns = sorted(required_columns - set(diagnostics.columns))
    if missing_columns:
        return [
            "cannot verify generalization fragility consistency, missing columns: "
            f"{missing_columns}"
        ]

    summary = json.loads(summary_path.read_text())
    required_keys = {
        "rows",
        "stable_rows",
        "watch_rows",
        "fragile_rows",
        "max_abs_gap_to_se_ratio",
        "most_fragile_context",
    }
    missing_keys = sorted(required_keys - set(summary))
    if missing_keys:
        return [
            "cannot verify generalization fragility consistency, missing summary keys: "
            f"{missing_keys}"
        ]

    errors = []
    ratio = diagnostics["abs_gap_to_se_ratio"].astype(float)
    se = diagnostics["heldout_directional_accuracy_se"].astype(float)
    gap = diagnostics["directional_accuracy_gap"].astype(float)

    invalid_se = diagnostics.loc[se < 0.0, ["scope", "context", "signal"]]
    if not invalid_se.empty:
        errors.append(
            "generalization fragility diagnostics contain negative heldout SE rows: "
            f"{_context_records(invalid_se)}"
        )

    expected_ratio = (gap.abs() / se).where(se > 0.0, 0.0)
    for index, expected_value in expected_ratio.items():
        found_value = ratio.loc[index]
        if abs(float(found_value) - float(expected_value)) > 1e-12:
            row = diagnostics.loc[index]
            errors.append(
                "generalization fragility ratio mismatch for "
                f"{row['scope']}:{row['context']}:{row['signal']}: "
                f"expected {float(expected_value)!r}, found {float(found_value)!r}"
            )

    labels = diagnostics["fragility_label"].astype(str)
    expected_labels = ratio.map(_expected_fragility_label)
    label_mismatches = diagnostics.loc[labels != expected_labels, ["scope", "context", "signal"]].copy()
    if not label_mismatches.empty:
        label_mismatches["expected_fragility_label"] = expected_labels.loc[label_mismatches.index]
        label_mismatches["found_fragility_label"] = labels.loc[label_mismatches.index]
        errors.append(
            "generalization fragility label mismatch: "
            f"{_context_records(label_mismatches)}"
        )

    if diagnostics.empty:
        expected_summary = {
            "rows": 0,
            "stable_rows": 0,
            "watch_rows": 0,
            "fragile_rows": 0,
            "max_abs_gap_to_se_ratio": 0.0,
            "most_fragile_context": "none",
        }
    else:
        worst = diagnostics.loc[ratio.idxmax()]
        expected_summary = {
            "rows": len(diagnostics),
            "stable_rows": int((labels == "stable").sum()),
            "watch_rows": int((labels == "watch").sum()),
            "fragile_rows": int((labels == "fragile").sum()),
            "max_abs_gap_to_se_ratio": float(ratio.max()),
            "most_fragile_context": f"{worst['scope']}:{worst['context']}:{worst['signal']}",
        }
    for key, expected_value in expected_summary.items():
        _append_fragility_mismatch(errors, f"summary.{key}", expected_value, summary.get(key))
    return errors


def verify_generalization_stability_confidence_intervals(output_dir: Path) -> list[str]:
    """Return errors for a missing or incomplete heldout confidence interval table."""
    path = output_dir / "generalization_stability_confidence_intervals.csv"
    if not path.exists():
        return [
            "missing generalization stability confidence intervals: "
            "generalization_stability_confidence_intervals.csv"
        ]

    columns = set(pd.read_csv(path, nrows=1).columns)
    required = {
        "scope",
        "context",
        "signal",
        "heldout_rows",
        "heldout_directional_accuracy",
        "heldout_directional_accuracy_se",
        "confidence_level",
        "heldout_directional_accuracy_ci_lower",
        "heldout_directional_accuracy_ci_upper",
        "heldout_directional_accuracy_ci_width",
        "directional_accuracy_gap",
        "gap_exceeds_ci_half_width",
    }
    missing = sorted(required - columns)
    if missing:
        return [f"incomplete generalization stability confidence intervals: {missing}"]
    return []


def verify_generalization_stability_confidence_summary(output_dir: Path) -> list[str]:
    """Return errors for a missing or incomplete heldout confidence summary."""
    path = output_dir / "generalization_stability_confidence_summary.json"
    if not path.exists():
        return [
            "missing generalization stability confidence summary: "
            "generalization_stability_confidence_summary.json"
        ]

    payload = json.loads(path.read_text())
    required = {
        "rows",
        "gap_exceeds_ci_half_width_rows",
        "mean_ci_width",
        "max_ci_width",
        "widest_interval_context",
    }
    missing = sorted(required - set(payload))
    if missing:
        return [f"incomplete generalization stability confidence summary: {missing}"]
    return []


def verify_generalization_stability_confidence_consistency(output_dir: Path) -> list[str]:
    """Return errors when heldout confidence interval artifacts are stale."""
    intervals_path = output_dir / "generalization_stability_confidence_intervals.csv"
    summary_path = output_dir / "generalization_stability_confidence_summary.json"
    if not intervals_path.exists() or not summary_path.exists():
        return []

    intervals = pd.read_csv(intervals_path)
    required_columns = {
        "scope",
        "context",
        "signal",
        "heldout_directional_accuracy",
        "heldout_directional_accuracy_se",
        "confidence_level",
        "heldout_directional_accuracy_ci_lower",
        "heldout_directional_accuracy_ci_upper",
        "heldout_directional_accuracy_ci_width",
        "directional_accuracy_gap",
        "gap_exceeds_ci_half_width",
    }
    missing_columns = sorted(required_columns - set(intervals.columns))
    if missing_columns:
        return [
            "cannot verify generalization stability confidence intervals, missing columns: "
            f"{missing_columns}"
        ]

    summary = json.loads(summary_path.read_text())
    required_keys = {
        "rows",
        "gap_exceeds_ci_half_width_rows",
        "mean_ci_width",
        "max_ci_width",
        "widest_interval_context",
    }
    missing_keys = sorted(required_keys - set(summary))
    if missing_keys:
        return [
            "cannot verify generalization stability confidence summary, missing keys: "
            f"{missing_keys}"
        ]

    errors = []
    accuracy = intervals["heldout_directional_accuracy"].astype(float)
    se = intervals["heldout_directional_accuracy_se"].astype(float)
    confidence_level = intervals["confidence_level"].astype(float)
    lower = intervals["heldout_directional_accuracy_ci_lower"].astype(float)
    upper = intervals["heldout_directional_accuracy_ci_upper"].astype(float)
    width = intervals["heldout_directional_accuracy_ci_width"].astype(float)
    gap = intervals["directional_accuracy_gap"].astype(float)

    expected_half_width = 1.96 * se
    expected_lower = (accuracy - expected_half_width).clip(lower=0.0)
    expected_upper = (accuracy + expected_half_width).clip(upper=1.0)
    expected_width = expected_upper - expected_lower
    expected_gap_flag = gap.abs() > expected_half_width

    for index, expected_value in expected_lower.items():
        row = intervals.loc[index]
        _append_fragility_mismatch(
            errors,
            f"confidence_interval.{row['scope']}:{row['context']}:{row['signal']}.lower",
            float(expected_value),
            float(lower.loc[index]),
        )
        _append_fragility_mismatch(
            errors,
            f"confidence_interval.{row['scope']}:{row['context']}:{row['signal']}.upper",
            float(expected_upper.loc[index]),
            float(upper.loc[index]),
        )
        _append_fragility_mismatch(
            errors,
            f"confidence_interval.{row['scope']}:{row['context']}:{row['signal']}.width",
            float(expected_width.loc[index]),
            float(width.loc[index]),
        )
        _append_fragility_mismatch(
            errors,
            f"confidence_interval.{row['scope']}:{row['context']}:{row['signal']}.confidence_level",
            0.950004209703559,
            float(confidence_level.loc[index]),
        )
        if bool(expected_gap_flag.loc[index]) != bool(intervals.loc[index, "gap_exceeds_ci_half_width"]):
            errors.append(
                "generalization stability confidence gap flag mismatch for "
                f"{row['scope']}:{row['context']}:{row['signal']}"
            )

    if intervals.empty:
        expected_summary = {
            "rows": 0,
            "gap_exceeds_ci_half_width_rows": 0,
            "mean_ci_width": 0.0,
            "max_ci_width": 0.0,
            "widest_interval_context": "none",
        }
    else:
        widest = intervals.loc[width.idxmax()]
        expected_summary = {
            "rows": len(intervals),
            "gap_exceeds_ci_half_width_rows": int(expected_gap_flag.sum()),
            "mean_ci_width": float(width.mean()),
            "max_ci_width": float(width.max()),
            "widest_interval_context": f"{widest['scope']}:{widest['context']}:{widest['signal']}",
        }
    for key, expected_value in expected_summary.items():
        _append_fragility_mismatch(errors, f"confidence_summary.{key}", expected_value, summary.get(key))
    return errors


def verify_lcri_generalization_gap_leaderboard(output_dir: Path) -> list[str]:
    """Return errors for a missing or incomplete LCRI gap leaderboard artifact."""
    path = output_dir / "lcri_generalization_gap_leaderboard.csv"
    if not path.exists():
        return ["missing LCRI generalization gap leaderboard: lcri_generalization_gap_leaderboard.csv"]

    columns = set(pd.read_csv(path, nrows=1).columns)
    required = {"scope", "context", "signal", "directional_accuracy_gap"}
    missing = sorted(required - columns)
    if missing:
        return [f"incomplete LCRI generalization gap leaderboard: {missing}"]
    return []


def verify_lcri_generalization_scope_summary(output_dir: Path) -> list[str]:
    """Return errors for a missing or incomplete LCRI scope summary artifact."""
    path = output_dir / "lcri_generalization_scope_summary.csv"
    if not path.exists():
        return ["missing LCRI generalization scope summary: lcri_generalization_scope_summary.csv"]

    columns = set(pd.read_csv(path, nrows=1).columns)
    required = {
        "scope",
        "rows",
        "mean_directional_accuracy_gap",
        "max_directional_accuracy_gap",
    }
    missing = sorted(required - columns)
    if missing:
        return [f"incomplete LCRI generalization scope summary: {missing}"]
    return []


def verify_lcri_worst_generalization_context(output_dir: Path) -> list[str]:
    """Return errors for a missing or incomplete worst LCRI gap context."""
    path = output_dir / "lcri_worst_generalization_context.json"
    if not path.exists():
        return ["missing worst LCRI generalization context: lcri_worst_generalization_context.json"]

    payload = json.loads(path.read_text())
    required = {"scope", "context", "directional_accuracy_gap"}
    missing = sorted(required - set(payload))
    if missing:
        return [f"incomplete worst LCRI generalization context: {missing}"]
    return []


def verify_lcri_generalization_severity(output_dir: Path) -> list[str]:
    """Return errors for a missing or incomplete LCRI severity artifact."""
    path = output_dir / "lcri_generalization_severity.csv"
    if not path.exists():
        return ["missing LCRI generalization severity: lcri_generalization_severity.csv"]

    columns = set(pd.read_csv(path, nrows=1).columns)
    required = {"scope", "context", "directional_accuracy_gap", "severity"}
    missing = sorted(required - columns)
    if missing:
        return [f"incomplete LCRI generalization severity: {missing}"]
    return []


def verify_lcri_fragility_gate_alignment(output_dir: Path) -> list[str]:
    """Return errors for missing or incomplete LCRI fragility/gate alignment rows."""
    path = output_dir / "lcri_fragility_gate_alignment.csv"
    if not path.exists():
        return ["missing LCRI fragility gate alignment: lcri_fragility_gate_alignment.csv"]

    columns = set(pd.read_csv(path, nrows=1).columns)
    required = {
        "scope",
        "context",
        "directional_accuracy_gap",
        "severity",
        "heldout_rows",
        "heldout_directional_accuracy_se",
        "abs_gap_to_se_ratio",
        "fragility_label",
        "alignment_label",
        "review_note",
    }
    missing = sorted(required - columns)
    if missing:
        return [f"incomplete LCRI fragility gate alignment: {missing}"]

    frame = pd.read_csv(path)
    errors = []
    for row in frame.to_dict("records"):
        expected = _expected_fragility_gate_alignment_label(row["severity"], row["fragility_label"])
        if str(row["alignment_label"]) != expected:
            errors.append(
                "LCRI fragility gate alignment mismatch for "
                f"{row['scope']}:{row['context']}: expected {expected!r}, "
                f"found {row['alignment_label']!r}"
            )
    return errors


def verify_lcri_fragility_gate_scorecard(output_dir: Path) -> list[str]:
    """Return errors when LCRI fragility/gate scorecard is missing or stale."""
    path = output_dir / "lcri_fragility_gate_scorecard.json"
    alignment_path = output_dir / "lcri_fragility_gate_alignment.csv"
    if not path.exists():
        return ["missing LCRI fragility gate scorecard: lcri_fragility_gate_scorecard.json"]

    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [f"invalid LCRI fragility gate scorecard JSON: {exc.msg}"]

    required = {
        "rows",
        "aligned_rows",
        "review_required_rows",
        "gate_blocks_stable_slice_rows",
        "uncertainty_fragile_noncritical_rows",
        "uncertainty_watch_stable_gap_rows",
        "critical_rows",
        "critical_stable_slice_share",
        "max_abs_gap_to_se_ratio",
        "worst_review_context",
    }
    missing = sorted(required - set(payload))
    if missing:
        return [f"incomplete LCRI fragility gate scorecard: {missing}"]
    if not alignment_path.exists():
        return []

    alignment = pd.read_csv(alignment_path)
    required_alignment = {"scope", "context", "severity", "abs_gap_to_se_ratio", "alignment_label"}
    missing_alignment = sorted(required_alignment - set(alignment.columns))
    if missing_alignment:
        return [f"cannot verify LCRI fragility gate scorecard, missing alignment columns: {missing_alignment}"]

    labels = alignment["alignment_label"].astype(str)
    severity = alignment["severity"].astype(str)
    critical = severity == "critical"
    critical_rows = int(critical.sum())
    gate_blocks_stable = labels == "gate_blocks_stable_slice"
    review_required = labels != "aligned"
    ratio = alignment["abs_gap_to_se_ratio"].astype(float)
    review_rows = alignment.loc[review_required].copy()
    if review_rows.empty:
        worst_review_context = "none"
    else:
        review_ratio = review_rows["abs_gap_to_se_ratio"].astype(float)
        worst = review_rows.loc[review_ratio.idxmax()]
        worst_review_context = f"{worst['scope']}:{worst['context']}:{worst['alignment_label']}"

    expected = {
        "rows": len(alignment),
        "aligned_rows": int((labels == "aligned").sum()),
        "review_required_rows": int(review_required.sum()),
        "gate_blocks_stable_slice_rows": int(gate_blocks_stable.sum()),
        "uncertainty_fragile_noncritical_rows": int((labels == "uncertainty_fragile_noncritical").sum()),
        "uncertainty_watch_stable_gap_rows": int((labels == "uncertainty_watch_stable_gap").sum()),
        "critical_rows": critical_rows,
        "critical_stable_slice_share": float(gate_blocks_stable.sum() / critical_rows) if critical_rows else 0.0,
        "max_abs_gap_to_se_ratio": float(ratio.max()) if len(ratio) else 0.0,
        "worst_review_context": worst_review_context,
    }
    errors = []
    for key, expected_value in expected.items():
        found = payload.get(key)
        if isinstance(expected_value, float):
            if abs(float(found) - expected_value) > 1e-9:
                errors.append(
                    f"LCRI fragility gate scorecard mismatch for {key}: "
                    f"expected {expected_value!r}, found {found!r}"
                )
        elif found != expected_value:
            errors.append(
                f"LCRI fragility gate scorecard mismatch for {key}: "
                f"expected {expected_value!r}, found {found!r}"
            )
    return errors


def verify_lcri_generalization_severity_by_scope(output_dir: Path) -> list[str]:
    """Return errors for a missing or incomplete LCRI severity scope artifact."""
    path = output_dir / "lcri_generalization_severity_by_scope.csv"
    if not path.exists():
        return ["missing LCRI generalization severity by scope: lcri_generalization_severity_by_scope.csv"]

    columns = set(pd.read_csv(path, nrows=1).columns)
    required = {"scope", "rows", "stable_rows", "warning_rows", "critical_rows"}
    missing = sorted(required - columns)
    if missing:
        return [f"incomplete LCRI generalization severity by scope: {missing}"]
    return []


def verify_lcri_generalization_severity_consistency(output_dir: Path) -> list[str]:
    """Return errors when LCRI severity rollups disagree with row-level severity.

    The row-level severity artifact is the source of truth for LCRI release-gate
    posture. This check recomputes the global and per-scope stable/warning/
    critical counts so stale summary JSON or scope rollups cannot silently pass
    structural verification.
    """
    severity_path = output_dir / "lcri_generalization_severity.csv"
    scope_path = output_dir / "lcri_generalization_severity_by_scope.csv"
    summary_path = output_dir / "lcri_generalization_severity_summary.json"
    if not severity_path.exists() or not scope_path.exists() or not summary_path.exists():
        return []

    severity = pd.read_csv(severity_path)
    scope_rollup = pd.read_csv(scope_path)
    required_severity_columns = {"scope", "severity"}
    missing_severity_columns = sorted(required_severity_columns - set(severity.columns))
    if missing_severity_columns:
        return [
            "cannot verify LCRI severity consistency, missing severity columns: "
            f"{missing_severity_columns}"
        ]

    required_scope_columns = {"scope", "rows", "stable_rows", "warning_rows", "critical_rows"}
    missing_scope_columns = sorted(required_scope_columns - set(scope_rollup.columns))
    if missing_scope_columns:
        return [
            "cannot verify LCRI severity consistency, missing scope rollup columns: "
            f"{missing_scope_columns}"
        ]

    summary = json.loads(summary_path.read_text())
    required_summary_keys = {"rows", "stable_rows", "warning_rows", "critical_rows"}
    missing_summary_keys = sorted(required_summary_keys - set(summary))
    if missing_summary_keys:
        return [
            "cannot verify LCRI severity consistency, missing summary keys: "
            f"{missing_summary_keys}"
        ]

    errors = []
    severity_values = severity["severity"].astype(str)
    valid_severities = {"stable", "warning", "critical"}
    unknown_severities = sorted(set(severity_values) - valid_severities)
    if unknown_severities:
        errors.append(f"unknown LCRI severity values: {unknown_severities}")

    expected_summary = {
        "rows": len(severity),
        "stable_rows": int((severity_values == "stable").sum()),
        "warning_rows": int((severity_values == "warning").sum()),
        "critical_rows": int((severity_values == "critical").sum()),
    }
    for key, expected_value in expected_summary.items():
        if summary.get(key) != expected_value:
            errors.append(
                f"LCRI severity summary mismatch for {key}: "
                f"expected {expected_value!r}, found {summary.get(key)!r}"
            )

    counts_by_scope = (
        severity.assign(_severity=severity_values)
        .groupby("scope", dropna=False)["_severity"]
        .value_counts()
        .unstack(fill_value=0)
    )
    for severity_name in valid_severities:
        if severity_name not in counts_by_scope.columns:
            counts_by_scope[severity_name] = 0
    expected_scope_counts = {
        str(scope): {
            "rows": int(row.sum()),
            "stable_rows": int(row["stable"]),
            "warning_rows": int(row["warning"]),
            "critical_rows": int(row["critical"]),
        }
        for scope, row in counts_by_scope[["stable", "warning", "critical"]].iterrows()
    }
    found_scopes = {str(scope) for scope in scope_rollup["scope"]}
    expected_scopes = set(expected_scope_counts)
    if found_scopes != expected_scopes:
        errors.append(
            "LCRI severity scope rollup scope mismatch: "
            f"expected {sorted(expected_scopes)}, found {sorted(found_scopes)}"
        )

    for row in scope_rollup.to_dict(orient="records"):
        scope = str(row["scope"])
        if scope not in expected_scope_counts:
            continue
        for key, expected_value in expected_scope_counts[scope].items():
            if int(row[key]) != expected_value:
                errors.append(
                    f"LCRI severity scope rollup mismatch for {scope}.{key}: "
                    f"expected {expected_value!r}, found {int(row[key])!r}"
                )
    return errors


def verify_lcri_generalization_scope_risk(output_dir: Path) -> list[str]:
    """Return errors for a missing or incomplete LCRI scope risk artifact."""
    path = output_dir / "lcri_generalization_scope_risk.csv"
    if not path.exists():
        return ["missing LCRI scope risk: lcri_generalization_scope_risk.csv"]

    columns = set(pd.read_csv(path, nrows=1).columns)
    required = {"scope", "rows", "warning_or_critical_share", "critical_share"}
    missing = sorted(required - columns)
    if missing:
        return [f"incomplete LCRI scope risk: {missing}"]
    return []


def verify_lcri_generalization_scope_gate_decisions(output_dir: Path) -> list[str]:
    """Return errors for missing LCRI scope gate decisions."""
    path = output_dir / "lcri_generalization_scope_gate_decisions.csv"
    if not path.exists():
        return ["missing LCRI scope gate decisions: lcri_generalization_scope_gate_decisions.csv"]

    columns = set(pd.read_csv(path, nrows=1).columns)
    required = {"scope", "rows", "decision", "reason"}
    missing = sorted(required - columns)
    if missing:
        return [f"incomplete LCRI scope gate decisions: {missing}"]
    return []


def verify_lcri_generalization_scope_gate_decision_summary(output_dir: Path) -> list[str]:
    """Return errors for a missing or incomplete LCRI scope gate summary."""
    path = output_dir / "lcri_generalization_scope_gate_decision_summary.json"
    if not path.exists():
        return [
            "missing LCRI scope gate decision summary: "
            "lcri_generalization_scope_gate_decision_summary.json"
        ]

    payload = json.loads(path.read_text())
    required = {
        "scopes",
        "pass_scopes",
        "warn_scopes",
        "block_scopes",
        "blocked_scope_names",
        "warn_scope_names",
    }
    missing = sorted(required - set(payload))
    if missing:
        return [f"incomplete LCRI scope gate decision summary: {missing}"]
    return []


def verify_lcri_generalization_scope_gate_consistency(output_dir: Path) -> list[str]:
    """Return errors when LCRI scope gate CSV and summary disagree.

    Column-level checks catch missing artifacts, but release review also needs the
    compact JSON summary to faithfully mirror the decision table. This verifier
    recomputes counts and scope-name lists from the CSV and compares them to the
    dashboard-facing summary payload.
    """
    decision_path = output_dir / "lcri_generalization_scope_gate_decisions.csv"
    summary_path = output_dir / "lcri_generalization_scope_gate_decision_summary.json"
    if not decision_path.exists() or not summary_path.exists():
        return []

    decisions = pd.read_csv(decision_path)
    required_columns = {"scope", "decision"}
    missing_columns = sorted(required_columns - set(decisions.columns))
    if missing_columns:
        return [f"cannot verify LCRI scope gate consistency, missing columns: {missing_columns}"]

    summary = json.loads(summary_path.read_text())
    required_keys = {
        "scopes",
        "pass_scopes",
        "warn_scopes",
        "block_scopes",
        "blocked_scope_names",
        "warn_scope_names",
    }
    missing_keys = sorted(required_keys - set(summary))
    if missing_keys:
        return [f"cannot verify LCRI scope gate consistency, missing summary keys: {missing_keys}"]

    decision_values = decisions["decision"].astype(str)
    blocked = sorted(decisions.loc[decision_values == "block", "scope"].astype(str))
    warned = sorted(decisions.loc[decision_values == "warn", "scope"].astype(str))
    expected = {
        "scopes": len(decisions),
        "pass_scopes": int((decision_values == "pass").sum()),
        "warn_scopes": int((decision_values == "warn").sum()),
        "block_scopes": int((decision_values == "block").sum()),
        "blocked_scope_names": ",".join(blocked) if blocked else "none",
        "warn_scope_names": ",".join(warned) if warned else "none",
    }

    errors = []
    for key, expected_value in expected.items():
        if summary.get(key) != expected_value:
            errors.append(
                f"LCRI scope gate summary mismatch for {key}: "
                f"expected {expected_value!r}, found {summary.get(key)!r}"
            )
    return errors


def verify_lcri_generalization_critical_contexts(output_dir: Path) -> list[str]:
    """Return errors for a missing or incomplete LCRI critical contexts artifact."""
    path = output_dir / "lcri_generalization_critical_contexts.csv"
    if not path.exists():
        return ["missing LCRI critical contexts: lcri_generalization_critical_contexts.csv"]

    columns = set(pd.read_csv(path, nrows=1).columns)
    required = {"scope", "context", "directional_accuracy_gap", "severity"}
    missing = sorted(required - columns)
    if missing:
        return [f"incomplete LCRI critical contexts: {missing}"]
    return []


def verify_lcri_generalization_blocker_summary(output_dir: Path) -> list[str]:
    """Return errors for a missing or incomplete LCRI blocker summary."""
    path = output_dir / "lcri_generalization_blocker_summary.json"
    if not path.exists():
        return ["missing LCRI blocker summary: lcri_generalization_blocker_summary.json"]

    payload = json.loads(path.read_text())
    required = {"critical_rows", "critical_scopes", "max_critical_gap", "max_critical_context"}
    missing = sorted(required - set(payload))
    if missing:
        return [f"incomplete LCRI blocker summary: {missing}"]
    return []


def verify_lcri_generalization_severity_summary(output_dir: Path) -> list[str]:
    """Return errors for a missing or incomplete LCRI severity summary."""
    path = output_dir / "lcri_generalization_severity_summary.json"
    if not path.exists():
        return ["missing LCRI generalization severity summary: lcri_generalization_severity_summary.json"]

    payload = json.loads(path.read_text())
    required = {
        "rows",
        "stable_rows",
        "warning_rows",
        "critical_rows",
        "passes_lcri_generalization_gate",
    }
    missing = sorted(required - set(payload))
    if missing:
        return [f"incomplete LCRI generalization severity summary: {missing}"]
    return []


def verify_lcri_generalization_gate_decision(output_dir: Path) -> list[str]:
    """Return errors for a missing or incomplete LCRI gate decision."""
    path = output_dir / "lcri_generalization_gate_decision.json"
    if not path.exists():
        return ["missing LCRI generalization gate decision: lcri_generalization_gate_decision.json"]

    payload = json.loads(path.read_text())
    required = {
        "passes",
        "decision",
        "rows_evaluated",
        "warning_rows",
        "critical_rows",
        "worst_scope",
        "worst_context",
        "worst_directional_accuracy_gap",
        "reason",
    }
    missing = sorted(required - set(payload))
    if missing:
        return [f"incomplete LCRI generalization gate decision: {missing}"]
    return []


def verify_lcri_generalization_gap_delta(output_dir: Path) -> list[str]:
    """Return errors for a missing or incomplete LCRI gap delta artifact."""
    path = output_dir / "lcri_generalization_gap_delta.csv"
    if not path.exists():
        return ["missing LCRI generalization gap delta: lcri_generalization_gap_delta.csv"]

    columns = set(pd.read_csv(path, nrows=1).columns)
    required = {
        "scope",
        "context",
        "raw_imbalance_directional_accuracy_gap",
        "lcri_directional_accuracy_gap",
        "raw_minus_lcri_directional_accuracy_gap",
    }
    missing = sorted(required - columns)
    if missing:
        return [f"incomplete LCRI generalization gap delta: {missing}"]
    return []


def verify_lcri_gap_delta_dominant_scopes(output_dir: Path) -> list[str]:
    """Return errors for a missing or incomplete LCRI dominant scopes payload."""
    path = output_dir / "lcri_gap_delta_dominant_scopes.json"
    if not path.exists():
        return ["missing LCRI gap delta dominant scopes: lcri_gap_delta_dominant_scopes.json"]

    payload = json.loads(path.read_text())
    required = {
        "best_scope",
        "best_mean_raw_minus_lcri_gap",
        "worst_scope",
        "worst_mean_raw_minus_lcri_gap",
    }
    missing = sorted(required - set(payload))
    if missing:
        return [f"incomplete LCRI gap delta dominant scopes: {missing}"]
    return []


def verify_lcri_gap_delta_flags(output_dir: Path) -> list[str]:
    """Return errors for a missing or incomplete LCRI gap delta flags artifact."""
    path = output_dir / "lcri_gap_delta_flags.csv"
    if not path.exists():
        return ["missing LCRI gap delta flags: lcri_gap_delta_flags.csv"]

    columns = set(pd.read_csv(path, nrows=1).columns)
    required = {
        "scope",
        "context",
        "raw_minus_lcri_directional_accuracy_gap",
        "stability_flag",
    }
    missing = sorted(required - columns)
    if missing:
        return [f"incomplete LCRI gap delta flags: {missing}"]
    return []


def verify_lcri_gap_delta_improvements(output_dir: Path) -> list[str]:
    """Return errors for a missing LCRI gap delta improvement artifact."""
    path = output_dir / "lcri_gap_delta_improvements.csv"
    if not path.exists():
        return ["missing LCRI gap delta improvements: lcri_gap_delta_improvements.csv"]

    columns = set(pd.read_csv(path, nrows=1).columns)
    required = {"scope", "context", "raw_minus_lcri_directional_accuracy_gap"}
    missing = sorted(required - columns)
    if missing:
        return [f"incomplete LCRI gap delta improvements: {missing}"]
    return []


def verify_lcri_gap_delta_regressions(output_dir: Path) -> list[str]:
    """Return errors for a missing LCRI gap delta regression artifact."""
    path = output_dir / "lcri_gap_delta_regressions.csv"
    if not path.exists():
        return ["missing LCRI gap delta regressions: lcri_gap_delta_regressions.csv"]

    columns = set(pd.read_csv(path, nrows=1).columns)
    required = {"scope", "context", "raw_minus_lcri_directional_accuracy_gap"}
    missing = sorted(required - columns)
    if missing:
        return [f"incomplete LCRI gap delta regressions: {missing}"]
    return []


def verify_lcri_gap_delta_scorecard(output_dir: Path) -> list[str]:
    """Return errors for a missing or incomplete LCRI gap delta scorecard."""
    path = output_dir / "lcri_gap_delta_scorecard.json"
    if not path.exists():
        return ["missing LCRI gap delta scorecard: lcri_gap_delta_scorecard.json"]

    payload = json.loads(path.read_text())
    required = {
        "rows",
        "mean_raw_minus_lcri_directional_accuracy_gap",
        "median_raw_minus_lcri_directional_accuracy_gap",
        "lcri_more_stable_share",
        "lcri_less_stable_share",
    }
    missing = sorted(required - set(payload))
    if missing:
        return [f"incomplete LCRI gap delta scorecard: {missing}"]
    return []


def verify_lcri_gap_delta_scope_extremes(output_dir: Path) -> list[str]:
    """Return errors for a missing LCRI gap delta scope extremes artifact."""
    path = output_dir / "lcri_gap_delta_scope_extremes.csv"
    if not path.exists():
        return ["missing LCRI gap delta scope extremes: lcri_gap_delta_scope_extremes.csv"]

    columns = set(pd.read_csv(path, nrows=1).columns)
    required = {
        "scope",
        "best_context",
        "worst_context",
        "best_raw_minus_lcri_gap",
        "worst_raw_minus_lcri_gap",
    }
    missing = sorted(required - columns)
    if missing:
        return [f"incomplete LCRI gap delta scope extremes: {missing}"]
    return []


def verify_lcri_gap_delta_scope_summary(output_dir: Path) -> list[str]:
    """Return errors for a missing LCRI gap delta scope summary."""
    path = output_dir / "lcri_gap_delta_scope_summary.csv"
    if not path.exists():
        return ["missing LCRI gap delta scope summary: lcri_gap_delta_scope_summary.csv"]

    columns = set(pd.read_csv(path, nrows=1).columns)
    required = {
        "scope",
        "rows",
        "mean_raw_minus_lcri_gap",
        "min_raw_minus_lcri_gap",
        "max_raw_minus_lcri_gap",
        "lcri_more_stable_share",
        "lcri_less_stable_share",
    }
    missing = sorted(required - columns)
    if missing:
        return [f"incomplete LCRI gap delta scope summary: {missing}"]
    return []


def verify_lcri_gap_delta_summary(output_dir: Path) -> list[str]:
    """Return errors for a missing or incomplete LCRI gap delta summary."""
    path = output_dir / "lcri_gap_delta_summary.json"
    if not path.exists():
        return ["missing LCRI gap delta summary: lcri_gap_delta_summary.json"]

    payload = json.loads(path.read_text())
    required = {
        "rows",
        "lcri_more_stable_rows",
        "lcri_less_stable_rows",
        "lcri_equal_stability_rows",
        "max_lcri_stability_edge",
        "max_lcri_stability_edge_context",
        "max_lcri_instability_edge",
        "max_lcri_instability_edge_context",
    }
    missing = sorted(required - set(payload))
    if missing:
        return [f"incomplete LCRI gap delta summary: {missing}"]
    return []


def verify_lcri_scope_stability_contradictions(output_dir: Path) -> list[str]:
    """Return errors for missing or incomplete cross-scope contradiction rows."""
    path = output_dir / "lcri_scope_stability_contradictions.csv"
    if not path.exists():
        return ["missing LCRI scope stability contradictions: lcri_scope_stability_contradictions.csv"]

    columns = set(pd.read_csv(path, nrows=1).columns)
    required = {
        "scope",
        "decision",
        "rows",
        "lcri_more_stable_share",
        "lcri_less_stable_share",
        "fragility_review_required_rows",
        "contradiction_label",
        "review_note",
    }
    missing = sorted(required - columns)
    if missing:
        return [f"incomplete LCRI scope stability contradictions: {missing}"]
    return []


def verify_lcri_scope_stability_contradiction_summary(output_dir: Path) -> list[str]:
    """Return errors for missing or incomplete cross-scope contradiction summary."""
    path = output_dir / "lcri_scope_stability_contradiction_summary.json"
    if not path.exists():
        return [
            "missing LCRI scope stability contradiction summary: "
            "lcri_scope_stability_contradiction_summary.json"
        ]

    payload = json.loads(path.read_text())
    required = {
        "scopes",
        "aligned_scopes",
        "contradiction_scopes",
        "gate_blocks_despite_relative_stability_scopes",
        "pass_scope_with_relative_regressions_scopes",
        "warning_scope_with_broad_relative_regression_scopes",
        "fragility_review_required_rows",
        "worst_contradiction_scope",
    }
    missing = sorted(required - set(payload))
    if missing:
        return [f"incomplete LCRI scope stability contradiction summary: {missing}"]
    return []


def verify_lcri_ci_gate_contradiction_diagnostics(output_dir: Path) -> list[str]:
    """Return errors for missing or incomplete LCRI CI/gate contradiction rows."""
    path = output_dir / "lcri_ci_gate_contradiction_diagnostics.csv"
    if not path.exists():
        return [
            "missing LCRI CI gate contradiction diagnostics: "
            "lcri_ci_gate_contradiction_diagnostics.csv"
        ]

    columns = set(pd.read_csv(path, nrows=1).columns)
    required = {
        "scope",
        "context",
        "severity",
        "directional_accuracy_gap",
        "heldout_rows",
        "confidence_level",
        "heldout_directional_accuracy_ci_width",
        "ci_half_width",
        "gap_exceeds_ci_half_width",
        "ci_gate_label",
        "review_priority",
        "review_note",
    }
    missing = sorted(required - columns)
    if missing:
        return [f"incomplete LCRI CI gate contradiction diagnostics: {missing}"]
    return []


def verify_lcri_ci_gate_contradiction_summary(output_dir: Path) -> list[str]:
    """Return errors for missing or incomplete LCRI CI/gate contradiction summary."""
    path = output_dir / "lcri_ci_gate_contradiction_summary.json"
    if not path.exists():
        return [
            "missing LCRI CI gate contradiction summary: "
            "lcri_ci_gate_contradiction_summary.json"
        ]

    payload = json.loads(path.read_text())
    required = {
        "rows",
        "aligned_rows",
        "contradiction_rows",
        "gate_blocks_inside_ci_rows",
        "gate_warns_inside_ci_rows",
        "stable_gap_outside_ci_rows",
        "max_review_priority",
        "worst_ci_gate_context",
    }
    missing = sorted(required - set(payload))
    if missing:
        return [f"incomplete LCRI CI gate contradiction summary: {missing}"]
    return []


def verify_lcri_ci_gate_contradiction_consistency(output_dir: Path) -> list[str]:
    """Return errors when CI/gate contradiction artifacts disagree with sources."""
    diagnostics_path = output_dir / "lcri_ci_gate_contradiction_diagnostics.csv"
    summary_path = output_dir / "lcri_ci_gate_contradiction_summary.json"
    severity_path = output_dir / "lcri_generalization_severity.csv"
    intervals_path = output_dir / "generalization_stability_confidence_intervals.csv"
    if not all(path.exists() for path in [diagnostics_path, summary_path, severity_path, intervals_path]):
        return []

    diagnostics = pd.read_csv(diagnostics_path)
    severity = pd.read_csv(severity_path)
    intervals = pd.read_csv(intervals_path)
    expected = lcri_ci_gate_contradiction_diagnostics(severity, intervals)
    errors: list[str] = []
    _compare_lcri_delta_numeric_records(
        errors,
        label="ci_gate_contradiction_diagnostics",
        expected=expected,
        found=diagnostics,
        key_columns=["scope", "context"],
    )

    summary = json.loads(summary_path.read_text())
    expected_summary = lcri_ci_gate_contradiction_summary(expected)
    for key, expected_value in expected_summary.items():
        _append_lcri_delta_mismatch(
            errors,
            f"ci_gate_contradiction_summary.{key}",
            expected_value,
            summary.get(key),
        )
    return errors


def verify_lcri_fracture_reversal_gate(output_dir: Path) -> list[str]:
    """Return errors when the fracture/reversal release gate is stale."""
    gate_path = output_dir / "lcri_fracture_reversal_gate.json"
    reversal_path = output_dir / "lcri_reversal_stress_concentration_summary.json"
    fracture_path = output_dir / "lcri_calibration_fracture_gate.json"
    heldout_reversal_path = output_dir / "heldout_lcri_reversal_stress_concentration_summary.json"
    if not gate_path.exists():
        return ["missing LCRI fracture reversal gate: lcri_fracture_reversal_gate.json"]
    if not reversal_path.exists() or not fracture_path.exists():
        return []

    gate = json.loads(gate_path.read_text())
    reversal_summary = json.loads(reversal_path.read_text())
    fracture_gate = json.loads(fracture_path.read_text())
    heldout_reversal_summary = (
        json.loads(heldout_reversal_path.read_text()) if heldout_reversal_path.exists() else None
    )
    expected = fracture_reversal_release_gate(
        reversal_summary,
        fracture_gate,
        heldout_reversal_summary=heldout_reversal_summary,
    )
    errors: list[str] = []
    missing = sorted(set(expected) - set(gate))
    if missing:
        errors.append(f"incomplete LCRI fracture reversal gate: {missing}")
    for key, expected_value in expected.items():
        _append_lcri_gate_mismatch(errors, f"fracture_reversal_gate.{key}", expected_value, gate.get(key))
    return errors


def verify_lcri_reversal_transition_gate_consistency(output_dir: Path) -> list[str]:
    """Return errors when transition-local gates contradict the release gate."""
    release_path = output_dir / "lcri_fracture_reversal_gate.json"
    if not release_path.exists():
        return []

    release_gate = json.loads(release_path.read_text())
    release_decision = str(release_gate.get("decision", ""))
    release_active = release_decision in {"review", "block"} or release_gate.get("passes") is False
    errors: list[str] = []
    required = {
        "transition",
        "total_reversal_coupling",
        "transition_stress_share",
        "release_gate_decision",
        "transition_gate_decision",
    }
    for artifact in [
        "lcri_reversal_transition_gate.csv",
        "heldout_lcri_reversal_transition_gate.csv",
    ]:
        path = output_dir / artifact
        if not path.exists():
            continue
        table = pd.read_csv(path)
        missing = sorted(required - set(table.columns))
        if missing:
            errors.append(f"incomplete LCRI reversal transition gate {artifact}: {missing}")
            continue
        if table.empty:
            continue

        decisions = set(table["transition_gate_decision"].astype(str))
        invalid_decisions = sorted(decisions - {"pass", "review"})
        if invalid_decisions:
            errors.append(f"invalid transition gate decision in {artifact}: {invalid_decisions}")

        recorded_release = set(table["release_gate_decision"].astype(str))
        if recorded_release != {release_decision}:
            errors.append(
                f"transition gate release mismatch in {artifact}: "
                f"expected {release_decision!r}, found {sorted(recorded_release)!r}"
            )

        stress_share = table["transition_stress_share"].astype(float)
        if not np.isfinite(stress_share.to_numpy()).all():
            errors.append(f"non-finite transition stress share in {artifact}")
        if ((stress_share < 0.0) | (stress_share > 1.0)).any():
            errors.append(f"transition stress share outside [0, 1] in {artifact}")

        total_coupling = table.get("total_reversal_coupling", pd.Series(0.0, index=table.index)).astype(
            float
        )
        if not np.isfinite(total_coupling.to_numpy()).all():
            errors.append(f"non-finite transition reversal coupling in {artifact}")
        if (total_coupling < 0.0).any():
            errors.append(f"negative transition reversal coupling in {artifact}")

        expected_review = release_active & (stress_share >= 0.50) & (total_coupling > 0.0)
        found_review = table["transition_gate_decision"].astype(str) == "review"
        if (expected_review & ~found_review).any():
            errors.append(f"active release gate missing high-stress review transition in {artifact}")
        if (~expected_review & found_review).any():
            errors.append(f"transition gate review decision lacks active high-stress support in {artifact}")
        if not release_active and "review" in decisions:
            errors.append(f"inactive release gate has review transitions in {artifact}")
    return errors


def verify_lcri_ci_confidence_coverage_scorecard(output_dir: Path) -> list[str]:
    """Return errors for missing or incomplete LCRI CI coverage scorecard."""
    path = output_dir / "lcri_ci_confidence_coverage_scorecard.csv"
    if not path.exists():
        return [
            "missing LCRI CI confidence coverage scorecard: "
            "lcri_ci_confidence_coverage_scorecard.csv"
        ]

    columns = set(pd.read_csv(path, nrows=1).columns)
    required = {
        "scope",
        "rows",
        "mean_ci_width",
        "max_ci_width",
        "wide_ci_rows",
        "wide_ci_share",
        "gap_exceeds_ci_half_width_rows",
        "gap_exceeds_ci_half_width_share",
        "ci_gate_contradiction_rows",
        "high_priority_ci_gate_rows",
        "max_ci_gate_review_priority",
        "coverage_label",
        "review_note",
    }
    missing = sorted(required - columns)
    if missing:
        return [f"incomplete LCRI CI confidence coverage scorecard: {missing}"]
    return []


def verify_lcri_ci_confidence_coverage_summary(output_dir: Path) -> list[str]:
    """Return errors for missing or incomplete LCRI CI coverage summary."""
    path = output_dir / "lcri_ci_confidence_coverage_summary.json"
    if not path.exists():
        return [
            "missing LCRI CI confidence coverage summary: "
            "lcri_ci_confidence_coverage_summary.json"
        ]

    payload = json.loads(path.read_text())
    required = {
        "scopes",
        "review_scopes",
        "blocking_review_scopes",
        "contradiction_review_scopes",
        "wide_ci_review_scopes",
        "total_ci_gate_contradiction_rows",
        "total_wide_ci_rows",
        "worst_ci_confidence_scope",
    }
    missing = sorted(required - set(payload))
    if missing:
        return [f"incomplete LCRI CI confidence coverage summary: {missing}"]
    return []


def verify_lcri_ci_confidence_coverage_consistency(output_dir: Path) -> list[str]:
    """Return errors when CI confidence coverage artifacts are stale."""
    scorecard_path = output_dir / "lcri_ci_confidence_coverage_scorecard.csv"
    summary_path = output_dir / "lcri_ci_confidence_coverage_summary.json"
    intervals_path = output_dir / "generalization_stability_confidence_intervals.csv"
    diagnostics_path = output_dir / "lcri_ci_gate_contradiction_diagnostics.csv"
    if not all(path.exists() for path in [scorecard_path, summary_path, intervals_path, diagnostics_path]):
        return []

    scorecard = pd.read_csv(scorecard_path)
    intervals = pd.read_csv(intervals_path)
    diagnostics = pd.read_csv(diagnostics_path)
    expected = lcri_ci_confidence_coverage_scorecard(intervals, diagnostics)
    errors: list[str] = []
    _compare_lcri_delta_numeric_records(
        errors,
        label="ci_confidence_coverage_scorecard",
        expected=expected,
        found=scorecard,
        key_columns=["scope"],
    )

    summary = json.loads(summary_path.read_text())
    expected_summary = lcri_ci_confidence_coverage_summary(expected)
    for key, expected_value in expected_summary.items():
        _append_lcri_delta_mismatch(
            errors,
            f"ci_confidence_coverage_summary.{key}",
            expected_value,
            summary.get(key),
        )
    return errors


def verify_lcri_contradiction_review_packet(output_dir: Path) -> list[str]:
    """Return errors when the evidence-linked contradiction review packet is stale."""
    packet_path = output_dir / "lcri_contradiction_review_packet.csv"
    contradictions_path = output_dir / "lcri_scope_stability_contradictions.csv"
    severity_path = output_dir / "lcri_generalization_severity.csv"
    delta_path = output_dir / "lcri_generalization_gap_delta.csv"
    fragility_path = output_dir / "lcri_fragility_gate_alignment.csv"
    if not packet_path.exists():
        return ["missing LCRI contradiction review packet: lcri_contradiction_review_packet.csv"]
    if not all(path.exists() for path in [contradictions_path, severity_path, delta_path, fragility_path]):
        return []

    packet = pd.read_csv(packet_path)
    required = {
        "scope",
        "contradiction_label",
        "decision",
        "scope_rows",
        "lcri_less_stable_share",
        "fragility_review_required_rows",
        "worst_gate_context",
        "worst_gate_severity",
        "worst_gate_directional_accuracy_gap",
        "worst_delta_context",
        "worst_raw_minus_lcri_directional_accuracy_gap",
        "worst_fragility_context",
        "worst_fragility_alignment_label",
        "worst_fragility_abs_gap_to_se_ratio",
        "review_priority",
    }
    missing = sorted(required - set(packet.columns))
    if missing:
        return [f"incomplete LCRI contradiction review packet: {missing}"]

    contradictions = pd.read_csv(contradictions_path)
    severity = pd.read_csv(severity_path)
    delta = pd.read_csv(delta_path)
    fragility = pd.read_csv(fragility_path)
    missing_source = []
    for label, frame, columns in [
        (
            "contradictions",
            contradictions,
            {
                "scope",
                "contradiction_label",
                "decision",
                "rows",
                "lcri_less_stable_share",
                "fragility_review_required_rows",
            },
        ),
        ("severity", severity, {"scope", "context", "directional_accuracy_gap", "severity"}),
        ("gap_delta", delta, {"scope", "context", "raw_minus_lcri_directional_accuracy_gap"}),
        ("fragility", fragility, {"scope", "context", "alignment_label", "abs_gap_to_se_ratio"}),
    ]:
        missing_columns = sorted(columns - set(frame.columns))
        if missing_columns:
            missing_source.append(f"{label}: {missing_columns}")
    if missing_source:
        return [f"cannot verify LCRI contradiction review packet, missing source columns: {missing_source}"]

    expected = _expected_lcri_contradiction_review_packet(
        contradictions,
        severity,
        delta,
        fragility,
    )
    errors: list[str] = []
    _compare_lcri_delta_numeric_records(
        errors,
        label="contradiction_review_packet",
        expected=expected,
        found=packet,
        key_columns=["scope"],
    )
    return errors


def verify_lcri_contradiction_review_packet_summary(output_dir: Path) -> list[str]:
    """Return errors when the compact contradiction packet summary is stale."""
    summary_path = output_dir / "lcri_contradiction_review_packet_summary.json"
    packet_path = output_dir / "lcri_contradiction_review_packet.csv"
    if not summary_path.exists():
        return [
            "missing LCRI contradiction review packet summary: "
            "lcri_contradiction_review_packet_summary.json"
        ]
    if not packet_path.exists():
        return []

    payload = json.loads(summary_path.read_text())
    required = {
        "scopes",
        "high_priority_scopes",
        "medium_priority_scopes",
        "low_priority_scopes",
        "fragility_review_required_rows",
        "max_review_priority",
        "worst_review_scope",
        "worst_fragility_scope",
    }
    missing = sorted(required - set(payload))
    if missing:
        return [f"incomplete LCRI contradiction review packet summary: {missing}"]

    packet = pd.read_csv(packet_path)
    expected = lcri_contradiction_review_packet_summary(packet)
    errors: list[str] = []
    for key, expected_value in expected.items():
        _append_lcri_delta_mismatch(
            errors,
            f"contradiction_review_packet_summary.{key}",
            expected_value,
            payload.get(key),
        )
    return errors


def verify_lcri_uncertainty_weighted_review_priority(output_dir: Path) -> list[str]:
    """Return errors when uncertainty-weighted review priorities are incomplete."""
    path = output_dir / "lcri_uncertainty_weighted_review_priority.csv"
    if not path.exists():
        return [
            "missing LCRI uncertainty-weighted review priority: "
            "lcri_uncertainty_weighted_review_priority.csv"
        ]

    columns = set(pd.read_csv(path, nrows=1).columns)
    required = {
        "scope",
        "contradiction_label",
        "base_review_priority",
        "fragility_review_required_rows",
        "worst_fragility_abs_gap_to_se_ratio",
        "coverage_label",
        "mean_ci_width",
        "max_ci_width",
        "wide_ci_share",
        "ci_gate_contradiction_rows",
        "high_priority_ci_gate_rows",
        "uncertainty_weighted_priority",
        "priority_label",
        "review_note",
    }
    missing = sorted(required - columns)
    if missing:
        return [f"incomplete LCRI uncertainty-weighted review priority: {missing}"]
    return []


def verify_lcri_uncertainty_weighted_review_priority_summary(output_dir: Path) -> list[str]:
    """Return errors when uncertainty-weighted review priority summary is incomplete."""
    path = output_dir / "lcri_uncertainty_weighted_review_priority_summary.json"
    if not path.exists():
        return [
            "missing LCRI uncertainty-weighted review priority summary: "
            "lcri_uncertainty_weighted_review_priority_summary.json"
        ]

    payload = json.loads(path.read_text())
    required = {
        "scopes",
        "critical_priority_scopes",
        "high_priority_scopes",
        "medium_priority_scopes",
        "low_priority_scopes",
        "max_uncertainty_weighted_priority",
        "worst_uncertainty_weighted_scope",
    }
    missing = sorted(required - set(payload))
    if missing:
        return [f"incomplete LCRI uncertainty-weighted review priority summary: {missing}"]
    return []


def verify_lcri_uncertainty_weighted_review_priority_consistency(output_dir: Path) -> list[str]:
    """Return errors when uncertainty-weighted priority artifacts are stale."""
    priority_path = output_dir / "lcri_uncertainty_weighted_review_priority.csv"
    summary_path = output_dir / "lcri_uncertainty_weighted_review_priority_summary.json"
    packet_path = output_dir / "lcri_contradiction_review_packet.csv"
    scorecard_path = output_dir / "lcri_ci_confidence_coverage_scorecard.csv"
    if not all(path.exists() for path in [priority_path, summary_path, packet_path, scorecard_path]):
        return []

    priority = pd.read_csv(priority_path)
    packet = pd.read_csv(packet_path)
    scorecard = pd.read_csv(scorecard_path)
    expected = lcri_uncertainty_weighted_review_priority(packet, scorecard)
    errors: list[str] = []
    _compare_lcri_delta_numeric_records(
        errors,
        label="uncertainty_weighted_review_priority",
        expected=expected,
        found=priority,
        key_columns=["scope"],
    )

    summary = json.loads(summary_path.read_text())
    expected_summary = lcri_uncertainty_weighted_review_priority_summary(expected)
    for key, expected_value in expected_summary.items():
        _append_lcri_delta_mismatch(
            errors,
            f"uncertainty_weighted_review_priority_summary.{key}",
            expected_value,
            summary.get(key),
        )
    return errors


def verify_lcri_cross_artifact_evidence_index(output_dir: Path) -> list[str]:
    """Return errors when cross-artifact evidence index rows are incomplete."""
    path = output_dir / "lcri_cross_artifact_evidence_index.csv"
    if not path.exists():
        return ["missing LCRI cross-artifact evidence index: lcri_cross_artifact_evidence_index.csv"]

    columns = set(pd.read_csv(path, nrows=1).columns)
    required = {
        "scope",
        "gate_decision",
        "severity_rows",
        "warning_rows",
        "critical_rows",
        "lcri_more_stable_share",
        "lcri_less_stable_share",
        "fragility_review_required_rows",
        "ci_gate_contradiction_rows",
        "high_priority_ci_gate_rows",
        "contradiction_label",
        "priority_label",
        "uncertainty_weighted_priority",
        "evidence_score",
        "evidence_label",
        "review_note",
    }
    missing = sorted(required - columns)
    if missing:
        return [f"incomplete LCRI cross-artifact evidence index: {missing}"]
    return []


def verify_lcri_cross_artifact_evidence_index_summary(output_dir: Path) -> list[str]:
    """Return errors when cross-artifact evidence index summary is incomplete."""
    path = output_dir / "lcri_cross_artifact_evidence_index_summary.json"
    if not path.exists():
        return [
            "missing LCRI cross-artifact evidence index summary: "
            "lcri_cross_artifact_evidence_index_summary.json"
        ]

    payload = json.loads(path.read_text())
    required = {
        "scopes",
        "urgent_scopes",
        "review_scopes",
        "monitor_scopes",
        "aligned_scopes",
        "max_evidence_score",
        "worst_evidence_scope",
    }
    missing = sorted(required - set(payload))
    if missing:
        return [f"incomplete LCRI cross-artifact evidence index summary: {missing}"]
    return []


def verify_lcri_cross_artifact_evidence_index_consistency(output_dir: Path) -> list[str]:
    """Return errors when cross-artifact evidence artifacts are stale."""
    index_path = output_dir / "lcri_cross_artifact_evidence_index.csv"
    summary_path = output_dir / "lcri_cross_artifact_evidence_index_summary.json"
    severity_path = output_dir / "lcri_generalization_severity_by_scope.csv"
    gate_path = output_dir / "lcri_generalization_scope_gate_decisions.csv"
    delta_path = output_dir / "lcri_gap_delta_scope_summary.csv"
    contradiction_path = output_dir / "lcri_scope_stability_contradictions.csv"
    ci_path = output_dir / "lcri_ci_confidence_coverage_scorecard.csv"
    priority_path = output_dir / "lcri_uncertainty_weighted_review_priority.csv"
    expected_paths = [
        index_path,
        summary_path,
        severity_path,
        gate_path,
        delta_path,
        contradiction_path,
        ci_path,
        priority_path,
    ]
    if not all(path.exists() for path in expected_paths):
        return []

    expected = lcri_cross_artifact_evidence_index(
        pd.read_csv(severity_path),
        pd.read_csv(gate_path),
        pd.read_csv(delta_path),
        pd.read_csv(contradiction_path),
        pd.read_csv(ci_path),
        pd.read_csv(priority_path),
    )
    found = pd.read_csv(index_path)
    errors: list[str] = []
    _compare_lcri_delta_numeric_records(
        errors,
        label="cross_artifact_evidence_index",
        expected=expected,
        found=found,
        key_columns=["scope"],
    )

    summary = json.loads(summary_path.read_text())
    expected_summary = lcri_cross_artifact_evidence_index_summary(expected)
    for key, expected_value in expected_summary.items():
        _append_lcri_delta_mismatch(
            errors,
            f"cross_artifact_evidence_index_summary.{key}",
            expected_value,
            summary.get(key),
        )
    return errors


def verify_lcri_evidence_release_checklist(output_dir: Path) -> list[str]:
    """Return errors when evidence-derived release checklist rows are incomplete."""
    path = output_dir / "lcri_evidence_release_checklist.csv"
    if not path.exists():
        return ["missing LCRI evidence release checklist: lcri_evidence_release_checklist.csv"]

    columns = set(pd.read_csv(path, nrows=1).columns)
    required = {
        "scope",
        "check_status",
        "checklist_item",
        "gate_decision",
        "evidence_label",
        "evidence_score",
        "priority_label",
        "required_action",
        "source_artifact",
    }
    missing = sorted(required - columns)
    if missing:
        return [f"incomplete LCRI evidence release checklist: {missing}"]
    return []


def verify_lcri_evidence_release_checklist_summary(output_dir: Path) -> list[str]:
    """Return errors when evidence-derived release checklist summary is incomplete."""
    path = output_dir / "lcri_evidence_release_checklist_summary.json"
    if not path.exists():
        return [
            "missing LCRI evidence release checklist summary: "
            "lcri_evidence_release_checklist_summary.json"
        ]

    payload = json.loads(path.read_text())
    required = {
        "items",
        "blocked_items",
        "review_items",
        "monitor_items",
        "ready_items",
        "max_evidence_score",
        "worst_check_scope",
        "release_ready",
    }
    missing = sorted(required - set(payload))
    if missing:
        return [f"incomplete LCRI evidence release checklist summary: {missing}"]
    return []


def verify_lcri_evidence_release_checklist_consistency(output_dir: Path) -> list[str]:
    """Return errors when evidence-derived release checklist artifacts are stale."""
    index_path = output_dir / "lcri_cross_artifact_evidence_index.csv"
    checklist_path = output_dir / "lcri_evidence_release_checklist.csv"
    summary_path = output_dir / "lcri_evidence_release_checklist_summary.json"
    if not all(path.exists() for path in [index_path, checklist_path, summary_path]):
        return []

    expected = lcri_evidence_release_checklist(pd.read_csv(index_path))
    found = pd.read_csv(checklist_path)
    errors: list[str] = []
    _compare_lcri_delta_numeric_records(
        errors,
        label="evidence_release_checklist",
        expected=expected,
        found=found,
        key_columns=["scope"],
    )

    summary = json.loads(summary_path.read_text())
    expected_summary = lcri_evidence_release_checklist_summary(expected)
    for key, expected_value in expected_summary.items():
        _append_lcri_delta_mismatch(
            errors,
            f"evidence_release_checklist_summary.{key}",
            expected_value,
            summary.get(key),
        )
    return errors


def verify_lcri_owner_handoff_packet(output_dir: Path) -> list[str]:
    """Return errors when owner handoff packet rows are incomplete."""
    path = output_dir / "lcri_owner_handoff_packet.csv"
    if not path.exists():
        return ["missing LCRI owner handoff packet: lcri_owner_handoff_packet.csv"]

    columns = set(pd.read_csv(path, nrows=1).columns)
    required = {
        "scope",
        "handoff_rank",
        "handoff_status",
        "owner_queue",
        "check_status",
        "gate_decision",
        "evidence_label",
        "evidence_score",
        "priority_label",
        "critical_rows",
        "warning_rows",
        "fragility_review_required_rows",
        "ci_gate_contradiction_rows",
        "high_priority_ci_gate_rows",
        "lcri_less_stable_share",
        "required_action",
        "evidence_source_artifact",
        "checklist_source_artifact",
    }
    missing = sorted(required - columns)
    if missing:
        return [f"incomplete LCRI owner handoff packet: {missing}"]
    return []


def verify_lcri_owner_handoff_packet_summary(output_dir: Path) -> list[str]:
    """Return errors when owner handoff packet summary is incomplete."""
    path = output_dir / "lcri_owner_handoff_packet_summary.json"
    if not path.exists():
        return [
            "missing LCRI owner handoff packet summary: "
            "lcri_owner_handoff_packet_summary.json"
        ]

    payload = json.loads(path.read_text())
    required = {
        "items",
        "immediate_items",
        "review_items",
        "monitor_items",
        "signoff_items",
        "max_evidence_score",
        "top_handoff_scope",
        "handoff_clear",
    }
    missing = sorted(required - set(payload))
    if missing:
        return [f"incomplete LCRI owner handoff packet summary: {missing}"]
    return []


def verify_lcri_owner_handoff_packet_consistency(output_dir: Path) -> list[str]:
    """Return errors when owner handoff artifacts are stale."""
    index_path = output_dir / "lcri_cross_artifact_evidence_index.csv"
    checklist_path = output_dir / "lcri_evidence_release_checklist.csv"
    packet_path = output_dir / "lcri_owner_handoff_packet.csv"
    summary_path = output_dir / "lcri_owner_handoff_packet_summary.json"
    if not all(path.exists() for path in [index_path, checklist_path, packet_path, summary_path]):
        return []

    expected = lcri_owner_handoff_packet(pd.read_csv(index_path), pd.read_csv(checklist_path))
    found = pd.read_csv(packet_path)
    errors: list[str] = []
    _compare_lcri_delta_numeric_records(
        errors,
        label="owner_handoff_packet",
        expected=expected,
        found=found,
        key_columns=["scope"],
    )

    summary = json.loads(summary_path.read_text())
    expected_summary = lcri_owner_handoff_packet_summary(expected)
    for key, expected_value in expected_summary.items():
        _append_lcri_delta_mismatch(
            errors,
            f"owner_handoff_packet_summary.{key}",
            expected_value,
            summary.get(key),
        )
    return errors


def verify_lcri_owner_handoff_markdown_packet(output_dir: Path) -> list[str]:
    """Return errors when owner handoff markdown is missing or stale."""
    markdown_path = output_dir / "lcri_owner_handoff_packet.md"
    packet_path = output_dir / "lcri_owner_handoff_packet.csv"
    summary_path = output_dir / "lcri_owner_handoff_packet_summary.json"
    if not markdown_path.exists():
        return ["missing LCRI owner handoff markdown packet: lcri_owner_handoff_packet.md"]
    if not packet_path.exists() or not summary_path.exists():
        return []

    text = markdown_path.read_text()
    errors: list[str] = []
    for heading in [
        "# LCRI Owner Handoff Packet",
        "## Queue summary",
        "## Owner queue",
        "## Source artifacts",
    ]:
        if heading not in text:
            errors.append(f"missing LCRI owner handoff markdown section: {heading}")

    summary = json.loads(summary_path.read_text())
    for key, value in summary.items():
        expected_line = f"- {key}: {_format_value(value)}"
        if expected_line not in text:
            errors.append(f"stale LCRI owner handoff markdown summary field: {key}")

    packet = pd.read_csv(packet_path)
    if packet.empty:
        if "_No owner handoff rows._" not in text:
            errors.append("stale LCRI owner handoff markdown empty queue")
    else:
        for record in packet.head(12).to_dict(orient="records"):
            rank = _format_value(record.get("handoff_rank"))
            scope = _format_value(record.get("scope"))
            status = _format_value(record.get("handoff_status"))
            evidence = _format_value(record.get("evidence_score"))
            for label, value in [
                ("handoff_rank", rank),
                ("scope", scope),
                ("handoff_status", status),
                ("evidence_score", evidence),
            ]:
                if value not in text:
                    errors.append(
                        "stale LCRI owner handoff markdown row "
                        f"{scope}:{label} expected {value}"
                    )
                    break

    for artifact in [
        "lcri_owner_handoff_packet.csv",
        "lcri_owner_handoff_packet_summary.json",
        "lcri_evidence_release_checklist.csv",
        "lcri_cross_artifact_evidence_index.csv",
    ]:
        if artifact not in text:
            errors.append(f"missing LCRI owner handoff markdown source artifact: {artifact}")
    return errors


def verify_lcri_evidence_lineage_map(output_dir: Path) -> list[str]:
    """Return errors when evidence lineage rows are incomplete."""
    path = output_dir / "lcri_evidence_lineage_map.csv"
    if not path.exists():
        return ["missing LCRI evidence lineage map: lcri_evidence_lineage_map.csv"]

    columns = set(pd.read_csv(path, nrows=1).columns)
    required = {
        "scope",
        "evidence_label",
        "check_status",
        "handoff_status",
        "evidence_score",
        "evidence_source_artifact",
        "checklist_source_artifact",
        "handoff_source_artifact",
        "lineage_status",
        "lineage_note",
    }
    missing = sorted(required - columns)
    if missing:
        return [f"incomplete LCRI evidence lineage map: {missing}"]
    return []


def verify_lcri_evidence_lineage_map_summary(output_dir: Path) -> list[str]:
    """Return errors when evidence lineage summary is incomplete."""
    path = output_dir / "lcri_evidence_lineage_map_summary.json"
    if not path.exists():
        return [
            "missing LCRI evidence lineage map summary: "
            "lcri_evidence_lineage_map_summary.json"
        ]

    payload = json.loads(path.read_text())
    required = {
        "scopes",
        "complete_scopes",
        "source_mismatch_scopes",
        "incomplete_scopes",
        "max_evidence_score",
        "worst_lineage_scope",
        "lineage_clear",
    }
    missing = sorted(required - set(payload))
    if missing:
        return [f"incomplete LCRI evidence lineage map summary: {missing}"]
    return []


def verify_lcri_evidence_lineage_map_consistency(output_dir: Path) -> list[str]:
    """Return errors when evidence lineage map artifacts are stale."""
    index_path = output_dir / "lcri_cross_artifact_evidence_index.csv"
    checklist_path = output_dir / "lcri_evidence_release_checklist.csv"
    handoff_path = output_dir / "lcri_owner_handoff_packet.csv"
    lineage_path = output_dir / "lcri_evidence_lineage_map.csv"
    summary_path = output_dir / "lcri_evidence_lineage_map_summary.json"
    if not all(
        path.exists()
        for path in [index_path, checklist_path, handoff_path, lineage_path, summary_path]
    ):
        return []

    expected = lcri_evidence_lineage_map(
        pd.read_csv(index_path),
        pd.read_csv(checklist_path),
        pd.read_csv(handoff_path),
    )
    found = pd.read_csv(lineage_path)
    errors: list[str] = []
    _compare_lcri_delta_numeric_records(
        errors,
        label="evidence_lineage_map",
        expected=expected,
        found=found,
        key_columns=["scope"],
    )

    summary = json.loads(summary_path.read_text())
    expected_summary = lcri_evidence_lineage_map_summary(expected)
    for key, expected_value in expected_summary.items():
        _append_lcri_delta_mismatch(
            errors,
            f"evidence_lineage_map_summary.{key}",
            expected_value,
            summary.get(key),
        )
    return errors


def verify_lcri_generalization_gate_decision_consistency(output_dir: Path) -> list[str]:
    """Return errors when owner-facing LCRI gate payloads disagree.

    The gate decision, blocker summary, worst context, severity rows, and
    severity summary are intentionally redundant for different review surfaces.
    Recompute the owner-facing JSON fields from the row-level CSVs so a stale
    release decision cannot silently survive verification.
    """
    severity_path = output_dir / "lcri_generalization_severity.csv"
    severity_summary_path = output_dir / "lcri_generalization_severity_summary.json"
    worst_context_path = output_dir / "lcri_worst_generalization_context.json"
    gate_decision_path = output_dir / "lcri_generalization_gate_decision.json"
    critical_contexts_path = output_dir / "lcri_generalization_critical_contexts.csv"
    blocker_summary_path = output_dir / "lcri_generalization_blocker_summary.json"
    expected_paths = [
        severity_path,
        severity_summary_path,
        worst_context_path,
        gate_decision_path,
        critical_contexts_path,
        blocker_summary_path,
    ]
    if not all(path.exists() for path in expected_paths):
        return []

    severity = pd.read_csv(severity_path)
    critical_contexts = pd.read_csv(critical_contexts_path)
    required_severity_columns = {"scope", "context", "directional_accuracy_gap", "severity"}
    missing_severity_columns = sorted(required_severity_columns - set(severity.columns))
    if missing_severity_columns:
        return [
            "cannot verify LCRI gate decision consistency, missing severity columns: "
            f"{missing_severity_columns}"
        ]
    missing_critical_columns = sorted(required_severity_columns - set(critical_contexts.columns))
    if missing_critical_columns:
        return [
            "cannot verify LCRI gate decision consistency, missing critical context columns: "
            f"{missing_critical_columns}"
        ]

    errors = []
    expected_critical = severity.loc[severity["severity"].astype(str) == "critical"]
    _compare_lcri_gate_records(
        errors,
        label="critical_contexts",
        expected=expected_critical,
        found=critical_contexts,
        columns=["scope", "context", "directional_accuracy_gap", "severity"],
    )

    severity_summary = json.loads(severity_summary_path.read_text())
    worst_context = json.loads(worst_context_path.read_text())
    gate_decision = json.loads(gate_decision_path.read_text())
    blocker_summary = json.loads(blocker_summary_path.read_text())

    key_errors = []
    required_summary_keys = {"rows", "warning_rows", "critical_rows", "passes_lcri_generalization_gate"}
    missing_summary_keys = sorted(required_summary_keys - set(severity_summary))
    if missing_summary_keys:
        key_errors.append(
            "cannot verify LCRI gate decision consistency, missing severity summary keys: "
            f"{missing_summary_keys}"
        )
    required_worst_keys = {"scope", "context", "directional_accuracy_gap"}
    missing_worst_keys = sorted(required_worst_keys - set(worst_context))
    if missing_worst_keys:
        key_errors.append(
            "cannot verify LCRI gate decision consistency, missing worst context keys: "
            f"{missing_worst_keys}"
        )
    required_gate_keys = {
        "passes",
        "decision",
        "rows_evaluated",
        "warning_rows",
        "critical_rows",
        "worst_scope",
        "worst_context",
        "worst_directional_accuracy_gap",
    }
    missing_gate_keys = sorted(required_gate_keys - set(gate_decision))
    if missing_gate_keys:
        key_errors.append(
            "cannot verify LCRI gate decision consistency, missing gate decision keys: "
            f"{missing_gate_keys}"
        )
    required_blocker_keys = {"critical_rows", "critical_scopes", "max_critical_gap", "max_critical_context"}
    missing_blocker_keys = sorted(required_blocker_keys - set(blocker_summary))
    if missing_blocker_keys:
        key_errors.append(
            "cannot verify LCRI gate decision consistency, missing blocker summary keys: "
            f"{missing_blocker_keys}"
        )
    if key_errors:
        return [*errors, *key_errors]

    passes = bool(severity_summary["passes_lcri_generalization_gate"])
    expected_gate = {
        "passes": passes,
        "decision": "pass" if passes else "block",
        "rows_evaluated": int(severity_summary["rows"]),
        "warning_rows": int(severity_summary["warning_rows"]),
        "critical_rows": int(severity_summary["critical_rows"]),
        "worst_scope": str(worst_context["scope"]),
        "worst_context": str(worst_context["context"]),
        "worst_directional_accuracy_gap": float(worst_context["directional_accuracy_gap"]),
    }
    for key, expected_value in expected_gate.items():
        _append_lcri_gate_mismatch(errors, f"gate_decision.{key}", expected_value, gate_decision.get(key))

    critical_gaps = critical_contexts["directional_accuracy_gap"].astype(float)
    if critical_contexts.empty:
        expected_blocker = {
            "critical_rows": 0,
            "critical_scopes": "none",
            "max_critical_gap": 0.0,
            "max_critical_context": "none",
        }
    else:
        worst_blocker = critical_contexts.loc[critical_gaps.idxmax()]
        critical_scopes = sorted({str(scope) for scope in critical_contexts["scope"]})
        expected_blocker = {
            "critical_rows": len(critical_contexts),
            "critical_scopes": ",".join(critical_scopes),
            "max_critical_gap": float(critical_gaps.max()),
            "max_critical_context": f"{worst_blocker['scope']}:{worst_blocker['context']}",
        }
    for key, expected_value in expected_blocker.items():
        _append_lcri_gate_mismatch(errors, f"blocker_summary.{key}", expected_value, blocker_summary.get(key))

    return errors


def verify_lcri_gap_delta_consistency(output_dir: Path) -> list[str]:
    """Return errors when LCRI gap-delta dashboards disagree with source rows.

    The delta table is the source of truth for relative LCRI stability versus raw
    imbalance. Recompute compact scorecards, row partitions, scope summaries, and
    scope extremes so stale improvement/regression dashboards cannot silently pass
    structural verification.
    """
    delta_path = output_dir / "lcri_generalization_gap_delta.csv"
    scorecard_path = output_dir / "lcri_gap_delta_scorecard.json"
    summary_path = output_dir / "lcri_gap_delta_summary.json"
    flags_path = output_dir / "lcri_gap_delta_flags.csv"
    improvements_path = output_dir / "lcri_gap_delta_improvements.csv"
    regressions_path = output_dir / "lcri_gap_delta_regressions.csv"
    scope_summary_path = output_dir / "lcri_gap_delta_scope_summary.csv"
    scope_extremes_path = output_dir / "lcri_gap_delta_scope_extremes.csv"
    dominant_scopes_path = output_dir / "lcri_gap_delta_dominant_scopes.json"
    expected_paths = [
        delta_path,
        scorecard_path,
        summary_path,
        flags_path,
        improvements_path,
        regressions_path,
        scope_summary_path,
        scope_extremes_path,
        dominant_scopes_path,
    ]
    if not all(path.exists() for path in expected_paths):
        return []

    delta = pd.read_csv(delta_path)
    column = "raw_minus_lcri_directional_accuracy_gap"
    required_delta_columns = {"scope", "context", column}
    missing_delta_columns = sorted(required_delta_columns - set(delta.columns))
    if missing_delta_columns:
        return [f"cannot verify LCRI gap delta consistency, missing columns: {missing_delta_columns}"]

    values = delta[column].astype(float)
    rows = len(delta)
    more_stable = values > 0.0
    less_stable = values < 0.0
    equal_stability = values == 0.0
    errors = []

    scorecard = json.loads(scorecard_path.read_text())
    expected_scorecard = {
        "rows": rows,
        "mean_raw_minus_lcri_directional_accuracy_gap": float(values.mean()) if rows else 0.0,
        "median_raw_minus_lcri_directional_accuracy_gap": float(values.median()) if rows else 0.0,
        "lcri_more_stable_share": float(more_stable.sum() / rows) if rows else 0.0,
        "lcri_less_stable_share": float(less_stable.sum() / rows) if rows else 0.0,
    }
    for key, expected_value in expected_scorecard.items():
        _append_lcri_delta_mismatch(errors, f"scorecard.{key}", expected_value, scorecard.get(key))

    summary = json.loads(summary_path.read_text())
    if rows:
        best = delta.loc[values.idxmax()]
        worst = delta.loc[values.idxmin()]
        expected_summary = {
            "rows": rows,
            "lcri_more_stable_rows": int(more_stable.sum()),
            "lcri_less_stable_rows": int(less_stable.sum()),
            "lcri_equal_stability_rows": int(equal_stability.sum()),
            "max_lcri_stability_edge": float(values.max()),
            "max_lcri_stability_edge_context": f"{best['scope']}:{best['context']}",
            "max_lcri_instability_edge": float(values.min()),
            "max_lcri_instability_edge_context": f"{worst['scope']}:{worst['context']}",
        }
    else:
        expected_summary = {
            "rows": 0,
            "lcri_more_stable_rows": 0,
            "lcri_less_stable_rows": 0,
            "lcri_equal_stability_rows": 0,
            "max_lcri_stability_edge": 0.0,
            "max_lcri_stability_edge_context": "none",
            "max_lcri_instability_edge": 0.0,
            "max_lcri_instability_edge_context": "none",
        }
    for key, expected_value in expected_summary.items():
        _append_lcri_delta_mismatch(errors, f"summary.{key}", expected_value, summary.get(key))

    flags = pd.read_csv(flags_path)
    if "stability_flag" not in flags.columns:
        errors.append("cannot verify LCRI gap delta consistency, missing flags.stability_flag")
    elif len(flags) != rows:
        errors.append(f"LCRI gap delta mismatch for flags.rows: expected {rows!r}, found {len(flags)!r}")
    else:
        expected_flags = delta[["scope", "context"]].copy()
        expected_flags["stability_flag"] = [
            "lcri_more_stable" if value > 0.0 else "lcri_less_stable" if value < 0.0 else "lcri_equal_stability"
            for value in values
        ]
        _compare_lcri_delta_records(
            errors,
            label="flags",
            expected=expected_flags,
            found=flags,
            columns=["scope", "context", "stability_flag"],
        )

    improvements = pd.read_csv(improvements_path)
    regressions = pd.read_csv(regressions_path)
    _compare_lcri_delta_records(
        errors,
        label="improvements",
        expected=delta.loc[more_stable, ["scope", "context"]],
        found=improvements,
        columns=["scope", "context"],
    )
    _compare_lcri_delta_records(
        errors,
        label="regressions",
        expected=delta.loc[less_stable, ["scope", "context"]],
        found=regressions,
        columns=["scope", "context"],
    )

    scope_summary = pd.read_csv(scope_summary_path)
    if rows:
        expected_scope_summary = (
            delta.groupby("scope", sort=True)[column]
            .agg(
                rows="count",
                mean_raw_minus_lcri_gap="mean",
                min_raw_minus_lcri_gap="min",
                max_raw_minus_lcri_gap="max",
            )
            .reset_index()
        )
        shares = (
            delta.assign(
                lcri_more_stable=more_stable,
                lcri_less_stable=less_stable,
            )
            .groupby("scope", sort=True)[["lcri_more_stable", "lcri_less_stable"]]
            .mean()
            .reset_index()
            .rename(
                columns={
                    "lcri_more_stable": "lcri_more_stable_share",
                    "lcri_less_stable": "lcri_less_stable_share",
                }
            )
        )
        expected_scope_summary = expected_scope_summary.merge(shares, on="scope", how="left")
    else:
        expected_scope_summary = pd.DataFrame(
            columns=[
                "scope",
                "rows",
                "mean_raw_minus_lcri_gap",
                "min_raw_minus_lcri_gap",
                "max_raw_minus_lcri_gap",
                "lcri_more_stable_share",
                "lcri_less_stable_share",
            ]
        )
    _compare_lcri_delta_numeric_records(
        errors,
        label="scope_summary",
        expected=expected_scope_summary,
        found=scope_summary,
        key_columns=["scope"],
    )

    scope_extremes = pd.read_csv(scope_extremes_path)
    expected_extremes = []
    for scope, group in delta.groupby("scope", sort=True):
        group_values = group[column].astype(float)
        best = group.loc[group_values.idxmax()]
        worst = group.loc[group_values.idxmin()]
        expected_extremes.append(
            {
                "scope": scope,
                "best_context": str(best["context"]),
                "best_raw_minus_lcri_gap": float(best[column]),
                "worst_context": str(worst["context"]),
                "worst_raw_minus_lcri_gap": float(worst[column]),
            }
        )
    _compare_lcri_delta_numeric_records(
        errors,
        label="scope_extremes",
        expected=pd.DataFrame(expected_extremes),
        found=scope_extremes,
        key_columns=["scope"],
    )

    dominant = json.loads(dominant_scopes_path.read_text())
    if expected_scope_summary.empty:
        expected_dominant = {
            "best_scope": "none",
            "best_mean_raw_minus_lcri_gap": 0.0,
            "worst_scope": "none",
            "worst_mean_raw_minus_lcri_gap": 0.0,
        }
    else:
        scope_values = expected_scope_summary["mean_raw_minus_lcri_gap"].astype(float)
        best_scope = expected_scope_summary.loc[scope_values.idxmax()]
        worst_scope = expected_scope_summary.loc[scope_values.idxmin()]
        expected_dominant = {
            "best_scope": str(best_scope["scope"]),
            "best_mean_raw_minus_lcri_gap": float(best_scope["mean_raw_minus_lcri_gap"]),
            "worst_scope": str(worst_scope["scope"]),
            "worst_mean_raw_minus_lcri_gap": float(worst_scope["mean_raw_minus_lcri_gap"]),
        }
    for key, expected_value in expected_dominant.items():
        _append_lcri_delta_mismatch(errors, f"dominant_scopes.{key}", expected_value, dominant.get(key))

    return errors


def verify_lcri_scope_stability_contradictions_consistency(output_dir: Path) -> list[str]:
    """Return errors when cross-scope contradiction rows disagree with source dashboards."""
    contradictions_path = output_dir / "lcri_scope_stability_contradictions.csv"
    summary_path = output_dir / "lcri_scope_stability_contradiction_summary.json"
    scope_decisions_path = output_dir / "lcri_generalization_scope_gate_decisions.csv"
    delta_scope_summary_path = output_dir / "lcri_gap_delta_scope_summary.csv"
    fragility_alignment_path = output_dir / "lcri_fragility_gate_alignment.csv"
    expected_paths = [
        contradictions_path,
        summary_path,
        scope_decisions_path,
        delta_scope_summary_path,
        fragility_alignment_path,
    ]
    if not all(path.exists() for path in expected_paths):
        return []

    contradictions = pd.read_csv(contradictions_path)
    scope_decisions = pd.read_csv(scope_decisions_path)
    delta_scope_summary = pd.read_csv(delta_scope_summary_path)
    fragility_alignment = pd.read_csv(fragility_alignment_path)
    errors = []

    required_contradiction_columns = {
        "scope",
        "decision",
        "rows",
        "lcri_more_stable_share",
        "lcri_less_stable_share",
        "fragility_review_required_rows",
        "contradiction_label",
    }
    required_scope_columns = {"scope", "rows", "decision"}
    required_delta_columns = {"scope", "lcri_more_stable_share", "lcri_less_stable_share"}
    required_fragility_columns = {"scope", "alignment_label"}
    missing = sorted(required_contradiction_columns - set(contradictions.columns))
    if missing:
        return [f"cannot verify LCRI scope stability contradictions, missing columns: {missing}"]
    missing = sorted(required_scope_columns - set(scope_decisions.columns))
    if missing:
        return [f"cannot verify LCRI scope stability contradictions, missing scope decision columns: {missing}"]
    missing = sorted(required_delta_columns - set(delta_scope_summary.columns))
    if missing:
        return [f"cannot verify LCRI scope stability contradictions, missing delta scope columns: {missing}"]
    missing = sorted(required_fragility_columns - set(fragility_alignment.columns))
    if missing:
        return [f"cannot verify LCRI scope stability contradictions, missing fragility columns: {missing}"]

    expected = scope_decisions[["scope", "rows", "decision"]].merge(
        delta_scope_summary[["scope", "lcri_more_stable_share", "lcri_less_stable_share"]],
        on="scope",
        how="left",
    )
    review_counts = (
        fragility_alignment.assign(
            fragility_review_required=lambda frame: frame["alignment_label"].astype(str) != "aligned"
        )
        .groupby("scope", sort=True)["fragility_review_required"]
        .sum()
        .reset_index()
        .rename(columns={"fragility_review_required": "fragility_review_required_rows"})
    )
    expected = expected.merge(review_counts, on="scope", how="left")
    expected[["lcri_more_stable_share", "lcri_less_stable_share"]] = expected[
        ["lcri_more_stable_share", "lcri_less_stable_share"]
    ].fillna(0.0)
    expected["fragility_review_required_rows"] = expected[
        "fragility_review_required_rows"
    ].fillna(0).astype(int)
    expected["contradiction_label"] = [
        _expected_scope_stability_contradiction_label(row) for row in expected.to_dict("records")
    ]

    found = contradictions.set_index("scope")
    for row in expected.to_dict("records"):
        scope = str(row["scope"])
        if scope not in found.index:
            errors.append(f"missing LCRI scope stability contradiction row for scope: {scope}")
            continue
        found_row = found.loc[scope]
        for key in ["decision", "rows", "fragility_review_required_rows", "contradiction_label"]:
            _append_lcri_scope_stability_mismatch(
                errors,
                f"{scope}.{key}",
                row[key],
                found_row[key],
            )
        for key in ["lcri_more_stable_share", "lcri_less_stable_share"]:
            _append_lcri_scope_stability_mismatch(
                errors,
                f"{scope}.{key}",
                float(row[key]),
                float(found_row[key]),
            )

    summary = json.loads(summary_path.read_text())
    labels = contradictions["contradiction_label"].astype(str)
    contradiction_rows = contradictions.loc[labels != "aligned"].copy()
    if contradiction_rows.empty:
        worst_scope = "none"
    else:
        worst = contradiction_rows.sort_values(
            ["fragility_review_required_rows", "lcri_less_stable_share"],
            ascending=[False, False],
        ).iloc[0]
        worst_scope = f"{worst['scope']}:{worst['contradiction_label']}"
    expected_summary = {
        "scopes": len(contradictions),
        "aligned_scopes": int((labels == "aligned").sum()),
        "contradiction_scopes": int((labels != "aligned").sum()),
        "gate_blocks_despite_relative_stability_scopes": int(
            (labels == "gate_blocks_despite_relative_stability").sum()
        ),
        "pass_scope_with_relative_regressions_scopes": int(
            (labels == "pass_scope_with_relative_regressions").sum()
        ),
        "warning_scope_with_broad_relative_regression_scopes": int(
            (labels == "warning_scope_with_broad_relative_regression").sum()
        ),
        "fragility_review_required_rows": int(
            contradictions["fragility_review_required_rows"].astype(int).sum()
        ),
        "worst_contradiction_scope": worst_scope,
    }
    for key, expected_value in expected_summary.items():
        _append_lcri_scope_stability_mismatch(errors, f"summary.{key}", expected_value, summary.get(key))
    return errors


def summarize_artifact_metadata(metadata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Summarize manifest artifact metadata for compact audit output."""
    if not metadata:
        return {
            "artifacts_with_metadata": 0,
            "total_size_bytes": 0,
            "largest_artifact": "none",
            "largest_artifact_size_bytes": 0,
        }

    sizes = {
        artifact: int(values.get("size_bytes", 0))
        for artifact, values in metadata.items()
    }
    largest_artifact = max(sizes, key=sizes.get)
    return {
        "artifacts_with_metadata": len(metadata),
        "total_size_bytes": int(sum(sizes.values())),
        "largest_artifact": largest_artifact,
        "largest_artifact_size_bytes": sizes[largest_artifact],
    }


def summarize_verification_errors(errors: list[str]) -> dict[str, Any]:
    """Summarize report verification errors by broad artifact family."""
    families = {
        "manifest": 0,
        "generalization": 0,
        "lcri_gate": 0,
        "alpha_event": 0,
        "figures": 0,
        "other": 0,
    }
    for error in errors:
        lower = error.lower()
        if (
            "manifest" in lower
            or "metadata summary" in lower
            or "sha256" in lower
            or "size mismatch" in lower
        ):
            families["manifest"] += 1
        elif "lcri" in lower and ("gate" in lower or "severity" in lower or "blocker" in lower):
            families["lcri_gate"] += 1
        elif "alpha event" in lower:
            families["alpha_event"] += 1
        elif "figure" in lower or ".png" in lower:
            families["figures"] += 1
        elif "generalization" in lower:
            families["generalization"] += 1
        else:
            families["other"] += 1
    return {
        "errors": len(errors),
        **families,
        "passes_verification": len(errors) == 0,
    }


def write_research_summary(
    path: Path,
    *,
    rows: int,
    train_rows: int,
    heldout_rows: int,
    seed: int,
    train_frac: float,
    metrics: pd.DataFrame,
    heldout_metrics: pd.DataFrame | None = None,
    generalization_gap: pd.DataFrame | None = None,
    baseline_tail_lift_diagnostics: pd.DataFrame | None = None,
    baseline_stress_residual_drift: pd.DataFrame | None = None,
    baseline_nonlinear_extrapolation_risk: pd.DataFrame | None = None,
    baseline_nonlinear_extrapolation_risk_summary: dict[str, Any] | None = None,
    baseline_regime_publishability_summary: dict[str, Any] | None = None,
    regime_generalization_gap: pd.DataFrame | None = None,
    transition_generalization_gap: pd.DataFrame | None = None,
    generalization_fragility_diagnostics: pd.DataFrame | None = None,
    generalization_fragility_summary: dict[str, Any] | None = None,
    generalization_overview: dict[str, Any] | None = None,
    generalization_gap_leaderboard: pd.DataFrame | None = None,
    lcri_generalization_gap_leaderboard: pd.DataFrame | None = None,
    lcri_generalization_scope_summary: pd.DataFrame | None = None,
    lcri_generalization_severity: pd.DataFrame | None = None,
    lcri_fragility_gate_alignment: pd.DataFrame | None = None,
    lcri_fragility_gate_scorecard: dict[str, Any] | None = None,
    lcri_ci_gate_contradiction_diagnostics: pd.DataFrame | None = None,
    lcri_ci_gate_contradiction_summary: dict[str, Any] | None = None,
    lcri_ci_confidence_coverage_scorecard: pd.DataFrame | None = None,
    lcri_ci_confidence_coverage_summary: dict[str, Any] | None = None,
    lcri_generalization_severity_by_scope: pd.DataFrame | None = None,
    lcri_generalization_severity_summary: dict[str, Any] | None = None,
    lcri_worst_generalization_context: dict[str, Any] | None = None,
    lcri_generalization_gate_decision: dict[str, Any] | None = None,
    lcri_generalization_gap_delta: pd.DataFrame | None = None,
    lcri_gap_delta_flags: pd.DataFrame | None = None,
    lcri_gap_delta_scorecard: dict[str, Any] | None = None,
    lcri_gap_delta_summary: dict[str, Any] | None = None,
    lcri_scope_stability_contradictions: pd.DataFrame | None = None,
    lcri_scope_stability_contradiction_summary: dict[str, Any] | None = None,
    lcri_contradiction_review_packet: pd.DataFrame | None = None,
    lcri_contradiction_review_packet_summary: dict[str, Any] | None = None,
    lcri_uncertainty_weighted_review_priority: pd.DataFrame | None = None,
    lcri_uncertainty_weighted_review_priority_summary: dict[str, Any] | None = None,
    lcri_cross_artifact_evidence_index: pd.DataFrame | None = None,
    lcri_cross_artifact_evidence_index_summary: dict[str, Any] | None = None,
    lcri_evidence_release_checklist: pd.DataFrame | None = None,
    lcri_evidence_release_checklist_summary: dict[str, Any] | None = None,
    lcri_owner_handoff_packet: pd.DataFrame | None = None,
    lcri_owner_handoff_packet_summary: dict[str, Any] | None = None,
    transition_lift: pd.DataFrame | None = None,
    transition_robustness: dict[str, Any] | None = None,
    heldout_transition_lift: pd.DataFrame | None = None,
    lcri_signal_monotonicity: pd.DataFrame | None = None,
    heldout_lcri_signal_monotonicity: pd.DataFrame | None = None,
    lcri_signal_monotonicity_summary: dict[str, Any] | None = None,
    heldout_lcri_signal_monotonicity_summary: dict[str, Any] | None = None,
    lcri_calibration_curve: pd.DataFrame | None = None,
    heldout_lcri_calibration_curve: pd.DataFrame | None = None,
    lcri_calibration_gate: dict[str, Any] | None = None,
    heldout_lcri_calibration_gate: dict[str, Any] | None = None,
    lcri_calibration_fracture_pressure: pd.DataFrame | None = None,
    heldout_lcri_calibration_fracture_pressure: pd.DataFrame | None = None,
    lcri_calibration_fracture_pressure_summary: dict[str, Any] | None = None,
    heldout_lcri_calibration_fracture_pressure_summary: dict[str, Any] | None = None,
    lcri_calibration_fracture_gate: dict[str, Any] | None = None,
    lcri_reversal_stress_summary: dict[str, Any] | None = None,
    heldout_lcri_reversal_stress_summary: dict[str, Any] | None = None,
    lcri_fracture_reversal_gate: dict[str, Any] | None = None,
    lcri_reversal_transition_gate: pd.DataFrame | None = None,
    heldout_lcri_reversal_transition_gate: pd.DataFrame | None = None,
    heldout_transition_robustness: dict[str, Any] | None = None,
    alpha_event_release_review_packet: pd.DataFrame | None = None,
    alpha_event_window_regime_summary: pd.DataFrame | None = None,
    alpha_event_window_summary: dict[str, Any] | None = None,
    alpha_event_drift_gate: dict[str, Any] | None = None,
    alpha_event_review_verification_summary: dict[str, Any] | None = None,
    hidden_resiliency_asymmetry_summary: dict[str, Any] | None = None,
    heldout_hidden_resiliency_asymmetry_summary: dict[str, Any] | None = None,
    adverse_selection_phase_shift_summary: pd.DataFrame | None = None,
    heldout_adverse_selection_phase_shift_summary: pd.DataFrame | None = None,
) -> None:
    """Write a compact markdown summary of the demo artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# LCRI Research Summary",
                "",
                "## Run",
                "",
                f"- rows: {rows}",
                f"- train rows: {train_rows}",
                f"- heldout rows: {heldout_rows}",
                f"- seed: {seed}",
                f"- train fraction: {train_frac:.2f}",
                "",
                "## Signal quality",
                "",
                _markdown_table(metrics),
                "",
                "## Heldout signal quality",
                "",
                _markdown_table(heldout_metrics) if heldout_metrics is not None else "_Not generated._",
                "",
                "## Signal generalization gap",
                "",
                _markdown_table(generalization_gap)
                if generalization_gap is not None
                else "_Not generated._",
                "",
                "## Nonlinear baseline tail lift diagnostics",
                "",
                _markdown_table(baseline_tail_lift_diagnostics)
                if baseline_tail_lift_diagnostics is not None
                else "_Not generated._",
                "",
                "## Nonlinear baseline stress residual drift",
                "",
                _markdown_table(baseline_stress_residual_drift)
                if baseline_stress_residual_drift is not None
                else "_Not generated._",
                "",
                "## Nonlinear baseline extrapolation risk",
                "",
                _markdown_table(baseline_nonlinear_extrapolation_risk)
                if baseline_nonlinear_extrapolation_risk is not None
                else "_Not generated._",
                "",
                "## Nonlinear baseline extrapolation risk summary",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (baseline_nonlinear_extrapolation_risk_summary or {}).items()
                ],
                "" if baseline_nonlinear_extrapolation_risk_summary else "_Not generated._",
                "",
                "## Nonlinear baseline regime publishability",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (baseline_regime_publishability_summary or {}).items()
                ],
                "" if baseline_regime_publishability_summary else "_Not generated._",
                "",
                "## Regime generalization gap",
                "",
                _markdown_table(regime_generalization_gap)
                if regime_generalization_gap is not None
                else "_Not generated._",
                "",
                "## Transition generalization gap",
                "",
                _markdown_table(transition_generalization_gap)
                if transition_generalization_gap is not None
                else "_Not generated._",
                "",
                "## Generalization fragility diagnostics",
                "",
                _markdown_table(generalization_fragility_diagnostics)
                if generalization_fragility_diagnostics is not None
                else "_Not generated._",
                "",
                "## Generalization fragility summary",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (generalization_fragility_summary or {}).items()
                ],
                "" if generalization_fragility_summary else "_Not generated._",
                "",
                "## Generalization overview",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (generalization_overview or {}).items()
                ],
                "" if generalization_overview else "_Not generated._",
                "",
                "## Generalization gap leaderboard",
                "",
                _markdown_table(generalization_gap_leaderboard)
                if generalization_gap_leaderboard is not None
                else "_Not generated._",
                "",
                "## LCRI generalization gap leaderboard",
                "",
                _markdown_table(lcri_generalization_gap_leaderboard)
                if lcri_generalization_gap_leaderboard is not None
                else "_Not generated._",
                "",
                "## LCRI generalization scope summary",
                "",
                _markdown_table(lcri_generalization_scope_summary)
                if lcri_generalization_scope_summary is not None
                else "_Not generated._",
                "",
                "## LCRI generalization severity",
                "",
                _markdown_table(lcri_generalization_severity)
                if lcri_generalization_severity is not None
                else "_Not generated._",
                "",
                "## LCRI fragility gate alignment",
                "",
                _markdown_table(lcri_fragility_gate_alignment)
                if lcri_fragility_gate_alignment is not None
                else "_Not generated._",
                "",
                "## LCRI fragility gate scorecard",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (lcri_fragility_gate_scorecard or {}).items()
                ],
                "" if lcri_fragility_gate_scorecard else "_Not generated._",
                "",
                "## LCRI CI gate contradiction diagnostics",
                "",
                _markdown_table(lcri_ci_gate_contradiction_diagnostics)
                if lcri_ci_gate_contradiction_diagnostics is not None
                else "_Not generated._",
                "",
                "## LCRI CI gate contradiction summary",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (lcri_ci_gate_contradiction_summary or {}).items()
                ],
                "" if lcri_ci_gate_contradiction_summary else "_Not generated._",
                "",
                "## LCRI CI confidence coverage scorecard",
                "",
                _markdown_table(lcri_ci_confidence_coverage_scorecard)
                if lcri_ci_confidence_coverage_scorecard is not None
                else "_Not generated._",
                "",
                "## LCRI CI confidence coverage summary",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (lcri_ci_confidence_coverage_summary or {}).items()
                ],
                "" if lcri_ci_confidence_coverage_summary else "_Not generated._",
                "",
                "## LCRI generalization severity by scope",
                "",
                _markdown_table(lcri_generalization_severity_by_scope)
                if lcri_generalization_severity_by_scope is not None
                else "_Not generated._",
                "",
                "## LCRI generalization severity summary",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (lcri_generalization_severity_summary or {}).items()
                ],
                "" if lcri_generalization_severity_summary else "_Not generated._",
                "",
                "## LCRI worst generalization context",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (lcri_worst_generalization_context or {}).items()
                ],
                "" if lcri_worst_generalization_context else "_Not generated._",
                "",
                "## LCRI generalization gate decision",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (lcri_generalization_gate_decision or {}).items()
                ],
                "" if lcri_generalization_gate_decision else "_Not generated._",
                "",
                "## LCRI generalization gap delta",
                "",
                _markdown_table(lcri_generalization_gap_delta)
                if lcri_generalization_gap_delta is not None
                else "_Not generated._",
                "",
                "## LCRI gap delta flags",
                "",
                _markdown_table(lcri_gap_delta_flags)
                if lcri_gap_delta_flags is not None
                else "_Not generated._",
                "",
                "## LCRI gap delta scorecard",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (lcri_gap_delta_scorecard or {}).items()
                ],
                "" if lcri_gap_delta_scorecard else "_Not generated._",
                "",
                "## LCRI gap delta summary",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (lcri_gap_delta_summary or {}).items()
                ],
                "" if lcri_gap_delta_summary else "_Not generated._",
                "",
                "## LCRI scope stability contradictions",
                "",
                _markdown_table(lcri_scope_stability_contradictions)
                if lcri_scope_stability_contradictions is not None
                else "_Not generated._",
                "",
                "## LCRI scope stability contradiction summary",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (lcri_scope_stability_contradiction_summary or {}).items()
                ],
                "" if lcri_scope_stability_contradiction_summary else "_Not generated._",
                "",
                "## LCRI contradiction review packet",
                "",
                _markdown_table(lcri_contradiction_review_packet)
                if lcri_contradiction_review_packet is not None
                else "_Not generated._",
                "",
                "## LCRI contradiction review packet summary",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (lcri_contradiction_review_packet_summary or {}).items()
                ],
                "" if lcri_contradiction_review_packet_summary else "_Not generated._",
                "",
                "## Alpha event release review packet",
                "",
                _markdown_table(alpha_event_release_review_packet)
                if alpha_event_release_review_packet is not None
                else "_Not generated._",
                "",
                "## Alpha event-window row regimes",
                "",
                _markdown_table(alpha_event_window_regime_summary)
                if alpha_event_window_regime_summary is not None
                else "_Not generated._",
                "",
                "## Alpha event window summary",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (alpha_event_window_summary or {}).items()
                ],
                "" if alpha_event_window_summary else "_Not generated._",
                "",
                "## Alpha event drift gate",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (alpha_event_drift_gate or {}).items()
                ],
                "" if alpha_event_drift_gate else "_Not generated._",
                "",
                "## Alpha event review verification summary",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (alpha_event_review_verification_summary or {}).items()
                ],
                "" if alpha_event_review_verification_summary else "_Not generated._",
                "",
                "## LCRI uncertainty-weighted review priority",
                "",
                _markdown_table(lcri_uncertainty_weighted_review_priority)
                if lcri_uncertainty_weighted_review_priority is not None
                else "_Not generated._",
                "",
                "## LCRI uncertainty-weighted review priority summary",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (lcri_uncertainty_weighted_review_priority_summary or {}).items()
                ],
                "" if lcri_uncertainty_weighted_review_priority_summary else "_Not generated._",
                "",
                "## LCRI cross-artifact evidence index",
                "",
                _markdown_table(lcri_cross_artifact_evidence_index)
                if lcri_cross_artifact_evidence_index is not None
                else "_Not generated._",
                "",
                "## LCRI cross-artifact evidence index summary",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (lcri_cross_artifact_evidence_index_summary or {}).items()
                ],
                "" if lcri_cross_artifact_evidence_index_summary else "_Not generated._",
                "",
                "## LCRI evidence release checklist",
                "",
                _markdown_table(lcri_evidence_release_checklist)
                if lcri_evidence_release_checklist is not None
                else "_Not generated._",
                "",
                "## LCRI evidence release checklist summary",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (lcri_evidence_release_checklist_summary or {}).items()
                ],
                "" if lcri_evidence_release_checklist_summary else "_Not generated._",
                "",
                "## LCRI owner handoff packet",
                "",
                _markdown_table(lcri_owner_handoff_packet)
                if lcri_owner_handoff_packet is not None
                else "_Not generated._",
                "",
                "## LCRI owner handoff packet summary",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (lcri_owner_handoff_packet_summary or {}).items()
                ],
                "" if lcri_owner_handoff_packet_summary else "_Not generated._",
                "",
                "## Transition lift",
                "",
                _markdown_table(transition_lift) if transition_lift is not None else "_Not generated._",
                "",
                "## Transition robustness",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (transition_robustness or {}).items()
                ],
                "",
                "## LCRI signal monotonicity",
                "",
                _markdown_table(lcri_signal_monotonicity)
                if lcri_signal_monotonicity is not None
                else "_Not generated._",
                "",
                "## LCRI signal monotonicity summary",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (lcri_signal_monotonicity_summary or {}).items()
                ],
                "" if lcri_signal_monotonicity_summary else "_Not generated._",
                "",
                "## Heldout transition lift",
                "",
                _markdown_table(heldout_transition_lift)
                if heldout_transition_lift is not None
                else "_Not generated._",
                "",
                "## Heldout transition robustness",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (heldout_transition_robustness or {}).items()
                ],
                "",
                "## Hidden resiliency asymmetry summary",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (hidden_resiliency_asymmetry_summary or {}).items()
                ],
                "",
                "## Heldout hidden resiliency asymmetry summary",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (heldout_hidden_resiliency_asymmetry_summary or {}).items()
                ],
                "",
                "## Adverse selection phase-shift summary",
                "",
                _markdown_table(adverse_selection_phase_shift_summary)
                if adverse_selection_phase_shift_summary is not None
                else "_Not generated._",
                "",
                "## Heldout adverse selection phase-shift summary",
                "",
                _markdown_table(heldout_adverse_selection_phase_shift_summary)
                if heldout_adverse_selection_phase_shift_summary is not None
                else "_Not generated._",
                "",
                "## LCRI calibration curve",
                "",
                _markdown_table(lcri_calibration_curve)
                if lcri_calibration_curve is not None
                else "_Not generated._",
                "",
                "## LCRI calibration gate",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (lcri_calibration_gate or {}).items()
                ],
                "" if lcri_calibration_gate else "_Not generated._",
                "",
                "## LCRI calibration fracture pressure",
                "",
                _markdown_table(lcri_calibration_fracture_pressure)
                if lcri_calibration_fracture_pressure is not None
                else "_Not generated._",
                "",
                "## LCRI calibration fracture pressure summary",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (lcri_calibration_fracture_pressure_summary or {}).items()
                ],
                "" if lcri_calibration_fracture_pressure_summary else "_Not generated._",
                "",
                "## LCRI calibration fracture gate",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (lcri_calibration_fracture_gate or {}).items()
                ],
                "" if lcri_calibration_fracture_gate else "_Not generated._",
                "",
                "## LCRI reversal stress concentration summary",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (lcri_reversal_stress_summary or {}).items()
                ],
                "" if lcri_reversal_stress_summary else "_Not generated._",
                "",
                "## Heldout LCRI reversal stress concentration summary",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (heldout_lcri_reversal_stress_summary or {}).items()
                ],
                "" if heldout_lcri_reversal_stress_summary else "_Not generated._",
                "",
                "## LCRI fracture reversal gate",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (lcri_fracture_reversal_gate or {}).items()
                ],
                "" if lcri_fracture_reversal_gate else "_Not generated._",
                "",
                "## LCRI reversal transition gate",
                "",
                _markdown_table(lcri_reversal_transition_gate)
                if lcri_reversal_transition_gate is not None
                else "_Not generated._",
                "",
                "## Heldout LCRI reversal transition gate",
                "",
                _markdown_table(heldout_lcri_reversal_transition_gate)
                if heldout_lcri_reversal_transition_gate is not None
                else "_Not generated._",
                "",
                "## Heldout LCRI signal monotonicity",
                "",
                _markdown_table(heldout_lcri_signal_monotonicity)
                if heldout_lcri_signal_monotonicity is not None
                else "_Not generated._",
                "",
                "## Heldout LCRI signal monotonicity summary",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (heldout_lcri_signal_monotonicity_summary or {}).items()
                ],
                "" if heldout_lcri_signal_monotonicity_summary else "_Not generated._",
                "",
                "## Heldout LCRI calibration curve",
                "",
                _markdown_table(heldout_lcri_calibration_curve)
                if heldout_lcri_calibration_curve is not None
                else "_Not generated._",
                "",
                "## Heldout LCRI calibration gate",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (heldout_lcri_calibration_gate or {}).items()
                ],
                "" if heldout_lcri_calibration_gate else "_Not generated._",
                "",
                "## Heldout LCRI calibration fracture pressure",
                "",
                _markdown_table(heldout_lcri_calibration_fracture_pressure)
                if heldout_lcri_calibration_fracture_pressure is not None
                else "_Not generated._",
                "",
                "## Heldout LCRI calibration fracture pressure summary",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (heldout_lcri_calibration_fracture_pressure_summary or {}).items()
                ],
                "" if heldout_lcri_calibration_fracture_pressure_summary else "_Not generated._",
                "",
            ]
        )
    )


def write_lcri_owner_handoff_markdown_packet(
    path: Path,
    *,
    packet: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    """Write a compact owner-review markdown packet from handoff artifacts.

    The CSV remains the machine-readable queue. This markdown is the owner-facing
    decision surface: status counts, top scope, and the first rows that need a
    release owner to waive, fix, review, monitor, or sign off.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# LCRI Owner Handoff Packet",
        "",
        "## Queue summary",
        "",
        *[f"- {key}: {_format_value(value)}" for key, value in summary.items()],
        "",
        "## Owner queue",
        "",
    ]
    if packet.empty:
        lines.append("_No owner handoff rows._")
    else:
        columns = [
            "handoff_rank",
            "scope",
            "handoff_status",
            "check_status",
            "gate_decision",
            "evidence_label",
            "evidence_score",
            "owner_queue",
            "required_action",
        ]
        lines.append(_markdown_table(packet[[column for column in columns if column in packet.columns]].head(12)))
    lines.extend(
        [
            "",
            "## Source artifacts",
            "",
            "- lcri_owner_handoff_packet.csv",
            "- lcri_owner_handoff_packet_summary.json",
            "- lcri_evidence_release_checklist.csv",
            "- lcri_cross_artifact_evidence_index.csv",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


_RESEARCH_SUMMARY_ARTIFACT_SECTIONS = {
    "Signal quality": "metrics.csv",
    "Heldout signal quality": "heldout_metrics.csv",
    "Signal generalization gap": "generalization_gap.csv",
    "Nonlinear baseline tail lift diagnostics": "baseline_tail_lift_diagnostics.csv",
    "Nonlinear baseline stress residual drift": "baseline_stress_residual_drift.csv",
    "Nonlinear baseline extrapolation risk": "baseline_nonlinear_extrapolation_risk.csv",
    "Nonlinear baseline extrapolation risk summary": "baseline_nonlinear_extrapolation_risk_summary.json",
    "Regime generalization gap": "regime_generalization_gap.csv",
    "Transition generalization gap": "transition_generalization_gap.csv",
    "Generalization fragility diagnostics": "generalization_fragility_diagnostics.csv",
    "Generalization fragility summary": "generalization_fragility_summary.json",
    "Generalization overview": "generalization_overview.json",
    "Generalization gap leaderboard": "generalization_gap_leaderboard.csv",
    "LCRI generalization gap leaderboard": "lcri_generalization_gap_leaderboard.csv",
    "LCRI generalization scope summary": "lcri_generalization_scope_summary.csv",
    "LCRI generalization severity": "lcri_generalization_severity.csv",
    "LCRI fragility gate alignment": "lcri_fragility_gate_alignment.csv",
    "LCRI fragility gate scorecard": "lcri_fragility_gate_scorecard.json",
    "LCRI CI gate contradiction diagnostics": "lcri_ci_gate_contradiction_diagnostics.csv",
    "LCRI CI gate contradiction summary": "lcri_ci_gate_contradiction_summary.json",
    "LCRI CI confidence coverage scorecard": "lcri_ci_confidence_coverage_scorecard.csv",
    "LCRI CI confidence coverage summary": "lcri_ci_confidence_coverage_summary.json",
    "LCRI generalization severity by scope": "lcri_generalization_severity_by_scope.csv",
    "LCRI generalization severity summary": "lcri_generalization_severity_summary.json",
    "LCRI worst generalization context": "lcri_worst_generalization_context.json",
    "LCRI generalization gate decision": "lcri_generalization_gate_decision.json",
    "LCRI generalization gap delta": "lcri_generalization_gap_delta.csv",
    "LCRI gap delta flags": "lcri_gap_delta_flags.csv",
    "LCRI gap delta scorecard": "lcri_gap_delta_scorecard.json",
    "LCRI gap delta summary": "lcri_gap_delta_summary.json",
    "LCRI scope stability contradictions": "lcri_scope_stability_contradictions.csv",
    "LCRI scope stability contradiction summary": "lcri_scope_stability_contradiction_summary.json",
    "LCRI contradiction review packet": "lcri_contradiction_review_packet.csv",
    "LCRI contradiction review packet summary": "lcri_contradiction_review_packet_summary.json",
    "Alpha event release review packet": "alpha_event_release_review_packet.csv",
    "Alpha event-window row regimes": "alpha_event_window_regime_summary.csv",
    "Alpha event window summary": "alpha_event_window_summary.json",
    "Alpha event drift gate": "alpha_event_drift_gate.json",
    "Alpha event review verification summary": "alpha_event_review_verification_summary.json",
    "LCRI uncertainty-weighted review priority": "lcri_uncertainty_weighted_review_priority.csv",
    "LCRI uncertainty-weighted review priority summary": "lcri_uncertainty_weighted_review_priority_summary.json",
    "LCRI cross-artifact evidence index": "lcri_cross_artifact_evidence_index.csv",
    "LCRI cross-artifact evidence index summary": "lcri_cross_artifact_evidence_index_summary.json",
    "LCRI evidence release checklist": "lcri_evidence_release_checklist.csv",
    "LCRI evidence release checklist summary": "lcri_evidence_release_checklist_summary.json",
    "LCRI owner handoff packet": "lcri_owner_handoff_packet.csv",
    "LCRI owner handoff packet summary": "lcri_owner_handoff_packet_summary.json",
    "Transition lift": "transition_lift.csv",
    "Transition robustness": "transition_robustness.json",
    "LCRI signal monotonicity": "lcri_signal_monotonicity.csv",
    "LCRI signal monotonicity summary": "lcri_signal_monotonicity_summary.json",
    "Heldout transition lift": "heldout_transition_lift.csv",
    "Heldout transition robustness": "heldout_transition_robustness.json",
    "Hidden resiliency asymmetry summary": "hidden_resiliency_asymmetry_summary.json",
    "Heldout hidden resiliency asymmetry summary": "heldout_hidden_resiliency_asymmetry_summary.json",
    "Adverse selection phase-shift summary": "adverse_selection_phase_shift_summary.csv",
    "Heldout adverse selection phase-shift summary": "heldout_adverse_selection_phase_shift_summary.csv",
    "Heldout LCRI signal monotonicity": "heldout_lcri_signal_monotonicity.csv",
    "Heldout LCRI signal monotonicity summary": "heldout_lcri_signal_monotonicity_summary.json",
    "LCRI calibration curve": "lcri_calibration_curve.csv",
    "LCRI calibration gate": "lcri_calibration_gate.json",
    "LCRI calibration fracture pressure": "lcri_calibration_fracture_pressure.csv",
    "LCRI calibration fracture pressure summary": "lcri_calibration_fracture_pressure_summary.json",
    "LCRI calibration fracture gate": "lcri_calibration_fracture_gate.json",
    "LCRI reversal stress concentration summary": "lcri_reversal_stress_concentration_summary.json",
    "Heldout LCRI reversal stress concentration summary": "heldout_lcri_reversal_stress_concentration_summary.json",
    "LCRI fracture reversal gate": "lcri_fracture_reversal_gate.json",
    "LCRI reversal transition gate": "lcri_reversal_transition_gate.csv",
    "Heldout LCRI reversal transition gate": "heldout_lcri_reversal_transition_gate.csv",
    "Heldout LCRI calibration curve": "heldout_lcri_calibration_curve.csv",
    "Heldout LCRI calibration gate": "heldout_lcri_calibration_gate.json",
    "Heldout LCRI calibration fracture pressure": "heldout_lcri_calibration_fracture_pressure.csv",
    "Heldout LCRI calibration fracture pressure summary": "heldout_lcri_calibration_fracture_pressure_summary.json",
}


def _is_safe_artifact_path(artifact: str) -> bool:
    path = Path(artifact)
    return bool(artifact) and not path.is_absolute() and ".." not in path.parts


def _artifact_family(artifact: str) -> str:
    if artifact.startswith("figures/"):
        return "figures"
    if artifact in {"lcri-model.json", "sample_snapshots.csv"}:
        return "model_data"
    if artifact in {"research_summary.md", "artifact_manifest.json"} or artifact.startswith("artifact_"):
        return "audit"
    if artifact.startswith("lcri_gap_delta") or "gap_delta" in artifact:
        return "lcri_gap_delta"
    if artifact.startswith("lcri_ci_gate_contradiction") or artifact.startswith("lcri_ci_confidence"):
        return "lcri_gate"
    if artifact.startswith("lcri_calibration_fracture") or artifact.startswith(
        "heldout_lcri_calibration_fracture"
    ):
        return "lcri_gate"
    if artifact.startswith("lcri_reversal_stress") or artifact.startswith(
        "heldout_lcri_reversal_stress"
    ):
        return "lcri_gate"
    if artifact == "lcri_fracture_reversal_gate.json" or "reversal_transition_gate" in artifact:
        return "lcri_gate"
    if artifact in {
        "lcri_contradiction_review_packet.csv",
        "lcri_contradiction_review_packet_summary.json",
        "lcri_uncertainty_weighted_review_priority.csv",
        "lcri_uncertainty_weighted_review_priority_summary.json",
        "lcri_cross_artifact_evidence_index.csv",
        "lcri_cross_artifact_evidence_index_summary.json",
        "lcri_evidence_release_checklist.csv",
        "lcri_evidence_release_checklist_summary.json",
        "lcri_owner_handoff_packet.csv",
        "lcri_owner_handoff_packet_summary.json",
        "lcri_owner_handoff_packet.md",
    }:
        return "lcri_gate"
    if artifact.startswith("lcri_generalization") or artifact.startswith("lcri_worst"):
        if any(token in artifact for token in ["gate", "severity", "blocker", "critical", "risk", "worst"]):
            return "lcri_gate"
        return "lcri_generalization"
    if "generalization" in artifact:
        return "generalization"
    if "transition" in artifact:
        return "transition"
    if "metrics" in artifact:
        return "metrics"
    return "other"


def _artifact_verification_role(artifact: str) -> str:
    if artifact.startswith("figures/"):
        return "visual_evidence"
    if artifact in {"artifact_manifest.json", "artifact_coverage_matrix.csv"}:
        return "manifest_audit"
    if artifact.startswith("artifact_"):
        return "manifest_audit"
    if "reversal_transition_gate" in artifact:
        return "transition_verification"
    if "passive_fill_event_transition" in artifact:
        return "transition_verification"
    if artifact in {
        "transition_metrics.csv",
        "heldout_transition_metrics.csv",
        "transition_generalization_gap.csv",
        "transition_lift.csv",
        "heldout_transition_lift.csv",
        "transition_robustness.json",
        "heldout_transition_robustness.json",
    }:
        return "transition_verification"
    if artifact in {"research_summary.md", "lcri_owner_handoff_packet.md"}:
        return "owner_readiness"
    if artifact.startswith("lcri_") or artifact.startswith("heldout_lcri_"):
        return "lcri_release_evidence"
    return "supporting_evidence"


def _compare_artifact_coverage_records(
    errors: list[str],
    *,
    expected: pd.DataFrame,
    found: pd.DataFrame,
) -> None:
    columns = [
        "artifact",
        "family",
        "verification_role",
        "extension",
        "in_research_summary",
        "is_figure",
        "has_manifest_metadata",
    ]
    missing_columns = sorted(set(columns) - set(found.columns))
    if missing_columns:
        errors.append(f"incomplete artifact coverage matrix: {missing_columns}")
        return

    expected_records = sorted(
        tuple(_normalize_artifact_coverage_value(record[column]) for column in columns)
        for record in expected[columns].to_dict(orient="records")
    )
    found_records = sorted(
        tuple(_normalize_artifact_coverage_value(record[column]) for column in columns)
        for record in found[columns].to_dict(orient="records")
    )
    if expected_records != found_records:
        errors.append("artifact coverage matrix mismatch against manifest artifacts")


def _normalize_artifact_coverage_value(value: object) -> object:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    return value


def _markdown_section_body(text: str, section: str) -> str | None:
    heading = f"## {section}"
    start = text.find(heading)
    if start < 0:
        return None
    body_start = text.find("\n", start)
    if body_start < 0:
        return ""
    next_heading = text.find("\n## ", body_start + 1)
    if next_heading < 0:
        return text[body_start:].strip()
    return text[body_start:next_heading].strip()


def _verify_summary_csv_section(section: str, artifact_path: Path, body: str) -> list[str]:
    frame = pd.read_csv(artifact_path)
    if frame.empty:
        return [] if "_No rows._" in body else [f"research summary section missing empty-table marker: {section}"]

    missing_columns = [column for column in frame.columns if f"| {column}" not in body and f" {column} |" not in body]
    if missing_columns:
        return [f"research summary section missing CSV columns for {section}: {missing_columns}"]

    expected_records = [
        tuple(_format_value(record[column]) for column in frame.columns)
        for record in frame.to_dict(orient="records")
    ]
    found_records = _markdown_table_records(body, list(frame.columns))
    if found_records is None:
        return [f"research summary section has unparsable markdown table for {section}"]
    if found_records != expected_records:
        return [f"research summary section CSV values mismatch for {section}"]
    return []


def _verify_summary_json_section(section: str, artifact_path: Path, body: str) -> list[str]:
    payload = json.loads(artifact_path.read_text())
    missing_keys = [key for key in payload if f"- {key}:" not in body]
    if missing_keys:
        return [f"research summary section missing JSON keys for {section}: {missing_keys}"]

    mismatched_keys = [
        key
        for key, value in payload.items()
        if f"- {key}: {_format_value(value)}" not in body
    ]
    if mismatched_keys:
        return [f"research summary section JSON values mismatch for {section}: {mismatched_keys}"]
    return []


def _markdown_table_records(body: str, columns: list[str]) -> list[tuple[str, ...]] | None:
    lines = [line.strip() for line in body.splitlines() if line.strip().startswith("|")]
    if len(lines) < 2:
        return None

    header = _markdown_row_cells(lines[0])
    if header != columns:
        return None
    records = []
    for line in lines[2:]:
        cells = _markdown_row_cells(line)
        if len(cells) != len(columns):
            return None
        records.append(tuple(cells))
    return records


def _markdown_row_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _validate_png_bytes(data: bytes) -> str | None:
    signature = b"\x89PNG\r\n\x1a\n"
    if not data.startswith(signature):
        return "missing PNG signature"
    if len(data) < 33:
        return "truncated PNG header"
    if data[12:16] != b"IHDR":
        return "missing IHDR chunk"
    width = int.from_bytes(data[16:20], byteorder="big")
    height = int.from_bytes(data[20:24], byteorder="big")
    if width <= 0 or height <= 0:
        return f"non-positive dimensions: {width}x{height}"
    if b"IEND" not in data[-32:]:
        return "missing IEND trailer"
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _append_lcri_gate_mismatch(
    errors: list[str],
    label: str,
    expected_value: object,
    found_value: object,
) -> None:
    if isinstance(expected_value, float):
        try:
            if abs(float(found_value) - expected_value) <= 1e-12:
                return
        except (TypeError, ValueError):
            pass
    elif found_value == expected_value:
        return
    errors.append(
        f"LCRI gate decision mismatch for {label}: "
        f"expected {expected_value!r}, found {found_value!r}"
    )


def _append_lcri_delta_mismatch(
    errors: list[str],
    label: str,
    expected_value: object,
    found_value: object,
) -> None:
    if isinstance(expected_value, float):
        try:
            if abs(float(found_value) - expected_value) <= 1e-12:
                return
        except (TypeError, ValueError):
            pass
    elif found_value == expected_value:
        return
    errors.append(
        f"LCRI gap delta mismatch for {label}: "
        f"expected {expected_value!r}, found {found_value!r}"
    )


def _append_lcri_scope_stability_mismatch(
    errors: list[str],
    label: str,
    expected_value: object,
    found_value: object,
) -> None:
    if isinstance(expected_value, float):
        try:
            if abs(float(found_value) - expected_value) <= 1e-12:
                return
        except (TypeError, ValueError):
            pass
    elif str(found_value) == str(expected_value):
        return
    errors.append(
        f"LCRI scope stability contradiction mismatch for {label}: "
        f"expected {expected_value!r}, found {found_value!r}"
    )


def _append_fragility_mismatch(
    errors: list[str],
    label: str,
    expected_value: object,
    found_value: object,
) -> None:
    if isinstance(expected_value, float):
        try:
            if abs(float(found_value) - expected_value) <= 1e-12:
                return
        except (TypeError, ValueError):
            pass
    elif found_value == expected_value:
        return
    errors.append(
        f"generalization fragility mismatch for {label}: "
        f"expected {expected_value!r}, found {found_value!r}"
    )


def _expected_fragility_label(abs_gap_to_se_ratio: float) -> str:
    if abs_gap_to_se_ratio >= 3.0:
        return "fragile"
    if abs_gap_to_se_ratio >= 1.96:
        return "watch"
    return "stable"


def _expected_fragility_gate_alignment_label(severity: object, fragility: object) -> str:
    severity_label = str(severity)
    fragility_label = str(fragility)
    if severity_label == "critical" and fragility_label == "stable":
        return "gate_blocks_stable_slice"
    if severity_label in {"stable", "warning"} and fragility_label == "fragile":
        return "uncertainty_fragile_noncritical"
    if severity_label == "stable" and fragility_label == "watch":
        return "uncertainty_watch_stable_gap"
    return "aligned"


def _expected_scope_stability_contradiction_label(row: dict[str, object]) -> str:
    decision = str(row.get("decision", ""))
    more_share = float(row.get("lcri_more_stable_share", 0.0))
    less_share = float(row.get("lcri_less_stable_share", 0.0))
    if decision == "block" and more_share >= 0.5:
        return "gate_blocks_despite_relative_stability"
    if decision == "pass" and less_share > 0.0:
        return "pass_scope_with_relative_regressions"
    if decision == "warn" and less_share >= 0.5 and more_share == 0.0:
        return "warning_scope_with_broad_relative_regression"
    return "aligned"


def _expected_lcri_contradiction_review_packet(
    contradictions: pd.DataFrame,
    severity: pd.DataFrame,
    delta: pd.DataFrame,
    fragility: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "scope",
        "contradiction_label",
        "decision",
        "scope_rows",
        "lcri_less_stable_share",
        "fragility_review_required_rows",
        "worst_gate_context",
        "worst_gate_severity",
        "worst_gate_directional_accuracy_gap",
        "worst_delta_context",
        "worst_raw_minus_lcri_directional_accuracy_gap",
        "worst_fragility_context",
        "worst_fragility_alignment_label",
        "worst_fragility_abs_gap_to_se_ratio",
        "review_priority",
    ]
    if contradictions.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for contradiction in contradictions.to_dict("records"):
        scope = str(contradiction["scope"])
        gate = _expected_worst_gate_row(severity, scope)
        delta_row = _expected_worst_delta_row(delta, scope)
        fragility_row = _expected_worst_fragility_row(fragility, scope)
        rows.append(
            {
                "scope": scope,
                "contradiction_label": str(contradiction["contradiction_label"]),
                "decision": str(contradiction["decision"]),
                "scope_rows": int(contradiction["rows"]),
                "lcri_less_stable_share": float(contradiction["lcri_less_stable_share"]),
                "fragility_review_required_rows": int(
                    contradiction["fragility_review_required_rows"]
                ),
                "worst_gate_context": gate["context"],
                "worst_gate_severity": gate["severity"],
                "worst_gate_directional_accuracy_gap": gate["directional_accuracy_gap"],
                "worst_delta_context": delta_row["context"],
                "worst_raw_minus_lcri_directional_accuracy_gap": delta_row[
                    "raw_minus_lcri_directional_accuracy_gap"
                ],
                "worst_fragility_context": fragility_row["context"],
                "worst_fragility_alignment_label": fragility_row["alignment_label"],
                "worst_fragility_abs_gap_to_se_ratio": fragility_row["abs_gap_to_se_ratio"],
                "review_priority": _expected_contradiction_review_priority(
                    str(contradiction["contradiction_label"]),
                    int(contradiction["fragility_review_required_rows"]),
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        [
            "review_priority",
            "fragility_review_required_rows",
            "worst_gate_directional_accuracy_gap",
            "lcri_less_stable_share",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def _expected_worst_gate_row(severity: pd.DataFrame, scope: str) -> dict[str, float | str]:
    rows = severity.loc[severity["scope"].astype(str) == scope]
    if rows.empty:
        return {"context": "none", "severity": "none", "directional_accuracy_gap": 0.0}
    values = rows["directional_accuracy_gap"].astype(float)
    row = rows.loc[values.idxmax()]
    return {
        "context": str(row["context"]),
        "severity": str(row["severity"]),
        "directional_accuracy_gap": float(row["directional_accuracy_gap"]),
    }


def _expected_worst_delta_row(delta: pd.DataFrame, scope: str) -> dict[str, float | str]:
    rows = delta.loc[delta["scope"].astype(str) == scope]
    if rows.empty:
        return {"context": "none", "raw_minus_lcri_directional_accuracy_gap": 0.0}
    values = rows["raw_minus_lcri_directional_accuracy_gap"].astype(float)
    row = rows.loc[values.idxmin()]
    return {
        "context": str(row["context"]),
        "raw_minus_lcri_directional_accuracy_gap": float(
            row["raw_minus_lcri_directional_accuracy_gap"]
        ),
    }


def _expected_worst_fragility_row(fragility: pd.DataFrame, scope: str) -> dict[str, float | str]:
    rows = fragility.loc[fragility["scope"].astype(str) == scope]
    if rows.empty:
        return {"context": "none", "alignment_label": "none", "abs_gap_to_se_ratio": 0.0}
    review_rows = rows.loc[rows["alignment_label"].astype(str) != "aligned"]
    candidate = review_rows if not review_rows.empty else rows
    values = candidate["abs_gap_to_se_ratio"].astype(float)
    row = candidate.loc[values.idxmax()]
    return {
        "context": str(row["context"]),
        "alignment_label": str(row["alignment_label"]),
        "abs_gap_to_se_ratio": float(row["abs_gap_to_se_ratio"]),
    }


def _expected_contradiction_review_priority(contradiction_label: str, fragility_review_rows: int) -> int:
    if contradiction_label != "aligned" and fragility_review_rows > 0:
        return 3
    if contradiction_label != "aligned":
        return 2
    if fragility_review_rows > 0:
        return 1
    return 0


def _context_records(frame: pd.DataFrame) -> list[dict[str, str]]:
    return [
        {column: str(value) for column, value in record.items()}
        for record in frame.to_dict(orient="records")
    ]


def _compare_lcri_gate_records(
    errors: list[str],
    *,
    label: str,
    expected: pd.DataFrame,
    found: pd.DataFrame,
    columns: list[str],
) -> None:
    missing_columns = sorted(set(columns) - set(found.columns))
    if missing_columns:
        errors.append(f"cannot verify LCRI gate decision consistency, missing {label} columns: {missing_columns}")
        return

    expected_records = sorted(
        tuple(_normalize_lcri_record_value(record[column]) for column in columns)
        for record in expected[columns].to_dict(orient="records")
    )
    found_records = sorted(
        tuple(_normalize_lcri_record_value(record[column]) for column in columns)
        for record in found[columns].to_dict(orient="records")
    )
    if expected_records != found_records:
        errors.append(
            f"LCRI gate decision mismatch for {label}: "
            f"expected {expected_records!r}, found {found_records!r}"
        )


def _normalize_lcri_record_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def _compare_lcri_delta_records(
    errors: list[str],
    *,
    label: str,
    expected: pd.DataFrame,
    found: pd.DataFrame,
    columns: list[str],
) -> None:
    missing_columns = sorted(set(columns) - set(found.columns))
    if missing_columns:
        errors.append(f"cannot verify LCRI gap delta consistency, missing {label} columns: {missing_columns}")
        return

    expected_records = sorted(
        tuple(str(record[column]) for column in columns)
        for record in expected[columns].to_dict(orient="records")
    )
    found_records = sorted(
        tuple(str(record[column]) for column in columns)
        for record in found[columns].to_dict(orient="records")
    )
    if expected_records != found_records:
        errors.append(
            f"LCRI gap delta mismatch for {label}: "
            f"expected {expected_records!r}, found {found_records!r}"
        )


def _compare_lcri_delta_numeric_records(
    errors: list[str],
    *,
    label: str,
    expected: pd.DataFrame,
    found: pd.DataFrame,
    key_columns: list[str],
) -> None:
    if expected.empty and found.empty:
        return
    missing_columns = sorted(set(expected.columns) - set(found.columns))
    if missing_columns:
        errors.append(f"cannot verify LCRI gap delta consistency, missing {label} columns: {missing_columns}")
        return

    expected_by_key = {
        tuple(str(record[column]) for column in key_columns): record
        for record in expected.to_dict(orient="records")
    }
    found_by_key = {
        tuple(str(record[column]) for column in key_columns): record
        for record in found.to_dict(orient="records")
    }
    if set(expected_by_key) != set(found_by_key):
        errors.append(
            f"LCRI gap delta mismatch for {label} keys: "
            f"expected {sorted(expected_by_key)}, found {sorted(found_by_key)}"
        )
        return

    for key, expected_record in expected_by_key.items():
        found_record = found_by_key[key]
        for column, expected_value in expected_record.items():
            if column in key_columns:
                continue
            _append_lcri_delta_mismatch(
                errors,
                f"{label}.{':'.join(key)}.{column}",
                expected_value,
                found_record.get(column),
            )


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"

    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.to_dict("records"):
        lines.append("| " + " | ".join(_format_value(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def _format_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)
