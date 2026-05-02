from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


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
    return [artifact for artifact in artifacts if not (output_dir / artifact).exists()]


def verify_artifact_manifest(output_dir: Path, manifest: dict[str, Any]) -> list[str]:
    """Return manifest verification errors for missing or changed artifacts."""
    artifacts = manifest.get("artifacts", [])
    metadata = manifest.get("artifact_metadata", {})
    errors = [f"missing artifact: {artifact}" for artifact in missing_artifacts(output_dir, artifacts)]

    current_metadata = collect_artifact_metadata(output_dir, artifacts)
    for artifact, expected in metadata.items():
        current = current_metadata.get(artifact)
        if current is None:
            continue
        if current.get("size_bytes") != expected.get("size_bytes"):
            errors.append(f"size mismatch: {artifact}")
        if current.get("sha256") != expected.get("sha256"):
            errors.append(f"sha256 mismatch: {artifact}")
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
    regime_generalization_gap: pd.DataFrame | None = None,
    transition_generalization_gap: pd.DataFrame | None = None,
    generalization_overview: dict[str, Any] | None = None,
    generalization_gap_leaderboard: pd.DataFrame | None = None,
    lcri_generalization_gap_leaderboard: pd.DataFrame | None = None,
    lcri_generalization_scope_summary: pd.DataFrame | None = None,
    lcri_worst_generalization_context: dict[str, Any] | None = None,
    lcri_generalization_gap_delta: pd.DataFrame | None = None,
    lcri_gap_delta_flags: pd.DataFrame | None = None,
    lcri_gap_delta_summary: dict[str, Any] | None = None,
    transition_lift: pd.DataFrame,
    transition_robustness: dict[str, Any],
    heldout_transition_lift: pd.DataFrame | None = None,
    heldout_transition_robustness: dict[str, Any] | None = None,
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
                "## LCRI worst generalization context",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (lcri_worst_generalization_context or {}).items()
                ],
                "" if lcri_worst_generalization_context else "_Not generated._",
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
                "## LCRI gap delta summary",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in (lcri_gap_delta_summary or {}).items()
                ],
                "" if lcri_gap_delta_summary else "_Not generated._",
                "",
                "## Transition lift",
                "",
                _markdown_table(transition_lift),
                "",
                "## Transition robustness",
                "",
                *[
                    f"- {key}: {_format_value(value)}"
                    for key, value in transition_robustness.items()
                ],
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
            ]
        )
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
