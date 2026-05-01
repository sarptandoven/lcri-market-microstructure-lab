from __future__ import annotations

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
    artifacts: list[str],
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
        "artifacts": artifacts,
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
    transition_lift: pd.DataFrame,
    transition_robustness: dict[str, Any],
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
            ]
        )
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


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
