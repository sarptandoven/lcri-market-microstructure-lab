from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from lcri_lab.evaluation import calibration_curve


def write_figures(
    frame: pd.DataFrame,
    regime_table: pd.DataFrame,
    output_dir: Path,
    transition_table: pd.DataFrame | None = None,
    heldout_transition_table: pd.DataFrame | None = None,
    heldout_frame: pd.DataFrame | None = None,
    generalization_gap: pd.DataFrame | None = None,
    regime_generalization_gap: pd.DataFrame | None = None,
    transition_generalization_gap: pd.DataFrame | None = None,
    generalization_fragility_diagnostics: pd.DataFrame | None = None,
    generalization_stability_confidence_intervals: pd.DataFrame | None = None,
    lcri_generalization_gap_delta: pd.DataFrame | None = None,
    lcri_generalization_severity_by_scope: pd.DataFrame | None = None,
    lcri_ci_gate_contradiction_diagnostics: pd.DataFrame | None = None,
    lcri_ci_confidence_coverage_scorecard: pd.DataFrame | None = None,
    lcri_gap_delta_scope_summary: pd.DataFrame | None = None,
    lcri_gap_delta_scope_extremes: pd.DataFrame | None = None,
    lcri_contradiction_review_packet: pd.DataFrame | None = None,
    lcri_uncertainty_weighted_review_priority: pd.DataFrame | None = None,
    lcri_cross_artifact_evidence_index: pd.DataFrame | None = None,
    lcri_evidence_release_checklist: pd.DataFrame | None = None,
    lcri_owner_handoff_packet: pd.DataFrame | None = None,
    lcri_evidence_lineage_map: pd.DataFrame | None = None,
    lcri_calibration_fracture_pressure: pd.DataFrame | None = None,
    lcri_reversal_transition_gate: pd.DataFrame | None = None,
) -> None:
    if frame.empty:
        raise ValueError("cannot write figures from an empty frame")
    if regime_table.empty:
        raise ValueError("cannot write figures from an empty regime table")
    output_dir.mkdir(parents=True, exist_ok=True)
    _scatter(frame, output_dir / "raw_vs_lcri_scatter.png")
    _regime_bars(regime_table, output_dir / "regime_signal_quality.png")
    if transition_table is not None:
        _transition_bars(
            transition_table,
            output_dir / "transition_signal_quality.png",
            title="Directional accuracy around regime transitions",
        )
    if heldout_transition_table is not None:
        _transition_bars(
            heldout_transition_table,
            output_dir / "heldout_transition_signal_quality.png",
            title="Heldout directional accuracy around regime transitions",
        )
    _calibration(frame, output_dir / "calibration_curve.png", title="Calibration curve")
    if heldout_frame is not None:
        _calibration(
            heldout_frame,
            output_dir / "heldout_calibration_curve.png",
            title="Heldout calibration curve",
        )
    if generalization_gap is not None:
        _generalization_gap_bars(
            generalization_gap,
            output_dir / "generalization_gap.png",
        )
    if regime_generalization_gap is not None:
        _regime_generalization_gap_bars(
            regime_generalization_gap,
            output_dir / "regime_generalization_gap.png",
        )
    if transition_generalization_gap is not None:
        _transition_generalization_gap_bars(
            transition_generalization_gap,
            output_dir / "transition_generalization_gap.png",
        )
    if generalization_fragility_diagnostics is not None:
        _generalization_fragility_bars(
            generalization_fragility_diagnostics,
            output_dir / "generalization_fragility_diagnostics.png",
        )
    if generalization_stability_confidence_intervals is not None:
        _generalization_stability_confidence_interval_bars(
            generalization_stability_confidence_intervals,
            output_dir / "generalization_stability_confidence_intervals.png",
        )
    if lcri_generalization_gap_delta is not None:
        _lcri_generalization_gap_delta_bars(
            lcri_generalization_gap_delta,
            output_dir / "lcri_generalization_gap_delta.png",
        )
    if lcri_generalization_severity_by_scope is not None:
        _lcri_generalization_severity_scope_bars(
            lcri_generalization_severity_by_scope,
            output_dir / "lcri_generalization_severity_by_scope.png",
        )
    if lcri_ci_gate_contradiction_diagnostics is not None:
        _lcri_ci_gate_contradiction_bars(
            lcri_ci_gate_contradiction_diagnostics,
            output_dir / "lcri_ci_gate_contradiction_diagnostics.png",
        )
    if lcri_ci_confidence_coverage_scorecard is not None:
        _lcri_ci_confidence_coverage_scorecard_bars(
            lcri_ci_confidence_coverage_scorecard,
            output_dir / "lcri_ci_confidence_coverage_scorecard.png",
        )
    if lcri_gap_delta_scope_summary is not None:
        _lcri_gap_delta_scope_summary_bars(
            lcri_gap_delta_scope_summary,
            output_dir / "lcri_gap_delta_scope_summary.png",
        )
    if lcri_gap_delta_scope_extremes is not None:
        _lcri_gap_delta_scope_extremes_bars(
            lcri_gap_delta_scope_extremes,
            output_dir / "lcri_gap_delta_scope_extremes.png",
        )
    if lcri_contradiction_review_packet is not None:
        _lcri_contradiction_review_packet_bars(
            lcri_contradiction_review_packet,
            output_dir / "lcri_contradiction_review_packet.png",
        )
    if lcri_uncertainty_weighted_review_priority is not None:
        _lcri_uncertainty_weighted_review_priority_bars(
            lcri_uncertainty_weighted_review_priority,
            output_dir / "lcri_uncertainty_weighted_review_priority.png",
        )
    if lcri_cross_artifact_evidence_index is not None:
        _lcri_cross_artifact_evidence_index_bars(
            lcri_cross_artifact_evidence_index,
            output_dir / "lcri_cross_artifact_evidence_index.png",
        )
    if lcri_evidence_release_checklist is not None:
        _lcri_evidence_release_checklist_bars(
            lcri_evidence_release_checklist,
            output_dir / "lcri_evidence_release_checklist.png",
        )
    if lcri_owner_handoff_packet is not None:
        _lcri_owner_handoff_packet_bars(
            lcri_owner_handoff_packet,
            output_dir / "lcri_owner_handoff_packet.png",
        )
    if lcri_evidence_lineage_map is not None:
        _lcri_evidence_lineage_map_bars(
            lcri_evidence_lineage_map,
            output_dir / "lcri_evidence_lineage_map.png",
        )
    if lcri_calibration_fracture_pressure is not None:
        _lcri_calibration_fracture_pressure_bars(
            lcri_calibration_fracture_pressure,
            output_dir / "lcri_calibration_fracture_pressure.png",
        )
    if lcri_reversal_transition_gate is not None:
        _lcri_reversal_transition_gate_bars(
            lcri_reversal_transition_gate,
            output_dir / "lcri_reversal_transition_gate.png",
        )


def _scatter(frame: pd.DataFrame, path: Path) -> None:
    sample = frame.sample(min(len(frame), 5000), random_state=11)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(sample["raw_imbalance"], sample["lcri"], s=8, alpha=0.25)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_title("Raw imbalance vs liquidity-conditioned residual imbalance")
    ax.set_xlabel("Raw imbalance")
    ax.set_ylabel("LCRI")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _lcri_calibration_fracture_pressure_bars(table: pd.DataFrame, path: Path) -> None:
    required = {"quantile", "fracture_pressure", "calibration_residual", "pressure_label"}
    if table.empty or not required.issubset(table.columns):
        return
    plot = table.sort_values("fracture_pressure", ascending=False).head(12).copy()
    colors = plot["pressure_label"].map(
        {
            "fractured_miscalibrated": "#e45756",
            "fractured_shape_only": "#f58518",
            "aligned": "#72b7b2",
        }
    ).fillna("#4c78a8")
    labels = [f"q{quantile}" for quantile in plot["quantile"].astype(str)]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, plot["fracture_pressure"].astype(float), color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("LCRI calibration-monotonicity fracture pressure")
    ax.set_xlabel("Signal quantile")
    ax.set_ylabel("Residual-weighted fracture pressure")
    for index, residual in enumerate(plot["calibration_residual"].astype(float)):
        ax.text(index, 0.0, f"resid {residual:+.2f}", rotation=90, va="bottom", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _lcri_reversal_transition_gate_bars(table: pd.DataFrame, path: Path) -> None:
    required = {"transition", "transition_stress_share", "transition_gate_decision"}
    if table.empty or not required.issubset(table.columns):
        return
    plot = table.sort_values("transition_stress_share", ascending=False).head(12).copy()
    colors = plot["transition_gate_decision"].map(
        {"review": "#e45756", "pass": "#72b7b2"}
    ).fillna("#4c78a8")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(plot["transition"].astype(str), plot["transition_stress_share"].astype(float), color=colors)
    ax.set_title("Combined gate reversal stress at regime transitions")
    ax.set_xlabel("Transition")
    ax.set_ylabel("Share of transition reversal coupling")
    ax.set_ylim(0.0, max(1.0, float(plot["transition_stress_share"].max()) * 1.1))
    ax.tick_params(axis="x", rotation=35)
    for index, decision in enumerate(plot["transition_gate_decision"].astype(str)):
        ax.text(index, 0.02, decision, ha="center", va="bottom", fontsize=8, rotation=90)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _regime_bars(regime_table: pd.DataFrame, path: Path) -> None:
    pivot = regime_table.pivot(index="regime", columns="signal", values="directional_accuracy")
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("Directional accuracy by liquidity regime")
    ax.set_xlabel("Regime")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.legend(title="Signal")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _transition_bars(transition_table: pd.DataFrame, path: Path, *, title: str) -> None:
    pivot = transition_table.pivot(
        index="segment", columns="signal", values="directional_accuracy"
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Segment")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.legend(title="Signal")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _generalization_gap_bars(gap_table: pd.DataFrame, path: Path) -> None:
    plot_columns = [
        "directional_accuracy_gap",
        "brier_score_gap",
        "rank_correlation_gap",
    ]
    available = [column for column in plot_columns if column in gap_table.columns]
    if not available:
        return

    pivot = gap_table.set_index("signal")[available]
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Full-sample to heldout generalization gaps")
    ax.set_xlabel("Signal")
    ax.set_ylabel("Gap")
    ax.legend(title="Metric")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _regime_generalization_gap_bars(gap_table: pd.DataFrame, path: Path) -> None:
    if "directional_accuracy_gap" not in gap_table.columns:
        return

    pivot = gap_table.pivot(
        index="regime", columns="signal", values="directional_accuracy_gap"
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Directional accuracy generalization gap by regime")
    ax.set_xlabel("Regime")
    ax.set_ylabel("Full-sample minus heldout accuracy")
    ax.legend(title="Signal")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _transition_generalization_gap_bars(gap_table: pd.DataFrame, path: Path) -> None:
    if "directional_accuracy_gap" not in gap_table.columns:
        return

    pivot = gap_table.pivot(
        index="segment", columns="signal", values="directional_accuracy_gap"
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Directional accuracy generalization gap by transition segment")
    ax.set_xlabel("Segment")
    ax.set_ylabel("Full-sample minus heldout accuracy")
    ax.legend(title="Signal")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _lcri_generalization_gap_delta_bars(gap_delta: pd.DataFrame, path: Path) -> None:
    column = "raw_minus_lcri_directional_accuracy_gap"
    if column not in gap_delta.columns:
        return

    table = gap_delta.copy()
    table["label"] = table["scope"].astype(str) + ": " + table["context"].astype(str)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(table["label"], table[column])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Raw imbalance minus LCRI generalization gap")
    ax.set_xlabel("Scope")
    ax.set_ylabel("Positive means LCRI degraded less")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _generalization_fragility_bars(diagnostics: pd.DataFrame, path: Path) -> None:
    required = {"scope", "context", "signal", "abs_gap_to_se_ratio", "fragility_label"}
    if diagnostics.empty or not required.issubset(diagnostics.columns):
        return

    table = diagnostics.copy()
    table["label"] = (
        table["scope"].astype(str)
        + ": "
        + table["context"].astype(str)
        + " / "
        + table["signal"].astype(str)
    )
    table = table.sort_values("abs_gap_to_se_ratio", ascending=False).head(12)
    colors = table["fragility_label"].map(
        {"stable": "#4c78a8", "watch": "#f58518", "fragile": "#e45756"}
    ).fillna("#9d9d9d")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(table["label"], table["abs_gap_to_se_ratio"], color=colors)
    ax.axvline(1.0, color="#f58518", linestyle="--", linewidth=1.0, label="watch")
    ax.axvline(2.0, color="#e45756", linestyle="--", linewidth=1.0, label="fragile")
    ax.invert_yaxis()
    ax.set_title("Generalization fragility scaled by heldout uncertainty")
    ax.set_xlabel("Absolute accuracy gap / heldout standard error")
    ax.set_ylabel("Context")
    ax.legend(title="threshold")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _generalization_stability_confidence_interval_bars(intervals: pd.DataFrame, path: Path) -> None:
    required = {
        "scope",
        "context",
        "signal",
        "heldout_directional_accuracy",
        "heldout_directional_accuracy_ci_lower",
        "heldout_directional_accuracy_ci_upper",
        "gap_exceeds_ci_half_width",
    }
    if intervals.empty or not required.issubset(intervals.columns):
        return

    table = intervals.copy()
    table["label"] = (
        table["scope"].astype(str)
        + ": "
        + table["context"].astype(str)
        + " / "
        + table["signal"].astype(str)
    )
    table["ci_width"] = (
        table["heldout_directional_accuracy_ci_upper"].astype(float)
        - table["heldout_directional_accuracy_ci_lower"].astype(float)
    )
    table = table.sort_values(
        ["gap_exceeds_ci_half_width", "ci_width"], ascending=[False, False]
    ).head(12)
    accuracy = table["heldout_directional_accuracy"].astype(float)
    lower_error = accuracy - table["heldout_directional_accuracy_ci_lower"].astype(float)
    upper_error = table["heldout_directional_accuracy_ci_upper"].astype(float) - accuracy
    colors = table["gap_exceeds_ci_half_width"].astype(bool).map(
        {True: "#e45756", False: "#4c78a8"}
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    y_positions = range(len(table))
    ax.barh(y_positions, accuracy, xerr=[lower_error, upper_error], color=colors, alpha=0.85)
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(table["label"])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_title("Heldout directional accuracy confidence intervals")
    ax.set_xlabel("Heldout directional accuracy with confidence interval")
    ax.set_ylabel("Context")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _lcri_gap_delta_scope_extremes_bars(scope_extremes: pd.DataFrame, path: Path) -> None:
    columns = ["best_raw_minus_lcri_gap", "worst_raw_minus_lcri_gap"]
    if "scope" not in scope_extremes.columns or not set(columns).issubset(scope_extremes.columns):
        return

    plot = scope_extremes.set_index("scope")[columns]
    fig, ax = plt.subplots(figsize=(8, 5))
    plot.plot(kind="bar", ax=ax)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Best and worst LCRI stability edge by scope")
    ax.set_xlabel("Scope")
    ax.set_ylabel("Raw gap minus LCRI gap")
    ax.legend(title="Context edge")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _lcri_gap_delta_scope_summary_bars(scope_summary: pd.DataFrame, path: Path) -> None:
    column = "mean_raw_minus_lcri_gap"
    if "scope" not in scope_summary.columns or column not in scope_summary.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(scope_summary["scope"], scope_summary[column])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Mean LCRI stability edge by scope")
    ax.set_xlabel("Scope")
    ax.set_ylabel("Raw gap minus LCRI gap")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _lcri_contradiction_review_packet_bars(packet: pd.DataFrame, path: Path) -> None:
    required = {
        "scope",
        "contradiction_label",
        "review_priority",
        "worst_gate_directional_accuracy_gap",
        "worst_raw_minus_lcri_directional_accuracy_gap",
        "worst_fragility_abs_gap_to_se_ratio",
    }
    if packet.empty or not required.issubset(packet.columns):
        return

    table = packet.copy()
    table["label"] = table["scope"].astype(str) + ": " + table["contradiction_label"].astype(str)
    table = table.sort_values(
        ["review_priority", "worst_fragility_abs_gap_to_se_ratio"], ascending=[False, False]
    ).head(12)
    colors = table["review_priority"].map({3: "#e45756", 2: "#f58518", 1: "#4c78a8"}).fillna(
        "#9d9d9d"
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    y_positions = range(len(table))
    ax.barh(y_positions, table["review_priority"], color=colors, alpha=0.85)
    ax.scatter(
        table["worst_fragility_abs_gap_to_se_ratio"],
        y_positions,
        color="#000000",
        marker="D",
        label="fragility SE ratio",
        zorder=3,
    )
    ax.scatter(
        table["worst_gate_directional_accuracy_gap"].abs() * 100.0,
        y_positions,
        color="#54a24b",
        marker="o",
        label="abs gate gap x100",
        zorder=3,
    )
    ax.scatter(
        table["worst_raw_minus_lcri_directional_accuracy_gap"].abs() * 100.0,
        y_positions,
        color="#b279a2",
        marker="^",
        label="abs raw-LCRI edge x100",
        zorder=3,
    )
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(table["label"])
    ax.invert_yaxis()
    ax.set_title("LCRI stability contradiction evidence packet")
    ax.set_xlabel("Priority plus scaled evidence markers")
    ax.set_ylabel("Scope contradiction")
    ax.set_xlim(left=0.0)
    ax.legend(title="evidence")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _lcri_uncertainty_weighted_review_priority_bars(
    priorities: pd.DataFrame, path: Path
) -> None:
    required = {
        "scope",
        "priority_label",
        "uncertainty_weighted_priority",
        "base_review_priority",
        "mean_ci_width",
        "wide_ci_share",
        "ci_gate_contradiction_rows",
    }
    if priorities.empty or not required.issubset(priorities.columns):
        return

    table = priorities.copy()
    table["label"] = table["scope"].astype(str) + ": " + table["priority_label"].astype(str)
    table = table.sort_values(
        ["uncertainty_weighted_priority", "ci_gate_contradiction_rows"],
        ascending=[False, False],
    ).head(12)
    colors = table["priority_label"].map(
        {"critical": "#e45756", "high": "#f58518", "medium": "#b279a2", "low": "#4c78a8"}
    ).fillna("#9d9d9d")

    fig, ax = plt.subplots(figsize=(11, 6))
    y_positions = range(len(table))
    ax.barh(
        y_positions,
        table["uncertainty_weighted_priority"].astype(float),
        color=colors,
        alpha=0.85,
    )
    ax.scatter(
        table["base_review_priority"].astype(float),
        y_positions,
        color="#000000",
        marker="D",
        label="base priority",
        zorder=3,
    )
    ax.scatter(
        table["mean_ci_width"].astype(float) * 10.0,
        y_positions,
        color="#54a24b",
        marker="o",
        label="mean CI width x10",
        zorder=3,
    )
    ax.scatter(
        table["wide_ci_share"].astype(float) * 10.0,
        y_positions,
        color="#72b7b2",
        marker="^",
        label="wide-CI share x10",
        zorder=3,
    )
    for y_pos, rows in zip(y_positions, table["ci_gate_contradiction_rows"].astype(int)):
        ax.text(0.0, y_pos, f" c{rows} ", va="center", ha="left", color="white", fontsize=8)
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(table["label"])
    ax.invert_yaxis()
    ax.set_title("LCRI uncertainty-weighted owner review priority")
    ax.set_xlabel("Weighted priority with scaled uncertainty markers")
    ax.set_ylabel("Review scope")
    ax.set_xlim(left=0.0)
    ax.legend(title="evidence")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _lcri_cross_artifact_evidence_index_bars(evidence_index: pd.DataFrame, path: Path) -> None:
    required = {
        "scope",
        "evidence_label",
        "evidence_score",
        "critical_rows",
        "gate_decision",
        "contradiction_label",
        "uncertainty_weighted_priority",
        "ci_gate_contradiction_rows",
    }
    if evidence_index.empty or not required.issubset(evidence_index.columns):
        return

    table = evidence_index.copy()
    table["label"] = table["scope"].astype(str) + ": " + table["evidence_label"].astype(str)
    table = table.sort_values(
        ["evidence_score", "critical_rows", "ci_gate_contradiction_rows"],
        ascending=[False, False, False],
    ).head(12)
    colors = table["evidence_label"].map(
        {"urgent": "#e45756", "review": "#f58518", "monitor": "#b279a2", "aligned": "#4c78a8"}
    ).fillna("#9d9d9d")

    fig, ax = plt.subplots(figsize=(11, 6))
    y_positions = range(len(table))
    ax.barh(
        y_positions,
        table["evidence_score"].astype(float),
        color=colors,
        alpha=0.85,
    )
    ax.scatter(
        table["uncertainty_weighted_priority"].astype(float),
        y_positions,
        color="#000000",
        marker="D",
        label="uncertainty priority",
        zorder=3,
    )
    ax.scatter(
        table["critical_rows"].astype(float),
        y_positions,
        color="#54a24b",
        marker="o",
        label="critical gate rows",
        zorder=3,
    )
    ax.scatter(
        table["ci_gate_contradiction_rows"].astype(float),
        y_positions,
        color="#72b7b2",
        marker="^",
        label="CI/gate contradictions",
        zorder=3,
    )
    for y_pos, blocked in zip(y_positions, table["gate_decision"].astype(str)):
        if blocked == "block":
            ax.text(
                0.0,
                y_pos,
                " block ",
                va="center",
                ha="left",
                color="white",
                fontsize=8,
            )
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(table["label"])
    ax.invert_yaxis()
    ax.set_title("LCRI cross-artifact evidence index")
    ax.set_xlabel("Evidence score with gate, CI, and uncertainty markers")
    ax.set_ylabel("Review scope")
    ax.set_xlim(left=0.0)
    ax.legend(title="evidence")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _lcri_evidence_release_checklist_bars(checklist: pd.DataFrame, path: Path) -> None:
    required = {
        "scope",
        "check_status",
        "evidence_label",
        "evidence_score",
        "required_action",
    }
    if checklist.empty or not required.issubset(checklist.columns):
        return

    table = checklist.copy()
    table["label"] = table["scope"].astype(str) + ": " + table["check_status"].astype(str)
    table["blocker_marker"] = (table["check_status"].astype(str) == "blocked").astype(int)
    table = table.sort_values(
        ["blocker_marker", "evidence_score", "scope"], ascending=[False, False, True]
    ).head(12)
    colors = table["check_status"].map(
        {
            "blocked": "#e45756",
            "needs_review": "#f58518",
            "monitor": "#b279a2",
            "ready": "#54a24b",
        }
    ).fillna("#9d9d9d")

    fig, ax = plt.subplots(figsize=(11, 6))
    y_positions = range(len(table))
    ax.barh(y_positions, table["evidence_score"].astype(float), color=colors, alpha=0.85)
    ax.scatter(
        table["blocker_marker"].astype(float),
        y_positions,
        color="#000000",
        marker="D",
        label="release blocker",
        zorder=3,
    )
    for y_pos, action in zip(y_positions, table["required_action"].astype(str)):
        ax.text(
            table["evidence_score"].astype(float).max() * 0.02,
            y_pos,
            action[:64],
            va="center",
            ha="left",
            color="white",
            fontsize=8,
        )
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(table["label"])
    ax.invert_yaxis()
    ax.set_title("LCRI evidence release checklist")
    ax.set_xlabel("Evidence score with release-blocker marker")
    ax.set_ylabel("Owner checklist scope")
    ax.set_xlim(left=0.0)
    ax.legend(title="release signal")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _lcri_owner_handoff_packet_bars(packet: pd.DataFrame, path: Path) -> None:
    required = {
        "scope",
        "handoff_rank",
        "handoff_status",
        "owner_queue",
        "evidence_score",
        "high_priority_ci_gate_rows",
    }
    if packet.empty or not required.issubset(packet.columns):
        return

    table = packet.copy()
    table["label"] = (
        table["handoff_rank"].astype(str)
        + ". "
        + table["scope"].astype(str)
        + ": "
        + table["handoff_status"].astype(str)
    )
    table["release_blocker_marker"] = (table["check_status"].astype(str) == "blocked").astype(int)
    table = table.sort_values("handoff_rank", ascending=True).head(12)
    colors = table["handoff_status"].map(
        {
            "immediate_owner_decision": "#e45756",
            "owner_review": "#f58518",
            "release_note_monitor": "#b279a2",
            "signoff_ready": "#54a24b",
        }
    ).fillna("#9d9d9d")

    fig, ax = plt.subplots(figsize=(12, 6))
    y_positions = range(len(table))
    ax.barh(y_positions, table["evidence_score"].astype(float), color=colors, alpha=0.85)
    ax.scatter(
        table["high_priority_ci_gate_rows"].astype(float),
        y_positions,
        color="#000000",
        marker="D",
        label="high-priority CI/gate rows",
        zorder=3,
    )
    ax.scatter(
        table["release_blocker_marker"].astype(float),
        y_positions,
        color="#4c78a8",
        marker="^",
        label="release blocker",
        zorder=3,
    )
    for y_pos, queue in zip(y_positions, table["owner_queue"].astype(str)):
        ax.text(
            table["evidence_score"].astype(float).max() * 0.02,
            y_pos,
            queue[:72],
            va="center",
            ha="left",
            color="white",
            fontsize=8,
        )
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(table["label"])
    ax.invert_yaxis()
    ax.set_title("LCRI owner handoff packet")
    ax.set_xlabel("Evidence score with CI/gate and blocker markers")
    ax.set_ylabel("Owner queue rank")
    ax.set_xlim(left=0.0)
    ax.legend(title="handoff signal")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _lcri_evidence_lineage_map_bars(lineage_map: pd.DataFrame, path: Path) -> None:
    required = {
        "scope",
        "lineage_status",
        "evidence_score",
        "evidence_label",
        "check_status",
        "handoff_status",
    }
    if lineage_map.empty or not required.issubset(lineage_map.columns):
        return

    table = lineage_map.copy()
    table["label"] = table["scope"].astype(str) + ": " + table["lineage_status"].astype(str)
    table["incomplete_marker"] = (
        table["lineage_status"].astype(str) == "incomplete_lineage"
    ).astype(int)
    table["mismatch_marker"] = (table["lineage_status"].astype(str) == "source_mismatch").astype(int)
    table = table.sort_values(
        ["mismatch_marker", "incomplete_marker", "evidence_score", "scope"],
        ascending=[False, False, False, True],
    ).head(12)
    colors = table["lineage_status"].map(
        {
            "source_mismatch": "#e45756",
            "incomplete_lineage": "#f58518",
            "complete": "#54a24b",
        }
    ).fillna("#9d9d9d")

    fig, ax = plt.subplots(figsize=(12, 6))
    y_positions = range(len(table))
    ax.barh(y_positions, table["evidence_score"].astype(float), color=colors, alpha=0.85)
    ax.scatter(
        table["mismatch_marker"].astype(float),
        y_positions,
        color="#000000",
        marker="D",
        label="stale source reference",
        zorder=3,
    )
    ax.scatter(
        table["incomplete_marker"].astype(float),
        y_positions,
        color="#4c78a8",
        marker="^",
        label="missing owner surface",
        zorder=3,
    )
    for y_pos, row in zip(y_positions, table.to_dict("records")):
        ax.text(
            table["evidence_score"].astype(float).max() * 0.02,
            y_pos,
            (
                f"evidence={row['evidence_label']} | "
                f"check={row['check_status']} | handoff={row['handoff_status']}"
            )[:82],
            va="center",
            ha="left",
            color="white",
            fontsize=8,
        )
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(table["label"])
    ax.invert_yaxis()
    ax.set_title("LCRI evidence lineage map")
    ax.set_xlabel("Evidence score with stale/missing lineage markers")
    ax.set_ylabel("Evidence-to-handoff chain")
    ax.set_xlim(left=0.0)
    ax.legend(title="lineage signal")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _lcri_ci_confidence_coverage_scorecard_bars(scorecard: pd.DataFrame, path: Path) -> None:
    required = {
        "scope",
        "coverage_label",
        "mean_ci_width",
        "max_ci_width",
        "wide_ci_share",
        "gap_exceeds_ci_half_width_share",
        "ci_gate_contradiction_rows",
        "high_priority_ci_gate_rows",
    }
    if scorecard.empty or not required.issubset(scorecard.columns):
        return

    table = scorecard.copy()
    table["label"] = table["scope"].astype(str) + ": " + table["coverage_label"].astype(str)
    table = table.sort_values(
        [
            "high_priority_ci_gate_rows",
            "ci_gate_contradiction_rows",
            "wide_ci_share",
            "max_ci_width",
        ],
        ascending=[False, False, False, False],
    ).head(12)
    colors = table["coverage_label"].map(
        {
            "blocking_ci_gate_review": "#e45756",
            "ci_gate_contradiction_review": "#f58518",
            "wide_ci_review": "#b279a2",
            "adequate_ci_coverage": "#4c78a8",
        }
    ).fillna("#9d9d9d")

    fig, ax = plt.subplots(figsize=(11, 6))
    y_positions = range(len(table))
    ax.barh(y_positions, table["max_ci_width"].astype(float), color=colors, alpha=0.85)
    ax.scatter(
        table["mean_ci_width"].astype(float),
        y_positions,
        color="#000000",
        marker="D",
        label="mean CI width",
        zorder=3,
    )
    ax.scatter(
        table["wide_ci_share"].astype(float),
        y_positions,
        color="#54a24b",
        marker="^",
        label="wide-CI share",
        zorder=3,
    )
    ax.scatter(
        table["gap_exceeds_ci_half_width_share"].astype(float),
        y_positions,
        color="#72b7b2",
        marker="o",
        label="gap outside half-width share",
        zorder=3,
    )
    for y_pos, rows in zip(y_positions, table["ci_gate_contradiction_rows"].astype(int)):
        ax.text(0.0, y_pos, f" c{rows} ", va="center", ha="left", color="white", fontsize=8)
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(table["label"])
    ax.invert_yaxis()
    ax.set_title("LCRI CI confidence coverage scorecard")
    ax.set_xlabel("Max CI width with coverage and contradiction markers")
    ax.set_ylabel("Review scope")
    ax.set_xlim(left=0.0)
    ax.legend(title="uncertainty evidence")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _lcri_ci_gate_contradiction_bars(diagnostics: pd.DataFrame, path: Path) -> None:
    required = {
        "scope",
        "context",
        "ci_gate_label",
        "review_priority",
        "directional_accuracy_gap",
        "ci_half_width",
    }
    if diagnostics.empty or not required.issubset(diagnostics.columns):
        return

    table = diagnostics.copy()
    table["label"] = table["scope"].astype(str) + ": " + table["context"].astype(str)
    table["abs_directional_accuracy_gap"] = table["directional_accuracy_gap"].astype(float).abs()
    table = table.sort_values(
        ["review_priority", "abs_directional_accuracy_gap"], ascending=[False, False]
    ).head(12)
    colors = table["ci_gate_label"].map(
        {
            "aligned": "#4c78a8",
            "gate_warns_inside_ci": "#f58518",
            "gate_blocks_inside_ci": "#e45756",
            "stable_gap_outside_ci": "#b279a2",
        }
    ).fillna("#9d9d9d")

    fig, ax = plt.subplots(figsize=(11, 6))
    y_positions = range(len(table))
    ax.barh(y_positions, table["abs_directional_accuracy_gap"], color=colors, alpha=0.85)
    ax.scatter(
        table["ci_half_width"].astype(float),
        y_positions,
        color="#000000",
        marker="D",
        label="CI half-width",
        zorder=3,
    )
    for y_pos, priority in zip(y_positions, table["review_priority"].astype(int)):
        ax.text(0.0, y_pos, f" p{priority} ", va="center", ha="left", color="white", fontsize=8)
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(table["label"])
    ax.invert_yaxis()
    ax.set_title("LCRI CI-versus-gate contradiction diagnostics")
    ax.set_xlabel("Absolute directional accuracy gap, with CI half-width marker")
    ax.set_ylabel("LCRI context")
    ax.set_xlim(left=0.0)
    ax.legend(title="uncertainty")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _lcri_generalization_severity_scope_bars(severity_scope: pd.DataFrame, path: Path) -> None:
    columns = ["stable_rows", "warning_rows", "critical_rows"]
    if "scope" not in severity_scope.columns or not set(columns).issubset(severity_scope.columns):
        return

    pivot = severity_scope.set_index("scope")[columns]
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("LCRI generalization severity by scope")
    ax.set_xlabel("Scope")
    ax.set_ylabel("Rows")
    ax.legend(title="Severity")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _calibration(frame: pd.DataFrame, path: Path, *, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for signal in ["raw_imbalance", "lcri"]:
        curve = calibration_curve(frame, signal=signal, bins=10)
        ax.plot(
            curve["predicted_probability"],
            curve["observed_frequency"],
            marker="o",
            label=signal,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
