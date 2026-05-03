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
    lcri_generalization_gap_delta: pd.DataFrame | None = None,
    lcri_generalization_severity_by_scope: pd.DataFrame | None = None,
    lcri_gap_delta_scope_summary: pd.DataFrame | None = None,
) -> None:
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
    if lcri_gap_delta_scope_summary is not None:
        _lcri_gap_delta_scope_summary_bars(
            lcri_gap_delta_scope_summary,
            output_dir / "lcri_gap_delta_scope_summary.png",
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
