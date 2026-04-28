from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from lcri_lab.evaluation import calibration_curve


def write_figures(frame: pd.DataFrame, regime_table: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _scatter(frame, output_dir / "raw_vs_lcri_scatter.png")
    _regime_bars(regime_table, output_dir / "regime_signal_quality.png")
    _calibration(frame, output_dir / "calibration_curve.png")


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


def _calibration(frame: pd.DataFrame, path: Path) -> None:
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
    ax.set_title("Calibration curve")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
