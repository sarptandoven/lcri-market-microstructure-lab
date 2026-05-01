from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_signals(frame: pd.DataFrame, signals: list[str] | None = None) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("cannot evaluate an empty frame")
    signals = signals or ["raw_imbalance", "lcri"]
    _require_columns(frame, [*signals, "future_direction"])
    rows = []
    for signal in signals:
        score = frame[signal].to_numpy(dtype=float)
        target = frame["future_direction"].to_numpy(dtype=float)
        probability = _logistic(_standardize(score))
        rows.append(
            {
                "signal": signal,
                "directional_accuracy": _directional_accuracy(score, target),
                "brier_score": float(np.mean((probability - target) ** 2)),
                "rank_correlation": _spearman(score, target),
                "mean_abs_score": float(np.mean(np.abs(score))),
            }
        )
    return pd.DataFrame(rows)


def compare_transmission_signal(frame: pd.DataFrame) -> dict[str, float]:
    """Compare LCRI and transmission pressure as directional signals."""
    _require_columns(frame, ["lcri", "transmission_pressure", "future_direction"])
    metrics = evaluate_signals(frame, signals=["lcri", "transmission_pressure"]).set_index("signal")
    lcri = metrics.loc["lcri"]
    transmission = metrics.loc["transmission_pressure"]
    return {
        "directional_accuracy_delta": float(
            transmission["directional_accuracy"] - lcri["directional_accuracy"]
        ),
        "brier_score_delta": float(transmission["brier_score"] - lcri["brier_score"]),
        "rank_correlation_delta": float(
            transmission["rank_correlation"] - lcri["rank_correlation"]
        ),
    }


def absorption_regime_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Evaluate LCRI and transmission pressure inside each absorption regime."""
    if frame.empty:
        raise ValueError("cannot evaluate an empty frame")
    _require_columns(
        frame,
        ["absorption_regime", "lcri", "transmission_pressure", "future_direction"],
    )

    rows = []
    for regime, group in frame.groupby("absorption_regime", sort=True):
        metrics = evaluate_signals(group, signals=["lcri", "transmission_pressure"])
        for row in metrics.to_dict("records"):
            row["absorption_regime"] = regime
            row["rows"] = len(group)
            rows.append(row)
    return pd.DataFrame(rows)[
        [
            "absorption_regime",
            "signal",
            "rows",
            "directional_accuracy",
            "brier_score",
            "rank_correlation",
            "mean_abs_score",
        ]
    ]


def regime_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("cannot evaluate an empty frame")
    _require_columns(frame, ["regime", "raw_imbalance", "lcri", "future_direction"])
    rows = []
    for regime, group in frame.groupby("regime", sort=True):
        metrics = evaluate_signals(group)
        for row in metrics.to_dict("records"):
            row["regime"] = regime
            row["rows"] = len(group)
            rows.append(row)
    return pd.DataFrame(rows)[
        [
            "regime",
            "signal",
            "rows",
            "directional_accuracy",
            "brier_score",
            "rank_correlation",
            "mean_abs_score",
        ]
    ]


def summarize_signal_lift(frame: pd.DataFrame) -> dict[str, float]:
    metrics = evaluate_signals(frame).set_index("signal")
    raw = metrics.loc["raw_imbalance"]
    lcri = metrics.loc["lcri"]
    return {
        "directional_accuracy_lift": float(
            lcri["directional_accuracy"] - raw["directional_accuracy"]
        ),
        "brier_score_reduction": float(raw["brier_score"] - lcri["brier_score"]),
        "rank_correlation_lift": float(lcri["rank_correlation"] - raw["rank_correlation"]),
    }




def feature_stability_report(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Report finite rates and distribution stability for selected features."""
    if frame.empty:
        raise ValueError("cannot evaluate an empty frame")
    if not columns:
        raise ValueError("columns must be non-empty")
    _require_columns(frame, ["regime", *columns])

    rows = []
    for regime, group in frame.groupby("regime", sort=True):
        for column in columns:
            values = group[column].to_numpy(dtype=float)
            finite = np.isfinite(values)
            finite_values = values[finite]
            rows.append(
                {
                    "regime": regime,
                    "feature": column,
                    "rows": len(group),
                    "finite_rate": float(np.mean(finite)),
                    "mean": float(np.mean(finite_values)) if len(finite_values) else 0.0,
                    "std": float(np.std(finite_values)) if len(finite_values) else 0.0,
                    "p05": float(np.quantile(finite_values, 0.05)) if len(finite_values) else 0.0,
                    "p95": float(np.quantile(finite_values, 0.95)) if len(finite_values) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def lcri_tail_diagnostics(
    frame: pd.DataFrame,
    thresholds: tuple[float, ...] = (1.0, 2.0, 3.0),
) -> pd.DataFrame:
    """Summarize future behavior in positive and negative LCRI tails."""
    if frame.empty:
        raise ValueError("cannot evaluate an empty frame")
    if not thresholds:
        raise ValueError("thresholds must be non-empty")
    _require_columns(frame, ["lcri", "future_direction"])

    future_ticks = None
    if "future_return_ticks" in frame.columns:
        future_ticks = frame["future_return_ticks"].to_numpy(dtype=float)
    score = frame["lcri"].to_numpy(dtype=float)
    target = frame["future_direction"].to_numpy(dtype=float)

    rows = []
    for threshold in thresholds:
        if threshold <= 0.0:
            raise ValueError("thresholds must be positive")
        for side, mask, expected_direction in [
            ("positive", score >= threshold, 1.0),
            ("negative", score <= -threshold, 0.0),
        ]:
            count = int(np.sum(mask))
            row = {
                "threshold": float(threshold),
                "side": side,
                "rows": count,
                "hit_rate": 0.0,
                "mean_future_return_ticks": 0.0,
            }
            if count:
                row["hit_rate"] = float(np.mean(target[mask] == expected_direction))
                if future_ticks is not None:
                    row["mean_future_return_ticks"] = float(np.mean(future_ticks[mask]))
            rows.append(row)
    return pd.DataFrame(rows)


def calibration_curve(frame: pd.DataFrame, signal: str, bins: int = 10) -> pd.DataFrame:
    if bins < 1:
        raise ValueError("bins must be at least 1")
    _require_columns(frame, [signal, "future_direction"])
    probability = _logistic(frame[signal].to_numpy(dtype=float))
    target = frame["future_direction"].to_numpy(dtype=float)
    bucket = np.clip(np.floor(probability * bins).astype(int), 0, bins - 1)
    rows = []
    for idx in range(bins):
        mask = bucket == idx
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin": idx,
                "predicted_probability": float(np.mean(probability[mask])),
                "observed_frequency": float(np.mean(target[mask])),
                "rows": int(np.sum(mask)),
            }
        )
    return pd.DataFrame(rows)


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"missing evaluation columns: {missing}")


def _directional_accuracy(score: np.ndarray, target: np.ndarray) -> float:
    prediction = (score > 0.0).astype(float)
    return float(np.mean(prediction == target))


def _standardize(score: np.ndarray) -> np.ndarray:
    scale = float(np.std(score))
    if scale == 0.0:
        return score
    return score / scale


def _logistic(score: np.ndarray) -> np.ndarray:
    clipped = np.clip(score, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _spearman(score: np.ndarray, target: np.ndarray) -> float:
    score_rank = pd.Series(score).rank(method="average").to_numpy(dtype=float)
    target_rank = pd.Series(target).rank(method="average").to_numpy(dtype=float)
    if np.std(score_rank) == 0.0 or np.std(target_rank) == 0.0:
        return 0.0
    return float(np.corrcoef(score_rank, target_rank)[0, 1])
