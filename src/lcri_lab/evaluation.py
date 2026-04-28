from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_signals(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for signal in ["raw_imbalance", "lcri"]:
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


def regime_metrics(frame: pd.DataFrame) -> pd.DataFrame:
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


def calibration_curve(frame: pd.DataFrame, signal: str, bins: int = 10) -> pd.DataFrame:
    if bins < 1:
        raise ValueError("bins must be at least 1")
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
