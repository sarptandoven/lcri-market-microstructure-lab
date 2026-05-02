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


def regime_generalization_gap(metrics: pd.DataFrame, heldout_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare full-sample and heldout metrics by regime and signal."""
    required = ["regime", "signal", "directional_accuracy", "brier_score", "rank_correlation"]
    _require_columns(metrics, required)
    _require_columns(heldout_metrics, required)

    full = metrics.set_index(["regime", "signal"])
    heldout = heldout_metrics.set_index(["regime", "signal"])
    keys = [key for key in full.index if key in heldout.index]
    rows = []
    for regime, signal in keys:
        rows.append(
            {
                "regime": regime,
                "signal": signal,
                "directional_accuracy_gap": float(
                    full.loc[(regime, signal), "directional_accuracy"]
                    - heldout.loc[(regime, signal), "directional_accuracy"]
                ),
                "brier_score_gap": float(
                    heldout.loc[(regime, signal), "brier_score"]
                    - full.loc[(regime, signal), "brier_score"]
                ),
                "rank_correlation_gap": float(
                    full.loc[(regime, signal), "rank_correlation"]
                    - heldout.loc[(regime, signal), "rank_correlation"]
                ),
            }
        )
    return pd.DataFrame(rows)


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


def signal_generalization_gap(metrics: pd.DataFrame, heldout_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare full-sample and heldout signal metrics by signal."""
    required = ["signal", "directional_accuracy", "brier_score", "rank_correlation"]
    _require_columns(metrics, required)
    _require_columns(heldout_metrics, required)

    full = metrics.set_index("signal")
    heldout = heldout_metrics.set_index("signal")
    signals = [signal for signal in full.index if signal in heldout.index]
    rows = []
    for signal in signals:
        rows.append(
            {
                "signal": signal,
                "directional_accuracy_gap": float(
                    full.loc[signal, "directional_accuracy"]
                    - heldout.loc[signal, "directional_accuracy"]
                ),
                "brier_score_gap": float(
                    heldout.loc[signal, "brier_score"] - full.loc[signal, "brier_score"]
                ),
                "rank_correlation_gap": float(
                    full.loc[signal, "rank_correlation"]
                    - heldout.loc[signal, "rank_correlation"]
                ),
            }
        )
    return pd.DataFrame(rows)


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


def transition_generalization_gap(
    metrics: pd.DataFrame,
    heldout_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Compare full-sample and heldout metrics by transition segment and signal."""
    required = ["segment", "signal", "directional_accuracy", "brier_score", "rank_correlation"]
    _require_columns(metrics, required)
    _require_columns(heldout_metrics, required)

    full = metrics.set_index(["segment", "signal"])
    heldout = heldout_metrics.set_index(["segment", "signal"])
    rows = []
    for segment, signal in [key for key in full.index if key in heldout.index]:
        rows.append(
            {
                "segment": segment,
                "signal": signal,
                "directional_accuracy_gap": float(
                    full.loc[(segment, signal), "directional_accuracy"]
                    - heldout.loc[(segment, signal), "directional_accuracy"]
                ),
                "brier_score_gap": float(
                    heldout.loc[(segment, signal), "brier_score"]
                    - full.loc[(segment, signal), "brier_score"]
                ),
                "rank_correlation_gap": float(
                    full.loc[(segment, signal), "rank_correlation"]
                    - heldout.loc[(segment, signal), "rank_correlation"]
                ),
            }
        )
    return pd.DataFrame(rows)


def transition_signal_lift(
    frame: pd.DataFrame,
    *,
    transition_col: str = "regime_changed",
) -> pd.DataFrame:
    """Summarize LCRI lift over raw imbalance during stable and transition periods."""
    metrics = transition_conditioned_metrics(frame, transition_col=transition_col)
    rows = []
    for segment, group in metrics.groupby("segment", sort=True):
        by_signal = group.set_index("signal")
        raw = by_signal.loc["raw_imbalance"]
        lcri = by_signal.loc["lcri"]
        rows.append(
            {
                "segment": segment,
                "rows": int(lcri["rows"]),
                "directional_accuracy_lift": float(
                    lcri["directional_accuracy"] - raw["directional_accuracy"]
                ),
                "brier_score_reduction": float(raw["brier_score"] - lcri["brier_score"]),
                "rank_correlation_lift": float(
                    lcri["rank_correlation"] - raw["rank_correlation"]
                ),
            }
        )
    return pd.DataFrame(rows)


def transition_robustness_summary(
    frame: pd.DataFrame,
    *,
    transition_col: str = "regime_changed",
    min_accuracy_lift: float = 0.0,
) -> dict[str, float | bool]:
    """Summarize whether LCRI lift survives both stable and transition periods."""
    lift = transition_signal_lift(frame, transition_col=transition_col).set_index("segment")
    stable = lift.loc["stable"] if "stable" in lift.index else None
    transition = lift.loc["transition"] if "transition" in lift.index else None

    stable_accuracy_lift = (
        float(stable["directional_accuracy_lift"]) if stable is not None else 0.0
    )
    transition_accuracy_lift = (
        float(transition["directional_accuracy_lift"]) if transition is not None else 0.0
    )
    stable_rows = int(stable["rows"]) if stable is not None else 0
    transition_rows = int(transition["rows"]) if transition is not None else 0

    return {
        "stable_rows": stable_rows,
        "transition_rows": transition_rows,
        "stable_directional_accuracy_lift": stable_accuracy_lift,
        "transition_directional_accuracy_lift": transition_accuracy_lift,
        "minimum_directional_accuracy_lift": min(
            stable_accuracy_lift,
            transition_accuracy_lift,
        ),
        "transition_to_stable_lift_ratio": _safe_scalar_divide(
            transition_accuracy_lift,
            stable_accuracy_lift,
        ),
        "passes_transition_robustness": bool(
            stable_rows > 0
            and transition_rows > 0
            and stable_accuracy_lift >= min_accuracy_lift
            and transition_accuracy_lift >= min_accuracy_lift
        ),
    }


def transition_conditioned_metrics(
    frame: pd.DataFrame,
    signals: list[str] | None = None,
    *,
    transition_col: str = "regime_changed",
) -> pd.DataFrame:
    """Evaluate signals separately around stable and transitioning liquidity states."""
    if frame.empty:
        raise ValueError("cannot evaluate an empty frame")
    signals = signals or ["raw_imbalance", "lcri"]
    _require_columns(frame, [*signals, "future_direction", transition_col])

    transition = frame[transition_col].to_numpy(dtype=float) > 0.0
    segments = [
        ("stable", ~transition),
        ("transition", transition),
    ]

    rows = []
    for segment, mask in segments:
        if not np.any(mask):
            continue
        metrics = evaluate_signals(frame.loc[mask], signals=signals)
        for row in metrics.to_dict("records"):
            row["segment"] = segment
            row["rows"] = int(np.sum(mask))
            rows.append(row)

    if not rows:
        raise ValueError("transition-conditioned evaluation has no rows")
    return pd.DataFrame(rows)[
        [
            "segment",
            "signal",
            "rows",
            "directional_accuracy",
            "brier_score",
            "rank_correlation",
            "mean_abs_score",
        ]
    ]


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


def evaluate_cost_aware_signals(
    frame: pd.DataFrame,
    signals: list[str] | None = None,
) -> pd.DataFrame:
    """Evaluate signals only where transaction-cost labels choose a side."""
    if frame.empty:
        raise ValueError("cannot evaluate an empty frame")
    signals = signals or ["raw_imbalance", "lcri"]
    _require_columns(frame, [*signals, "cost_aware_direction"])

    tradable = frame["cost_aware_direction"].to_numpy(dtype=float) != -1.0
    if not np.any(tradable):
        raise ValueError("cost-aware evaluation has no tradable rows")

    rows = []
    target = frame.loc[tradable, "cost_aware_direction"].to_numpy(dtype=float)
    for signal in signals:
        score = frame.loc[tradable, signal].to_numpy(dtype=float)
        probability = _logistic(_standardize(score))
        rows.append(
            {
                "signal": signal,
                "rows": int(np.sum(tradable)),
                "abstained_rows": int(len(frame) - np.sum(tradable)),
                "directional_accuracy": _directional_accuracy(score, target),
                "brier_score": float(np.mean((probability - target) ** 2)),
                "rank_correlation": _spearman(score, target),
            }
        )
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


def _safe_scalar_divide(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return float(numerator / denominator)


def _spearman(score: np.ndarray, target: np.ndarray) -> float:
    score_rank = pd.Series(score).rank(method="average").to_numpy(dtype=float)
    target_rank = pd.Series(target).rank(method="average").to_numpy(dtype=float)
    if np.std(score_rank) == 0.0 or np.std(target_rank) == 0.0:
        return 0.0
    return float(np.corrcoef(score_rank, target_rank)[0, 1])
