from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FillProbabilityConfig:
    """Controls for snapshot-based passive fill and adverse-selection proxies."""

    queue_position_fraction: float = 0.50
    min_probability: float = 0.01
    max_probability: float = 0.99
    pressure_scale: float = 1.0
    queue_penalty_scale: float = 1.0
    adverse_selection_scale: float = 1.0

    def __post_init__(self) -> None:
        for name, value in {
            "queue_position_fraction": self.queue_position_fraction,
            "min_probability": self.min_probability,
            "max_probability": self.max_probability,
            "pressure_scale": self.pressure_scale,
            "queue_penalty_scale": self.queue_penalty_scale,
            "adverse_selection_scale": self.adverse_selection_scale,
        }.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.queue_position_fraction <= 1.0:
            raise ValueError("queue_position_fraction must be in [0.0, 1.0]")
        if not 0.0 <= self.min_probability < 1.0:
            raise ValueError("min_probability must be in [0.0, 1.0)")
        if not 0.0 < self.max_probability <= 1.0:
            raise ValueError("max_probability must be in (0.0, 1.0]")
        if self.min_probability >= self.max_probability:
            raise ValueError("min_probability must be less than max_probability")
        if self.pressure_scale <= 0.0:
            raise ValueError("pressure_scale must be positive")
        if self.queue_penalty_scale < 0.0:
            raise ValueError("queue_penalty_scale must be non-negative")
        if self.adverse_selection_scale < 0.0:
            raise ValueError("adverse_selection_scale must be non-negative")


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing {label} columns: {missing}")


def _finite_values(frame: pd.DataFrame, columns: list[str], label: str) -> pd.DataFrame:
    values = frame[columns].astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"{label} inputs must be finite")
    return values


def _logistic(value: pd.Series) -> pd.Series:
    clipped = value.clip(-40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _clip_probability(value: pd.Series, config: FillProbabilityConfig) -> pd.Series:
    return value.clip(config.min_probability, config.max_probability)


def add_queue_position_features(
    frame: pd.DataFrame,
    *,
    levels: int = 5,
    queue_position_fraction: float = 0.50,
) -> pd.DataFrame:
    """Approximate queue-ahead state from visible top-of-book depth.

    The repo currently works with L2 snapshots rather than event-level order-add,
    cancel, and trade messages. This function creates a transparent passive-order
    proxy: a child order is assumed to join at a fixed fraction through the best
    bid/ask queue, then normalizes that visible queue-ahead by same-side depth.
    """
    if levels < 1:
        raise ValueError("levels must be positive")
    if not math.isfinite(queue_position_fraction):
        raise ValueError("queue_position_fraction must be finite")
    if not 0.0 <= queue_position_fraction <= 1.0:
        raise ValueError("queue_position_fraction must be in [0.0, 1.0]")

    required = {"bid_sz_1", "ask_sz_1"}
    required.update({f"bid_sz_{level}" for level in range(1, levels + 1)})
    required.update({f"ask_sz_{level}" for level in range(1, levels + 1)})
    _require_columns(frame, required, "queue position")
    values = _finite_values(frame, sorted(required), "queue position")
    if (values < 0.0).any().any():
        raise ValueError("queue position sizes must be non-negative")

    output = frame.copy()
    bid_total = pd.Series(0.0, index=frame.index)
    ask_total = pd.Series(0.0, index=frame.index)
    for level in range(1, levels + 1):
        bid_total = bid_total + values[f"bid_sz_{level}"]
        ask_total = ask_total + values[f"ask_sz_{level}"]
    bid_ahead = values["bid_sz_1"] * queue_position_fraction
    ask_ahead = values["ask_sz_1"] * queue_position_fraction

    output["bid_queue_ahead"] = bid_ahead
    output["ask_queue_ahead"] = ask_ahead
    output["bid_queue_share"] = np.divide(
        bid_ahead,
        bid_total.replace(0.0, np.nan),
    ).fillna(0.0)
    output["ask_queue_share"] = np.divide(
        ask_ahead,
        ask_total.replace(0.0, np.nan),
    ).fillna(0.0)
    output["queue_position_imbalance"] = bid_ahead - ask_ahead
    return output


def add_passive_fill_probabilities(
    frame: pd.DataFrame,
    *,
    config: FillProbabilityConfig | None = None,
    pressure_col: str = "lcri",
) -> pd.DataFrame:
    """Estimate passive bid/ask fill probability and adverse-fill risk.

    Negative residual pressure implies sell pressure that can deplete the bid and
    fill passive buys. Positive residual pressure analogously depletes the ask and
    fills passive sells. The queue-share penalty makes the proxy execution-aware:
    deeper position in the visible queue lowers fill odds even when pressure is high.
    """
    config = config or FillProbabilityConfig()
    required = {
        pressure_col,
        "bid_queue_share",
        "ask_queue_share",
        "spread_ticks",
        "volatility",
        "replenishment_rate",
    }
    _require_columns(frame, required, "fill probability")
    values = _finite_values(frame, sorted(required), "fill probability")

    pressure = values[pressure_col]
    bid_queue_share = values["bid_queue_share"].clip(lower=0.0)
    ask_queue_share = values["ask_queue_share"].clip(lower=0.0)
    spread_ticks = values["spread_ticks"].clip(lower=0.0)
    volatility = values["volatility"].clip(lower=0.0)
    replenishment = values["replenishment_rate"].clip(lower=0.0)

    spread_stress = np.log1p(spread_ticks)
    volatility_stress = np.log1p(volatility)
    thin_book_stress = 1.0 / (1.0 + replenishment)

    bid_depletion_pressure = (-pressure / config.pressure_scale) + 0.25 * spread_stress
    ask_depletion_pressure = (pressure / config.pressure_scale) + 0.25 * spread_stress

    bid_fill_logit = (
        bid_depletion_pressure
        + 0.35 * volatility_stress
        + 0.35 * thin_book_stress
        - config.queue_penalty_scale * bid_queue_share
    )
    ask_fill_logit = (
        ask_depletion_pressure
        + 0.35 * volatility_stress
        + 0.35 * thin_book_stress
        - config.queue_penalty_scale * ask_queue_share
    )

    bid_fill = _clip_probability(_logistic(bid_fill_logit), config)
    ask_fill = _clip_probability(_logistic(ask_fill_logit), config)

    bid_adverse_logit = (
        bid_depletion_pressure * config.adverse_selection_scale
        + 0.50 * volatility_stress
        + 0.20 * spread_stress
        - 0.40 * replenishment
    )
    ask_adverse_logit = (
        ask_depletion_pressure * config.adverse_selection_scale
        + 0.50 * volatility_stress
        + 0.20 * spread_stress
        - 0.40 * replenishment
    )

    output = frame.copy()
    output["bid_depletion_pressure"] = bid_depletion_pressure
    output["ask_depletion_pressure"] = ask_depletion_pressure
    output["bid_fill_probability"] = bid_fill
    output["ask_fill_probability"] = ask_fill
    output["bid_adverse_fill_probability"] = _clip_probability(_logistic(bid_adverse_logit), config)
    output["ask_adverse_fill_probability"] = _clip_probability(_logistic(ask_adverse_logit), config)
    output["fill_probability_imbalance"] = ask_fill - bid_fill
    output["passive_fill_regime"] = np.select(
        [bid_fill > ask_fill, ask_fill > bid_fill],
        ["bid_depletion", "ask_depletion"],
        default="balanced",
    )
    return output


def add_execution_adjusted_edge(
    frame: pd.DataFrame,
    *,
    signal_col: str = "lcri",
    probability_col: str = "lcri_probability",
    long_net_col: str = "long_net_return_ticks",
    short_net_col: str = "short_net_return_ticks",
) -> pd.DataFrame:
    """Convert directional LCRI edge into passive-fill-adjusted tradable edge."""
    required = {
        signal_col,
        probability_col,
        long_net_col,
        short_net_col,
        "bid_fill_probability",
        "ask_fill_probability",
        "bid_adverse_fill_probability",
        "ask_adverse_fill_probability",
    }
    _require_columns(frame, required, "execution edge")
    values = _finite_values(frame, sorted(required), "execution edge")

    signal = values[signal_col]
    long_return = values[long_net_col]
    short_return = values[short_net_col]
    bid_fill = values["bid_fill_probability"].clip(0.0, 1.0)
    ask_fill = values["ask_fill_probability"].clip(0.0, 1.0)
    bid_adverse = values["bid_adverse_fill_probability"].clip(0.0, 1.0)
    ask_adverse = values["ask_adverse_fill_probability"].clip(0.0, 1.0)

    long_edge = bid_fill * long_return - bid_adverse * long_return.abs()
    short_edge = ask_fill * short_return - ask_adverse * short_return.abs()
    best_edge = np.maximum(long_edge, short_edge)
    best_side = np.select(
        [(long_edge > 0.0) & (long_edge >= short_edge), (short_edge > 0.0) & (short_edge > long_edge)],
        ["long", "short"],
        default="abstain",
    )

    output = frame.copy()
    output["long_fill_adjusted_edge_ticks"] = long_edge
    output["short_fill_adjusted_edge_ticks"] = short_edge
    output["best_execution_side"] = best_side
    output["execution_adjusted_edge_ticks"] = best_edge
    output["execution_adjusted_lcri_score"] = np.select(
        [best_side == "long", best_side == "short"],
        [signal, signal],
        default=0.0,
    )
    return output


def passive_fill_edge_curve(
    frame: pd.DataFrame,
    *,
    bins: int = 5,
    side_col: str = "best_execution_side",
    long_return_col: str = "long_net_return_ticks",
    short_return_col: str = "short_net_return_ticks",
) -> pd.DataFrame:
    """Bin tradable passive opportunities by predicted fill quality.

    The curve is a lightweight calibration surface for the snapshot-based fill
    proxy. It keeps only rows where `side_col` is ``long`` or ``short``, selects
    the side-appropriate fill/adverse-fill probability and realized return, then
    reports whether higher predicted fill buckets also carry healthier realized
    execution edge. The result is intended for research review dashboards rather
    than as a formal fill simulator.
    """
    if not isinstance(bins, int) or isinstance(bins, bool):
        raise ValueError("bins must be an integer")
    if bins < 1:
        raise ValueError("bins must be at least 1")

    columns = [
        side_col,
        "bid_fill_probability",
        "ask_fill_probability",
        "bid_adverse_fill_probability",
        "ask_adverse_fill_probability",
        "execution_adjusted_edge_ticks",
        long_return_col,
        short_return_col,
    ]
    _require_columns(frame, set(columns), "passive fill edge curve")
    if frame.empty:
        return _empty_passive_fill_edge_curve()

    values = _finite_values(
        frame,
        [
            "bid_fill_probability",
            "ask_fill_probability",
            "bid_adverse_fill_probability",
            "ask_adverse_fill_probability",
            "execution_adjusted_edge_ticks",
            long_return_col,
            short_return_col,
        ],
        "passive fill edge curve",
    )
    side = frame[side_col].astype(str)
    tradable = side.isin(["long", "short"])
    if not bool(tradable.any()):
        return _empty_passive_fill_edge_curve()

    selected = pd.DataFrame(index=frame.index[tradable])
    selected_side = side.loc[tradable]
    selected["side"] = selected_side
    selected["predicted_fill_probability"] = np.where(
        selected_side == "long",
        values.loc[tradable, "bid_fill_probability"],
        values.loc[tradable, "ask_fill_probability"],
    )
    selected["adverse_fill_probability"] = np.where(
        selected_side == "long",
        values.loc[tradable, "bid_adverse_fill_probability"],
        values.loc[tradable, "ask_adverse_fill_probability"],
    )
    selected["realized_edge_ticks"] = np.where(
        selected_side == "long",
        values.loc[tradable, long_return_col],
        values.loc[tradable, short_return_col],
    )
    selected["execution_adjusted_edge_ticks"] = values.loc[tradable, "execution_adjusted_edge_ticks"]
    selected["bin"] = _rank_probability_bins(selected["predicted_fill_probability"], bins)

    rows: list[dict[str, float | int]] = []
    for bin_id, group in selected.groupby("bin", sort=True):
        rows.append(
            {
                "bin": int(bin_id),
                "rows": len(group),
                "long_rows": int((group["side"] == "long").sum()),
                "short_rows": int((group["side"] == "short").sum()),
                "mean_predicted_fill_probability": float(group["predicted_fill_probability"].mean()),
                "mean_adverse_fill_probability": float(group["adverse_fill_probability"].mean()),
                "mean_realized_edge_ticks": float(group["realized_edge_ticks"].mean()),
                "positive_edge_rate": float((group["realized_edge_ticks"] > 0.0).mean()),
                "mean_execution_adjusted_edge_ticks": float(
                    group["execution_adjusted_edge_ticks"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)[list(_empty_passive_fill_edge_curve().columns)]


def passive_fill_calibration_curve(
    frame: pd.DataFrame,
    *,
    bins: int = 5,
    side_col: str = "best_execution_side",
    regime_col: str | None = None,
    bid_realized_col: str = "bid_realized_fill",
    ask_realized_col: str = "ask_realized_fill",
) -> pd.DataFrame:
    """Calibrate side-specific passive fill probabilities against realized fills.

    Rows are reduced to executable ``long``/``short`` decisions, mapped to the
    matching bid/ask predicted fill probability and realized fill flag, then
    rank-binned within each regime. This is the bridge from the snapshot proxy to
    event-level add/cancel/trade validation: high predicted buckets should carry
    higher realized fill rates with small calibration error and Brier loss.
    """
    if not isinstance(bins, int) or isinstance(bins, bool):
        raise ValueError("bins must be an integer")
    if bins < 1:
        raise ValueError("bins must be at least 1")

    required = {
        side_col,
        "bid_fill_probability",
        "ask_fill_probability",
        bid_realized_col,
        ask_realized_col,
    }
    if regime_col is not None:
        required.add(regime_col)
    _require_columns(frame, required, "passive fill calibration")
    if frame.empty:
        return _empty_passive_fill_calibration_curve()

    values = _finite_values(
        frame,
        ["bid_fill_probability", "ask_fill_probability", bid_realized_col, ask_realized_col],
        "passive fill calibration",
    )
    side = frame[side_col].astype(str)
    tradable = side.isin(["long", "short"])
    if not bool(tradable.any()):
        return _empty_passive_fill_calibration_curve()

    selected = pd.DataFrame(index=frame.index[tradable])
    selected_side = side.loc[tradable]
    selected["side"] = selected_side
    selected["regime"] = (
        frame.loc[tradable, regime_col].astype(str) if regime_col is not None else "all"
    )
    selected["predicted_fill_probability"] = np.where(
        selected_side == "long",
        values.loc[tradable, "bid_fill_probability"],
        values.loc[tradable, "ask_fill_probability"],
    )
    selected["realized_fill"] = np.where(
        selected_side == "long",
        values.loc[tradable, bid_realized_col],
        values.loc[tradable, ask_realized_col],
    )
    if not selected["realized_fill"].between(0.0, 1.0).all():
        raise ValueError("passive fill calibration realized fills must be in [0, 1]")
    if not selected["predicted_fill_probability"].between(0.0, 1.0).all():
        raise ValueError("passive fill calibration probabilities must be in [0, 1]")

    rows: list[dict[str, float | int | str]] = []
    for regime, regime_group in selected.groupby("regime", sort=True):
        regime_group = regime_group.copy()
        regime_group["bin"] = _rank_probability_bins(
            regime_group["predicted_fill_probability"], bins
        )
        for bin_id, group in regime_group.groupby("bin", sort=True):
            predicted = group["predicted_fill_probability"]
            realized = group["realized_fill"]
            fill_rate = float(realized.mean())
            mean_prediction = float(predicted.mean())
            error = fill_rate - mean_prediction
            rows.append(
                {
                    "regime": str(regime),
                    "bin": int(bin_id),
                    "rows": int(len(group)),
                    "long_rows": int((group["side"] == "long").sum()),
                    "short_rows": int((group["side"] == "short").sum()),
                    "mean_predicted_fill_probability": mean_prediction,
                    "realized_fill_rate": fill_rate,
                    "calibration_error": error,
                    "absolute_calibration_error": abs(error),
                    "brier_score": float(((predicted - realized) ** 2).mean()),
                }
            )
    return pd.DataFrame(rows)[list(_empty_passive_fill_calibration_curve().columns)]


def passive_fill_calibration_summary(curve: pd.DataFrame) -> dict[str, float | int | str]:
    """Summarize passive-fill calibration curve quality with row weighting."""
    if curve.empty:
        return _empty_passive_fill_calibration_summary()
    required = {
        "regime",
        "bin",
        "rows",
        "mean_predicted_fill_probability",
        "realized_fill_rate",
        "calibration_error",
        "absolute_calibration_error",
        "brier_score",
    }
    _require_columns(curve, required, "passive fill calibration summary")
    values = _finite_values(
        curve,
        [
            "rows",
            "mean_predicted_fill_probability",
            "realized_fill_rate",
            "calibration_error",
            "absolute_calibration_error",
            "brier_score",
        ],
        "passive fill calibration summary",
    )
    if (values["rows"] < 0.0).any():
        raise ValueError("passive fill calibration summary rows must be non-negative")
    total_rows = int(values["rows"].sum())
    if total_rows == 0:
        return _empty_passive_fill_calibration_summary()
    weights = values["rows"] / total_rows
    worst_idx = values["absolute_calibration_error"].idxmax()
    return {
        "rows": total_rows,
        "bins": int(len(curve)),
        "regimes": int(curve["regime"].astype(str).nunique()),
        "weighted_mean_predicted_fill_probability": float(
            (values["mean_predicted_fill_probability"] * weights).sum()
        ),
        "weighted_realized_fill_rate": float((values["realized_fill_rate"] * weights).sum()),
        "weighted_calibration_error": float((values["calibration_error"] * weights).sum()),
        "expected_calibration_error": float(
            (values["absolute_calibration_error"] * weights).sum()
        ),
        "weighted_brier_score": float((values["brier_score"] * weights).sum()),
        "worst_regime": str(curve.loc[worst_idx, "regime"]),
        "worst_absolute_calibration_error": float(values.loc[worst_idx, "absolute_calibration_error"]),
    }


def passive_fill_event_window_diagnostics(
    frame: pd.DataFrame,
    *,
    threshold: float = 0.75,
    window: int = 3,
    side_col: str = "best_execution_side",
    long_return_col: str = "long_net_return_ticks",
    short_return_col: str = "short_net_return_ticks",
    regime_col: str | None = None,
) -> pd.DataFrame:
    """Measure realized edge drift around high-probability passive-fill events.

    The fill model can look attractive exactly when adverse-selection risk is
    highest. This diagnostic isolates rows where the side-specific passive-fill
    probability breaches ``threshold`` and compares realized, side-consistent
    edge before and after the event. Grouping by an optional regime column turns
    the output into an event-window regime table for publishability review.
    """
    if not isinstance(window, int) or isinstance(window, bool):
        raise ValueError("window must be an integer")
    if window < 1:
        raise ValueError("window must be at least 1")
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be finite and between 0 and 1")

    required = {
        side_col,
        "bid_fill_probability",
        "ask_fill_probability",
        "bid_adverse_fill_probability",
        "ask_adverse_fill_probability",
        "execution_adjusted_edge_ticks",
        long_return_col,
        short_return_col,
    }
    if regime_col is not None:
        required.add(regime_col)
    _require_columns(frame, required, "passive fill event window")
    if frame.empty:
        return _empty_passive_fill_event_windows()

    numeric_columns = [
        "bid_fill_probability",
        "ask_fill_probability",
        "bid_adverse_fill_probability",
        "ask_adverse_fill_probability",
        "execution_adjusted_edge_ticks",
        long_return_col,
        short_return_col,
    ]
    values = _finite_values(frame, numeric_columns, "passive fill event window")
    side = frame[side_col].astype(str).reset_index(drop=True)
    original_index = pd.Series(frame.index, index=range(len(frame)))
    event_fill = pd.Series(
        np.select(
            [side == "long", side == "short"],
            [values["bid_fill_probability"], values["ask_fill_probability"]],
            default=0.0,
        ),
        index=range(len(frame)),
    )
    event_adverse = pd.Series(
        np.select(
            [side == "long", side == "short"],
            [values["bid_adverse_fill_probability"], values["ask_adverse_fill_probability"]],
            default=0.0,
        ),
        index=range(len(frame)),
    )
    event_edge = values["execution_adjusted_edge_ticks"].reset_index(drop=True)
    long_returns = values[long_return_col].reset_index(drop=True)
    short_returns = values[short_return_col].reset_index(drop=True)
    regimes = (
        frame[regime_col].astype(str).reset_index(drop=True)
        if regime_col is not None
        else pd.Series("all", index=range(len(frame)))
    )

    event_positions = event_fill.index[(side.isin(["long", "short"])) & (event_fill >= threshold)]
    rows: list[dict[str, float | int | str]] = []
    for position in event_positions:
        event_side = side.iloc[position]
        realized = long_returns if event_side == "long" else short_returns
        pre_start = max(0, int(position) - window)
        post_end = min(len(frame), int(position) + window + 1)
        pre = realized.iloc[pre_start:position]
        post = realized.iloc[int(position) + 1 : post_end]
        pre_sum = float(pre.sum()) if len(pre) else 0.0
        post_sum = float(post.sum()) if len(post) else 0.0
        rows.append(
            {
                "event_index": original_index.iloc[position],
                "event_side": str(event_side),
                "event_regime": str(regimes.iloc[position]),
                "event_fill_probability": float(event_fill.iloc[position]),
                "event_adverse_fill_probability": float(event_adverse.iloc[position]),
                "event_edge_ticks": float(event_edge.iloc[position]),
                "pre_realized_edge_sum": pre_sum,
                "post_realized_edge_sum": post_sum,
                "post_minus_pre_realized_edge": float(post_sum - pre_sum),
                "window_rows": int(len(pre) + len(post) + 1),
            }
        )
    if not rows:
        return _empty_passive_fill_event_windows()
    return pd.DataFrame(rows)[list(_empty_passive_fill_event_windows().columns)]


def passive_fill_event_regime_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate passive-fill event windows by execution regime."""
    columns = list(_empty_passive_fill_event_regime_summary().columns)
    if events.empty:
        return _empty_passive_fill_event_regime_summary()
    required = {
        "event_regime",
        "event_fill_probability",
        "event_adverse_fill_probability",
        "event_edge_ticks",
        "post_minus_pre_realized_edge",
    }
    _require_columns(events, required, "passive fill event regime summary")
    values = _finite_values(
        events,
        [
            "event_fill_probability",
            "event_adverse_fill_probability",
            "event_edge_ticks",
            "post_minus_pre_realized_edge",
        ],
        "passive fill event regime summary",
    )
    data = values.copy()
    data["event_regime"] = events["event_regime"].astype(str)

    rows: list[dict[str, float | int | str]] = []
    for regime, group in data.groupby("event_regime", sort=True):
        drift = group["post_minus_pre_realized_edge"]
        adverse = int((drift < 0.0).sum())
        rows.append(
            {
                "event_regime": str(regime),
                "events": int(len(group)),
                "adverse_post_edge_events": adverse,
                "adverse_post_edge_share": float(adverse / len(group)),
                "mean_event_fill_probability": float(group["event_fill_probability"].mean()),
                "mean_event_adverse_fill_probability": float(
                    group["event_adverse_fill_probability"].mean()
                ),
                "mean_event_edge_ticks": float(group["event_edge_ticks"].mean()),
                "mean_post_minus_pre_realized_edge": float(drift.mean()),
                "worst_post_minus_pre_realized_edge": float(drift.min()),
            }
        )
    return pd.DataFrame(rows)[columns].sort_values(
        ["adverse_post_edge_share", "worst_post_minus_pre_realized_edge"],
        ascending=[False, True],
        ignore_index=True,
    )


def _empty_passive_fill_edge_curve() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "bin",
            "rows",
            "long_rows",
            "short_rows",
            "mean_predicted_fill_probability",
            "mean_adverse_fill_probability",
            "mean_realized_edge_ticks",
            "positive_edge_rate",
            "mean_execution_adjusted_edge_ticks",
        ]
    )


def _empty_passive_fill_calibration_curve() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "regime",
            "bin",
            "rows",
            "long_rows",
            "short_rows",
            "mean_predicted_fill_probability",
            "realized_fill_rate",
            "calibration_error",
            "absolute_calibration_error",
            "brier_score",
        ]
    )


def _empty_passive_fill_calibration_summary() -> dict[str, float | int | str]:
    return {
        "rows": 0,
        "bins": 0,
        "regimes": 0,
        "weighted_mean_predicted_fill_probability": 0.0,
        "weighted_realized_fill_rate": 0.0,
        "weighted_calibration_error": 0.0,
        "expected_calibration_error": 0.0,
        "weighted_brier_score": 0.0,
        "worst_regime": "none",
        "worst_absolute_calibration_error": 0.0,
    }


def _empty_passive_fill_event_windows() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_index",
            "event_side",
            "event_regime",
            "event_fill_probability",
            "event_adverse_fill_probability",
            "event_edge_ticks",
            "pre_realized_edge_sum",
            "post_realized_edge_sum",
            "post_minus_pre_realized_edge",
            "window_rows",
        ]
    )


def _empty_passive_fill_event_regime_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_regime",
            "events",
            "adverse_post_edge_events",
            "adverse_post_edge_share",
            "mean_event_fill_probability",
            "mean_event_adverse_fill_probability",
            "mean_event_edge_ticks",
            "mean_post_minus_pre_realized_edge",
            "worst_post_minus_pre_realized_edge",
        ]
    )


def _rank_probability_bins(probability: pd.Series, bins: int) -> pd.Series:
    effective_bins = min(bins, len(probability))
    ranks = probability.rank(method="first")
    bin_ids = pd.qcut(ranks, q=effective_bins, labels=False, duplicates="drop")
    return bin_ids.astype(int) + 1


def execution_publishability_review_packet(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pre/post-execution publishability conflicts for review.

    The packet cross-tabs the pre-execution ``publishable_side`` gate against the
    queue/adverse-fill-aware ``best_execution_side``. It is intentionally small
    enough to ship as a demo/report artifact while still exposing the failure
    modes that matter for market microstructure review: signals that abstain
    after queue-position adjustment, side flips, and opportunities surfaced only
    by the execution layer.
    """
    columns = list(_empty_execution_publishability_review_packet().columns)
    required = {
        "publishable_side",
        "best_execution_side",
        "execution_adjusted_edge_ticks",
        "long_fill_adjusted_edge_ticks",
        "short_fill_adjusted_edge_ticks",
        "bid_fill_probability",
        "ask_fill_probability",
        "bid_adverse_fill_probability",
        "ask_adverse_fill_probability",
    }
    if frame.empty:
        return _empty_execution_publishability_review_packet()
    _require_columns(frame, required, "execution publishability review")
    values = _finite_values(
        frame,
        [
            "execution_adjusted_edge_ticks",
            "long_fill_adjusted_edge_ticks",
            "short_fill_adjusted_edge_ticks",
            "bid_fill_probability",
            "ask_fill_probability",
            "bid_adverse_fill_probability",
            "ask_adverse_fill_probability",
        ],
        "execution publishability review",
    )

    publishable_side = frame["publishable_side"].astype(str)
    best_side = frame["best_execution_side"].astype(str)
    row_state = pd.DataFrame(
        {
            "publishable_side": publishable_side,
            "best_execution_side": best_side,
            "execution_adjusted_edge_ticks": values["execution_adjusted_edge_ticks"],
            "best_fill_probability": _side_probability(
                best_side,
                bid=values["bid_fill_probability"],
                ask=values["ask_fill_probability"],
            ),
            "best_adverse_fill_probability": _side_probability(
                best_side,
                bid=values["bid_adverse_fill_probability"],
                ask=values["ask_adverse_fill_probability"],
            ),
            "publishable_fill_probability": _side_probability(
                publishable_side,
                bid=values["bid_fill_probability"],
                ask=values["ask_fill_probability"],
            ),
            "publishable_edge_ticks": _side_probability(
                publishable_side,
                bid=values["long_fill_adjusted_edge_ticks"],
                ask=values["short_fill_adjusted_edge_ticks"],
            ),
        }
    )
    row_state["edge_drag_ticks"] = (
        row_state["execution_adjusted_edge_ticks"] - row_state["publishable_edge_ticks"]
    )
    row_state["is_conflict"] = row_state["publishable_side"] != row_state["best_execution_side"]

    rows: list[dict[str, float | int | str]] = []
    for (published, best), group in row_state.groupby(
        ["publishable_side", "best_execution_side"], sort=True
    ):
        conflict_rows = int(group["is_conflict"].sum())
        rows.append(
            {
                "publishable_side": str(published),
                "best_execution_side": str(best),
                "rows": len(group),
                "conflict_rows": conflict_rows,
                "conflict_share": float(group["is_conflict"].mean()),
                "mean_execution_adjusted_edge_ticks": float(
                    group["execution_adjusted_edge_ticks"].mean()
                ),
                "mean_best_fill_probability": float(group["best_fill_probability"].mean()),
                "mean_best_adverse_fill_probability": float(
                    group["best_adverse_fill_probability"].mean()
                ),
                "mean_publishable_fill_probability": float(
                    group["publishable_fill_probability"].mean()
                ),
                "mean_edge_drag_ticks": float(group["edge_drag_ticks"].mean()),
                "review_priority": _execution_review_priority(str(published), str(best)),
                "review_note": _execution_review_note(str(published), str(best)),
            }
        )

    packet = pd.DataFrame(rows)[columns]
    return packet.sort_values(
        ["review_priority", "conflict_rows", "rows", "publishable_side", "best_execution_side"],
        ascending=[False, False, False, True, True],
        ignore_index=True,
    )


def _side_probability(side: pd.Series, *, bid: pd.Series, ask: pd.Series) -> pd.Series:
    return pd.Series(
        np.select([side == "long", side == "short"], [bid, ask], default=0.0),
        index=side.index,
    )


def _execution_review_priority(publishable_side: str, best_execution_side: str) -> int:
    if publishable_side == best_execution_side:
        return 0
    if publishable_side in {"long", "short"}:
        return 3
    if best_execution_side in {"long", "short"}:
        return 2
    return 1


def _execution_review_note(publishable_side: str, best_execution_side: str) -> str:
    if publishable_side == best_execution_side:
        return f"pre-execution and execution-aware gates agree on {best_execution_side}"
    if publishable_side in {"long", "short"} and best_execution_side == "abstain":
        return (
            f"pre-execution {publishable_side} signal abstains after "
            "queue/adverse-fill adjustment"
        )
    if publishable_side in {"long", "short"} and best_execution_side in {"long", "short"}:
        return f"pre-execution {publishable_side} signal flips to {best_execution_side} after execution adjustment"
    if publishable_side == "abstain" and best_execution_side in {"long", "short"}:
        return f"execution layer surfaces {best_execution_side} opportunity despite pre-execution abstain"
    return f"publishability gate changes from {publishable_side} to {best_execution_side} after execution adjustment"


def _empty_execution_publishability_review_packet() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
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
        ]
    )


def _empty_execution_summary() -> dict[str, float | int | str]:
    return {
        "rows": 0,
        "tradable_rows": 0,
        "abstain_rows": 0,
        "tradable_share": 0.0,
        "mean_execution_adjusted_edge_ticks": 0.0,
        "median_execution_adjusted_edge_ticks": 0.0,
        "mean_bid_fill_probability": 0.0,
        "mean_ask_fill_probability": 0.0,
        "mean_adverse_fill_probability": 0.0,
        "publishable_side_conflict_rows": 0,
        "publishable_side_conflict_share": 0.0,
        "dominant_execution_side": "none",
    }


def execution_adjusted_edge_summary(frame: pd.DataFrame) -> dict[str, float | int | str]:
    """Summarize execution-aware tradability after queue-fill adjustment.

    The optional `publishable_side` column is treated as the pre-execution gate.
    When present, conflict counts expose rows where a signal that looked
    publishable before queue/adverse-fill adjustment changes side or abstains.
    """
    if frame.empty:
        return _empty_execution_summary()

    required = {
        "best_execution_side",
        "execution_adjusted_edge_ticks",
        "bid_fill_probability",
        "ask_fill_probability",
        "bid_adverse_fill_probability",
        "ask_adverse_fill_probability",
    }
    _require_columns(frame, required, "execution summary")
    values = _finite_values(
        frame,
        [
            "execution_adjusted_edge_ticks",
            "bid_fill_probability",
            "ask_fill_probability",
            "bid_adverse_fill_probability",
            "ask_adverse_fill_probability",
        ],
        "execution summary",
    )

    side = frame["best_execution_side"].astype(str)
    tradable = side != "abstain"
    adverse = 0.5 * (values["bid_adverse_fill_probability"] + values["ask_adverse_fill_probability"])
    side_counts = side[tradable].value_counts()
    dominant_side = "none" if side_counts.empty else str(side_counts.idxmax())

    conflict_rows = 0
    conflict_share = 0.0
    if "publishable_side" in frame.columns:
        publishable_side = frame["publishable_side"].astype(str)
        conflict = publishable_side != side
        conflict_rows = int(conflict.sum())
        conflict_share = float(conflict.mean())

    return {
        "rows": len(frame),
        "tradable_rows": int(tradable.sum()),
        "abstain_rows": int((~tradable).sum()),
        "tradable_share": float(tradable.mean()),
        "mean_execution_adjusted_edge_ticks": float(values["execution_adjusted_edge_ticks"].mean()),
        "median_execution_adjusted_edge_ticks": float(values["execution_adjusted_edge_ticks"].median()),
        "mean_bid_fill_probability": float(values["bid_fill_probability"].mean()),
        "mean_ask_fill_probability": float(values["ask_fill_probability"].mean()),
        "mean_adverse_fill_probability": float(adverse.mean()),
        "publishable_side_conflict_rows": conflict_rows,
        "publishable_side_conflict_share": conflict_share,
        "dominant_execution_side": dominant_side,
    }
