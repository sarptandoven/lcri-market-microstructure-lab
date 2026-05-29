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
