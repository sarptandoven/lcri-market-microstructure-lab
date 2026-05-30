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


def add_queue_position_realized_fill_proxy(
    frame: pd.DataFrame,
    *,
    horizon: int = 1,
    group_cols: str | list[str] | tuple[str, ...] | None = None,
    bid_realized_col: str = "bid_realized_fill",
    ask_realized_col: str = "ask_realized_fill",
) -> pd.DataFrame:
    """Infer passive fill labels from visible queue depletion over a snapshot horizon.

    Event-level add/cancel/trade data is the gold standard for queue priority, but
    the lab's core artifacts often operate on L2 snapshots. This proxy marks a
    passive bid as filled when any of the next ``horizon`` snapshots either loses
    the current best bid price level or depletes at least the estimated bid queue
    ahead; passive asks are handled symmetrically when the best ask level is lost
    upward or same-price visible size depletion clears the estimated ask queue
    ahead. ``group_cols`` prevents fill labels from leaking across independent
    symbols, venues, or sessions when a batch contains interleaved books.
    """
    if not isinstance(horizon, int) or isinstance(horizon, bool):
        raise ValueError("horizon must be an integer")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    required = {
        "bid_px_1",
        "ask_px_1",
        "bid_sz_1",
        "ask_sz_1",
        "bid_queue_ahead",
        "ask_queue_ahead",
    }
    _require_columns(frame, required, "queue position realized fill proxy")
    if group_cols is None:
        grouping_columns: list[str] = []
    elif isinstance(group_cols, str):
        grouping_columns = [group_cols]
    else:
        grouping_columns = list(group_cols)
    if group_cols is not None and not grouping_columns:
        raise ValueError("group_cols must be a non-empty sequence when provided")
    if grouping_columns:
        _require_columns(frame, set(grouping_columns), "queue position realized fill proxy group")
    if frame.empty:
        output = frame.copy()
        output["bid_visible_depletion"] = pd.Series(dtype=float)
        output["ask_visible_depletion"] = pd.Series(dtype=float)
        output["bid_queue_depletion_ratio"] = pd.Series(dtype=float)
        output["ask_queue_depletion_ratio"] = pd.Series(dtype=float)
        output[bid_realized_col] = pd.Series(dtype=float)
        output[ask_realized_col] = pd.Series(dtype=float)
        return output

    values = _finite_values(frame, sorted(required), "queue position realized fill proxy")
    size_columns = ["bid_sz_1", "ask_sz_1", "bid_queue_ahead", "ask_queue_ahead"]
    if (values[size_columns] < 0.0).any().any():
        raise ValueError("queue position realized fill proxy sizes must be non-negative")

    def future_value(column: str, offset: int) -> pd.Series:
        if not grouping_columns:
            return values[column].shift(-offset)
        keys = [frame[group_col] for group_col in grouping_columns]
        return values[column].groupby(keys, sort=False, dropna=False).shift(-offset)

    bid_depletion = pd.Series(0.0, index=frame.index)
    ask_depletion = pd.Series(0.0, index=frame.index)
    has_future = pd.Series(False, index=frame.index)
    for offset in range(1, horizon + 1):
        future_bid_px = future_value("bid_px_1", offset)
        future_ask_px = future_value("ask_px_1", offset)
        future_bid_sz = future_value("bid_sz_1", offset)
        future_ask_sz = future_value("ask_sz_1", offset)
        valid_future = (
            future_bid_px.notna()
            & future_ask_px.notna()
            & future_bid_sz.notna()
            & future_ask_sz.notna()
        )
        has_future = has_future | valid_future

        bid_level_lost = valid_future & (future_bid_px < values["bid_px_1"])
        ask_level_lost = valid_future & (future_ask_px > values["ask_px_1"])
        bid_same_level = valid_future & (future_bid_px == values["bid_px_1"])
        ask_same_level = valid_future & (future_ask_px == values["ask_px_1"])

        bid_same_depletion = (values["bid_sz_1"] - future_bid_sz).clip(lower=0.0).fillna(0.0)
        ask_same_depletion = (values["ask_sz_1"] - future_ask_sz).clip(lower=0.0).fillna(0.0)
        bid_step_depletion = pd.Series(
            np.where(
                bid_level_lost,
                values["bid_sz_1"],
                np.where(bid_same_level, bid_same_depletion, 0.0),
            ),
            index=frame.index,
        )
        ask_step_depletion = pd.Series(
            np.where(
                ask_level_lost,
                values["ask_sz_1"],
                np.where(ask_same_level, ask_same_depletion, 0.0),
            ),
            index=frame.index,
        )
        bid_depletion = bid_depletion.combine(bid_step_depletion, max)
        ask_depletion = ask_depletion.combine(ask_step_depletion, max)

    bid_queue = values["bid_queue_ahead"]
    ask_queue = values["ask_queue_ahead"]
    bid_ratio = pd.Series(
        np.where(
            bid_queue > 0.0,
            bid_depletion / bid_queue.replace(0.0, np.nan),
            np.where(bid_depletion > 0.0, 1.0, 0.0),
        ),
        index=frame.index,
    )
    ask_ratio = pd.Series(
        np.where(
            ask_queue > 0.0,
            ask_depletion / ask_queue.replace(0.0, np.nan),
            np.where(ask_depletion > 0.0, 1.0, 0.0),
        ),
        index=frame.index,
    )

    output = frame.copy()
    output["bid_visible_depletion"] = bid_depletion
    output["ask_visible_depletion"] = ask_depletion
    output["bid_queue_depletion_ratio"] = bid_ratio
    output["ask_queue_depletion_ratio"] = ask_ratio
    bid_realized = has_future & np.where(bid_queue > 0.0, bid_depletion >= bid_queue, bid_depletion > 0.0)
    ask_realized = has_future & np.where(ask_queue > 0.0, ask_depletion >= ask_queue, ask_depletion > 0.0)
    output[bid_realized_col] = bid_realized.astype(float)
    output[ask_realized_col] = ask_realized.astype(float)
    return output


def add_event_level_realized_fill_proxy(
    snapshots: pd.DataFrame,
    events: pd.DataFrame,
    *,
    horizon: float,
    group_cols: str | list[str] | tuple[str, ...] | None = None,
    timestamp_col: str = "timestamp",
    event_type_col: str = "event_type",
    event_side_col: str = "side",
    event_price_col: str = "price",
    event_size_col: str = "size",
    bid_realized_col: str = "bid_realized_fill",
    ask_realized_col: str = "ask_realized_fill",
    trade_event_types: tuple[str, ...] = ("trade",),
    cancel_event_types: tuple[str, ...] = ("cancel", "delete"),
    bid_trade_sides: tuple[str, ...] = ("sell", "bid"),
    ask_trade_sides: tuple[str, ...] = ("buy", "ask"),
    bid_cancel_sides: tuple[str, ...] = ("bid",),
    ask_cancel_sides: tuple[str, ...] = ("ask",),
) -> pd.DataFrame:
    """Label passive fills from event-level queue-depleting trades and cancels.

    This is the live-data counterpart to the snapshot depletion proxy. For each
    snapshot, the function measures queue depletion at the displayed best bid/ask
    over ``(snapshot_time, snapshot_time + horizon]``. By default, sell trades and
    bid-side cancels consume bid queue-ahead while buy trades and ask-side cancels
    consume ask queue-ahead. Venue-specific event-type and side aliases can be
    supplied for feeds that encode aggressor side or book side differently.
    Optional grouping columns keep symbols, venues, or sessions from sharing event
    flow in batched data.
    """
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be a finite positive value")
    if group_cols is None:
        grouping_columns: list[str] = []
    elif isinstance(group_cols, str):
        grouping_columns = [group_cols]
    else:
        grouping_columns = list(group_cols)
    if group_cols is not None and not grouping_columns:
        raise ValueError("group_cols must be a non-empty sequence when provided")

    snapshot_required = {
        timestamp_col,
        "bid_px_1",
        "ask_px_1",
        "bid_queue_ahead",
        "ask_queue_ahead",
        *grouping_columns,
    }
    event_required = {
        timestamp_col,
        event_type_col,
        event_side_col,
        event_price_col,
        event_size_col,
        *grouping_columns,
    }
    _require_columns(snapshots, snapshot_required, "event-level realized fill snapshot")
    _require_columns(events, event_required, "event-level realized fill event")

    output = snapshots.copy()
    output["bid_event_depletion"] = pd.Series(0.0, index=snapshots.index)
    output["ask_event_depletion"] = pd.Series(0.0, index=snapshots.index)
    output["bid_event_depletion_ratio"] = pd.Series(0.0, index=snapshots.index)
    output["ask_event_depletion_ratio"] = pd.Series(0.0, index=snapshots.index)
    output[bid_realized_col] = pd.Series(0.0, index=snapshots.index)
    output[ask_realized_col] = pd.Series(0.0, index=snapshots.index)
    if snapshots.empty:
        return output

    snapshot_numeric = _finite_values(
        snapshots,
        [timestamp_col, "bid_px_1", "ask_px_1", "bid_queue_ahead", "ask_queue_ahead"],
        "event-level realized fill snapshot",
    )
    event_numeric = _finite_values(
        events,
        [timestamp_col, event_price_col, event_size_col],
        "event-level realized fill event",
    )
    if (snapshot_numeric[["bid_queue_ahead", "ask_queue_ahead"]] < 0.0).any().any():
        raise ValueError("event-level realized fill snapshot queues must be non-negative")
    if (event_numeric[event_size_col] < 0.0).any():
        raise ValueError("event-level realized fill event sizes must be non-negative")

    def normalized_aliases(values: tuple[str, ...], label: str) -> set[str]:
        if not values:
            raise ValueError(f"{label} must contain at least one alias")
        aliases = {str(value).lower() for value in values}
        if any(alias == "" for alias in aliases):
            raise ValueError(f"{label} aliases must be non-empty")
        return aliases

    trade_type_aliases = normalized_aliases(trade_event_types, "trade_event_types")
    cancel_type_aliases = normalized_aliases(cancel_event_types, "cancel_event_types")
    bid_trade_side_aliases = normalized_aliases(bid_trade_sides, "bid_trade_sides")
    ask_trade_side_aliases = normalized_aliases(ask_trade_sides, "ask_trade_sides")
    bid_cancel_side_aliases = normalized_aliases(bid_cancel_sides, "bid_cancel_sides")
    ask_cancel_side_aliases = normalized_aliases(ask_cancel_sides, "ask_cancel_sides")

    event_types = events[event_type_col].astype(str).str.lower()
    event_sides = events[event_side_col].astype(str).str.lower()
    bid_depleting_events = (event_types.isin(trade_type_aliases) & event_sides.isin(bid_trade_side_aliases)) | (
        event_types.isin(cancel_type_aliases) & event_sides.isin(bid_cancel_side_aliases)
    )
    ask_depleting_events = (event_types.isin(trade_type_aliases) & event_sides.isin(ask_trade_side_aliases)) | (
        event_types.isin(cancel_type_aliases) & event_sides.isin(ask_cancel_side_aliases)
    )

    bid_depletions: list[float] = []
    ask_depletions: list[float] = []
    for _, row in snapshots.iterrows():
        start = float(row[timestamp_col])
        event_window = (event_numeric[timestamp_col] > start) & (
            event_numeric[timestamp_col] <= start + horizon
        )
        if grouping_columns:
            for group_col in grouping_columns:
                event_window = event_window & (events[group_col] == row[group_col])

        bid_at_price = event_numeric[event_price_col] == float(row["bid_px_1"])
        ask_at_price = event_numeric[event_price_col] == float(row["ask_px_1"])
        bid_size = event_numeric.loc[event_window & bid_at_price & bid_depleting_events, event_size_col].sum()
        ask_size = event_numeric.loc[event_window & ask_at_price & ask_depleting_events, event_size_col].sum()
        bid_depletions.append(float(bid_size))
        ask_depletions.append(float(ask_size))

    bid_depletion = pd.Series(bid_depletions, index=snapshots.index)
    ask_depletion = pd.Series(ask_depletions, index=snapshots.index)
    bid_queue = snapshot_numeric["bid_queue_ahead"]
    ask_queue = snapshot_numeric["ask_queue_ahead"]
    bid_ratio = pd.Series(
        np.where(
            bid_queue > 0.0,
            bid_depletion / bid_queue.replace(0.0, np.nan),
            np.where(bid_depletion > 0.0, 1.0, 0.0),
        ),
        index=snapshots.index,
    ).fillna(0.0)
    ask_ratio = pd.Series(
        np.where(
            ask_queue > 0.0,
            ask_depletion / ask_queue.replace(0.0, np.nan),
            np.where(ask_depletion > 0.0, 1.0, 0.0),
        ),
        index=snapshots.index,
    ).fillna(0.0)

    output["bid_event_depletion"] = bid_depletion
    output["ask_event_depletion"] = ask_depletion
    output["bid_event_depletion_ratio"] = bid_ratio
    output["ask_event_depletion_ratio"] = ask_ratio
    output[bid_realized_col] = np.where(bid_queue > 0.0, bid_depletion >= bid_queue, bid_depletion > 0.0).astype(float)
    output[ask_realized_col] = np.where(ask_queue > 0.0, ask_depletion >= ask_queue, ask_depletion > 0.0).astype(float)
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


def queue_position_fraction_sweep(
    frame: pd.DataFrame,
    *,
    fractions: list[float] | tuple[float, ...] = (0.0, 0.25, 0.50, 0.75, 1.0),
    levels: int = 5,
    fill_config: FillProbabilityConfig | None = None,
    pressure_col: str = "lcri",
    signal_col: str = "lcri",
    probability_col: str = "lcri_probability",
    long_net_col: str = "long_net_return_ticks",
    short_net_col: str = "short_net_return_ticks",
) -> pd.DataFrame:
    """Stress-test passive execution quality across assumed queue placement.

    Snapshot data cannot observe true order priority, so passive fill quality should
    not be reported for a single hard-coded queue assumption. This sweep recomputes
    queue-ahead features, passive fill odds, and execution-adjusted edge at several
    join-depth fractions, producing a placement-sensitivity artifact for research
    review and demos.
    """
    if isinstance(fractions, (str, bytes)):
        raise ValueError("fractions must be a non-empty sequence of finite values")
    fractions = list(fractions)
    if not fractions:
        raise ValueError("fractions must be a non-empty sequence")
    for fraction in fractions:
        if not math.isfinite(float(fraction)):
            raise ValueError("queue_position_fraction values must be finite")
        if not 0.0 <= float(fraction) <= 1.0:
            raise ValueError("queue_position_fraction values must be in [0.0, 1.0]")
    if frame.empty:
        return _empty_queue_position_fraction_sweep()

    rows: list[dict[str, float | int | str]] = []
    for fraction in fractions:
        queued = add_queue_position_features(
            frame,
            levels=levels,
            queue_position_fraction=float(fraction),
        )
        filled = add_passive_fill_probabilities(
            queued,
            config=fill_config,
            pressure_col=pressure_col,
        )
        executed = add_execution_adjusted_edge(
            filled,
            signal_col=signal_col,
            probability_col=probability_col,
            long_net_col=long_net_col,
            short_net_col=short_net_col,
        )
        summary = execution_adjusted_edge_summary(executed)
        rows_count = int(summary["rows"])
        abstain_rows = int(summary["abstain_rows"])
        rows.append(
            {
                "queue_position_fraction": float(fraction),
                "rows": rows_count,
                "mean_bid_queue_share": float(executed["bid_queue_share"].mean()),
                "mean_ask_queue_share": float(executed["ask_queue_share"].mean()),
                "mean_bid_fill_probability": float(summary["mean_bid_fill_probability"]),
                "mean_ask_fill_probability": float(summary["mean_ask_fill_probability"]),
                "mean_fill_probability_imbalance": float(
                    executed["fill_probability_imbalance"].mean()
                ),
                "mean_bid_adverse_fill_probability": float(
                    executed["bid_adverse_fill_probability"].mean()
                ),
                "mean_ask_adverse_fill_probability": float(
                    executed["ask_adverse_fill_probability"].mean()
                ),
                "mean_execution_adjusted_edge_ticks": float(
                    summary["mean_execution_adjusted_edge_ticks"]
                ),
                "tradable_share": float(summary["tradable_share"]),
                "abstain_share": float(abstain_rows / rows_count) if rows_count else 0.0,
                "dominant_execution_side": str(summary["dominant_execution_side"]),
            }
        )
    return pd.DataFrame(rows)[list(_empty_queue_position_fraction_sweep().columns)]


def queue_position_capacity_frontier(
    sweep: pd.DataFrame,
    *,
    min_edge_ticks: float = 0.0,
    min_tradable_share: float = 0.50,
) -> dict[str, float | int | str]:
    """Find the deepest queue placement that preserves executable LCRI edge.

    ``queue_position_fraction_sweep`` shows how passive edge decays as assumed
    queue priority worsens. This reducer turns that curve into an actionable
    capacity frontier: the deepest quote-placement fraction that still clears a
    minimum execution-adjusted edge and tradable-share gate. It is intentionally
    threshold-based so publishability/demo artifacts can state whether alpha only
    works at unrealistic front-of-queue placement or survives meaningful queue
    depth.
    """
    if not math.isfinite(min_edge_ticks):
        raise ValueError("min_edge_ticks must be finite")
    if not math.isfinite(min_tradable_share):
        raise ValueError("min_tradable_share must be finite")
    if not 0.0 <= min_tradable_share <= 1.0:
        raise ValueError("min_tradable_share must be in [0.0, 1.0]")

    empty = _empty_queue_position_capacity_frontier()
    if sweep.empty:
        return empty

    required = {
        "queue_position_fraction",
        "rows",
        "mean_execution_adjusted_edge_ticks",
        "tradable_share",
        "dominant_execution_side",
    }
    _require_columns(sweep, required, "queue position capacity frontier")
    values = _finite_values(
        sweep,
        [
            "queue_position_fraction",
            "rows",
            "mean_execution_adjusted_edge_ticks",
            "tradable_share",
        ],
        "queue position capacity frontier",
    )
    if not values["queue_position_fraction"].between(0.0, 1.0).all():
        raise ValueError("queue position capacity frontier fractions must be in [0, 1]")
    if (values["rows"] < 0.0).any():
        raise ValueError("queue position capacity frontier rows must be non-negative")
    if not values["tradable_share"].between(0.0, 1.0).all():
        raise ValueError("queue position capacity frontier tradable shares must be in [0, 1]")

    data = values.copy()
    data["dominant_execution_side"] = sweep["dominant_execution_side"].astype(str)
    data = data.sort_values("queue_position_fraction", ignore_index=True)
    front = data.iloc[0]
    viable = data[
        (data["mean_execution_adjusted_edge_ticks"] >= min_edge_ticks)
        & (data["tradable_share"] >= min_tradable_share)
    ]

    if viable.empty:
        result = empty.copy()
        result.update(
            {
                "rows": int(len(data)),
                "front_queue_position_fraction": float(front["queue_position_fraction"]),
                "front_mean_execution_adjusted_edge_ticks": float(
                    front["mean_execution_adjusted_edge_ticks"]
                ),
                "front_tradable_share": float(front["tradable_share"]),
                "capacity_label": "no_viable_passive_capacity",
            }
        )
        return result

    capacity = viable.iloc[-1]
    edge_decay = float(
        front["mean_execution_adjusted_edge_ticks"]
        - capacity["mean_execution_adjusted_edge_ticks"]
    )
    tradable_decay = float(front["tradable_share"] - capacity["tradable_share"])
    max_fraction = float(capacity["queue_position_fraction"])
    result = {
        "rows": int(len(data)),
        "viable_rows": int(len(viable)),
        "front_queue_position_fraction": float(front["queue_position_fraction"]),
        "max_viable_queue_position_fraction": max_fraction,
        "front_mean_execution_adjusted_edge_ticks": float(
            front["mean_execution_adjusted_edge_ticks"]
        ),
        "max_viable_mean_execution_adjusted_edge_ticks": float(
            capacity["mean_execution_adjusted_edge_ticks"]
        ),
        "edge_decay_to_capacity_ticks": edge_decay,
        "front_tradable_share": float(front["tradable_share"]),
        "max_viable_tradable_share": float(capacity["tradable_share"]),
        "tradable_share_decay_to_capacity": tradable_decay,
        "dominant_execution_side_at_capacity": str(capacity["dominant_execution_side"]),
        "capacity_label": _queue_capacity_label(
            max_fraction=max_fraction,
            edge_decay=edge_decay,
            tradable_decay=tradable_decay,
        ),
    }
    return result


def queue_position_capacity_stability(
    research_frontier: dict[str, float | int | str],
    heldout_frontier: dict[str, float | int | str],
    *,
    max_fraction_gap: float = 0.10,
    max_edge_gap_ticks: float = 0.10,
    max_tradable_share_gap: float = 0.05,
) -> dict[str, float | int | str | bool]:
    """Compare in-sample and heldout passive queue-capacity frontiers.

    Capacity can look publishable if only the research segment clears the passive
    edge/tradability gates. This reducer turns the train/heldout frontier pair
    into an explicit stability gate: how much queue depth, edge, tradable share,
    and side selection survive out of sample.
    """
    for name, value in {
        "max_fraction_gap": max_fraction_gap,
        "max_edge_gap_ticks": max_edge_gap_ticks,
        "max_tradable_share_gap": max_tradable_share_gap,
    }.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    required = {
        "viable_rows",
        "max_viable_queue_position_fraction",
        "max_viable_mean_execution_adjusted_edge_ticks",
        "max_viable_tradable_share",
        "dominant_execution_side_at_capacity",
        "capacity_label",
    }
    for label, frontier in {
        "research": research_frontier,
        "heldout": heldout_frontier,
    }.items():
        missing = sorted(required - set(frontier))
        if missing:
            raise ValueError(f"missing {label} capacity frontier keys: {missing}")

    research_fraction = float(research_frontier["max_viable_queue_position_fraction"])
    heldout_fraction = float(heldout_frontier["max_viable_queue_position_fraction"])
    research_edge = float(research_frontier["max_viable_mean_execution_adjusted_edge_ticks"])
    heldout_edge = float(heldout_frontier["max_viable_mean_execution_adjusted_edge_ticks"])
    research_tradable = float(research_frontier["max_viable_tradable_share"])
    heldout_tradable = float(heldout_frontier["max_viable_tradable_share"])
    research_viable_rows = int(research_frontier["viable_rows"])
    heldout_viable_rows = int(heldout_frontier["viable_rows"])

    numeric_values = [
        research_fraction,
        heldout_fraction,
        research_edge,
        heldout_edge,
        research_tradable,
        heldout_tradable,
    ]
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError("capacity frontier numeric values must be finite")
    if not 0.0 <= research_fraction <= 1.0 or not 0.0 <= heldout_fraction <= 1.0:
        raise ValueError("capacity frontier fractions must be in [0.0, 1.0]")
    if not 0.0 <= research_tradable <= 1.0 or not 0.0 <= heldout_tradable <= 1.0:
        raise ValueError("capacity frontier tradable shares must be in [0.0, 1.0]")
    if research_viable_rows < 0 or heldout_viable_rows < 0:
        raise ValueError("capacity frontier viable rows must be non-negative")

    fraction_gap = heldout_fraction - research_fraction
    edge_gap = heldout_edge - research_edge
    tradable_gap = heldout_tradable - research_tradable
    viable_row_gap = heldout_viable_rows - research_viable_rows
    dominant_side_changed = (
        str(research_frontier["dominant_execution_side_at_capacity"])
        != str(heldout_frontier["dominant_execution_side_at_capacity"])
    )
    label = _queue_capacity_stability_label(
        fraction_gap=fraction_gap,
        edge_gap=edge_gap,
        tradable_gap=tradable_gap,
        viable_row_gap=viable_row_gap,
        dominant_side_changed=dominant_side_changed,
        heldout_label=str(heldout_frontier["capacity_label"]),
        max_fraction_gap=max_fraction_gap,
        max_edge_gap_ticks=max_edge_gap_ticks,
        max_tradable_share_gap=max_tradable_share_gap,
    )

    return {
        "research_capacity_label": str(research_frontier["capacity_label"]),
        "heldout_capacity_label": str(heldout_frontier["capacity_label"]),
        "capacity_fraction_gap": float(fraction_gap),
        "capacity_edge_gap_ticks": float(edge_gap),
        "capacity_tradable_share_gap": float(tradable_gap),
        "capacity_viable_row_gap": int(viable_row_gap),
        "dominant_side_changed": dominant_side_changed,
        "capacity_stability_label": label,
    }


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


def queue_position_fill_surface(
    frame: pd.DataFrame,
    *,
    queue_bins: int = 4,
    probability_bins: int = 4,
    side_col: str = "best_execution_side",
    regime_col: str | None = None,
    bid_realized_col: str = "bid_realized_fill",
    ask_realized_col: str = "ask_realized_fill",
) -> pd.DataFrame:
    """Cross queue position depth with passive-fill calibration quality.

    This diagnostic surfaces whether the fill proxy is only calibrated for front-
    of-queue quotes or remains reliable deeper in the visible queue. Tradable rows
    are mapped to side-specific queue share, fill probability, and realized fill,
    then rank-binned within each regime across queue depth and predicted fill.
    """
    if not isinstance(queue_bins, int) or isinstance(queue_bins, bool):
        raise ValueError("queue_bins must be an integer")
    if not isinstance(probability_bins, int) or isinstance(probability_bins, bool):
        raise ValueError("probability_bins must be an integer")
    if queue_bins < 1:
        raise ValueError("queue_bins must be at least 1")
    if probability_bins < 1:
        raise ValueError("probability_bins must be at least 1")

    required = {
        side_col,
        "bid_queue_share",
        "ask_queue_share",
        "bid_fill_probability",
        "ask_fill_probability",
        bid_realized_col,
        ask_realized_col,
        "execution_adjusted_edge_ticks",
    }
    if regime_col is not None:
        required.add(regime_col)
    _require_columns(frame, required, "queue position fill surface")
    if frame.empty:
        return _empty_queue_position_fill_surface()

    numeric_columns = [
        "bid_queue_share",
        "ask_queue_share",
        "bid_fill_probability",
        "ask_fill_probability",
        bid_realized_col,
        ask_realized_col,
        "execution_adjusted_edge_ticks",
    ]
    values = _finite_values(frame, numeric_columns, "queue position fill surface")
    side = frame[side_col].astype(str)
    tradable = side.isin(["long", "short"])
    if not bool(tradable.any()):
        return _empty_queue_position_fill_surface()

    selected = pd.DataFrame(index=frame.index[tradable])
    selected_side = side.loc[tradable]
    selected["regime"] = (
        frame.loc[tradable, regime_col].astype(str) if regime_col is not None else "all"
    )
    selected["queue_share"] = np.where(
        selected_side == "long",
        values.loc[tradable, "bid_queue_share"],
        values.loc[tradable, "ask_queue_share"],
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
    selected["execution_adjusted_edge_ticks"] = values.loc[
        tradable, "execution_adjusted_edge_ticks"
    ]
    if not selected["queue_share"].ge(0.0).all():
        raise ValueError("queue position fill surface queue shares must be non-negative")
    if not selected["predicted_fill_probability"].between(0.0, 1.0).all():
        raise ValueError("queue position fill surface probabilities must be in [0, 1]")
    if not selected["realized_fill"].between(0.0, 1.0).all():
        raise ValueError("queue position fill surface realized fills must be in [0, 1]")

    rows: list[dict[str, float | int | str]] = []
    for regime, regime_group in selected.groupby("regime", sort=True):
        regime_group = regime_group.copy()
        regime_group["queue_bin"] = _rank_probability_bins(
            regime_group["queue_share"], queue_bins
        )
        regime_group["fill_probability_bin"] = _rank_probability_bins(
            regime_group["predicted_fill_probability"], probability_bins
        )
        for (queue_bin, fill_bin), group in regime_group.groupby(
            ["queue_bin", "fill_probability_bin"], sort=True
        ):
            predicted = group["predicted_fill_probability"]
            realized = group["realized_fill"]
            fill_rate = float(realized.mean())
            mean_prediction = float(predicted.mean())
            error = fill_rate - mean_prediction
            rows.append(
                {
                    "regime": str(regime),
                    "queue_bin": int(queue_bin),
                    "fill_probability_bin": int(fill_bin),
                    "rows": int(len(group)),
                    "mean_queue_share": float(group["queue_share"].mean()),
                    "mean_predicted_fill_probability": mean_prediction,
                    "realized_fill_rate": fill_rate,
                    "calibration_error": error,
                    "absolute_calibration_error": abs(error),
                    "brier_score": float(((predicted - realized) ** 2).mean()),
                    "mean_execution_adjusted_edge_ticks": float(
                        group["execution_adjusted_edge_ticks"].mean()
                    ),
                }
            )
    if not rows:
        return _empty_queue_position_fill_surface()
    return pd.DataFrame(rows)[list(_empty_queue_position_fill_surface().columns)]


def queue_position_fill_calibration_surface(
    frame: pd.DataFrame,
    *,
    queue_bins: int = 4,
    probability_bins: int = 4,
    side_col: str = "best_execution_side",
    regime_col: str | None = None,
    bid_realized_col: str = "bid_realized_fill",
    ask_realized_col: str = "ask_realized_fill",
) -> pd.DataFrame:
    """Diagnose realized passive fills across side-specific queue depth.

    ``queue_position_fill_surface`` collapses long and short opportunities inside
    a regime-level grid. This side-aware companion keeps the execution side in the
    artifact so reviewers can see whether predicted bid/ask fill probabilities are
    calibrated differently as orders sit deeper in their respective visible queues.
    ``execution_adjusted_edge_ticks`` is optional; when present it is averaged into
    the surface so queue-depth calibration can be tied directly to realized edge.
    """
    if not isinstance(queue_bins, int) or isinstance(queue_bins, bool):
        raise ValueError("queue_bins must be an integer")
    if not isinstance(probability_bins, int) or isinstance(probability_bins, bool):
        raise ValueError("probability_bins must be an integer")
    if queue_bins < 1:
        raise ValueError("queue_bins must be at least 1")
    if probability_bins < 1:
        raise ValueError("probability_bins must be at least 1")

    required = {
        side_col,
        "bid_queue_share",
        "ask_queue_share",
        "bid_fill_probability",
        "ask_fill_probability",
        bid_realized_col,
        ask_realized_col,
    }
    if regime_col is not None:
        required.add(regime_col)
    _require_columns(frame, required, "queue position fill calibration surface")
    columns = list(_empty_queue_position_fill_calibration_surface().columns)
    if frame.empty:
        return _empty_queue_position_fill_calibration_surface()

    numeric_columns = [
        "bid_queue_share",
        "ask_queue_share",
        "bid_fill_probability",
        "ask_fill_probability",
        bid_realized_col,
        ask_realized_col,
    ]
    has_edge = "execution_adjusted_edge_ticks" in frame.columns
    if has_edge:
        numeric_columns.append("execution_adjusted_edge_ticks")
    values = _finite_values(frame, numeric_columns, "queue position fill calibration surface")
    side = frame[side_col].astype(str)
    tradable = side.isin(["long", "short"])
    if not bool(tradable.any()):
        return _empty_queue_position_fill_calibration_surface()

    selected = pd.DataFrame(index=frame.index[tradable])
    selected_side = side.loc[tradable]
    selected["regime"] = (
        frame.loc[tradable, regime_col].astype(str) if regime_col is not None else "all"
    )
    selected["best_execution_side"] = selected_side
    selected["queue_share"] = np.where(
        selected_side == "long",
        values.loc[tradable, "bid_queue_share"],
        values.loc[tradable, "ask_queue_share"],
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
    selected["execution_adjusted_edge_ticks"] = (
        values.loc[tradable, "execution_adjusted_edge_ticks"] if has_edge else 0.0
    )
    if not selected["queue_share"].ge(0.0).all():
        raise ValueError("queue position fill calibration surface queue shares must be non-negative")
    if not selected["predicted_fill_probability"].between(0.0, 1.0).all():
        raise ValueError("queue position fill calibration surface probabilities must be in [0, 1]")
    if not selected["realized_fill"].between(0.0, 1.0).all():
        raise ValueError("queue position fill calibration surface realized fills must be in [0, 1]")

    rows: list[dict[str, float | int | str]] = []
    for (regime, execution_side), side_group in selected.groupby(
        ["regime", "best_execution_side"], sort=True
    ):
        side_group = side_group.copy()
        side_group["queue_share_bin"] = _rank_probability_bins(
            side_group["queue_share"], queue_bins
        )
        side_group["fill_probability_bin"] = _rank_probability_bins(
            side_group["predicted_fill_probability"], probability_bins
        )
        for (queue_bin, fill_bin), group in side_group.groupby(
            ["queue_share_bin", "fill_probability_bin"], sort=True
        ):
            predicted = group["predicted_fill_probability"]
            realized = group["realized_fill"]
            fill_rate = float(realized.mean())
            mean_prediction = float(predicted.mean())
            error = fill_rate - mean_prediction
            rows.append(
                {
                    "regime": str(regime),
                    "best_execution_side": str(execution_side),
                    "queue_share_bin": int(queue_bin),
                    "fill_probability_bin": int(fill_bin),
                    "rows": int(len(group)),
                    "mean_queue_share": float(group["queue_share"].mean()),
                    "mean_predicted_fill_probability": mean_prediction,
                    "realized_fill_rate": fill_rate,
                    "calibration_error": error,
                    "absolute_calibration_error": abs(error),
                    "brier_score": float(((predicted - realized) ** 2).mean()),
                    "mean_execution_adjusted_edge_ticks": float(
                        group["execution_adjusted_edge_ticks"].mean()
                    ),
                }
            )
    if not rows:
        return _empty_queue_position_fill_calibration_surface()
    output = pd.DataFrame(rows)[columns]
    if regime_col is not None:
        output = output.rename(columns={"regime": regime_col})
    return output


def queue_position_edge_decay(surface: pd.DataFrame, *, min_rows: int = 1) -> pd.DataFrame:
    """Summarize how passive execution edge decays as queue position gets deeper.

    ``queue_position_fill_surface`` gives a two-dimensional calibration grid. This
    reducer collapses the fill-probability axis to a queue-depth frontier per
    regime, then compares front-of-queue and back-of-queue realized fill, predicted
    fill, calibration error, and execution-adjusted edge. The output is designed
    to answer the placement question: does a signal remain investable deeper in
    the visible queue, or is it mostly a front-of-queue opportunity?
    """
    if not isinstance(min_rows, int) or isinstance(min_rows, bool):
        raise ValueError("min_rows must be an integer")
    if min_rows < 1:
        raise ValueError("min_rows must be at least 1")
    columns = list(_empty_queue_position_edge_decay().columns)
    if surface.empty:
        return _empty_queue_position_edge_decay()

    required = {
        "regime",
        "queue_bin",
        "rows",
        "mean_queue_share",
        "mean_predicted_fill_probability",
        "realized_fill_rate",
        "absolute_calibration_error",
        "mean_execution_adjusted_edge_ticks",
    }
    _require_columns(surface, required, "queue position edge decay")
    values = _finite_values(
        surface,
        [
            "queue_bin",
            "rows",
            "mean_queue_share",
            "mean_predicted_fill_probability",
            "realized_fill_rate",
            "absolute_calibration_error",
            "mean_execution_adjusted_edge_ticks",
        ],
        "queue position edge decay",
    )
    if (values["rows"] < 0.0).any():
        raise ValueError("queue position edge decay rows must be non-negative")
    if not values["mean_predicted_fill_probability"].between(0.0, 1.0).all():
        raise ValueError("queue position edge decay predicted probabilities must be in [0, 1]")
    if not values["realized_fill_rate"].between(0.0, 1.0).all():
        raise ValueError("queue position edge decay realized fill rates must be in [0, 1]")

    data = values.copy()
    data["regime"] = surface["regime"].astype(str)
    rows: list[dict[str, float | int | str | bool]] = []
    for regime, regime_group in data.groupby("regime", sort=True):
        queue_rows = []
        for queue_bin, queue_group in regime_group.groupby("queue_bin", sort=True):
            total_rows = float(queue_group["rows"].sum())
            if total_rows < float(min_rows):
                continue
            weights = queue_group["rows"] / total_rows if total_rows > 0.0 else queue_group["rows"]
            queue_rows.append(
                {
                    "queue_bin": int(queue_bin),
                    "rows": int(total_rows),
                    "mean_queue_share": float((queue_group["mean_queue_share"] * weights).sum()),
                    "mean_predicted_fill_probability": float(
                        (queue_group["mean_predicted_fill_probability"] * weights).sum()
                    ),
                    "realized_fill_rate": float((queue_group["realized_fill_rate"] * weights).sum()),
                    "absolute_calibration_error": float(
                        (queue_group["absolute_calibration_error"] * weights).sum()
                    ),
                    "mean_execution_adjusted_edge_ticks": float(
                        (queue_group["mean_execution_adjusted_edge_ticks"] * weights).sum()
                    ),
                }
            )
        if not queue_rows:
            continue
        frontier = pd.DataFrame(queue_rows).sort_values("queue_bin").reset_index(drop=True)
        front = frontier.iloc[0]
        back = frontier.iloc[-1]
        edges = frontier["mean_execution_adjusted_edge_ticks"]
        worst_idx = edges.idxmin()
        edge_decay = float(front["mean_execution_adjusted_edge_ticks"] - back["mean_execution_adjusted_edge_ticks"])
        fill_decay = float(front["realized_fill_rate"] - back["realized_fill_rate"])
        calibration_widening = float(
            back["absolute_calibration_error"] - front["absolute_calibration_error"]
        )
        monotonic_edge_decay = bool((edges.diff().fillna(0.0) <= 1e-12).all())
        rows.append(
            {
                "regime": str(regime),
                "queue_bins": int(len(frontier)),
                "rows": int(frontier["rows"].sum()),
                "front_queue_bin": int(front["queue_bin"]),
                "back_queue_bin": int(back["queue_bin"]),
                "front_mean_queue_share": float(front["mean_queue_share"]),
                "back_mean_queue_share": float(back["mean_queue_share"]),
                "fill_rate_decay": fill_decay,
                "predicted_fill_decay": float(
                    front["mean_predicted_fill_probability"]
                    - back["mean_predicted_fill_probability"]
                ),
                "edge_decay_ticks": edge_decay,
                "calibration_error_widening": calibration_widening,
                "monotonic_edge_decay": monotonic_edge_decay,
                "worst_queue_bin": int(frontier.loc[worst_idx, "queue_bin"]),
                "worst_mean_execution_adjusted_edge_ticks": float(edges.loc[worst_idx]),
                "queue_decay_label": _queue_decay_label(
                    edge_decay=edge_decay,
                    fill_decay=fill_decay,
                    calibration_widening=calibration_widening,
                ),
            }
        )
    if not rows:
        return _empty_queue_position_edge_decay()
    return pd.DataFrame(rows)[columns].sort_values(
        ["edge_decay_ticks", "fill_rate_decay", "regime"],
        ascending=[False, False, True],
        ignore_index=True,
    )


def _empty_queue_position_execution_quality_gate() -> dict[str, float | int | str]:
    return {
        "surface_rows": 0,
        "decay_rows": 0,
        "surface_regimes": 0,
        "decay_regimes": 0,
        "eligible_regimes": 0,
        "blocked_regimes": 0,
        "weighted_absolute_calibration_error": 0.0,
        "weighted_brier_score": 0.0,
        "weighted_edge_decay_ticks": 0.0,
        "worst_calibration_regime": "none",
        "worst_decay_regime": "none",
        "max_regime_absolute_calibration_error": 0.0,
        "max_calibration_error_widening": 0.0,
        "non_monotonic_decay_regimes": 0,
        "quality_gate_label": "empty_queue_execution_surface",
    }


def queue_position_execution_quality_gate(
    surface: pd.DataFrame,
    decay: pd.DataFrame,
    *,
    max_expected_calibration_error: float = 0.15,
    max_expected_brier_score: float = 0.15,
    max_calibration_widening: float = 0.10,
) -> dict[str, float | int | str]:
    """Gate queue-position fill surfaces before treating passive edge as publishable.

    The surface-level calibration check asks whether predicted passive fills are
    empirically reliable across queue-depth buckets. The decay check asks whether
    edge degrades monotonically as quote placement moves deeper into the visible
    queue. Combining both prevents a superficially attractive LCRI signal from
    passing review when its execution evidence is driven by miscalibrated or
    non-monotone queue-depth regimes.
    """
    for name, value in {
        "max_expected_calibration_error": max_expected_calibration_error,
        "max_expected_brier_score": max_expected_brier_score,
        "max_calibration_widening": max_calibration_widening,
    }.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be a finite non-negative value")
    if surface.empty and decay.empty:
        return _empty_queue_position_execution_quality_gate()

    surface_required = {"regime", "rows", "absolute_calibration_error", "brier_score"}
    decay_required = {
        "regime",
        "rows",
        "edge_decay_ticks",
        "calibration_error_widening",
        "monotonic_edge_decay",
    }
    _require_columns(surface, surface_required, "queue execution quality surface")
    _require_columns(decay, decay_required, "queue execution quality decay")
    surface_values = _finite_values(
        surface,
        ["rows", "absolute_calibration_error", "brier_score"],
        "queue execution quality surface",
    )
    decay_values = _finite_values(
        decay,
        ["rows", "edge_decay_ticks", "calibration_error_widening"],
        "queue execution quality decay",
    )
    if (surface_values["rows"] < 0.0).any() or (decay_values["rows"] < 0.0).any():
        raise ValueError("queue execution quality rows must be non-negative")
    if (surface_values[["absolute_calibration_error", "brier_score"]] < 0.0).any().any():
        raise ValueError("queue execution quality calibration metrics must be non-negative")

    surface_data = surface_values.copy()
    surface_data["regime"] = surface["regime"].astype(str)
    regime_surface = surface_data.groupby("regime", sort=True).apply(
        _weighted_queue_quality_surface_row,
        include_groups=False,
    )
    if isinstance(regime_surface, pd.Series):
        regime_surface = regime_surface.unstack()
    regime_surface = regime_surface.reset_index()

    decay_data = decay_values.copy()
    decay_data["regime"] = decay["regime"].astype(str)
    decay_data["monotonic_edge_decay"] = decay["monotonic_edge_decay"].astype(bool)

    surface_rows = int(surface_values["rows"].sum())
    decay_rows = int(decay_values["rows"].sum())
    surface_weights = surface_values["rows"] / surface_rows if surface_rows > 0 else surface_values["rows"]
    decay_weights = decay_values["rows"] / decay_rows if decay_rows > 0 else decay_values["rows"]

    if regime_surface.empty:
        worst_calibration_regime = "none"
        max_regime_error = 0.0
        high_error_regimes: set[str] = set()
    else:
        worst_idx = regime_surface["absolute_calibration_error"].idxmax()
        worst_calibration_regime = str(regime_surface.loc[worst_idx, "regime"])
        max_regime_error = float(regime_surface["absolute_calibration_error"].max())
        high_error_regimes = set(
            regime_surface.loc[
                (regime_surface["absolute_calibration_error"] > max_expected_calibration_error)
                | (regime_surface["brier_score"] > max_expected_brier_score),
                "regime",
            ].astype(str)
        )

    if decay_data.empty:
        worst_decay_regime = "none"
        max_widening = 0.0
        non_monotonic = 0
        decay_block_regimes: set[str] = set()
    else:
        worst_decay_idx = decay_data["calibration_error_widening"].idxmax()
        worst_decay_regime = str(decay_data.loc[worst_decay_idx, "regime"])
        max_widening = float(decay_data["calibration_error_widening"].max())
        non_monotonic = int((~decay_data["monotonic_edge_decay"]).sum())
        decay_block_regimes = set(
            decay_data.loc[
                (decay_data["calibration_error_widening"] > max_calibration_widening)
                | (~decay_data["monotonic_edge_decay"]),
                "regime",
            ].astype(str)
        )

    blocked_regimes = high_error_regimes | decay_block_regimes
    eligible_regimes = set(surface_data["regime"]) | set(decay_data["regime"])
    label = _queue_execution_quality_label(
        blocked_regimes=len(blocked_regimes),
        eligible_regimes=len(eligible_regimes),
        weighted_calibration_error=float(
            (surface_values["absolute_calibration_error"] * surface_weights).sum()
        ),
        weighted_brier_score=float((surface_values["brier_score"] * surface_weights).sum()),
        max_expected_calibration_error=max_expected_calibration_error,
        max_expected_brier_score=max_expected_brier_score,
    )

    return {
        "surface_rows": surface_rows,
        "decay_rows": decay_rows,
        "surface_regimes": int(surface_data["regime"].nunique()),
        "decay_regimes": int(decay_data["regime"].nunique()),
        "eligible_regimes": len(eligible_regimes),
        "blocked_regimes": len(blocked_regimes),
        "weighted_absolute_calibration_error": float(
            (surface_values["absolute_calibration_error"] * surface_weights).sum()
        ),
        "weighted_brier_score": float((surface_values["brier_score"] * surface_weights).sum()),
        "weighted_edge_decay_ticks": float((decay_values["edge_decay_ticks"] * decay_weights).sum()),
        "worst_calibration_regime": worst_calibration_regime,
        "worst_decay_regime": worst_decay_regime,
        "max_regime_absolute_calibration_error": max_regime_error,
        "max_calibration_error_widening": max_widening,
        "non_monotonic_decay_regimes": non_monotonic,
        "quality_gate_label": label,
    }


def _weighted_queue_quality_surface_row(group: pd.DataFrame) -> pd.Series:
    total_rows = float(group["rows"].sum())
    weights = group["rows"] / total_rows if total_rows > 0.0 else group["rows"]
    return pd.Series(
        {
            "rows": int(total_rows),
            "absolute_calibration_error": float((group["absolute_calibration_error"] * weights).sum()),
            "brier_score": float((group["brier_score"] * weights).sum()),
        }
    )


def _queue_execution_quality_label(
    *,
    blocked_regimes: int,
    eligible_regimes: int,
    weighted_calibration_error: float,
    weighted_brier_score: float,
    max_expected_calibration_error: float,
    max_expected_brier_score: float,
) -> str:
    if eligible_regimes == 0:
        return "empty_queue_execution_surface"
    if blocked_regimes > 0:
        return "queue_execution_blocked"
    if (
        weighted_calibration_error > max_expected_calibration_error * 0.75
        or weighted_brier_score > max_expected_brier_score * 0.75
    ):
        return "queue_execution_review"
    return "queue_execution_publishable"


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


def event_level_passive_fill_horizon_sweep(
    snapshots: pd.DataFrame,
    events: pd.DataFrame,
    *,
    horizons: list[float] | tuple[float, ...] = (0.5, 1.0, 2.0, 5.0),
    bins: int = 5,
    group_cols: str | list[str] | tuple[str, ...] | None = None,
    side_col: str = "best_execution_side",
    regime_col: str | None = None,
    timestamp_col: str = "timestamp",
    event_type_col: str = "event_type",
    event_side_col: str = "side",
    event_price_col: str = "price",
    event_size_col: str = "size",
) -> pd.DataFrame:
    """Calibrate passive fill probabilities against event-level queue depletion horizons.

    Snapshot horizon sweeps are useful when only L2 states are available, but a
    publishable execution claim should prefer add/cancel/trade message evidence
    when available. This sweep relabels each passive quote with event-level queue
    depletion over multiple time horizons, then reports calibration drift versus
    the shortest event window.
    """
    if isinstance(horizons, (str, bytes)):
        raise ValueError("horizons must be a non-empty sequence of finite positive values")
    horizons = list(horizons)
    if not horizons:
        raise ValueError("horizons must be a non-empty sequence")
    for horizon in horizons:
        if not math.isfinite(float(horizon)) or float(horizon) <= 0.0:
            raise ValueError("horizon values must be finite positive values")
    if len(set(float(horizon) for horizon in horizons)) != len(horizons):
        raise ValueError("horizon values must be unique")
    if not isinstance(bins, int) or isinstance(bins, bool):
        raise ValueError("bins must be an integer")
    if bins < 1:
        raise ValueError("bins must be at least 1")

    rows: list[dict[str, float | int | str]] = []
    anchor_realized_rate: float | None = None
    anchor_brier: float | None = None
    for index, horizon in enumerate(sorted(float(value) for value in horizons)):
        realized = add_event_level_realized_fill_proxy(
            snapshots,
            events,
            horizon=horizon,
            group_cols=group_cols,
            timestamp_col=timestamp_col,
            event_type_col=event_type_col,
            event_side_col=event_side_col,
            event_price_col=event_price_col,
            event_size_col=event_size_col,
            bid_realized_col="_event_sweep_bid_realized_fill",
            ask_realized_col="_event_sweep_ask_realized_fill",
        )
        curve = passive_fill_calibration_curve(
            realized,
            bins=bins,
            side_col=side_col,
            regime_col=regime_col,
            bid_realized_col="_event_sweep_bid_realized_fill",
            ask_realized_col="_event_sweep_ask_realized_fill",
        )
        summary = passive_fill_calibration_summary(curve)
        realized_rate = float(summary["weighted_realized_fill_rate"])
        brier = float(summary["weighted_brier_score"])
        if index == 0:
            anchor_realized_rate = realized_rate
            anchor_brier = brier
        assert anchor_realized_rate is not None
        assert anchor_brier is not None
        realized_gap = realized_rate - anchor_realized_rate
        brier_gap = brier - anchor_brier
        rows.append(
            {
                "horizon": float(horizon),
                "event_depletion_source": "events",
                "rows": int(summary["rows"]),
                "bins": int(summary["bins"]),
                "regimes": int(summary["regimes"]),
                "weighted_mean_predicted_fill_probability": float(
                    summary["weighted_mean_predicted_fill_probability"]
                ),
                "weighted_realized_fill_rate": realized_rate,
                "weighted_calibration_error": float(summary["weighted_calibration_error"]),
                "expected_calibration_error": float(summary["expected_calibration_error"]),
                "weighted_brier_score": brier,
                "worst_absolute_calibration_error": float(
                    summary["worst_absolute_calibration_error"]
                ),
                "realized_fill_rate_gap_vs_shortest": float(realized_gap),
                "brier_score_gap_vs_shortest": float(brier_gap),
                "horizon_stability_label": _passive_fill_horizon_stability_label(
                    index=index,
                    realized_gap=realized_gap,
                    brier_gap=brier_gap,
                ),
            }
        )
    return pd.DataFrame(rows)[list(_empty_event_level_passive_fill_horizon_sweep().columns)]


def passive_fill_realization_horizon_sweep(
    frame: pd.DataFrame,
    *,
    horizons: list[int] | tuple[int, ...] = (1, 2, 3, 5),
    bins: int = 5,
    group_cols: str | list[str] | tuple[str, ...] | None = None,
    side_col: str = "best_execution_side",
    regime_col: str | None = None,
) -> pd.DataFrame:
    """Stress-test passive fill calibration across realized-depletion horizons.

    Queue-position labels inferred from snapshots are horizon-sensitive: one-step
    depletion is conservative, while longer horizons can include slower queue
    clearance that a passive order could plausibly capture. This sweep recomputes
    realized bid/ask fill labels for each horizon, recalibrates the side-selected
    fill probabilities, and reports drift versus the shortest horizon so research
    review can spot alpha that only looks tradable after delayed fills.
    """
    if isinstance(horizons, (str, bytes)):
        raise ValueError("horizons must be a non-empty sequence of positive integers")
    horizons = list(horizons)
    if not horizons:
        raise ValueError("horizons must be a non-empty sequence")
    for horizon in horizons:
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
            raise ValueError("horizon values must be positive integers")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizon values must be unique")
    if not isinstance(bins, int) or isinstance(bins, bool):
        raise ValueError("bins must be an integer")
    if bins < 1:
        raise ValueError("bins must be at least 1")

    rows: list[dict[str, float | int | str]] = []
    anchor_realized_rate: float | None = None
    anchor_brier: float | None = None
    for index, horizon in enumerate(sorted(horizons)):
        realized = add_queue_position_realized_fill_proxy(
            frame,
            horizon=horizon,
            group_cols=group_cols,
            bid_realized_col="_sweep_bid_realized_fill",
            ask_realized_col="_sweep_ask_realized_fill",
        )
        curve = passive_fill_calibration_curve(
            realized,
            bins=bins,
            side_col=side_col,
            regime_col=regime_col,
            bid_realized_col="_sweep_bid_realized_fill",
            ask_realized_col="_sweep_ask_realized_fill",
        )
        summary = passive_fill_calibration_summary(curve)
        realized_rate = float(summary["weighted_realized_fill_rate"])
        brier = float(summary["weighted_brier_score"])
        if index == 0:
            anchor_realized_rate = realized_rate
            anchor_brier = brier
        assert anchor_realized_rate is not None
        assert anchor_brier is not None
        realized_gap = realized_rate - anchor_realized_rate
        brier_gap = brier - anchor_brier
        rows.append(
            {
                "horizon": int(horizon),
                "rows": int(summary["rows"]),
                "bins": int(summary["bins"]),
                "regimes": int(summary["regimes"]),
                "weighted_mean_predicted_fill_probability": float(
                    summary["weighted_mean_predicted_fill_probability"]
                ),
                "weighted_realized_fill_rate": realized_rate,
                "weighted_calibration_error": float(summary["weighted_calibration_error"]),
                "expected_calibration_error": float(summary["expected_calibration_error"]),
                "weighted_brier_score": brier,
                "worst_absolute_calibration_error": float(
                    summary["worst_absolute_calibration_error"]
                ),
                "realized_fill_rate_gap_vs_shortest": float(realized_gap),
                "brier_score_gap_vs_shortest": float(brier_gap),
                "horizon_stability_label": _passive_fill_horizon_stability_label(
                    index=index,
                    realized_gap=realized_gap,
                    brier_gap=brier_gap,
                ),
            }
        )
    return pd.DataFrame(rows)[list(_empty_passive_fill_realization_horizon_sweep().columns)]


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
        event_regime = str(regimes.iloc[position])
        pre_regime = _modal_window_regime(regimes.iloc[pre_start:position], fallback=event_regime)
        post_regime = _modal_window_regime(
            regimes.iloc[int(position) + 1 : post_end], fallback=event_regime
        )
        rows.append(
            {
                "event_index": original_index.iloc[position],
                "event_side": str(event_side),
                "event_regime": event_regime,
                "pre_window_regime": pre_regime,
                "post_window_regime": post_regime,
                "regime_transition": f"{pre_regime}->{post_regime}",
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


def passive_fill_event_transition_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate passive-fill event toxicity by pre/post regime transition.

    Event-window failures often appear at liquidity-state boundaries rather than
    inside a single named regime. This table groups high-probability passive-fill
    events by the modal pre-window to post-window regime path so reviewers can see
    whether queue fills become toxic specifically during calm→thin or thin→stress
    transitions.
    """
    columns = list(_empty_passive_fill_event_transition_summary().columns)
    if events.empty:
        return _empty_passive_fill_event_transition_summary()
    required = {
        "event_regime",
        "regime_transition",
        "event_fill_probability",
        "event_adverse_fill_probability",
        "event_edge_ticks",
        "post_minus_pre_realized_edge",
    }
    _require_columns(events, required, "passive fill event transition summary")
    values = _finite_values(
        events,
        [
            "event_fill_probability",
            "event_adverse_fill_probability",
            "event_edge_ticks",
            "post_minus_pre_realized_edge",
        ],
        "passive fill event transition summary",
    )
    data = values.copy()
    data["event_regime"] = events["event_regime"].astype(str)
    data["regime_transition"] = events["regime_transition"].astype(str)

    rows: list[dict[str, float | int | str]] = []
    for transition, group in data.groupby("regime_transition", sort=True):
        drift = group["post_minus_pre_realized_edge"]
        adverse = int((drift < 0.0).sum())
        rows.append(
            {
                "regime_transition": str(transition),
                "event_regimes": int(group["event_regime"].nunique()),
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
        [
            "adverse_post_edge_share",
            "worst_post_minus_pre_realized_edge",
            "mean_event_adverse_fill_probability",
        ],
        ascending=[False, True, False],
        ignore_index=True,
    )


def passive_fill_event_lifecycle_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate passive-fill event toxicity by full pre/event/post regime path.

    Transition summaries collapse the event regime out of the state path. This
    lifecycle view keeps the modal pre-window regime, event-row regime, and modal
    post-window regime together so reviewers can spot toxic fills that are benign
    for the same pre→post transition in one event state but fragile in another.
    """
    columns = list(_empty_passive_fill_event_lifecycle_summary().columns)
    if events.empty:
        return _empty_passive_fill_event_lifecycle_summary()
    required = {
        "pre_window_regime",
        "event_regime",
        "post_window_regime",
        "regime_transition",
        "event_fill_probability",
        "event_adverse_fill_probability",
        "event_edge_ticks",
        "pre_realized_edge_sum",
        "post_realized_edge_sum",
        "post_minus_pre_realized_edge",
        "window_rows",
    }
    _require_columns(events, required, "passive fill event lifecycle summary")
    values = _finite_values(
        events,
        [
            "event_fill_probability",
            "event_adverse_fill_probability",
            "event_edge_ticks",
            "pre_realized_edge_sum",
            "post_realized_edge_sum",
            "post_minus_pre_realized_edge",
            "window_rows",
        ],
        "passive fill event lifecycle summary",
    )
    if (values["window_rows"] < 1.0).any():
        raise ValueError("passive fill event lifecycle summary window_rows must be positive")

    data = values.copy()
    data["pre_window_regime"] = events["pre_window_regime"].astype(str)
    data["event_regime"] = events["event_regime"].astype(str)
    data["post_window_regime"] = events["post_window_regime"].astype(str)
    data["regime_transition"] = events["regime_transition"].astype(str)
    data["lifecycle_path"] = (
        data["pre_window_regime"]
        + "|"
        + data["event_regime"]
        + "|"
        + data["post_window_regime"]
    )

    rows: list[dict[str, float | int | str]] = []
    for lifecycle_path, group in data.groupby("lifecycle_path", sort=True):
        drift = group["post_minus_pre_realized_edge"]
        adverse = int((drift < 0.0).sum())
        adverse_share = float(adverse / len(group))
        mean_drift = float(drift.mean())
        rows.append(
            {
                "lifecycle_path": str(lifecycle_path),
                "pre_window_regime": str(group["pre_window_regime"].iloc[0]),
                "event_regime": str(group["event_regime"].iloc[0]),
                "post_window_regime": str(group["post_window_regime"].iloc[0]),
                "regime_transitions": int(group["regime_transition"].nunique()),
                "events": int(len(group)),
                "adverse_post_edge_events": adverse,
                "adverse_post_edge_share": adverse_share,
                "mean_event_fill_probability": float(group["event_fill_probability"].mean()),
                "mean_event_adverse_fill_probability": float(
                    group["event_adverse_fill_probability"].mean()
                ),
                "mean_event_edge_ticks": float(group["event_edge_ticks"].mean()),
                "mean_pre_realized_edge_sum": float(group["pre_realized_edge_sum"].mean()),
                "mean_post_realized_edge_sum": float(group["post_realized_edge_sum"].mean()),
                "mean_post_minus_pre_realized_edge": mean_drift,
                "worst_post_minus_pre_realized_edge": float(drift.min()),
                "mean_window_rows": float(group["window_rows"].mean()),
                "lifecycle_toxicity_label": _passive_fill_lifecycle_toxicity_label(
                    adverse_share=adverse_share,
                    mean_post_minus_pre_edge=mean_drift,
                ),
            }
        )
    return pd.DataFrame(rows)[columns].sort_values(
        [
            "adverse_post_edge_share",
            "worst_post_minus_pre_realized_edge",
            "mean_event_adverse_fill_probability",
        ],
        ascending=[False, True, False],
        ignore_index=True,
    )


def _passive_fill_lifecycle_toxicity_label(
    *, adverse_share: float, mean_post_minus_pre_edge: float
) -> str:
    if adverse_share >= 0.5 and mean_post_minus_pre_edge < 0.0:
        return "toxic_transition_lifecycle"
    if adverse_share > 0.0 or mean_post_minus_pre_edge < 0.0:
        return "mixed_transition_lifecycle"
    return "benign_transition_lifecycle"


def _modal_window_regime(values: pd.Series, *, fallback: str) -> str:
    if values.empty:
        return fallback
    counts = values.astype(str).value_counts(sort=True)
    return str(counts.index[0])


def passive_fill_event_toxicity_scorecard(
    regime_summary: pd.DataFrame,
    *,
    max_adverse_post_edge_share: float = 0.60,
    min_mean_post_minus_pre_edge: float = -0.25,
    min_events: int = 1,
) -> dict[str, float | int | str]:
    """Gate passive-fill event windows for execution-aware publishability.

    High fill probability is only useful if the subsequent event window is not
    systematically toxic. This scorecard consumes ``passive_fill_event_regime_summary``
    and flags regimes where high-probability fills are followed by excessive
    adverse edge drift or a strongly negative mean post-minus-pre realized edge.
    """
    if not math.isfinite(max_adverse_post_edge_share) or not 0.0 <= max_adverse_post_edge_share <= 1.0:
        raise ValueError("max_adverse_post_edge_share must be finite and between 0 and 1")
    if not math.isfinite(min_mean_post_minus_pre_edge):
        raise ValueError("min_mean_post_minus_pre_edge must be finite")
    if not isinstance(min_events, int) or isinstance(min_events, bool) or min_events < 1:
        raise ValueError("min_events must be a positive integer")
    if regime_summary.empty:
        return _empty_passive_fill_event_toxicity_scorecard()

    required = {
        "event_regime",
        "events",
        "adverse_post_edge_share",
        "mean_event_fill_probability",
        "mean_event_adverse_fill_probability",
        "mean_post_minus_pre_realized_edge",
        "worst_post_minus_pre_realized_edge",
    }
    _require_columns(regime_summary, required, "passive fill event toxicity")
    values = _finite_values(
        regime_summary,
        [
            "events",
            "adverse_post_edge_share",
            "mean_event_fill_probability",
            "mean_event_adverse_fill_probability",
            "mean_post_minus_pre_realized_edge",
            "worst_post_minus_pre_realized_edge",
        ],
        "passive fill event toxicity",
    )
    if (values["events"] < 0.0).any():
        raise ValueError("passive fill event toxicity events must be non-negative")

    total_events = int(values["events"].sum())
    if total_events == 0:
        return _empty_passive_fill_event_toxicity_scorecard()

    data = values.copy()
    data["event_regime"] = regime_summary["event_regime"].astype(str)
    weights = data["events"] / total_events
    eligible = data[data["events"] >= min_events]
    if eligible.empty:
        scorecard = _empty_passive_fill_event_toxicity_scorecard()
        scorecard.update(
            {
                "rows": int(len(regime_summary)),
                "regimes": int(data["event_regime"].nunique()),
                "total_events": total_events,
                "weighted_mean_event_fill_probability": float(
                    (data["mean_event_fill_probability"] * weights).sum()
                ),
                "weighted_mean_event_adverse_fill_probability": float(
                    (data["mean_event_adverse_fill_probability"] * weights).sum()
                ),
                "weighted_mean_post_minus_pre_realized_edge": float(
                    (data["mean_post_minus_pre_realized_edge"] * weights).sum()
                ),
                "event_toxicity_label": "insufficient_event_windows",
            }
        )
        return scorecard

    blocked = eligible[
        (eligible["adverse_post_edge_share"] > max_adverse_post_edge_share)
        | (eligible["mean_post_minus_pre_realized_edge"] < min_mean_post_minus_pre_edge)
    ]
    worst_idx = eligible.sort_values(
        ["adverse_post_edge_share", "mean_post_minus_pre_realized_edge"],
        ascending=[False, True],
    ).index[0]
    label = "event_window_blocker" if not blocked.empty else "event_window_pass"

    return {
        "rows": int(len(regime_summary)),
        "regimes": int(data["event_regime"].nunique()),
        "total_events": total_events,
        "eligible_regimes": int(len(eligible)),
        "blocked_regimes": int(len(blocked)),
        "worst_regime": str(data.loc[worst_idx, "event_regime"]),
        "worst_adverse_post_edge_share": float(data.loc[worst_idx, "adverse_post_edge_share"]),
        "worst_mean_post_minus_pre_realized_edge": float(
            data.loc[worst_idx, "mean_post_minus_pre_realized_edge"]
        ),
        "worst_post_minus_pre_realized_edge": float(
            data.loc[worst_idx, "worst_post_minus_pre_realized_edge"]
        ),
        "weighted_mean_event_fill_probability": float(
            (data["mean_event_fill_probability"] * weights).sum()
        ),
        "weighted_mean_event_adverse_fill_probability": float(
            (data["mean_event_adverse_fill_probability"] * weights).sum()
        ),
        "weighted_mean_post_minus_pre_realized_edge": float(
            (data["mean_post_minus_pre_realized_edge"] * weights).sum()
        ),
        "event_toxicity_label": label,
    }


def passive_fill_event_transition_scorecard(
    transition_summary: pd.DataFrame,
    *,
    max_adverse_post_edge_share: float = 0.60,
    min_mean_post_minus_pre_edge: float = -0.25,
    min_events: int = 1,
) -> dict[str, float | int | str]:
    """Gate passive-fill event-window toxicity at regime-transition boundaries.

    Regime summaries can hide boundary-specific failures. This scorecard consumes
    ``passive_fill_event_transition_summary`` and promotes pre→post regime paths
    to first-class publishability gates, blocking transitions where high-probability
    fills are followed by excessive adverse drift.
    """
    if not math.isfinite(max_adverse_post_edge_share) or not 0.0 <= max_adverse_post_edge_share <= 1.0:
        raise ValueError("max_adverse_post_edge_share must be finite and between 0 and 1")
    if not math.isfinite(min_mean_post_minus_pre_edge):
        raise ValueError("min_mean_post_minus_pre_edge must be finite")
    if not isinstance(min_events, int) or isinstance(min_events, bool) or min_events < 1:
        raise ValueError("min_events must be a positive integer")
    if transition_summary.empty:
        return _empty_passive_fill_event_transition_scorecard()

    required = {
        "regime_transition",
        "events",
        "adverse_post_edge_share",
        "mean_event_fill_probability",
        "mean_event_adverse_fill_probability",
        "mean_post_minus_pre_realized_edge",
        "worst_post_minus_pre_realized_edge",
    }
    _require_columns(
        transition_summary,
        required,
        "passive fill event transition toxicity",
    )
    values = _finite_values(
        transition_summary,
        [
            "events",
            "adverse_post_edge_share",
            "mean_event_fill_probability",
            "mean_event_adverse_fill_probability",
            "mean_post_minus_pre_realized_edge",
            "worst_post_minus_pre_realized_edge",
        ],
        "passive fill event transition toxicity",
    )
    if (values["events"] < 0.0).any():
        raise ValueError("passive fill event transition toxicity events must be non-negative")

    total_events = int(values["events"].sum())
    if total_events == 0:
        return _empty_passive_fill_event_transition_scorecard()

    data = values.copy()
    data["regime_transition"] = transition_summary["regime_transition"].astype(str)
    weights = data["events"] / total_events
    eligible = data[data["events"] >= min_events]
    if eligible.empty:
        scorecard = _empty_passive_fill_event_transition_scorecard()
        scorecard.update(
            {
                "rows": int(len(transition_summary)),
                "transitions": int(data["regime_transition"].nunique()),
                "total_events": total_events,
                "weighted_mean_event_fill_probability": float(
                    (data["mean_event_fill_probability"] * weights).sum()
                ),
                "weighted_mean_event_adverse_fill_probability": float(
                    (data["mean_event_adverse_fill_probability"] * weights).sum()
                ),
                "weighted_mean_post_minus_pre_realized_edge": float(
                    (data["mean_post_minus_pre_realized_edge"] * weights).sum()
                ),
                "transition_toxicity_label": "insufficient_transition_event_windows",
            }
        )
        return scorecard

    blocked = eligible[
        (eligible["adverse_post_edge_share"] > max_adverse_post_edge_share)
        | (eligible["mean_post_minus_pre_realized_edge"] < min_mean_post_minus_pre_edge)
    ]
    worst_idx = eligible.sort_values(
        ["adverse_post_edge_share", "mean_post_minus_pre_realized_edge"],
        ascending=[False, True],
    ).index[0]
    label = (
        "transition_event_window_blocker"
        if not blocked.empty
        else "transition_event_window_pass"
    )

    return {
        "rows": int(len(transition_summary)),
        "transitions": int(data["regime_transition"].nunique()),
        "total_events": total_events,
        "eligible_transitions": int(len(eligible)),
        "blocked_transitions": int(len(blocked)),
        "worst_transition": str(data.loc[worst_idx, "regime_transition"]),
        "worst_adverse_post_edge_share": float(data.loc[worst_idx, "adverse_post_edge_share"]),
        "worst_mean_post_minus_pre_realized_edge": float(
            data.loc[worst_idx, "mean_post_minus_pre_realized_edge"]
        ),
        "worst_post_minus_pre_realized_edge": float(
            data.loc[worst_idx, "worst_post_minus_pre_realized_edge"]
        ),
        "weighted_mean_event_fill_probability": float(
            (data["mean_event_fill_probability"] * weights).sum()
        ),
        "weighted_mean_event_adverse_fill_probability": float(
            (data["mean_event_adverse_fill_probability"] * weights).sum()
        ),
        "weighted_mean_post_minus_pre_realized_edge": float(
            (data["mean_post_minus_pre_realized_edge"] * weights).sum()
        ),
        "transition_toxicity_label": label,
    }


def _empty_queue_position_fraction_sweep() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "queue_position_fraction",
            "rows",
            "mean_bid_queue_share",
            "mean_ask_queue_share",
            "mean_bid_fill_probability",
            "mean_ask_fill_probability",
            "mean_fill_probability_imbalance",
            "mean_bid_adverse_fill_probability",
            "mean_ask_adverse_fill_probability",
            "mean_execution_adjusted_edge_ticks",
            "tradable_share",
            "abstain_share",
            "dominant_execution_side",
        ]
    )


def _empty_queue_position_capacity_frontier() -> dict[str, float | int | str]:
    return {
        "rows": 0,
        "viable_rows": 0,
        "front_queue_position_fraction": 0.0,
        "max_viable_queue_position_fraction": 0.0,
        "front_mean_execution_adjusted_edge_ticks": 0.0,
        "max_viable_mean_execution_adjusted_edge_ticks": 0.0,
        "edge_decay_to_capacity_ticks": 0.0,
        "front_tradable_share": 0.0,
        "max_viable_tradable_share": 0.0,
        "tradable_share_decay_to_capacity": 0.0,
        "dominant_execution_side_at_capacity": "none",
        "capacity_label": "empty_sweep",
    }


def _queue_capacity_label(*, max_fraction: float, edge_decay: float, tradable_decay: float) -> str:
    if max_fraction >= 0.75 and edge_decay <= 0.25 and tradable_decay <= 0.15:
        return "deep_queue_resilient_capacity"
    if max_fraction >= 0.50:
        return "queue_capacity_constrained"
    return "front_queue_only_capacity"


def _queue_capacity_stability_label(
    *,
    fraction_gap: float,
    edge_gap: float,
    tradable_gap: float,
    viable_row_gap: int,
    dominant_side_changed: bool,
    heldout_label: str,
    max_fraction_gap: float,
    max_edge_gap_ticks: float,
    max_tradable_share_gap: float,
) -> str:
    if heldout_label in {"empty_sweep", "no_viable_passive_capacity"}:
        return "capacity_not_replicated"
    if (
        fraction_gap < -max_fraction_gap
        or edge_gap < -max_edge_gap_ticks
        or tradable_gap < -max_tradable_share_gap
        or viable_row_gap < 0
        or dominant_side_changed
    ):
        return "capacity_fragile"
    return "capacity_stable"


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


def _empty_queue_position_fill_surface() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "regime",
            "queue_bin",
            "fill_probability_bin",
            "rows",
            "mean_queue_share",
            "mean_predicted_fill_probability",
            "realized_fill_rate",
            "calibration_error",
            "absolute_calibration_error",
            "brier_score",
            "mean_execution_adjusted_edge_ticks",
        ]
    )


def _empty_queue_position_fill_calibration_surface() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "regime",
            "best_execution_side",
            "queue_share_bin",
            "fill_probability_bin",
            "rows",
            "mean_queue_share",
            "mean_predicted_fill_probability",
            "realized_fill_rate",
            "calibration_error",
            "absolute_calibration_error",
            "brier_score",
            "mean_execution_adjusted_edge_ticks",
        ]
    )


def _empty_queue_position_edge_decay() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "regime",
            "queue_bins",
            "rows",
            "front_queue_bin",
            "back_queue_bin",
            "front_mean_queue_share",
            "back_mean_queue_share",
            "fill_rate_decay",
            "predicted_fill_decay",
            "edge_decay_ticks",
            "calibration_error_widening",
            "monotonic_edge_decay",
            "worst_queue_bin",
            "worst_mean_execution_adjusted_edge_ticks",
            "queue_decay_label",
        ]
    )


def _queue_decay_label(*, edge_decay: float, fill_decay: float, calibration_widening: float) -> str:
    if calibration_widening > 0.100000000001:
        return "calibration_watch"
    if edge_decay > 0.0 and fill_decay >= 0.0:
        return "front_queue_preferred"
    if edge_decay <= 0.0 and fill_decay <= 0.0:
        return "deep_queue_resilient"
    return "mixed_queue_response"


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


def _empty_passive_fill_realization_horizon_sweep() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "horizon",
            "rows",
            "bins",
            "regimes",
            "weighted_mean_predicted_fill_probability",
            "weighted_realized_fill_rate",
            "weighted_calibration_error",
            "expected_calibration_error",
            "weighted_brier_score",
            "worst_absolute_calibration_error",
            "realized_fill_rate_gap_vs_shortest",
            "brier_score_gap_vs_shortest",
            "horizon_stability_label",
        ]
    )


def _empty_event_level_passive_fill_horizon_sweep() -> pd.DataFrame:
    columns = list(_empty_passive_fill_realization_horizon_sweep().columns)
    columns.insert(1, "event_depletion_source")
    return pd.DataFrame(columns=columns)


def _passive_fill_horizon_stability_label(
    *, index: int, realized_gap: float, brier_gap: float
) -> str:
    if index == 0:
        return "anchor_horizon"
    if realized_gap >= 0.10 and brier_gap <= 0.0:
        return "later_fill_realization"
    if realized_gap <= -0.10 or brier_gap >= 0.05:
        return "horizon_fragile"
    return "horizon_stable"


def _empty_passive_fill_event_windows() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "event_index",
            "event_side",
            "event_regime",
            "pre_window_regime",
            "post_window_regime",
            "regime_transition",
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


def _empty_passive_fill_event_transition_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "regime_transition",
            "event_regimes",
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


def _empty_passive_fill_event_lifecycle_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "lifecycle_path",
            "pre_window_regime",
            "event_regime",
            "post_window_regime",
            "regime_transitions",
            "events",
            "adverse_post_edge_events",
            "adverse_post_edge_share",
            "mean_event_fill_probability",
            "mean_event_adverse_fill_probability",
            "mean_event_edge_ticks",
            "mean_pre_realized_edge_sum",
            "mean_post_realized_edge_sum",
            "mean_post_minus_pre_realized_edge",
            "worst_post_minus_pre_realized_edge",
            "mean_window_rows",
            "lifecycle_toxicity_label",
        ]
    )


def _empty_passive_fill_event_toxicity_scorecard() -> dict[str, float | int | str]:
    return {
        "rows": 0,
        "regimes": 0,
        "total_events": 0,
        "eligible_regimes": 0,
        "blocked_regimes": 0,
        "worst_regime": "none",
        "worst_adverse_post_edge_share": 0.0,
        "worst_mean_post_minus_pre_realized_edge": 0.0,
        "worst_post_minus_pre_realized_edge": 0.0,
        "weighted_mean_event_fill_probability": 0.0,
        "weighted_mean_event_adverse_fill_probability": 0.0,
        "weighted_mean_post_minus_pre_realized_edge": 0.0,
        "event_toxicity_label": "empty_event_windows",
    }


def _empty_passive_fill_event_transition_scorecard() -> dict[str, float | int | str]:
    return {
        "rows": 0,
        "transitions": 0,
        "total_events": 0,
        "eligible_transitions": 0,
        "blocked_transitions": 0,
        "worst_transition": "none",
        "worst_adverse_post_edge_share": 0.0,
        "worst_mean_post_minus_pre_realized_edge": 0.0,
        "worst_post_minus_pre_realized_edge": 0.0,
        "weighted_mean_event_fill_probability": 0.0,
        "weighted_mean_event_adverse_fill_probability": 0.0,
        "weighted_mean_post_minus_pre_realized_edge": 0.0,
        "transition_toxicity_label": "empty_transition_event_windows",
    }


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


def _empty_execution_adjusted_lcri_quantile_diagnostics() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "bucket",
            "rows",
            "mean_abs_lcri",
            "mean_abs_execution_adjusted_lcri_score",
            "signal_survival_ratio",
            "tradable_share",
            "mean_execution_adjusted_edge_ticks",
            "edge_drag_vs_raw_abs_lcri",
        ]
    )


def execution_adjusted_lcri_quantile_diagnostics(
    frame: pd.DataFrame,
    *,
    bins: int = 5,
    signal_col: str = "lcri",
    execution_signal_col: str = "execution_adjusted_lcri_score",
) -> pd.DataFrame:
    """Measure how much raw LCRI strength survives passive execution constraints.

    The table sorts rows by absolute raw LCRI, buckets them into equal-count quantiles,
    and reports the execution-adjusted signal retained after fill/adverse-fill gating.
    It makes the key publishability question auditable: are the largest residual
    imbalances still tradable after queue position, or is apparent alpha mostly eaten
    by passive-fill friction and abstention?
    """
    if not isinstance(bins, int) or isinstance(bins, bool) or bins < 1:
        raise ValueError("bins must be a positive integer")
    if frame.empty:
        return _empty_execution_adjusted_lcri_quantile_diagnostics()

    required = {
        signal_col,
        execution_signal_col,
        "execution_adjusted_edge_ticks",
        "best_execution_side",
    }
    _require_columns(frame, required, "execution-adjusted LCRI quantile diagnostics")
    values = _finite_values(
        frame,
        [signal_col, execution_signal_col, "execution_adjusted_edge_ticks"],
        "execution-adjusted LCRI quantile diagnostics",
    )

    diagnostics_frame = pd.DataFrame(index=frame.index)
    diagnostics_frame["abs_lcri"] = values[signal_col].abs()
    diagnostics_frame["abs_execution_adjusted_lcri_score"] = values[execution_signal_col].abs()
    diagnostics_frame["execution_adjusted_edge_ticks"] = values["execution_adjusted_edge_ticks"]
    diagnostics_frame["tradable"] = frame["best_execution_side"].astype(str) != "abstain"

    actual_bins = min(bins, len(diagnostics_frame))
    ranks = diagnostics_frame["abs_lcri"].rank(method="first")
    diagnostics_frame["bucket_id"] = pd.qcut(ranks, q=actual_bins, labels=False, duplicates="drop")
    actual_bins = int(diagnostics_frame["bucket_id"].max()) + 1

    if actual_bins == 1:
        labels = ["all_abs_lcri"]
    elif actual_bins == 3:
        labels = ["low_abs_lcri", "mid_abs_lcri", "high_abs_lcri"]
    else:
        labels = [f"abs_lcri_q{index + 1:02d}" for index in range(actual_bins)]

    rows: list[dict[str, float | int | str]] = []
    for bucket_id, group in diagnostics_frame.groupby("bucket_id", sort=True):
        mean_abs_lcri = float(group["abs_lcri"].mean())
        mean_abs_execution_signal = float(group["abs_execution_adjusted_lcri_score"].mean())
        rows.append(
            {
                "bucket": labels[int(bucket_id)],
                "rows": int(len(group)),
                "mean_abs_lcri": mean_abs_lcri,
                "mean_abs_execution_adjusted_lcri_score": mean_abs_execution_signal,
                "signal_survival_ratio": (
                    mean_abs_execution_signal / mean_abs_lcri if mean_abs_lcri > 0.0 else 0.0
                ),
                "tradable_share": float(group["tradable"].mean()),
                "mean_execution_adjusted_edge_ticks": float(
                    group["execution_adjusted_edge_ticks"].mean()
                ),
                "edge_drag_vs_raw_abs_lcri": float(
                    mean_abs_lcri - group["execution_adjusted_edge_ticks"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)[list(_empty_execution_adjusted_lcri_quantile_diagnostics().columns)]
